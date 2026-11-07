"""Data ingestion, cleaning, chunking, train/val/test split and data-quality report.

Design notes (see docs/PROJECT_DEFENSE_GUIDE.md):
* The dataset is a Q&A corpus, so one record = (question, answer). We call it an "incident record".
* Cleaning removes leaked chain-of-thought at *sentence* level instead of deleting whole rows, and keeps the FULL answer
  (the prototype truncated at 1,500 chars, which hid more than half of the CVE mentions).
* The split is created once, stratified, deterministic, and stored in the corpus (`split` column): every model uses it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import cfg, path
from .textutil import split_sentences

HF_DATASET = "AlicanKiraz0/Cybersecurity-Dataset-v1"
_Q_COLS = ["user", "question", "instruction", "prompt", "input"]
_A_COLS = ["assistant", "answer", "response", "output"]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.S | re.I)         # unclosed tag: the generation was cut inside the reasoning
_THINK_CLOSE_ONLY = re.compile(r"\A.*</think>", re.S | re.I)  # reasoning without an opening tag
_CODE_BLOCK = re.compile(r"```.*?```", re.S)
# first-person planning / meta-talk typical of leaked chain-of-thought; legitimate technical prose almost never contains it
_REASONING = re.compile(
    r"\b(the user (is|wants|asks|asked|needs|might|probably|mentioned|specifically)|I need to|I should|I'll|I will|I'm going to|"
    r"I want to|let me|let's (see|think|start|break)|I recall|I remember|I think|I wonder|I'm not sure|"
    r"hmm|okay,|alright,|wait,|maybe I|first, I|so, I|now, I)\b", re.I)


def clean_text(t: str, strip_reasoning: bool = True) -> tuple[str, int]:
    """Return (clean_text, number_of_reasoning_sentences_removed)."""
    t = str(t)
    t = _THINK_BLOCK.sub(" ", t)
    t = _THINK_OPEN.sub(" ", t)
    t = _THINK_CLOSE_ONLY.sub(" ", t)
    t = _CODE_BLOCK.sub(" [code] ", t)
    t = t.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').replace("\\'", "'")
    t = re.sub(r"\s+", " ", t).strip()
    removed = 0
    if strip_reasoning:
        keep = []
        for s in split_sentences(t):
            if _REASONING.search(s):
                removed += 1
            else:
                keep.append(s)
        t = " ".join(keep)
    return t, removed


def load_raw(source: str | Path | None = None) -> pd.DataFrame:
    """Load the raw dataset from a local file (.parquet/.csv/.jsonl) or from Hugging Face when source == 'hf'."""
    source = source or path("data")
    if str(source) == "hf":
        from datasets import load_dataset
        return load_dataset(HF_DATASET, split="train").to_pandas()
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}. Put the parquet there or run with source='hf' (needs `datasets`).")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".jsonl", ".json"):
        return pd.read_json(p, lines=True)
    return pd.read_csv(p)


def build_corpus(raw: pd.DataFrame, limit: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Clean and standardise -> DataFrame[id, question, answer, text, n_reasoning_removed]."""
    seed = cfg("data", "seed") if seed is None else seed
    qc = next(c for c in _Q_COLS if c in raw.columns)
    ac = next(c for c in _A_COLS if c in raw.columns)
    df = raw[[qc, ac]].rename(columns={qc: "question", ac: "answer"}).dropna().copy()
    q = [clean_text(x, strip_reasoning=False) for x in df.question]
    a = [clean_text(x) for x in df.answer]
    df["question"] = [x[0] for x in q]
    df["answer"] = [x[0] for x in a]
    df["n_reasoning_removed"] = [x[1] for x in a]
    n0 = len(df)
    df = df[(df.question.str.len() >= cfg("data", "min_question_chars")) & (df.answer.str.len() >= cfg("data", "min_answer_chars"))]
    df = df.drop_duplicates("question").drop_duplicates("answer").reset_index(drop=True)
    df.attrs["dropped_rows"] = n0 - len(df)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=seed).reset_index(drop=True)
    df.insert(0, "id", np.arange(len(df)))
    df["text"] = df["question"] + " " + df["answer"]     # FULL text, not truncated
    return df


def assign_splits(df: pd.DataFrame, labels: pd.Series, seed: int | None = None) -> pd.Series:
    """Deterministic stratified train/val/test assignment (shared by every model, no leakage across experiments)."""
    seed = cfg("data", "seed") if seed is None else seed
    test_size, val_size = cfg("data", "test_size"), cfg("data", "val_size")
    idx = np.arange(len(df))
    rest, test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size), random_state=seed, stratify=labels.iloc[rest])
    s = pd.Series("train", index=df.index)
    s.iloc[val] = "val"
    s.iloc[test] = "test"
    return s


def make_chunks(df: pd.DataFrame, target_chars: int | None = None) -> pd.DataFrame:
    """Sentence-aligned chunks of the answer (course Chap. 4: chunking). Chunk 0 always starts with the question."""
    target_chars = target_chars or cfg("data", "chunk_chars")
    rows = []
    for r in df.itertuples():
        sents = split_sentences(r.answer)
        cur, k = [], 0
        for s in sents:
            if cur and sum(len(x) + 1 for x in cur) + len(s) > target_chars:
                rows.append((r.id, k, " ".join(cur)))
                cur, k = [], k + 1
            cur.append(s)
        if cur:
            rows.append((r.id, k, " ".join(cur)))
    ch = pd.DataFrame(rows, columns=["incident_id", "order", "text"])
    ch.insert(0, "chunk_id", np.arange(len(ch)))
    return ch


def quality_report(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    """Facts about the dataset used on the dashboard, in the audit and in the report (all measured, none assumed)."""
    qc = next(c for c in _Q_COLS if c in raw.columns)
    ac = next(c for c in _A_COLS if c in raw.columns)
    ans = raw[ac].dropna().astype(str)
    full = (raw[qc].fillna("") + " " + raw[ac].fillna("")).astype(str)
    return {
        "raw_rows": int(len(raw)),
        "rows_after_cleaning": int(len(df)),
        "columns": list(raw.columns),
        "rows_with_null": int(raw.isna().any(axis=1).sum()),
        "duplicate_questions": int(raw[qc].duplicated().sum()),
        "answers_with_think_tag": int(ans.str.contains("<think>", case=False).sum()),
        "answers_with_unclosed_think": int((ans.str.contains("<think>", case=False) & ~ans.str.contains("</think>", case=False)).sum()),
        "rows_with_reasoning_sentences_removed": int((df.n_reasoning_removed > 0).sum()),
        "reasoning_sentences_removed_total": int(df.n_reasoning_removed.sum()),
        "rows_with_literal_backslash_n": int(ans.str.contains("\\n", regex=False).sum()),
        "rows_with_cve_in_raw_text": int(full.str.contains(r"CVE-\d{4}-\d{4,7}", regex=True).sum()),
        "rows_with_ip_in_raw_text": int(full.str.contains(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", regex=True).sum()),
        "question_chars_mean": round(float(df.question.str.len().mean()), 1),
        "answer_chars_mean": round(float(df.answer.str.len().mean()), 1),
        "answer_chars_p95": round(float(df.answer.str.len().quantile(0.95)), 1),
        "nature": "LLM-generated Q&A about security concepts (not real incident reports); no ground-truth labels.",
    }
