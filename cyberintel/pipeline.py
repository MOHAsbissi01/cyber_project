"""Pipeline orchestrator: runs the stages and writes artifacts/ (cached objects) and reports/ (metrics, tables, figures).

    python -m cyberintel.pipeline corpus ner classify         # selected stages
    python -m cyberintel.pipeline all                         # everything (the UI only reads what this writes)

Stages are independent and idempotent; each one reads what the previous ones wrote (see STAGES order).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import cfg, path


# ----------------------------------------------------------------------------- io helpers
def art(name: str) -> Path:
    return path("artifacts", name, create=True)


def rep(name: str) -> Path:
    return path("reports", name, create=True)


def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_corpus() -> pd.DataFrame:
    p = art("corpus.parquet")
    if not p.exists():
        raise FileNotFoundError("artifacts/corpus.parquet missing: run `python -m cyberintel.pipeline corpus` first.")
    return pd.read_parquet(p)


def load_gold() -> pd.DataFrame | None:
    from .labels import load_gold as _lg
    return _lg(path("annotations", "gold_labels.csv"))


# ----------------------------------------------------------------------------- stages
def stage_corpus(source: str | None = None, limit: int | None = None):
    """Load -> clean -> weak labels -> stratified split -> chunks -> data-quality report."""
    from .data import assign_splits, build_corpus, load_raw, make_chunks, quality_report
    from .labels import CLASS_DEFINITIONS, weak_labels

    raw = load_raw(source)
    df = build_corpus(raw, limit=limit)
    df["category"] = weak_labels(df)
    df["split"] = assign_splits(df, df.category)
    df.to_parquet(art("corpus.parquet"), index=False)
    make_chunks(df).to_parquet(art("chunks.parquet"), index=False)
    q = quality_report(raw, df)
    q["class_definitions"] = CLASS_DEFINITIONS
    q["class_counts"] = df.category.value_counts().to_dict()
    q["split_counts"] = df.split.value_counts().to_dict()
    save_json(q, rep("data_quality.json"))
    print(f"[corpus] {len(raw)} raw -> {len(df)} rows ; classes={df.category.nunique()} ; splits={q['split_counts']}")


STAGES = {"corpus": stage_corpus}
# "all" = the fast, offline stages in dependency order. The Transformer experiments (`python -m cyberintel.transformers_exp minilm|distilbert`)
# need model downloads (and DistilBERT ~45 min on CPU), so they are run on request; `classify` merges their results when present.
ALL_ORDER = ["corpus", "ner", "topics", "classify", "kg", "rag_index", "gnn", "linkpred", "qa_eval"]


def register(name: str):
    def deco(fn):
        STAGES[name] = fn
        return fn
    return deco


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="*", default=["all"])
    ap.add_argument("--source", default=None, help="dataset path or 'hf' (default: config paths.data)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    from . import pipeline as P            # canonical module (when run with -m this file is __main__, a different copy)
    from . import pipeline_stages  # noqa: F401  (registers the remaining stages into P.STAGES)
    order = P.ALL_ORDER if "all" in a.stages else a.stages
    for s in order:
        if s not in P.STAGES:
            sys.exit(f"unknown stage {s!r}; available: {list(P.STAGES)}")
        t0 = time.time()
        P.STAGES[s](**({"source": a.source, "limit": a.limit} if s == "corpus" else {}))
        print(f"[{s}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
