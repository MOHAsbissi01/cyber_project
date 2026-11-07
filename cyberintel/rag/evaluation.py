"""RAG vs Graph-RAG evaluation on the hand-written test set (annotations/qa_testset.json). Course Chap. 4 metrics:
retriever = Precision@k / Recall@k; generator/answer = correctness vs gold keywords, faithfulness = grounding check.

Metric definitions
  hit@5, P@5, MRR : over incidents returned as sources (relevant = regex-defined set, independent of the system's NER)
  R@5 (capped)    : |relevant ∩ top5| / min(|relevant|, 5)   (plain recall@5 is meaningless when |relevant| >> 5)
  answer_correct  : every keyword group has >= 1 alternative present in the answer text (case-insensitive)
  abstention      : on unanswerable questions the system must refuse; on answerable ones it must not
  grounded        : share of cited claims that are verbatim in the cited source (verify_grounding)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

MODES = ("vector", "hybrid", "graph")


def relevant_ids(df: pd.DataFrame, item: dict) -> set[int]:
    if not item.get("relevant_regex"):
        return set()
    col = df.question if item["relevance_field"] == "question" else df.text
    return set(df.id[col.str.contains(item["relevant_regex"], case=False, regex=True)].tolist())


def keyword_score(answer: str, groups: list[list[str]]) -> tuple[bool, float]:
    if not groups:
        return True, 1.0
    a = answer.lower()
    hits = [any(k.lower() in a for k in g) for g in groups]
    return all(hits), sum(hits) / len(hits)


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    m = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(engine, df: pd.DataFrame, questions: list[dict], k: int = 5) -> dict:
    rows = []
    for item in questions:
        rel = relevant_ids(df, item)
        for mode in MODES:
            t0 = time.time()
            r = engine.answer(item["question"], mode, k)
            lat = time.time() - t0
            row = {"id": item["id"], "kind": item["kind"], "split": item["split"], "mode": mode, "abstained": r.abstained,
                   "expect_abstain": item["expect_abstain"], "kb_only": item.get("kb_only", False), "latency_s": lat, "intent": r.intent,
                   "expected_intent": item.get("expected_intent"), "grounded": (r.grounding or {}).get("grounded_fraction"),
                   "n_facts": len(r.facts)}
            if not item["expect_abstain"]:
                ok, part = keyword_score(r.answer if not r.abstained else "", item["answer_keywords"])
                row["answer_correct"], row["answer_partial"] = bool(ok and not r.abstained), part if not r.abstained else 0.0
                if rel and not row["kb_only"]:
                    top = r.incidents[:k]
                    hits = [i in rel for i in top]
                    row["hit@5"] = float(any(hits))
                    row["P@5"] = float(sum(hits) / k)
                    row["R@5"] = float(sum(hits) / min(len(rel), k))
                    row["MRR"] = next((1.0 / (p + 1) for p, h in enumerate(hits) if h), 0.0)
            rows.append(row)
    R = pd.DataFrame(rows)
    out = {"n_questions": len(questions), "k": k, "systems": {}, "by_kind": {}, "failures": []}
    for mode in MODES:
        m = R[R["mode"] == mode]
        ans = m[~m.expect_abstain]
        un = m[m.expect_abstain]
        ret = ans.dropna(subset=["hit@5"]) if "hit@5" in ans else ans.iloc[0:0]
        s = {"answerable": int(len(ans)), "unanswerable": int(len(un)),
             "answer_correct": _mean_ci(ans.answer_correct.astype(float)), "answer_partial": _mean_ci(ans.answer_partial),
             "false_abstention_rate": round(float(ans.abstained.mean()), 4) if len(ans) else None,
             "abstain_recall_on_unanswerable": round(float(un.abstained.mean()), 4) if len(un) else None,
             "retrieval_questions": int(len(ret)),
             "hit@5": _mean_ci(ret["hit@5"]) if len(ret) else None, "P@5": _mean_ci(ret["P@5"]) if len(ret) else None,
             "R@5_capped": _mean_ci(ret["R@5"]) if len(ret) else None, "MRR": _mean_ci(ret["MRR"]) if len(ret) else None,
             "grounded_fraction_mean": round(float(m.grounded.dropna().mean()), 4) if m.grounded.notna().any() else None,
             "mean_latency_s": round(float(m.latency_s.mean()), 3)}
        if mode == "graph":
            sub = m[m.expected_intent.notna()]
            s["intent_accuracy"] = round(float((sub.intent == sub.expected_intent).mean()), 4) if len(sub) else None
        out["systems"][mode] = s
        for kind, g in ans.groupby("kind"):
            out["by_kind"].setdefault(kind, {})[mode] = {"n": int(len(g)), "answer_correct": round(float(g.answer_correct.mean()), 3)}
    # paired difference graph - hybrid (answer correctness), bootstrap over questions
    a = R[(R["mode"] == "graph") & ~R.expect_abstain].set_index("id").answer_correct.astype(float)
    b = R[(R["mode"] == "hybrid") & ~R.expect_abstain].set_index("id").answer_correct.astype(float)
    d = (a - b).dropna().values
    lo, hi = bootstrap_ci(list(d))
    out["paired_diff_graph_minus_hybrid_answer_correct"] = {"mean": round(float(d.mean()), 4), "ci95": [round(lo, 3), round(hi, 3)], "n": int(len(d))}
    for mode in ("graph",):
        for item in questions:
            r = R[(R.id == item["id"]) & (R["mode"] == mode)].iloc[0]
            bad = (item["expect_abstain"] and not r.abstained) or (not item["expect_abstain"] and (r.abstained or not r.answer_correct))
            if bad:
                ans = engine.answer(item["question"], mode, k)
                out["failures"].append({"id": item["id"], "question": item["question"], "expect_abstain": item["expect_abstain"],
                                        "abstained": bool(r.abstained), "intent": r.intent, "answer": ans.answer[:300]})
    out["per_question"] = R.drop(columns=["grounded"]).round(4).to_dict("records")
    return out


def _mean_ci(x) -> dict:
    v = [float(i) for i in x if pd.notna(i)]
    lo, hi = bootstrap_ci(v)
    return {"mean": round(float(np.mean(v)), 4) if v else None, "ci95": [round(lo, 3), round(hi, 3)], "n": len(v)}
