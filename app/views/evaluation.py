"""Model evaluation: every number shown here is read from reports/*.json written by the pipeline (nothing is hard-coded)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import ui  # noqa: E402

FIG = ui.ROOT / "reports" / "figures"

ui.setup("Model evaluation", "📊")
svc = ui.service()
if svc is None:
    st.stop()
st.caption("Training, validation and test are strictly separated (one stratified split, hyper-parameters chosen on validation only). "
           "Classification is reported on the weak-label test set AND on an independent gold subset.")


def missing(stage: str):
    st.info(f"This report has not been generated yet. Run `python -m cyberintel.pipeline {stage}`.")


def mean_std(d):
    return f"{d['mean']:.3f} ± {d['std']:.3f}" if d else "-"


tab_cls, tab_ner, tab_gnn, tab_qa, tab_graph = st.tabs(["Classification", "NER", "Graph ML (GNN / GTN)", "Retrieval & QA", "Graph statistics"])

# ----------------------------------------------------------------------------- classification
with tab_cls:
    cls = svc.report("classification.json")
    if not cls:
        missing("classify")
    else:
        gi = cls.get("gold_info")
        if gi:
            st.markdown(f"**Gold subset:** {gi['n']} test-split records labelled independently of the keyword rules "
                        f"(annotator: {', '.join(gi['sources'])}; the student's own review is still pending).")
        rv = cls.get("rules_vs_gold")
        if rv:
            st.markdown(f"The keyword rules that produced the weak labels agree with the gold labels on **{rv['agreement']:.1%}** of records "
                        f"(Cohen's κ = {rv['cohen_kappa']:.2f}). Weak-label scores therefore overstate real topic accuracy.")
        rows = []
        for name, v in cls["models"].items():
            g = v.get("test_gold", {})
            rows.append({"model": name, "val macro-F1": v["val_macro_f1"], "test macro-F1 (weak)": v["test_weak"]["macro_f1"],
                         "test acc (weak)": v["test_weak"]["accuracy"], "gold macro-F1": g.get("macro_f1"), "gold acc": g.get("accuracy")})
        tbl = pd.DataFrame(rows).sort_values("val macro-F1", ascending=False)
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption("Model selection on **val**. 'weak' = agreement with the keyword rules; 'gold' = independent labels. "
                   "The gold subset is small (about 180 records), so differences below ~0.05 are within noise.")
        sel = st.selectbox("Inspect a model", [r["model"] for r in rows])
        v = cls["models"][sel]
        which = st.radio("Evaluated on", [w for w in ("test_weak", "test_gold") if w in v], horizontal=True,
                         format_func=lambda w: {"test_weak": "weak-label test set", "test_gold": "gold subset"}[w])
        pc = pd.DataFrame(v[which]["per_class"]).T[["precision", "recall", "f1-score", "support"]]
        st.dataframe(pc, width="stretch")
        cm_img = FIG / ("cm_tfidf_logreg_weak.png" if which == "test_weak" else "cm_tfidf_logreg_gold.png")
        if sel == "tfidf_logreg" and cm_img.exists():
            st.image(str(cm_img), caption="Confusion matrix (rows = true, columns = predicted)", width="stretch")
        pa = cls.get("preprocessing_ablation")
        if pa:
            st.markdown("**Effect of preprocessing (AA3)**")
            st.dataframe(pd.DataFrame({k: v for k, v in pa.items() if isinstance(v, dict)}).T, width="stretch")
            st.caption(pa["note"])
        tp = svc.report("topics.json")
        if tp:
            with st.expander("Unsupervised topics (LDA) vs the keyword taxonomy"):
                st.markdown(f"NMI between LDA topics and weak labels: **{tp['nmi_vs_weak_labels']}**")
                st.dataframe(pd.DataFrame([{"topic": t["topic"], "incidents": t["n_incidents"], "dominant weak label": t["dominant_weak_category"],
                                            "top words": ", ".join(t["top_words"][:8])} for t in tp["topics"]]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- NER
with tab_ner:
    ner = svc.report("ner.json")
    if not ner:
        missing("ner")
    else:
        rows = []
        for s, label in (("dev", "dev (lexicon tuned on these: not a generalisation estimate)"), ("test", "test (optimistic, see note)"),
                         ("final", "fresh sample, unbiased")):
            o = ner["gold_eval"][s]["overall"]
            rows.append({"set": label, "passages": ner["gold_eval"][s]["n_passages"], "precision": o["precision"], "recall": o["recall"],
                         "F1": o["f1"], "TP/FP/FN": f"{o['tp']}/{o['fp']}/{o['fn']}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pa = ner["precision_audit"]
        st.markdown(f"**Precision audit over {pa['n']} random mentions of all types:** {pa['correct']} correct → **{pa['precision']:.1%}**")
        st.caption("Reading: the rule-based NER is precise but its recall is limited by the closed vocabulary (long tail of product and attack names). "
                   "Entity-level metrics: an entity counts as correct if its type and canonical name match.")
        fin = ner["gold_eval"]["final"]
        st.markdown("**Per type (fresh sample)**")
        st.dataframe(pd.DataFrame(fin["per_type"]).T, width="stretch")
        with st.expander("Errors on the fresh sample"):
            st.dataframe(pd.DataFrame(fin["errors_sample"]), hide_index=True, width="stretch")
        c = ner["corpus"]
        st.markdown(f"**Corpus profile:** {c['mentions']:,} mentions · {c['unique_cves']} distinct CVE ids · "
                    f"{c['mitre_ids_with_name']}/{c['mitre_ids_found']} ATT&CK ids mapped to technique names")

# ----------------------------------------------------------------------------- graph ML
with tab_gnn:
    g = svc.report("gnn.json")
    lp = svc.report("linkpred.json")
    if not g:
        missing("gnn")
    else:
        st.markdown("**Task:** node classification of incidents. Same features, split and loss for every model; "
                    f"mean ± std over {len(g['seeds'])} seeds; labels of validation/test nodes never used for training.")
        rows = [{"model": r["model"], "features": r["features"], "graph": r["graph"], "test macro-F1 (weak)": mean_std(r["test_macro_f1"]),
                 "gold macro-F1": mean_std(r.get("gold_macro_f1")), "gold acc": mean_std(r.get("gold_acc"))} for r in g["main"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Why does the graph not help much? Edge homophily** (share of edges joining two incidents with the same weak label)")
        st.dataframe(pd.DataFrame([{"graph": k, **(v if isinstance(v, dict) else {"edge_homophily": v})} for k, v in g["homophily"].items()]),
                     hide_index=True, width="stretch")
        st.caption("A GNN smooths each node with its neighbours; that only helps if neighbours usually share the label. "
                   "The similarity graph is mostly redundant with the text features it was built from.")
        with st.expander("Graph and GTN ablations, label propagation"):
            ab = [{"model": r["model"], "graph": r["graph"], "features": r["features"], "test macro-F1": mean_std(r["test_macro_f1"]),
                   "gold macro-F1": mean_std(r.get("gold_macro_f1"))} for r in g["graph_ablation"] + g["gtn_ablation"]]
            st.dataframe(pd.DataFrame(ab), hide_index=True, width="stretch")
            st.markdown("Label propagation (graph only, no text):")
            st.dataframe(pd.DataFrame([{"graph": r["graph"], "test macro-F1": r["test_macro_f1"]["mean"],
                                        "gold macro-F1": r.get("gold_macro_f1", {}).get("mean")} for r in g["label_propagation"]]),
                         hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        for col, f, cap in ((c1, "gtn_tsne_by_community.png", "Graph-Transformer embeddings by graph community"),
                            (c2, "gtn_tsne_by_category.png", "Same embeddings by weak category")):
            if (FIG / f).exists():
                col.image(str(FIG / f), caption=cap, width="stretch")
    if lp:
        st.markdown("**Link prediction (relationship discovery)** on held-out entity–entity relations")
        st.dataframe(pd.DataFrame([{"method": k, "AUC": mean_std(v["auc"]), "AP": mean_std(v["ap"])} for k, v in lp["methods"].items()]),
                     hide_index=True, width="stretch")
        st.caption(f"{lp['n_entities']} entities, {lp['n_positive_edges']} relations; {lp['protocol']}. Structure-only heuristics are near chance because "
                   "the entity graph is sparse and star-shaped; the signal is in the semantic node features.")

# ----------------------------------------------------------------------------- QA
with tab_qa:
    qa = svc.report("qa_eval.json")
    pre = svc.report("qa_eval_before_fixes.json")
    if not qa:
        missing("qa_eval")
    else:
        st.markdown("Hand-written test set (≥ 50 questions) with relevance defined by regexes over the raw corpus (independent of the system's NER) "
                    "and gold answer keywords. **vector** = dense retrieval · **hybrid** = dense + lexical · **graph** = Graph-RAG.")
        which = st.radio("Question set", ["main", "paraphrase"], horizontal=True,
                         format_func=lambda x: {"main": "Main set (51 questions)", "paraphrase": "Paraphrase set (18 fresh phrasings, incl. French)"}[x])
        for label, src in (("After fixing the failures found (tuned on these questions)", qa), ("Before any fix (unbiased)", pre)):
            if not src:
                continue
            r = src[which]
            f = lambda d: f"{d['mean']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]" if d else "-"
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame([{"system": m, "answer correct": f(s["answer_correct"]), "hit@5": f(s["hit@5"]), "P@5": f(s["P@5"]),
                                        "MRR": f(s["MRR"]), "abstains when it should": s["abstain_recall_on_unanswerable"],
                                        "false abstentions": s["false_abstention_rate"], "grounded": s["grounded_fraction_mean"]}
                                       for m, s in r["systems"].items()]), hide_index=True, width="stretch")
        r = qa[which]
        d = r["paired_diff_graph_minus_hybrid_answer_correct"]
        st.markdown(f"Paired difference Graph-RAG − hybrid in answer correctness: **{d['mean']:+.2f}** (95% bootstrap CI {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f}, n={d['n']}).")
        st.markdown("**Answer correctness by question type**")
        st.dataframe(pd.DataFrame({k: {m: v[m]["answer_correct"] for m in v} for k, v in r["by_kind"].items()}).T, width="stretch")
        st.caption("The graph's advantage comes from structured questions (software ↔ vulnerability), where relations and the reference KB hold facts "
                   "that passage retrieval cannot assemble. On open topical questions the methods are close.")
        ui.caveat("Small test sets written by the system's author; grounding is 1.0 by construction for the extractive generator "
                  "(the check guards against bugs). Treat the numbers as evidence of behaviour on these questions, not as a general benchmark.")
        with st.expander(f"Remaining failures of Graph-RAG ({len(r['failures'])})"):
            ui.df_or_info(pd.DataFrame(r["failures"]), "No failure on this set after the fixes.")
        if pre and pre[which]["failures"]:
            with st.expander("Failures found BEFORE the fixes (these led to the rule changes)"):
                st.dataframe(pd.DataFrame(pre[which]["failures"]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- graph stats
with tab_graph:
    gs = svc.report("graph_stats.json")
    if not gs:
        missing("kg")
    else:
        c = st.columns(4)
        ui.kpi(c[0], "Nodes", f"{gs['nodes']:,}")
        ui.kpi(c[1], "Edges", f"{gs['edges']:,}")
        ui.kpi(c[2], "Semantic edges", gs["semantic_edges"], str(gs["semantic_edge_sources"]))
        ui.kpi(c[3], "Components", gs["connected_components"], f"largest {gs['largest_component']:,}")
        a, b = st.columns(2)
        a.markdown("**Node types**"); a.bar_chart(pd.Series(gs["node_types"]).sort_values(), horizontal=True)
        b.markdown("**Edge relations**"); b.bar_chart(pd.Series(gs["edge_relations"]).sort_values(), horizontal=True)
        eg = gs["entity_graph"]
        st.markdown(f"**Entity graph:** {eg['nodes']} nodes · {eg['edges']} edges · {eg['components']} components (largest {eg['largest_component']}) · "
                    f"density {eg['density']} · {eg['isolated_entities']} entities without any semantic relation")
        st.markdown("**Fact checking of corpus relations:** " + ", ".join(f"{k}: {v}" for k, v in gs["claims_check"].items()))
        if (FIG / "entity_graph.png").exists():
            st.image(str(FIG / "entity_graph.png"), caption="Highest-degree entities (colour = type, arrows = relation direction)", width="stretch")
