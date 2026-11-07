"""Pipeline stages (registered into cyberintel.pipeline.STAGES). Each stage reads artifacts/ and writes artifacts/ + reports/."""
from __future__ import annotations

import json
import pickle
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from . import viz
from .config import cfg, path
from .pipeline import art, load_corpus, load_gold, load_json, register, rep, save_json

NER_TYPES = ["CVE", "MITRE_TECHNIQUE", "IP", "HASH", "MALWARE", "THREAT_ACTOR", "SOFTWARE", "TOOL", "TECHNOLOGY", "ATTACK", "VULN_NAME",
             "DEFENSE", "TACTIC"]   # TARGET (generic nouns) is excluded from NER evaluation


# ----------------------------------------------------------------------------- NER
@register("ner")
def stage_ner(**_):
    from .kg import build_mentions
    from .lexicon import MITRE_NAMES
    from .ner import evaluate_ner
    df = load_corpus()
    m = build_mentions(df)
    m.to_parquet(art("mentions.parquet"), index=False)
    uniq = m.drop_duplicates(["incident_id", "type", "name"])
    nz = uniq[uniq.type != "TARGET"].incident_id.nunique()
    mit = uniq[uniq.type == "MITRE_TECHNIQUE"].name.unique()
    out = {"corpus": {"incidents": len(df), "mentions": len(m), "mentions_by_type": m.type.value_counts().to_dict(),
                      "unique_entities_by_type": uniq.drop_duplicates(["type", "name"]).type.value_counts().to_dict(),
                      "incidents_with_specific_entity": int(nz), "incidents_without_specific_entity": int(len(df) - nz),
                      "unique_cves": int(uniq[uniq.type == "CVE"].name.nunique()),
                      "mitre_ids_found": int(len(mit)), "mitre_ids_with_name": int(sum(x in MITRE_NAMES for x in mit))}}
    gold = json.loads(path("annotations", "ner_gold.json").read_text(encoding="utf-8"))["passages"]
    out["gold_eval"] = {}
    for s, note in (("dev", "lexicon tuned on these errors: NOT a generalisation estimate"),
                    ("test", "optimistic: lexicon extended after reading these passages"),
                    ("final", "UNBIASED: drawn and annotated after the lexicon freeze")):
        r = evaluate_ner([p for p in gold if p["set"] == s], types=NER_TYPES)
        r["note"] = note
        out["gold_eval"][s] = r
    pa = json.loads(path("annotations", "ner_precision_audit.json").read_text(encoding="utf-8"))
    out["precision_audit"] = {k: pa[k] for k in ("n", "correct", "precision")} | {"note": pa["_comment"]}
    save_json(out, rep("ner.json"))
    f = out["gold_eval"]["final"]["overall"]
    print(f"[ner] {len(m)} mentions, {out['corpus']['unique_cves']} unique CVEs ; final-set P={f['precision']} R={f['recall']} F1={f['f1']}")


# ----------------------------------------------------------------------------- classification
@register("classify")
def stage_classify(**_):
    from .classify import preprocessing_ablation, train_and_evaluate
    from .labels import CLASSES
    df = load_corpus()
    classes = sorted(df.category.unique().tolist())
    gold = load_gold()
    res, serving = train_and_evaluate(df, classes, gold)
    joblib.dump(serving, art("clf_serving.joblib"))
    res["preprocessing_ablation"] = preprocessing_ablation(df, classes)
    # merge Transformer experiments when they exist (transformers_exp.py)
    for name in ("minilm_logreg", "distilbert"):
        r = load_json(rep(f"{name}.json"))
        preds = rep(f"{name}_preds.csv")
        if r is None:
            continue
        entry = {"val_macro_f1": r["val_macro_f1"], "test_weak": r["test_weak"], "params": r.get("params", {}),
                 "seconds": r.get("seconds"), "device": r.get("device", "cpu"), "base": r.get("encoder") or r.get("base_model")}
        if gold is not None and preds.exists():
            from .classify import metrics_block
            p = pd.read_csv(preds)
            g = gold.merge(p, on="id")
            idx = {c: i for i, c in enumerate(classes)}
            entry["test_gold"] = metrics_block(g.gold_label.map(idx).values, g.pred.map(idx).values, classes)
        res["models"][name] = entry
    res["gold_info"] = None if gold is None else {"n": int(len(gold)), "sources": gold.gold_source.value_counts().to_dict(),
                                                  "class_counts": gold.gold_label.value_counts().to_dict()}
    save_json(res, rep("classification.json"))
    rows = []
    for k, v in res["models"].items():
        rows.append({"model": k, "val_macro_f1": v["val_macro_f1"], "test_weak_macro_f1": v["test_weak"]["macro_f1"],
                     "test_weak_acc": v["test_weak"]["accuracy"],
                     "gold_macro_f1": v.get("test_gold", {}).get("macro_f1"), "gold_acc": v.get("test_gold", {}).get("accuracy")})
    tbl = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    tbl.to_csv(rep("classification_table.csv"), index=False)
    best = res["models"]["tfidf_logreg"]
    viz.plot_confusion(best["test_weak"]["confusion_matrix"], classes, rep("figures/cm_tfidf_logreg_weak.png"),
                       "TF-IDF + LogReg - test set (weak labels)")
    if "test_gold" in best:
        viz.plot_confusion(best["test_gold"]["confusion_matrix"], classes, rep("figures/cm_tfidf_logreg_gold.png"),
                           "TF-IDF + LogReg - gold subset of the test set")
    print(tbl.to_string(index=False))


@register("topics")
def stage_topics(**_):
    """LDA topic modelling (course Chap. 1 NLU task): unsupervised discovery, compared with the weak taxonomy."""
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import normalized_mutual_info_score
    df = load_corpus()
    cv = CountVectorizer(stop_words="english", min_df=8, max_df=0.4, max_features=8000, token_pattern=r"[a-zA-Z][a-zA-Z0-9\-]{2,}")
    X = cv.fit_transform(df.text)
    K = 10
    lda = LatentDirichletAllocation(n_components=K, random_state=cfg("data", "seed"), learning_method="batch", max_iter=25).fit(X)
    vocab = np.array(cv.get_feature_names_out())
    topics = [{"topic": k, "top_words": vocab[np.argsort(-lda.components_[k])[:10]].tolist()} for k in range(K)]
    dom = lda.transform(X).argmax(1)
    ct = pd.crosstab(dom, df.category)
    for t in topics:
        t["n_incidents"] = int((dom == t["topic"]).sum())
        t["dominant_weak_category"] = ct.loc[t["topic"]].idxmax() if t["topic"] in ct.index else ""
    save_json({"k": K, "topics": topics, "nmi_vs_weak_labels": round(float(normalized_mutual_info_score(df.category, dom)), 4),
               "note": "Unsupervised topics only partly align with the keyword taxonomy."}, rep("topics.json"))
    print("[topics] NMI vs weak labels:", round(float(normalized_mutual_info_score(df.category, dom)), 3))


# ----------------------------------------------------------------------------- knowledge graph
@register("kg")
def stage_kg(**_):
    from . import graph_analysis as ga
    from . import kg as K
    from .embeddings import Embedder
    df = load_corpus()
    mentions = pd.read_parquet(art("mentions.parquet"))
    emb = Embedder("tfidf").fit(df.text)                       # whole corpus: this is the production index (no labels involved)
    E = emb.encode(df.text)
    np.save(art("embeddings_lsa.npy"), E)
    joblib.dump(emb, art("embedder_lsa.joblib"))
    kb = K.load_kb()
    rel = K.extract_relations(df, mentions, kb)
    rel.to_parquet(art("relations.parquet"), index=False)
    A_sim, A_ent = K.knn_adjacency(E), K.entity_adjacency(mentions, len(df))
    import scipy.sparse as sp
    sp.save_npz(art("A_sim.npz"), A_sim)
    sp.save_npz(art("A_ent.npz"), A_ent)
    G = K.build_graph(df, mentions, rel, A_sim)
    with open(art("kg.pkl"), "wb") as f:
        pickle.dump(G, f)
    stats = ga.graph_stats(G)
    # incident clusters on the combined graph
    A = A_sim + A_ent
    comm = ga.incident_communities(A, seed=cfg("data", "seed"))
    from sklearn.feature_extraction.text import TfidfVectorizer
    tv = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5, sublinear_tf=True).fit(df.text)
    cdesc, cq = ga.describe_communities(df, comm, mentions, tv)
    cdesc.to_csv(rep("incident_communities.csv"), index=False)
    pd.DataFrame({"incident": df.id, "community": comm, "category": df.category}).to_csv(art("incident_clusters.csv"), index=False)
    np.save(art("communities.npy"), comm)
    ecomm = ga.entity_communities(G)
    ecomm.to_csv(rep("entity_communities.csv"), index=False)
    rt = ga.relations_table(G)
    rt.to_csv(rep("relations.csv"), index=False)
    stats["incident_communities"] = cq
    stats["relation_edges_by_name"] = rel.groupby(["relation", "source"]).size().rename("n").reset_index().to_dict("records")
    stats["claims_check"] = rel[rel.source == "corpus"].status.value_counts().to_dict()
    stats["similarity_graph"] = {"edges": int(A_sim.nnz // 2), "edge_homophily_weak_labels": round(_homophily(A_sim, df.category.values), 4)}
    stats["shared_entity_graph"] = {"edges": int(A_ent.nnz // 2), "edge_homophily_weak_labels": round(_homophily(A_ent, df.category.values), 4)}
    save_json(stats, rep("graph_stats.json"))
    ga.export_graphml(G, path("reports", "export/incident_graph.graphml", create=True))
    ga.export_neo4j(G, path("reports", "export/neo4j", create=True))
    viz.plot_entity_graph(K.entity_graph(G), rep("figures/entity_graph.png"), title="Cybersecurity knowledge graph - entities and relations")
    print(f"[kg] nodes={stats['nodes']} edges={stats['edges']} semantic={stats['semantic_edges']} comms={cq['n_communities']} "
          f"claims={stats['claims_check']}")


def _homophily(A, y) -> float:
    """Fraction of edges joining two nodes of the same (weak) label: how much a label-agnostic graph 'agrees' with the labels."""
    import scipy.sparse as sp
    C = sp.triu(A).tocoo()
    return float((y[C.row] == y[C.col]).mean()) if C.nnz else 0.0


# ----------------------------------------------------------------------------- graph ML (AA7)
@register("gnn")
def stage_gnn(**_):
    """Node classification: MLP vs GCN/SAGE/GAT/GTN under one protocol; graph and GTN ablations; label propagation; homophily."""
    import time

    import scipy.sparse as sp
    import torch

    from . import gnn as G
    from . import kg as K
    from .embeddings import Embedder
    torch.set_num_threads(max(2, (torch.get_num_threads() or 4)))
    df = load_corpus()
    classes = sorted(df.category.unique().tolist())
    y = df.category.map({c: i for i, c in enumerate(classes)}).values
    split = df.split.values
    seeds = cfg("gnn", "seeds")
    hp = {k: cfg("gnn", k) for k in ("hidden", "epochs", "lr", "dropout")} | {"wd": cfg("gnn", "weight_decay")}
    gold = load_gold()
    gidx = gy = None
    if gold is not None and len(gold):
        gidx = gold.id.values
        gy = gold.gold_label.map({c: i for i, c in enumerate(classes)}).values
    A_sim, A_ent = sp.load_npz(art("A_sim.npz")), sp.load_npz(art("A_ent.npz"))
    E_all = np.load(art("embeddings_lsa.npy"))
    feats = {"lsa": Embedder("tfidf").fit(df.text[split == "train"]).encode(df.text)}   # fitted on TRAIN only
    if art("embeddings_minilm.npy").exists():
        feats["minilm"] = np.load(art("embeddings_minilm.npy"))
    n = len(df)
    out = {"classes": classes, "hyperparameters": hp, "seeds": seeds, "n_nodes": n,
           "graphs": {"similarity": {"edges": int(A_sim.nnz // 2)}, "shared_entity": {"edges": int(A_ent.nnz // 2)}},
           "homophily": {}, "main": [], "graph_ablation": [], "gtn_ablation": [], "label_propagation": [],
           "protocol": "same split for all models; class-weighted loss; model selection on val macro-F1; mean+-std over seeds; "
                       "labels of val/test nodes never used for training (transductive)."}
    for name, A in (("similarity", A_sim), ("shared_entity", A_ent), ("sim+ent", A_sim + A_ent)):
        C = sp.triu(A).tocoo()
        out["homophily"][name] = {"edges": int(C.nnz), "edge_homophily": round(float((y[C.row] == y[C.col]).mean()), 4)}
    out["homophily"]["random_baseline"] = round(float(((np.bincount(y) / n) ** 2).sum()), 4)
    graphs = {"sim+ent": G.build_graph_inputs(A_sim, A_ent, n)}

    def run(model, fname, gname, graph, pe_dim=0, tag="", sd=None):
        t0, preds, hid, ebias = time.time(), [], None, None
        for s in (sd or seeds):
            r = G.train_node_classifier(model, feats[fname], y, split, graph, len(classes), s, pe_dim=pe_dim, **{
                "hidden": hp["hidden"], "epochs": hp["epochs"], "lr": hp["lr"], "wd": hp["wd"], "dropout": hp["dropout"]})
            preds.append(r["pred"]); hid = r["hidden"]; ebias = r["attn_edge_bias"]
        res = {"model": model + tag, "features": fname, "graph": gname, "seconds": round(time.time() - t0, 1),
               **G.summarize(preds, y, split, gidx, gy)}
        if ebias:
            res["learned_edge_type_bias_[sim,ent]_per_layer_per_head"] = ebias
        print(f"   {model+tag:10s} {fname:7s} {gname:8s} test-F1 {res['test_macro_f1']['mean']:.3f}+-{res['test_macro_f1']['std']:.3f}"
              + (f"  gold-F1 {res['gold_macro_f1']['mean']:.3f}" if 'gold_macro_f1' in res else ""), flush=True)
        return res, hid

    hidden_keep = {}
    # 1. main comparison
    for fname in feats:
        for model in ("mlp", "gcn", "sage", "gat", "gtn"):
            res, hid = run(model, fname, "none" if model == "mlp" else "sim+ent", graphs["sim+ent"])
            out["main"].append(res)
            hidden_keep[(model, fname)] = hid
    # 2. graph ablation (best available features)
    fb = "minilm" if "minilm" in feats else "lsa"
    variants = {"similarity(k=8)": G.build_graph_inputs(A_sim, None, n), "shared_entity": G.build_graph_inputs(None, A_ent, n),
                "similarity(k=4)": G.build_graph_inputs(K.knn_adjacency(E_all, k=4), None, n),
                "similarity(k=16)": G.build_graph_inputs(K.knn_adjacency(E_all, k=16), None, n)}
    for gname, g in variants.items():
        for model in ("gcn", "gtn"):
            res, _ = run(model, fb, gname, g, sd=seeds[:3])
            out["graph_ablation"].append(res)
    # 3. GTN ablation: positional encoding, depth
    gpe = G.build_graph_inputs(A_sim, A_ent, n, pe_k=8)
    res, _ = run("gtn", fb, "sim+ent", gpe, pe_dim=8, tag="+lapPE", sd=seeds[:3]); out["gtn_ablation"].append(res)
    # 4. label propagation (graph only, no features)
    for gname, A in (("similarity", A_sim), ("shared_entity", A_ent), ("sim+ent", A_sim + A_ent)):
        p = G.label_propagation(A, y, split, len(classes))
        out["label_propagation"].append({"graph": gname, **G.summarize([p], y, split, gidx, gy)})
    # figures + embeddings of the best graph model for the UI
    from sklearn.manifold import TSNE
    key = ("gtn", fb)
    Z = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(hidden_keep[key])
    np.save(art("gtn_tsne.npy"), Z)
    comm = np.load(art("communities.npy"))
    viz.plot_embedding_2d(Z, [f"c{c}" for c in comm], rep("figures/gtn_tsne_by_community.png"), "GTN hidden embeddings (t-SNE) coloured by graph community")
    viz.plot_embedding_2d(Z, df.category.tolist(), rep("figures/gtn_tsne_by_category.png"), "GTN hidden embeddings (t-SNE) coloured by weak category")
    names = [f"{r['model']}\n{r['features']}" for r in out["main"]]
    viz.plot_bar(names, [r["test_macro_f1"]["mean"] for r in out["main"]], rep("figures/gnn_comparison.png"),
                 "Node classification, test macro-F1 (weak labels), mean +- std over seeds", "macro-F1",
                 errs=[r["test_macro_f1"]["std"] for r in out["main"]])
    save_json(out, rep("gnn.json"))


# ----------------------------------------------------------------------------- RAG index
@register("rag_index")
def stage_rag_index(**_):
    """Chunk index: lexical TF-IDF over the full vocabulary + dense sentence-transformer embeddings (cached model)."""
    from .rag.retrieval import ChunkIndex
    df = load_corpus()
    chunks = pd.read_parquet(art("chunks.parquet"))
    dense = cfg("rag", "dense") if "dense" in load_config_section("rag") else "st"
    try:
        idx = ChunkIndex(chunks, dict(zip(df.id, df.question)), dense=dense)
    except Exception as e:  # noqa: BLE001 - e.g. model not downloadable offline: degrade to lexical-only and say so
        print(f"[rag_index] dense encoder unavailable ({type(e).__name__}); building lexical-only index")
        idx, dense = ChunkIndex(chunks, dict(zip(df.id, df.question)), dense="none"), "none"
    idx.save(art("rag_index.joblib"))
    print(f"[rag_index] {len(chunks)} chunks, dense={dense}")


def load_config_section(name):
    from .config import load_config
    return load_config()[name]


@register("linkpred")
def stage_linkpred(**_):
    from .linkpred import run_link_prediction
    with open(art("kg.pkl"), "rb") as f:
        G = pickle.load(f)
    mentions = pd.read_parquet(art("mentions.parquet"))
    res = run_link_prediction(G, mentions, np.load(art("embeddings_lsa.npy")))
    save_json(res, rep("linkpred.json"))
    print(f"[linkpred] {res['n_entities']} entities, {res['n_positive_edges']} edges")
    for k, v in res["methods"].items():
        print(f"   {k:28s} AUC {v['auc']['mean']:.3f}+-{v['auc']['std']:.3f}  AP {v['ap']['mean']:.3f}+-{v['ap']['std']:.3f}")


@register("qa_eval")
def stage_qa_eval(tag: str = "", **_):
    """Main set (dev+test, phrasing close to the router's patterns) and PARAPHRASE set (fresh phrasings, French) are reported separately."""
    from .rag.evaluation import evaluate
    from .service import CyberIntelService
    svc = CyberIntelService()
    qs = json.loads(path("annotations", "qa_testset.json").read_text(encoding="utf-8"))["questions"]
    res = {"main": evaluate(svc.engine, svc.df, [q for q in qs if q["split"] != "paraphrase"]),
           "paraphrase": evaluate(svc.engine, svc.df, [q for q in qs if q["split"] == "paraphrase"]),
           "thresholds": {"min_dense_score": cfg("rag", "min_dense_score"), "min_lexical_score": cfg("rag", "min_lexical_score"),
                          "note": "set a priori, not tuned; the unknown-term abstention rule was designed after seeing dev questions U01/U02"},
           "graph_backend": svc.store.name}
    save_json(res, rep(f"qa_eval{tag}.json"))
    f = lambda d: f"{d['mean']:.3f}" if d else "  -  "
    for part in ("main", "paraphrase"):
        print(f"[qa_eval:{part}] n={res[part]['n_questions']}  system   answer_correct          hit@5  P@5    MRR    abstain_recall false_abst grounded")
        for m, s_ in res[part]["systems"].items():
            print(f"   {m:8s} {s_['answer_correct']['mean']:.3f} [{s_['answer_correct']['ci95'][0]:.2f},{s_['answer_correct']['ci95'][1]:.2f}]   "
                  f"{f(s_['hit@5'])}  {f(s_['P@5'])}  {f(s_['MRR'])}   {s_['abstain_recall_on_unanswerable']}   {s_['false_abstention_rate']}   {s_['grounded_fraction_mean']}")
        print("   paired graph-hybrid:", res[part]["paired_diff_graph_minus_hybrid_answer_correct"], "| graph failures:",
              [x["id"] for x in res[part]["failures"]])
