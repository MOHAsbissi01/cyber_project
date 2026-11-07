"""Pipeline complet S2 : donnees -> labels -> NER -> classification -> graphe -> communautes -> GCN -> Graph-RAG.

Exemples :
  python run_pipeline.py --source sample                 # test hors-ligne (donnees synthetiques)
  python run_pipeline.py --source hf                     # vrai dataset Hugging Face
  python run_pipeline.py --source hf --embedder st       # embeddings sentence-transformers
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

import graph_tools as gt
from cyberlib import Embedder, extract_entities, load_data, weak_label
from qa import GraphRAG, retrieval_eval

DEMO_QUESTIONS = [
    "Quelle vulnerabilite affecte log4j ?",
    "Which vulnerability affects apache?",
    "What does ransomware target?",
    "Quels incidents sont similaires a l'incident 3 ?",
    "How does a reflected XSS attack work in a browser?",
    "How should a SOC respond to phishing?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="sample")
    ap.add_argument("--embedder", default="tfidf", choices=["tfidf", "st"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--llm", action="store_true", help="generation via API Claude (ANTHROPIC_API_KEY requis)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    t0 = time.time()
    metrics = {}

    # 1. donnees + labels faibles + NER
    df = load_data(a.source, a.limit)
    df["category"] = [weak_label(q, r) for q, r in zip(df.question, df.answer)]
    df["entities"] = [extract_entities(t) for t in df.text]
    print(f"[1] {len(df)} incidents ; categories :\n{df.category.value_counts().to_string()}")
    ent_rows = [{"id": r.id, "type": t, "entity": n} for r in df.itertuples() for t, n in r.entities]
    ent_tbl = pd.DataFrame(ent_rows)
    ent_tbl.to_csv(out / "entities.csv", index=False)
    if len(ent_tbl):
        print("    entites par type :", ent_tbl.type.value_counts().to_dict())
    metrics["n_incidents"] = len(df)
    metrics["n_entity_mentions"] = len(ent_tbl)

    # 2. classification de textes (baseline TF-IDF + LogReg), entrainee sur la REPONSE seule
    #    (le label faible est derive surtout de la question -> limite la fuite triviale)
    lab = df.category.astype("category")
    y = lab.cat.codes.values
    names = list(lab.cat.categories)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=y if min(np.bincount(y)) > 1 else None)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    Xa = vec.fit_transform(df.answer)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xa[tr], y[tr])
    p = clf.predict(Xa[te])
    rep = classification_report(y[te], p, labels=range(len(names)), target_names=names, zero_division=0)
    (out / "classification_report.txt").write_text(rep, encoding="utf-8")
    metrics["tfidf_logreg"] = {"macro_f1": round(f1_score(y[te], p, average="macro"), 4),
                               "acc": round(float((p == y[te]).mean()), 4)}
    print("[2] TF-IDF+LogReg :", metrics["tfidf_logreg"])

    # 3. embeddings + graphes
    emb = Embedder(a.embedder)
    E = emb.fit_transform(df.text.tolist())
    G = gt.build_graph(df, min_cooc=2)
    A_sim, A_ent = gt.knn_adjacency(E), gt.entity_adjacency(df)
    gt.add_similarity_edges(G, A_sim)
    part = gt.detect_communities(G)
    comm = pd.DataFrame(gt.describe_communities(G, part))
    comm.to_csv(out / "communities.csv", index=False)
    rel = pd.DataFrame(gt.top_relations(G, 50))
    rel.to_csv(out / "top_relations.csv", index=False)
    pd.DataFrame([{"incident": int(n[4:]), "community": c, "category": G.nodes[n]["category"]}
                  for n, c in part.items() if n.startswith("INC:")]).to_csv(out / "incident_clusters.csv", index=False)
    metrics["graph"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "communities": len(comm)}
    print(f"[3] graphe : {metrics['graph']}")
    print(comm.head(8).to_string(index=False))
    gt.export_graphml(G, out / "incident_graph.graphml")
    gt.export_neo4j(G, out / "neo4j")
    gt.plot_entity_graph(G, part, out / "entity_graph.png")

    # 4. GNN (GCN) vs MLP : classification de noeuds-incidents
    A = A_sim + A_ent
    metrics["gnn"] = {}
    for name, use_graph in (("mlp_no_graph", False), ("gcn", True)):
        r = gt.train_node_classifier(E, A, y, len(names), use_graph=use_graph)
        metrics["gnn"][name] = {k: r[k] for k in ("val_acc", "test_acc", "test_macro_f1")}
        if use_graph:
            np.save(out / "gcn_hidden_embeddings.npy", r["hidden"])
    print("[4] GNN :", json.dumps(metrics["gnn"]))

    # 5. QA : RAG vs Graph-RAG
    rag = GraphRAG(df, G, emb, E)
    metrics["retrieval_hit@5_same_category"] = retrieval_eval(rag, df, te[:200])
    print("[5] retrieval hit@5 :", metrics["retrieval_hit@5_same_category"])
    md = ["# Demo QA (RAG vs Graph-RAG)\n"]
    for q in DEMO_QUESTIONS:
        for mode in ("vector", "graph"):
            r = rag.answer(q, mode, use_llm=a.llm)
            md += [f"## [{mode}] {q}", f"entites detectees: {r['entities']}", "", r["answer"], "",
                   "sources: " + str(r["sources"]), ("faits: " + " | ".join(r["facts"][:4])) if r["facts"] else "", ""]
    (out / "qa_demo.md").write_text("\n".join(md), encoding="utf-8")

    metrics["seconds"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Termine en {metrics['seconds']}s -> {out.resolve()}")


if __name__ == "__main__":
    main()
