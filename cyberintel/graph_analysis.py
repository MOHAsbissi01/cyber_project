"""Graph analysis: statistics, centrality, communities (clusters), and exports (GraphML for Gephi, CSV + Cypher for Neo4j)."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .kg import SEMANTIC_RELATIONS, entity_graph


# ----------------------------------------------------------------------------- statistics
def graph_stats(G: nx.MultiDiGraph) -> dict:
    types = Counter(d["type"] for _, d in G.nodes(data=True))
    rels = Counter(d["relation"] for _, _, d in G.edges(data=True))
    src = Counter(d["source"] for _, _, d in G.edges(data=True) if d["relation"] in SEMANTIC_RELATIONS)
    status = Counter(d.get("status", "") for _, _, d in G.edges(data=True) if d["relation"] in SEMANTIC_RELATIONS)
    U = nx.Graph(G)                                         # undirected view for connectivity
    comps = sorted((len(c) for c in nx.connected_components(U)), reverse=True)
    H = entity_graph(G)
    Hu = H.to_undirected()
    ecomps = sorted((len(c) for c in nx.connected_components(Hu)), reverse=True)
    pr = nx.pagerank(H, weight="weight") if H.number_of_edges() else {}
    hubs = sorted(((H.nodes[n]["label"], H.nodes[n]["type"], round(v, 4)) for n, v in pr.items()), key=lambda x: -x[2])[:12]
    deg = sorted(((H.nodes[n]["label"], H.nodes[n]["type"], int(Hu.degree(n))) for n in Hu), key=lambda x: -x[2])[:12]
    return {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "node_types": dict(types), "edge_relations": dict(rels),
        "semantic_edges": int(sum(rels[r] for r in SEMANTIC_RELATIONS)), "semantic_edge_sources": dict(src),
        "semantic_edge_status": dict(status),
        "connected_components": len(comps), "largest_component": comps[0] if comps else 0,
        "entity_graph": {"nodes": H.number_of_nodes(), "edges": H.number_of_edges(), "components": len(ecomps),
                         "largest_component": ecomps[0] if ecomps else 0, "density": round(nx.density(H), 5),
                         "isolated_entities": int(sum(1 for n in Hu if Hu.degree(n) == 0))},
        "pagerank_top": hubs, "degree_top": deg,
    }


# ----------------------------------------------------------------------------- communities
def _to_nx(A: sp.spmatrix) -> nx.Graph:
    A = sp.triu(A).tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(A.shape[0]))
    G.add_weighted_edges_from(zip(A.row.tolist(), A.col.tolist(), A.data.tolist()))
    return G


def incident_communities(A: sp.spmatrix, seed: int = 0, resolution: float = 1.0) -> np.ndarray:
    """Louvain clusters of incidents on the combined similarity + shared-entity graph. Returns community id per incident."""
    comms = nx.community.louvain_communities(_to_nx(A), weight="weight", seed=seed, resolution=resolution)
    comms = sorted(comms, key=len, reverse=True)
    lab = np.zeros(A.shape[0], dtype=int)
    for ci, c in enumerate(comms):
        lab[list(c)] = ci
    return lab


def describe_communities(df: pd.DataFrame, comm: np.ndarray, mentions: pd.DataFrame, text_vec=None, top: int = 6) -> tuple[pd.DataFrame, dict]:
    """One row per community: size, dominant (weak) category + purity, top specific entities, top TF-IDF terms."""
    d = df.assign(community=comm)
    ment = mentions[mentions.type != "TARGET"].merge(d[["id", "community"]], left_on="incident_id", right_on="id")
    terms = {}
    if text_vec is not None:
        X = text_vec.transform(d.text)
        vocab = np.array(text_vec.get_feature_names_out())
        for c in sorted(d.community.unique()):
            m = np.asarray(X[(d.community == c).values].mean(0)).ravel()
            terms[c] = vocab[np.argsort(-m)[:top]].tolist()
    rows = []
    for c, g in d.groupby("community"):
        cats = g.category.value_counts()
        ents = ment[ment.community == c].groupby(["type", "name"]).incident_id.nunique().sort_values(ascending=False).head(top)
        rows.append({"community": int(c), "n_incidents": len(g), "dominant_category": cats.index[0],
                     "category_purity": round(float(cats.iloc[0] / len(g)), 3),
                     "top_entities": "; ".join(f"{n} [{t}]" for (t, n) in ents.index),
                     "top_terms": ", ".join(terms.get(c, []))})
    out = pd.DataFrame(rows).sort_values("n_incidents", ascending=False)
    quality = {"n_communities": int(out.shape[0]),
               "nmi_vs_weak_labels": round(float(normalized_mutual_info_score(df.category, comm)), 4),
               "ari_vs_weak_labels": round(float(adjusted_rand_score(df.category, comm)), 4),
               "weighted_purity": round(float((out.category_purity * out.n_incidents).sum() / out.n_incidents.sum()), 4)}
    return out, quality


def entity_communities(G: nx.MultiDiGraph, seed: int = 0) -> pd.DataFrame:
    """Communities of entities on the semantic (entity-entity) graph: groups of related attacks / defences / software."""
    Hu = entity_graph(G).to_undirected()
    Hu.remove_nodes_from([n for n in list(Hu) if Hu.degree(n) == 0])
    comms = sorted(nx.community.louvain_communities(Hu, weight="weight", seed=seed), key=len, reverse=True)
    rows = []
    for ci, c in enumerate(comms):
        top = sorted(c, key=lambda n: -Hu.degree(n, weight="weight"))[:8]
        rows.append({"community": ci, "size": len(c), "members": "; ".join(f"{Hu.nodes[n]['label']} [{Hu.nodes[n]['type']}]" for n in top)})
    return pd.DataFrame(rows)


def relations_table(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for a, b, d in G.edges(data=True):
        if d["relation"] in SEMANTIC_RELATIONS:
            rows.append({"head": G.nodes[a]["label"], "head_type": G.nodes[a]["type"], "relation": d["relation"],
                         "tail": G.nodes[b]["label"], "tail_type": G.nodes[b]["type"], "support": d["support"], "lift": d["lift"],
                         "source": d["source"], "status": d.get("status", "")})
    return pd.DataFrame(rows).sort_values(["support", "lift"], ascending=False)


# ----------------------------------------------------------------------------- exports
def export_graphml(G: nx.MultiDiGraph, path: Path):
    H = nx.MultiDiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **{k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in d.items()})
    for a, b, k, d in G.edges(keys=True, data=True):
        if d["relation"] == "MENTIONS" or d["relation"] in SEMANTIC_RELATIONS or d["relation"] == "SIMILAR_TO":
            H.add_edge(a, b, key=k, **{kk: (vv if isinstance(vv, (int, float, str, bool)) else str(vv)) for kk, vv in d.items() if kk != "evidence"})
    nx.write_graphml(H, path)


def export_neo4j(G: nx.MultiDiGraph, outdir: Path):
    """nodes.csv + one edges_<REL>.csv per relation type + import.cypher (LOAD CSV). Relation types are static in Cypher,
    hence one statement per type. Same data is loaded by scripts/neo4j_import.py through the Python driver."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "label", "category", "n_incidents", "kb_name", "kb_summary", "question"])
        for n, d in G.nodes(data=True):
            w.writerow([n, d["type"], d.get("label", ""), d.get("category", ""), d.get("n_incidents", ""), d.get("kb_name", ""),
                        d.get("kb_summary", ""), d.get("label", "") if d["type"] == "INCIDENT" else ""])
    by = defaultdict(list)
    for a, b, d in G.edges(data=True):
        by[d["relation"]].append((a, b, d))
    cypher = ["// Generated by cyberintel.graph_analysis.export_neo4j. Copy the CSVs to Neo4j's import/ folder, then run this script.",
              "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;",
              "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS r",
              "MERGE (n:Node {id: r.id}) SET n.type = r.type, n.label = r.label, n.category = r.category,",
              "  n.n_incidents = toInteger(r.n_incidents), n.kb_name = r.kb_name, n.kb_summary = r.kb_summary;",
              "MATCH (n:Node) WHERE n.type = 'INCIDENT' SET n:Incident;",
              "MATCH (n:Node) WHERE n.type <> 'INCIDENT' SET n:Entity;"]
    for rel, items in sorted(by.items()):
        with open(out / f"edges_{rel}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "target", "weight", "support", "lift", "origin", "status", "evidence"])
            for a, b, d in items:
                w.writerow([a, b, d.get("weight", 1), d.get("support", 0), d.get("lift", 0), d.get("source", ""), d.get("status", ""),
                            d.get("evidence", "")])
        cypher += [f"LOAD CSV WITH HEADERS FROM 'file:///edges_{rel}.csv' AS r",
                   "MATCH (a:Node {id: r.source}), (b:Node {id: r.target})",
                   f"MERGE (a)-[e:{rel}]->(b) SET e.weight = toFloat(r.weight), e.support = toInteger(r.support), e.lift = toFloat(r.lift),",
                   "  e.origin = r.origin, e.status = r.status, e.evidence = r.evidence;"]
    cypher += ["// Example: which CVEs affect Windows, with provenance",
               "// MATCH (c:Entity {type:'CVE'})-[e:AFFECTS]->(s:Entity {id:'SOFTWARE:Windows'}) RETURN c.label, e.origin, e.status, e.support ORDER BY e.support DESC;"]
    (out / "import.cypher").write_text("\n".join(cypher) + "\n", encoding="utf-8")
