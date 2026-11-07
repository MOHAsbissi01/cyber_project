"""Backend service layer: the single object the UI (and any future API) talks to. It owns the cached artifacts and exposes
plain-Python methods returning JSON-like dicts / DataFrames. No UI code here; no heavy computation at start-up (artifacts are
built once by `python -m cyberintel.pipeline`)."""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from .classify import predict_proba
from .config import cfg, path
from .kg import SEMANTIC_RELATIONS, edge_evidence, entity_graph, nid
from .ner import extract_entities
from .pipeline import art, load_json, rep
from .preprocessing import preprocessing_demo
from .rag.graph_store import NetworkXStore, open_store
from .rag.qa import QAEngine, make_generator
from .rag.retrieval import ChunkIndex

REQUIRED = ["corpus.parquet", "mentions.parquet", "kg.pkl", "rag_index.joblib", "clf_serving.joblib", "embeddings_lsa.npy",
            "embedder_lsa.joblib", "relations.parquet", "communities.npy"]


class ArtifactsMissing(RuntimeError):
    pass


class CyberIntelService:
    def __init__(self, store: str = "networkx", llm: str | None = None):
        missing = [f for f in REQUIRED if not art(f).exists()]
        if missing:
            raise ArtifactsMissing("Missing artifacts: " + ", ".join(missing) + ". Run `python -m cyberintel.pipeline all` first.")
        self.df = pd.read_parquet(art("corpus.parquet"))
        self.mentions = pd.read_parquet(art("mentions.parquet"))
        with open(art("kg.pkl"), "rb") as f:
            self.G: nx.MultiDiGraph = pickle.load(f)
        self.E = np.load(art("embeddings_lsa.npy"))
        self.embedder = joblib.load(art("embedder_lsa.joblib"))
        self.clf = joblib.load(art("clf_serving.joblib"))
        self.communities = np.load(art("communities.npy"))
        self.relations = pd.read_parquet(art("relations.parquet"))
        self.index = ChunkIndex.load(art("rag_index.joblib"))
        self.store, self.store_note = open_store(store, self.G)
        self.engine = QAEngine(self.df, self.index, self.store, self.E, make_generator(llm))
        self._H = None

    # ------------------------------------------------------------------ reports (written by the pipeline)
    def report(self, name: str):
        return load_json(rep(name))

    # ------------------------------------------------------------------ dashboard / dataset
    def overview(self) -> dict:
        gs = self.report("graph_stats.json") or {}
        ner = self.report("ner.json") or {}
        return {
            "incidents": len(self.df), "mentions": len(self.mentions),
            "unique_entities": int(self.mentions.drop_duplicates(["type", "name"]).shape[0]),
            "specific_entities": int(self.mentions[self.mentions.type != "TARGET"].drop_duplicates(["type", "name"]).shape[0]),
            "graph_nodes": self.G.number_of_nodes(), "graph_edges": self.G.number_of_edges(),
            "semantic_relations": gs.get("semantic_edges"), "communities": int(self.communities.max() + 1),
            "category_counts": self.df.category.value_counts().to_dict(),
            "entity_type_counts": self.mentions.drop_duplicates(["type", "name"]).type.value_counts().to_dict(),
            "claims_check": gs.get("claims_check", {}), "graph_stats": gs, "ner": ner.get("corpus", {}),
            "graph_backend": self.store.name, "graph_backend_note": self.store_note,
        }

    def dataset_page(self, category: str | None = None, split: str | None = None, query: str | None = None,
                     limit: int = 200) -> pd.DataFrame:
        d = self.df
        if category:
            d = d[d.category == category]
        if split:
            d = d[d.split == split]
        if query:
            d = d[d.text.str.contains(query, case=False, regex=False)]
        out = d[["id", "category", "split", "question"]].head(limit).copy()
        out["chars"] = d.answer.str.len().head(limit).values
        return out

    def incident(self, i: int) -> dict:
        r = self.df.iloc[i]
        ents = self.store.entities_of_incident(i, exclude_types=())
        return {"id": int(r.id), "question": r.question, "answer": r.answer, "category": r.category, "split": r.split,
                "community": int(self.communities[i]), "entities": ents, "similar": self.store.similar_incidents(i, 5)}

    # ------------------------------------------------------------------ incident analysis
    def analyze_incident(self, text: str, k: int = 5) -> dict:
        if not text or not text.strip():
            raise ValueError("Empty text.")
        probs = predict_proba(self.clf, [text]).iloc[0].sort_values(ascending=False)
        ents = extract_entities(text)
        grouped: dict[str, list[str]] = defaultdict(list)
        for e in ents:
            if e.name not in grouped[e.type]:
                grouped[e.type].append(e.name)
        # similar incidents: hybrid retrieval over chunks, aggregated per incident (max chunk score)
        best: dict[int, float] = {}
        for row, score, *_ in self.index.hybrid(text, 60):
            inc = self.index.incident_of(row)
            best[inc] = max(best.get(inc, 0.0), score)
        top = sorted(best.items(), key=lambda x: -x[1])[:k]
        norm = max((s for _, s in top), default=1.0)
        similar = [{"id": i, "relevance": round(s / norm, 3), "category": self.df.category[i], "question": self.df.question[i]} for i, s in top]
        # knowledge-graph context for the recognised entities (relations with provenance)
        context = []
        for e in ents:
            eid = nid(e.type, e.name)
            if e.type in ("TARGET",) or not self.store.has_entity(eid):
                continue
            for r in self.store.relations(eid, None, "both", 3):
                context.append({"entity": e.name, "relation": r["relation"], "head": self.G.nodes[r["head"]]["label"],
                                "tail": self.G.nodes[r["tail"]]["label"], "support": r["support"], "source": r["source"], "status": r["status"]})
        seen, dedup = set(), []
        for c in context:
            key = (c["head"], c["relation"], c["tail"])
            if key not in seen:
                seen.add(key); dedup.append(c)
        return {"category": probs.index[0], "probabilities": probs.round(4).to_dict(), "confidence": float(probs.iloc[0]),
                "entities": dict(grouped), "similar": similar, "graph_context": dedup[:12], "preprocessing": preprocessing_demo(text[:300]),
                "top_terms": (self.report("classification.json") or {}).get("top_terms", {}).get(probs.index[0], [])}

    # ------------------------------------------------------------------ knowledge graph
    @property
    def H(self) -> nx.DiGraph:
        if self._H is None:
            self._H = entity_graph(self.G)
        return self._H

    def search_entities(self, query: str = "", types: list[str] | None = None, limit: int = 50) -> pd.DataFrame:
        rows = [{"id": n, "label": d["label"], "type": d["type"], "incidents": d.get("n_incidents", 0),
                 "relations": int(self.H.degree(n)) if n in self.H else 0}
                for n, d in self.G.nodes(data=True)
                if d["type"] != "INCIDENT" and (not types or d["type"] in types) and query.lower() in d["label"].lower()]
        return pd.DataFrame(rows).sort_values(["relations", "incidents"], ascending=False).head(limit) if rows else pd.DataFrame(
            columns=["id", "label", "type", "incidents", "relations"])

    def entity_profile(self, eid: str) -> dict | None:
        meta = self.store.entity(eid)
        if not meta:
            return None
        rels = self.store.relations(eid, None, "both", 60)
        incs = self.store.incidents_of(eid, 15)
        return {"entity": meta, "relations": rels,
                "incidents": [{"id": i, "mentions": int(w), "category": self.df.category[i], "question": self.df.question[i]} for i, w in incs],
                "n_incidents": meta.get("n_incidents", 0)}

    def subgraph(self, eid: str, max_nodes: int = 25) -> nx.DiGraph:
        """Entity neighbourhood (semantic relations only) for the interactive graph view."""
        H = self.H
        if eid not in H:
            return nx.DiGraph()
        nbrs = set(H.successors(eid)) | set(H.predecessors(eid))
        nbrs = set(sorted(nbrs, key=lambda n: -max(H[eid][n]["weight"] if H.has_edge(eid, n) else 0, H[n][eid]["weight"] if H.has_edge(n, eid) else 0))[:max_nodes])
        return H.subgraph({eid} | nbrs).copy()

    def global_entity_graph(self, types: list[str] | None = None, min_support: int = 1, max_nodes: int = 80) -> nx.DiGraph:
        H = self.H.copy()
        H.remove_nodes_from([n for n, d in H.nodes(data=True) if types and d["type"] not in types])
        H.remove_edges_from([(a, b) for a, b, d in H.edges(data=True) if d["weight"] < min_support])
        H.remove_nodes_from([n for n in list(H) if H.degree(n) == 0])
        keep = sorted(H.nodes, key=lambda n: -H.degree(n, weight="weight"))[:max_nodes]
        return H.subgraph(keep).copy()

    def communities_table(self) -> pd.DataFrame:
        p = rep("incident_communities.csv")
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def community_incidents(self, c: int, limit: int = 25) -> pd.DataFrame:
        d = self.df[self.communities == c]
        return d[["id", "category", "question"]].head(limit)

    def relations_table(self, relation: str | None = None, source: str | None = None, status: str | None = None) -> pd.DataFrame:
        r = self.relations
        if relation:
            r = r[r.relation == relation]
        if source:
            r = r[r.source == source]
        if status:
            r = r[r.status == status]
        return r[["head_type", "head", "relation", "tail_type", "tail", "source", "support", "lift", "status"]]

    # ------------------------------------------------------------------ question answering
    def answer(self, question: str, mode: str = "graph", k: int = 5) -> dict:
        if not question or not question.strip():
            raise ValueError("Empty question.")
        return self.engine.answer(question, mode, k).to_dict()

    def compare(self, question: str, k: int = 5) -> dict[str, dict]:
        return {m: self.answer(question, m, k) for m in ("vector", "hybrid", "graph")}
