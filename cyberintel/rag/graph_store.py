"""Graph access layer for Graph-RAG. One interface, two interchangeable implementations that return identical structures:

* NetworkXStore : in-memory (default, offline, used by the UI and the tests)
* Neo4jStore    : Cypher queries on a Neo4j server (course Chap. 4). Load the graph with `python scripts/neo4j_import.py`.

QA code only talks to this interface, so Graph-RAG works the same with or without a database.
"""
from __future__ import annotations

import json
import os
from typing import Protocol

import networkx as nx

from ..config import cfg
from ..kg import SEMANTIC_RELATIONS, edge_evidence, entity_graph


def _parse_inc(node_id: str) -> int:
    return int(node_id.split(":", 1)[1])


class GraphStore(Protocol):
    name: str

    def has_entity(self, eid: str) -> bool: ...
    def entity(self, eid: str) -> dict | None: ...
    def find_entities(self, text: str, types: list[str] | None = None, limit: int = 10) -> list[dict]: ...
    def relations(self, eid: str, relation: str | None = None, direction: str = "both", limit: int = 30) -> list[dict]: ...
    def incidents_of(self, eid: str, limit: int = 100) -> list[tuple[int, float]]: ...
    def entities_of_incident(self, incident_id: int, exclude_types: tuple = ("TARGET",)) -> list[dict]: ...
    def similar_incidents(self, incident_id: int, k: int = 5) -> list[tuple[int, float]]: ...
    def path_between(self, a: str, b: str, max_len: int = 4) -> list[dict] | None: ...


# ============================================================================ in-memory
class NetworkXStore:
    name = "networkx"

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self._H = None

    @property
    def H(self) -> nx.DiGraph:
        if self._H is None:
            self._H = entity_graph(self.G)
        return self._H

    def has_entity(self, eid):
        return eid in self.G and self.G.nodes[eid]["type"] != "INCIDENT"

    def entity(self, eid):
        if not self.has_entity(eid):
            return None
        d = self.G.nodes[eid]
        return {"id": eid, **{k: d.get(k) for k in ("type", "label", "name", "n_incidents", "kb_name", "kb_summary", "kb_weakness")}}

    def find_entities(self, text, types=None, limit=10):
        t = text.lower()
        out = [{"id": n, "type": d["type"], "label": d["label"], "n_incidents": d.get("n_incidents", 0)}
               for n, d in self.G.nodes(data=True)
               if d["type"] != "INCIDENT" and (not types or d["type"] in types) and t in d["label"].lower()]
        return sorted(out, key=lambda x: -(x["n_incidents"] or 0))[:limit]

    def _edge(self, a, b, d, other, direction):
        return {"head": a, "tail": b, "relation": d["relation"], "source": d["source"], "support": d.get("support", 0),
                "lift": d.get("lift", 0.0), "status": d.get("status", ""), "kb_note": d.get("kb_note", ""),
                "evidence": edge_evidence(d), "other": other, "direction": direction,
                "other_label": self.G.nodes[other]["label"], "other_type": self.G.nodes[other]["type"]}

    def relations(self, eid, relation=None, direction="both", limit=30):
        if eid not in self.G:
            return []
        out = []
        if direction in ("out", "both"):
            for _, b, d in self.G.out_edges(eid, data=True):
                if d["relation"] in SEMANTIC_RELATIONS and (relation is None or d["relation"] == relation):
                    out.append(self._edge(eid, b, d, b, "out"))
        if direction in ("in", "both"):
            for a, _, d in self.G.in_edges(eid, data=True):
                if d["relation"] in SEMANTIC_RELATIONS and (relation is None or d["relation"] == relation):
                    out.append(self._edge(a, eid, d, a, "in"))
        out.sort(key=lambda r: (-r["support"], -r["lift"]))
        return out[:limit]

    def incidents_of(self, eid, limit=100):
        if eid not in self.G:
            return []
        res = [(_parse_inc(a), d["weight"]) for a, _, d in self.G.in_edges(eid, data=True) if d["relation"] == "MENTIONS"]
        return sorted(res, key=lambda x: -x[1])[:limit]

    def entities_of_incident(self, incident_id, exclude_types=("TARGET",)):
        n = f"INC:{incident_id}"
        if n not in self.G:
            return []
        return [{"id": b, "type": self.G.nodes[b]["type"], "label": self.G.nodes[b]["label"]} for _, b, d in self.G.out_edges(n, data=True)
                if d["relation"] == "MENTIONS" and self.G.nodes[b]["type"] not in exclude_types]

    def similar_incidents(self, incident_id, k=5):
        n = f"INC:{incident_id}"
        if n not in self.G:
            return []
        res = [(_parse_inc(b), d["weight"]) for _, b, d in self.G.out_edges(n, data=True) if d["relation"] == "SIMILAR_TO"]
        res += [(_parse_inc(a), d["weight"]) for a, _, d in self.G.in_edges(n, data=True) if d["relation"] == "SIMILAR_TO"]
        return sorted(res, key=lambda x: -x[1])[:k]

    def path_between(self, a, b, max_len=4):
        U = self.H.to_undirected()
        if a not in U or b not in U:
            return None
        try:
            p = nx.shortest_path(U, a, b)
        except nx.NetworkXNoPath:
            return None
        if len(p) - 1 > max_len:
            return None
        steps = []
        for x, y in zip(p, p[1:]):
            e = self.H[x][y] if self.H.has_edge(x, y) else self.H[y][x]
            steps.append({"from": self.G.nodes[x]["label"], "to": self.G.nodes[y]["label"], "relations": e["relations"], "support": e["weight"]})
        return steps


# ============================================================================ Neo4j
class Neo4jStore:
    """Cypher implementation. NOT exercised against a live server in this delivery's automated tests (no Neo4j available when it was
    written): see docs/NEO4J.md for the procedure to load the graph and verify. Returned structures match NetworkXStore."""
    name = "neo4j"

    def __init__(self, uri=None, user=None, password=None, database=None):
        from neo4j import GraphDatabase
        c = cfg("neo4j")
        self.db = database or c["database"]
        pw = password if password is not None else os.environ.get(c["password_env"], "")
        self.driver = GraphDatabase.driver(uri or c["uri"], auth=(user or c["user"], pw))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def _q(self, cypher: str, **params) -> list[dict]:
        with self.driver.session(database=self.db) as s:
            return [r.data() for r in s.run(cypher, **params)]

    def has_entity(self, eid):
        return bool(self._q("MATCH (n:Entity {id:$id}) RETURN n.id AS id LIMIT 1", id=eid))

    def entity(self, eid):
        r = self._q("MATCH (n:Entity {id:$id}) RETURN n.id AS id, n.type AS type, n.label AS label, n.n_incidents AS n_incidents, "
                    "n.kb_name AS kb_name, n.kb_summary AS kb_summary LIMIT 1", id=eid)
        return r[0] if r else None

    def find_entities(self, text, types=None, limit=10):
        return self._q("MATCH (n:Entity) WHERE toLower(n.label) CONTAINS toLower($t) AND ($types IS NULL OR n.type IN $types) "
                       "RETURN n.id AS id, n.type AS type, n.label AS label, n.n_incidents AS n_incidents "
                       "ORDER BY n.n_incidents DESC LIMIT $limit", t=text, types=types, limit=limit)

    def relations(self, eid, relation=None, direction="both", limit=30):
        if relation is not None and relation not in SEMANTIC_RELATIONS:       # relationship types cannot be Cypher parameters:
            raise ValueError(f"unknown relation type {relation!r}")           # interpolate only whitelisted names (no injection)
        rels = "|".join(SEMANTIC_RELATIONS) if relation is None else relation
        parts = []
        if direction in ("out", "both"):
            parts.append(f"MATCH (a:Entity {{id:$id}})-[e:{rels}]->(b:Entity) RETURN a.id AS head, b.id AS tail, b.id AS other, "
                         "b.label AS other_label, b.type AS other_type, type(e) AS relation, e.origin AS source, e.support AS support, "
                         "e.lift AS lift, e.status AS status, e.evidence AS evidence, 'out' AS direction")
        if direction in ("in", "both"):
            parts.append(f"MATCH (a:Entity)-[e:{rels}]->(b:Entity {{id:$id}}) RETURN a.id AS head, b.id AS tail, a.id AS other, "
                         "a.label AS other_label, a.type AS other_type, type(e) AS relation, e.origin AS source, e.support AS support, "
                         "e.lift AS lift, e.status AS status, e.evidence AS evidence, 'in' AS direction")
        rows = self._q(" UNION ".join(parts), id=eid)
        for r in rows:
            r["evidence"] = json.loads(r["evidence"]) if r.get("evidence") else []
            r["kb_note"] = ""
        rows.sort(key=lambda r: (-(r["support"] or 0), -(r["lift"] or 0)))
        return rows[:limit]

    def incidents_of(self, eid, limit=100):
        rows = self._q("MATCH (i:Incident)-[m:MENTIONS]->(e:Entity {id:$id}) RETURN i.id AS id, m.weight AS w ORDER BY w DESC LIMIT $limit",
                       id=eid, limit=limit)
        return [(_parse_inc(r["id"]), float(r["w"])) for r in rows]

    def entities_of_incident(self, incident_id, exclude_types=("TARGET",)):
        return self._q("MATCH (i:Incident {id:$id})-[:MENTIONS]->(e:Entity) WHERE NOT e.type IN $ex "
                       "RETURN e.id AS id, e.type AS type, e.label AS label", id=f"INC:{incident_id}", ex=list(exclude_types))

    def similar_incidents(self, incident_id, k=5):
        rows = self._q("MATCH (i:Incident {id:$id})-[s:SIMILAR_TO]-(j:Incident) RETURN j.id AS id, s.weight AS w ORDER BY w DESC LIMIT $k",
                       id=f"INC:{incident_id}", k=k)
        return [(_parse_inc(r["id"]), float(r["w"])) for r in rows]

    def path_between(self, a, b, max_len=4):
        rels = "|".join(SEMANTIC_RELATIONS)
        rows = self._q(f"MATCH p = shortestPath((x:Entity {{id:$a}})-[:{rels}*..{int(max_len)}]-(y:Entity {{id:$b}})) "
                       "RETURN [n IN nodes(p) | n.label] AS labels, [r IN relationships(p) | type(r)] AS rels", a=a, b=b)
        if not rows:
            return None
        L, R = rows[0]["labels"], rows[0]["rels"]
        return [{"from": L[i], "to": L[i + 1], "relations": [R[i]], "support": None} for i in range(len(R))]


def open_store(kind: str, G: nx.MultiDiGraph | None = None):
    """kind: 'networkx' | 'neo4j'. Falls back to networkx with a clear message when Neo4j is unreachable."""
    if kind == "neo4j":
        try:
            return Neo4jStore(), None
        except Exception as e:  # noqa: BLE001 - any connection/driver error must degrade gracefully
            if G is None:
                raise
            return NetworkXStore(G), f"Neo4j unavailable ({type(e).__name__}: {str(e)[:120]}); using the in-memory graph."
    return NetworkXStore(G), None
