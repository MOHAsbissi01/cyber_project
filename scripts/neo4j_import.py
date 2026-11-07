"""Load the knowledge graph into Neo4j with the official Python driver, and (optionally) verify that the Cypher store answers like the
in-memory store.

    docker compose up -d                                  # starts Neo4j 5 (see docker-compose.yml), password from NEO4J_PASSWORD
    set NEO4J_PASSWORD=...                                # PowerShell: $env:NEO4J_PASSWORD="..."
    python scripts/neo4j_import.py --reset                # (re)load nodes + relationships
    python scripts/neo4j_import.py --check                # compare Neo4jStore vs NetworkXStore on sample queries

STATUS: written and syntax-checked, but NOT executed against a live Neo4j server in the delivery environment (Docker daemon was not running).
Run `--check` on your machine; it prints PASS/FAIL per query and exits non-zero on any mismatch.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyberintel.kg import SEMANTIC_RELATIONS  # noqa: E402
from cyberintel.pipeline import art  # noqa: E402
from cyberintel.rag.graph_store import NetworkXStore, Neo4jStore  # noqa: E402

REL_TYPES = set(SEMANTIC_RELATIONS) | {"MENTIONS", "SIMILAR_TO"}      # whitelist: relationship types are interpolated into Cypher
BATCH = 2000


def load(store: Neo4jStore, G, reset: bool):
    if reset:
        store._q("MATCH (n:Node) DETACH DELETE n")
    store._q("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
    nodes = [{"id": n, "type": d["type"], "label": d.get("label", ""), "category": d.get("category", ""), "n_incidents": int(d.get("n_incidents") or 0),
              "kb_name": d.get("kb_name", ""), "kb_summary": d.get("kb_summary", "")} for n, d in G.nodes(data=True)]
    for i in range(0, len(nodes), BATCH):
        store._q("UNWIND $rows AS r MERGE (n:Node {id: r.id}) SET n.type = r.type, n.label = r.label, n.category = r.category, "
                 "n.n_incidents = r.n_incidents, n.kb_name = r.kb_name, n.kb_summary = r.kb_summary", rows=nodes[i:i + BATCH])
    store._q("MATCH (n:Node) WHERE n.type = 'INCIDENT' SET n:Incident")
    store._q("MATCH (n:Node) WHERE n.type <> 'INCIDENT' SET n:Entity")
    by: dict[str, list] = {}
    for a, b, d in G.edges(data=True):
        by.setdefault(d["relation"], []).append({"s": a, "t": b, "w": float(d.get("weight", 1)), "support": int(d.get("support", 0)),
                                                 "lift": float(d.get("lift", 0)), "origin": d.get("source", ""), "status": d.get("status", ""),
                                                 "evidence": d.get("evidence", "") or ""})
    for rel, rows in by.items():
        assert rel in REL_TYPES, rel
        for i in range(0, len(rows), BATCH):
            store._q(f"UNWIND $rows AS r MATCH (a:Node {{id: r.s}}), (b:Node {{id: r.t}}) MERGE (a)-[e:{rel}]->(b) "
                     "SET e.weight = r.w, e.support = r.support, e.lift = r.lift, e.origin = r.origin, e.status = r.status, e.evidence = r.evidence",
                     rows=rows[i:i + BATCH])
        print(f"  {rel}: {len(rows)} relationships")
    print(f"loaded {len(nodes)} nodes")


def check(neo: Neo4jStore, mem: NetworkXStore) -> int:
    """Same question to both stores; compare the answers (sets of related entities / incident ids)."""
    probes = [("ATTACK:ROP", "MITIGATES", "in"), ("SOFTWARE:Windows", "AFFECTS", "in"), ("CVE:CVE-2016-5195", "AFFECTS", "out"),
              ("ATTACK:phishing", "TARGETS", "out"), ("MALWARE:Mimikatz", None, "both")]
    bad = 0
    for eid, rel, direction in probes:
        a = {(r["head"], r["relation"], r["tail"]) for r in neo.relations(eid, rel, direction, 500)}
        b = {(r["head"], r["relation"], r["tail"]) for r in mem.relations(eid, rel, direction, 500)}
        ok = a == b
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  relations({eid}, {rel}, {direction}): neo4j={len(a)} memory={len(b)}")
    for eid in ("MALWARE:Mimikatz", "ATTACK:XSS", "SOFTWARE:Windows"):
        a, b = {i for i, _ in neo.incidents_of(eid, 1000)}, {i for i, _ in mem.incidents_of(eid, 1000)}
        ok = a == b
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  incidents_of({eid}): neo4j={len(a)} memory={len(b)}")
    a = {i for i, _ in neo.similar_incidents(3, 8)}
    b = {i for i, _ in mem.similar_incidents(3, 8)}
    print(f"{'PASS' if a == b else 'FAIL'}  similar_incidents(3): neo4j={sorted(a)} memory={sorted(b)}")
    bad += a != b
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete existing :Node data first")
    ap.add_argument("--check", action="store_true", help="after loading (or alone with --no-load) compare with the in-memory store")
    ap.add_argument("--no-load", action="store_true")
    a = ap.parse_args()
    with open(art("kg.pkl"), "rb") as f:
        G = pickle.load(f)
    neo = Neo4jStore()
    if not a.no_load:
        load(neo, G, a.reset)
    if a.check:
        sys.exit(1 if check(neo, NetworkXStore(G)) else 0)
