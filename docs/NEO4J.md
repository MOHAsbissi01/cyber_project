# Neo4j (course Chapter 4) — status and procedure

## What exists

| Piece | File | Status |
|---|---|---|
| Graph export as CSV + `import.cypher` (`LOAD CSV`, one statement per relationship type) | `reports/export/neo4j/` (written by stage `kg`) | **generated and content-tested** (`tests/test_integration.py`), **not executed on a server** |
| Python loader (driver, batched `UNWIND`) | `scripts/neo4j_import.py` | syntax-checked, **not executed on a server** |
| Cypher retrieval layer implementing the `GraphStore` interface | `cyberintel/rag/graph_store.py::Neo4jStore` | written, **not executed on a server** |
| In-memory twin with identical outputs | `NetworkXStore` | tested; used by the app by default |
| Equivalence self-check (Neo4j answers == in-memory answers) | `python scripts/neo4j_import.py --check` | ready, to be run by you |

Why not tested: when this work was done, Docker Desktop was installed but its daemon was not running and no Neo4j server was installed; the choice made
with the student was "write the Cypher layer + in-memory fallback and be explicit that it was not run against a live server".

## How to run it (≈ 10 minutes)

```powershell
$env:NEO4J_PASSWORD = "choose-a-password"
docker compose up -d                                  # Neo4j 5 on bolt://localhost:7687, browser http://localhost:7474
python scripts/neo4j_import.py --reset --check        # loads 2.7k nodes / 26k relationships, then prints PASS/FAIL per query
streamlit run app/main.py                             # sidebar -> Backend settings -> "Use Neo4j for graph queries"
```
If the connection fails the app shows a banner and keeps working on the in-memory graph.

## Data model

* Every node has label `:Node` plus `:Incident` or `:Entity`; properties `id`, `type`, `label`, `n_incidents`, `category` (incidents), `kb_name`/`kb_summary` (CVE nodes).
* Relationship types: `MENTIONS` (Incident → Entity), `SIMILAR_TO` (Incident ↔ Incident), and the semantic relations
  `TARGETS`, `AFFECTS`, `EXPLOITS`, `ALIAS_OF`, `MAPS_TO`, `PART_OF`, `MITIGATES`, `USES` (Entity → Entity) with properties
  `support`, `lift`, `origin` (`corpus` | `reference_kb`), `status` (`verified` | `differs_from_kb` | `contradicted_by_kb` | `unverified` | `reference`), `evidence` (JSON).

## Example Cypher (also used by the retrieval layer)

```cypher
// CVEs that affect Windows, with provenance and the KB verdict
MATCH (c:Entity {type:'CVE'})-[e:AFFECTS]->(s:Entity {id:'SOFTWARE:Windows'})
RETURN c.label, e.origin, e.status, e.support ORDER BY e.support DESC;

// What mitigates ROP?
MATCH (d:Entity)-[e:MITIGATES]->(a:Entity {id:'ATTACK:ROP'}) RETURN d.label, e.support ORDER BY e.support DESC;

// Indirect connection between two entities (<= 4 semantic hops)
MATCH p = shortestPath((x:Entity {id:'ATTACK:Kerberoasting'})-[:TARGETS|USES|MITIGATES|AFFECTS|EXPLOITS*..4]-(y:Entity {id:'TECHNOLOGY:Kerberos'}))
RETURN [n IN nodes(p) | n.label], [r IN relationships(p) | type(r)];

// Incidents that mention an entity
MATCH (i:Incident)-[m:MENTIONS]->(e:Entity {id:'MALWARE:Mimikatz'}) RETURN i.id, i.label ORDER BY m.weight DESC LIMIT 10;
```
