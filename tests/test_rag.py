"""RAG unit tests on the tiny corpus (lexical-only index, in-memory graph store)."""
import pytest

from cyberintel import kg as K
from cyberintel.data import make_chunks
from cyberintel.rag.graph_store import NetworkXStore
from cyberintel.rag.qa import ExtractiveGenerator, QAEngine, verify_grounding
from cyberintel.rag.retrieval import ChunkIndex, rrf


@pytest.fixture(scope="module")
def engine(mini_df):
    m = K.build_mentions(mini_df)
    G = K.build_graph(mini_df, m, K.extract_relations(mini_df, m))
    idx = ChunkIndex(make_chunks(mini_df, 300), dict(zip(mini_df.id, mini_df.question)), dense="none")
    return QAEngine(mini_df, idx, NetworkXStore(G), None, ExtractiveGenerator())


def test_rrf_prefers_documents_ranked_high_in_both_lists():
    s = rrf([[1, 2, 3], [3, 1, 4]], k=10)
    assert s[1] > s[2] and s[1] > s[4] and s[3] > s[2]


def test_lexical_search_finds_rare_identifier(mini_df):
    idx = ChunkIndex(make_chunks(mini_df, 300), dict(zip(mini_df.id, mini_df.question)), dense="none")
    hits = idx.lexical("log4shell", 3)
    assert hits and idx.incident_of(hits[0][0]) == 0


def test_intent_detection(engine):
    for q, expected in [("Which vulnerabilities affect log4j?", "vulns_of_software"), ("Which software does CVE-2016-5195 affect?", "software_of_cve"),
                        ("How can ROP attacks be mitigated?", "mitigations"), ("What does phishing target?", "targets_of_attack"),
                        ("Which incidents are related to Mimikatz?", "incidents_of_entity"),
                        ("Which incidents are similar to incident 3?", "similar_incidents")]:
        ents = engine.link_entities(q)
        assert engine.detect_intent(q, ents)[0] == expected, q


def test_graph_answer_has_facts_with_provenance_and_is_grounded(engine):
    r = engine.answer("How can ROP attacks be mitigated?", "graph")
    assert not r.abstained and "ASLR mitigates ROP" in {f.text for f in r.facts}     # DEP/CFG are seen once only: below min_support
    assert all(f.evidence for f in r.facts if f.source == "corpus")
    assert r.grounding["grounded_fraction"] == 1.0


def test_kb_answers_even_when_corpus_has_a_single_mention(engine):
    r = engine.answer("Which software does CVE-2021-44228 affect?", "graph")
    assert not r.abstained and any("Log4j" in f.text for f in r.facts)


def test_abstains_on_unknown_or_offtopic_questions(engine):
    for q in ("Which vulnerability affects Zorbatron-9000 firmware?", "What is the capital of France?", "Which CVE affects SAP HANA 2.0?"):
        r = engine.answer(q, "graph")
        assert r.abstained and r.answer.startswith("I cannot answer") and not r.facts, q


def test_verify_grounding_detects_fabricated_sentences(engine, mini_df):
    r = engine.answer("How can ROP attacks be mitigated?", "graph")
    assert verify_grounding(r, mini_df)["grounded_fraction"] == 1.0
    r.passages[0].sentences = ["This sentence was invented and exists nowhere in the corpus."]
    assert verify_grounding(r, mini_df)["grounded_fraction"] < 1.0


def test_store_interface_paths_and_similarity(engine):
    st = engine.store
    assert st.has_entity("ATTACK:ROP") and not st.has_entity("ATTACK:does-not-exist")
    assert st.incidents_of("MALWARE:Mimikatz")
    assert st.relations("ATTACK:ROP", "MITIGATES", "in")


def test_neo4j_store_mapping_with_a_fake_driver():
    """No server here: this checks only the Python side of Neo4jStore (queries are parameterised, rows -> same structure as NetworkXStore,
    evidence JSON parsed, ordering). The Cypher itself is validated by `scripts/neo4j_import.py --check` on a live server."""
    import json

    from cyberintel.rag.graph_store import Neo4jStore
    seen = []

    def fake_q(cypher, **params):
        seen.append((cypher, params))
        if "RETURN a.id AS head" in cypher:
            return [{"head": "DEFENSE:ASLR", "tail": "ATTACK:ROP", "other": "DEFENSE:ASLR", "other_label": "ASLR", "other_type": "DEFENSE",
                     "relation": "MITIGATES", "source": "corpus", "support": 29, "lift": 2.7, "status": "unverified",
                     "evidence": json.dumps([{"incident": 7, "sentence": "ASLR mitigates ROP."}]), "direction": "in"},
                    {"head": "DEFENSE:DEP/NX", "tail": "ATTACK:ROP", "other": "DEFENSE:DEP/NX", "other_label": "DEP/NX", "other_type": "DEFENSE",
                     "relation": "MITIGATES", "source": "corpus", "support": 40, "lift": 2.0, "status": "unverified", "evidence": None,
                     "direction": "in"}]
        if "MENTIONS" in cypher:
            return [{"id": "INC:12", "w": 3.0}, {"id": "INC:5", "w": 1.0}]
        return []

    store = object.__new__(Neo4jStore)
    store._q = fake_q
    rels = store.relations("ATTACK:ROP", "MITIGATES", "in", 10)
    assert [r["other_label"] for r in rels] == ["DEP/NX", "ASLR"]                       # sorted by support
    assert rels[1]["evidence"][0]["incident"] == 7 and rels[0]["evidence"] == []         # JSON parsed, missing evidence -> []
    assert "MITIGATES" in seen[0][0] and seen[0][1] == {"id": "ATTACK:ROP"}              # parameterised, relationship type from the whitelist
    assert store.incidents_of("MALWARE:Mimikatz") == [(12, 3.0), (5, 1.0)]


def test_neo4j_store_rejects_non_whitelisted_relation_types():
    from cyberintel.rag.graph_store import Neo4jStore
    store = object.__new__(Neo4jStore)
    store._q = lambda *a, **k: []
    with pytest.raises(ValueError):
        store.relations("ATTACK:ROP", "MITIGATES]->() DETACH DELETE n //", "in")
