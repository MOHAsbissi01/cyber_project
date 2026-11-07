"""Integration tests on the BUILT system (artifacts + reports). Skipped when `python -m cyberintel.pipeline all` has not been run."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_split_integrity_and_label_sources(svc):
    df = svc.df
    assert set(df.split) == {"train", "val", "test"} and df.id.is_unique and (df.id.values == range(len(df))).all()
    from cyberintel.labels import load_gold
    gold = load_gold(ROOT / "annotations" / "gold_labels.csv")
    assert gold is not None and len(gold) > 100
    assert (df.set_index("id").loc[gold.id, "split"] == "test").all()            # gold rows come from the TEST split only


def test_service_overview_and_dataset(svc):
    ov = svc.overview()
    assert ov["incidents"] == len(svc.df) and ov["semantic_relations"] > 100 and ov["communities"] > 3
    assert len(svc.dataset_page(category="xss", limit=5)) == 5


def test_incident_analysis_end_to_end(svc):
    r = svc.analyze_incident("Attackers exploited CVE-2021-44228 in Log4j and used Mimikatz to dump credentials on a Windows server.")
    assert r["category"] in svc.clf["classes"] and abs(sum(r["probabilities"].values()) - 1) < 1e-3
    assert "CVE" in r["entities"] and "MALWARE" in r["entities"] and r["similar"] and r["graph_context"]
    with pytest.raises(ValueError):
        svc.analyze_incident("   ")


def test_graph_rag_beats_plain_retrieval_on_structured_questions(svc):
    # the CVE of OpenSSL's flaw (Heartbleed) is only in the reference KB: passage retrieval cannot produce it, the graph can
    q = "Which CVE affects OpenSSL?"
    g, h = svc.answer(q, "graph"), svc.answer(q, "hybrid")
    assert "CVE-2014-0160" in g["answer"] and "CVE-2014-0160" not in h["answer"]
    assert g["facts"] and not h["facts"] and any(f["source"] in ("reference_kb", "corpus+reference_kb") for f in g["facts"])


def test_contradictions_in_the_corpus_are_flagged(svc):
    r = svc.answer("What is EternalBlue?", "graph")
    assert any(f["status"] == "contradicted_by_kb" for f in r["facts"]) and any(f["status"] == "verified" for f in r["facts"])


def test_abstention_and_empty_input(svc):
    assert svc.answer("What is the recipe for chocolate cake?", "graph")["abstained"]
    with pytest.raises(ValueError):
        svc.answer("  ")


def test_entity_search_and_profile(svc):
    hits = svc.search_entities("windows")
    assert len(hits) >= 1
    p = svc.entity_profile(hits.iloc[0].id)
    assert p and p["entity"]["type"] == "SOFTWARE" and p["n_incidents"] > 10
    assert svc.subgraph("SOFTWARE:Windows").number_of_nodes() > 1


def test_reports_are_consistent_with_corpus(svc):
    cls = json.loads((ROOT / "reports" / "classification.json").read_text(encoding="utf-8"))
    n_test = int((svc.df.split == "test").sum())
    assert cls["models"]["tfidf_logreg"]["test_weak"]["n"] == n_test
    gs = json.loads((ROOT / "reports" / "graph_stats.json").read_text(encoding="utf-8"))
    assert gs["nodes"] == svc.G.number_of_nodes() and gs["edges"] == svc.G.number_of_edges()


def test_qa_eval_report_has_sets_and_systems():
    qa = json.loads((ROOT / "reports" / "qa_eval.json").read_text(encoding="utf-8"))
    assert qa["main"]["n_questions"] >= 30 and set(qa["main"]["systems"]) == {"vector", "hybrid", "graph"}
    assert qa["paraphrase"]["n_questions"] >= 10


def test_neo4j_cypher_export_is_complete():
    cy = (ROOT / "reports" / "export" / "neo4j" / "import.cypher").read_text(encoding="utf-8")
    assert "LOAD CSV" in cy and "AFFECTS" in cy and "MENTIONS" in cy and "SIMILAR_TO" in cy
