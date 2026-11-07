import json

import numpy as np
import pandas as pd
import pytest

from cyberintel import kg as K


@pytest.fixture(scope="module")
def built(mini_df):
    m = K.build_mentions(mini_df)
    rel = K.extract_relations(mini_df, m)
    return m, rel, K.build_graph(mini_df, m, rel)


def test_mentions_have_sentence_index(built):
    m, _, _ = built
    assert {"incident_id", "type", "name", "start", "end", "sent"} <= set(m.columns) and (m.sent >= 0).all()


def test_sentence_level_relations_with_evidence(built):
    _, rel, _ = built
    row = rel[(rel["head"] == "ASLR") & (rel.relation == "MITIGATES") & (rel["tail"] == "ROP")]
    assert len(row) == 1 and row.iloc[0].support >= 1
    ev = json.loads(row.iloc[0].evidence)
    assert ev and "ASLR" in ev[0]["sentence"]


def test_min_support_filters_single_cooccurrences(built):
    _, rel, _ = built
    corpus = rel[rel.source == "corpus"]
    generic = corpus[~corpus.head_type.isin(K.NEAREST_ONLY_HEADS)]
    assert (generic.support >= 2).all()                        # non-CVE relations need >= min_support incidents


def test_cve_links_to_nearest_software_and_is_verified_by_kb(built):
    _, rel, _ = built
    r = rel[(rel["head"] == "CVE-2021-44228") & (rel.source == "corpus")]
    assert "Apache Log4j" in set(r["tail"])
    assert r[r["tail"] == "Apache Log4j"].iloc[0].status == "verified"
    d = rel[(rel["head"] == "Dirty COW") & (rel.relation == "ALIAS_OF") & (rel.source == "corpus")]
    assert d.iloc[0]["tail"] == "CVE-2016-5195" and d.iloc[0].status == "verified"


def test_kb_verification_flags_wrong_claims():
    kb = K.load_kb()
    claims = pd.DataFrame([
        {"head_type": "CVE", "head": "CVE-2017-0144", "relation": "AFFECTS", "tail_type": "SOFTWARE", "tail": "Linux"},
        {"head_type": "VULN_NAME", "head": "EternalBlue", "relation": "ALIAS_OF", "tail_type": "CVE", "tail": "CVE-2017-0290"},
        {"head_type": "CVE", "head": "CVE-2099-0001", "relation": "AFFECTS", "tail_type": "SOFTWARE", "tail": "Windows"}])
    out = K.verify_against_kb(claims, kb)
    assert list(out.status) == ["differs_from_kb", "contradicted_by_kb", "unverified"]


def test_graph_structure_and_provenance(built):
    _, _, G = built
    types = {d["type"] for _, d in G.nodes(data=True)}
    assert "INCIDENT" in types and "CVE" in types
    assert sum(1 for *_, d in G.edges(data=True) if d["relation"] == "MENTIONS") > 0
    for a, b, d in G.edges(data=True):
        if d["relation"] in K.SEMANTIC_RELATIONS:
            assert d["source"] in ("corpus", "reference_kb") and "status" in d
    assert G.nodes["CVE:CVE-2021-44228"].get("kb_name") == "Log4Shell"


def test_knn_and_entity_adjacency_shapes(built, mini_df):
    m, _, _ = built
    E = np.random.RandomState(0).randn(len(mini_df), 8)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    A = K.knn_adjacency(E, k=3, thr=-1)
    assert A.shape == (len(mini_df),) * 2 and (A != A.T).nnz == 0
    assert K.entity_adjacency(m, len(mini_df)).shape == (len(mini_df),) * 2
