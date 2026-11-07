"""Knowledge-graph construction (course Chap. 1 'text -> graph' and Chap. 4 Graph-RAG pipeline: NER -> relation extraction -> graph).

Pipeline
  1. mentions   : NER with character offsets for every incident record (ner.extract_entities)
  2. relations  : SENTENCE-level relation extraction. Two entities form a candidate relation only if they occur in the same
                  sentence within `max_pair_chars` characters; the relation LABEL comes from a typed schema (head type, tail type).
                  Each relation is aggregated with support (#incidents), lift (association strength) and evidence sentences.
  3. reference  : a small curated KB (resources/kb.json) adds CVE -> software facts (provenance 'reference_kb') and lets us
                  VERIFY corpus claims: verified / differs_from_kb / contradicted_by_kb / unverified.
  4. graph      : directed multigraph: Incident -MENTIONS-> Entity, Entity -REL-> Entity (+ SIMILAR_TO between incidents).

Why sentence level (not document level as in the prototype): two entities in the same 3,000-character answer are rarely related;
inside one sentence and close together they usually are. Provenance (evidence sentences) makes every edge auditable.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import cfg
from .lexicon import LEXICON, LOW_INFORMATION, MITRE_NAMES
from .ner import entity_label, extract_entities
from .textutil import sentence_spans

KB_PATH = Path(__file__).parent / "resources" / "kb.json"

# (head type, tail type) -> relation. Directed: the head is the actor/cause, the tail the object/target.
SCHEMA: dict[tuple[str, str], str] = {
    ("ATTACK", "TARGET"): "TARGETS", ("MALWARE", "TARGET"): "TARGETS", ("ATTACK", "SOFTWARE"): "TARGETS",
    ("MALWARE", "SOFTWARE"): "TARGETS", ("ATTACK", "TECHNOLOGY"): "TARGETS", ("THREAT_ACTOR", "TARGET"): "TARGETS",
    ("CVE", "SOFTWARE"): "AFFECTS", ("CVE", "TECHNOLOGY"): "AFFECTS", ("VULN_NAME", "SOFTWARE"): "AFFECTS",
    ("VULN_NAME", "TECHNOLOGY"): "AFFECTS",
    ("ATTACK", "CVE"): "EXPLOITS", ("MALWARE", "CVE"): "EXPLOITS", ("VULN_NAME", "CVE"): "ALIAS_OF",
    ("ATTACK", "MITRE_TECHNIQUE"): "MAPS_TO", ("MITRE_TECHNIQUE", "TACTIC"): "PART_OF",
    ("DEFENSE", "ATTACK"): "MITIGATES", ("DEFENSE", "VULN_NAME"): "MITIGATES",
    ("THREAT_ACTOR", "MALWARE"): "USES", ("THREAT_ACTOR", "ATTACK"): "USES", ("MALWARE", "ATTACK"): "USES",
    ("ATTACK", "TOOL"): "USES", ("MALWARE", "TOOL"): "USES", ("THREAT_ACTOR", "TOOL"): "USES",
}
NEAREST_ONLY_HEADS = {"CVE", "VULN_NAME"}          # rare, precise heads: link to the closest tail only ("CVE-2017-5123 (Firefox) and CVE-2020-6463 (Chrome)")
SEMANTIC_RELATIONS = sorted(set(SCHEMA.values()))
EVIDENCE_MAX = 3


def load_kb(path: Path = KB_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_name(name: str) -> tuple[str, str]:
    """Entity type of a canonical name using the lexicon (SOFTWARE first); unknown names default to SOFTWARE."""
    for t in ("SOFTWARE", "TECHNOLOGY", "DEFENSE", "TOOL", "VULN_NAME"):
        if name in LEXICON[t]:
            return t, name
    return "SOFTWARE", name


def nid(t: str, name: str) -> str:
    return f"{t}:{name}"


# ----------------------------------------------------------------------------- 1. mentions
def build_mentions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        spans = sentence_spans(r.text)
        starts = np.array([a for a, _ in spans])
        for e in extract_entities(r.text):
            k = int(np.searchsorted(starts, e.start, side="right") - 1)
            rows.append((int(r.id), e.type, e.name, e.start, e.end, max(k, 0)))
    return pd.DataFrame(rows, columns=["incident_id", "type", "name", "start", "end", "sent"])


# ----------------------------------------------------------------------------- 2. relations
def extract_relations(df: pd.DataFrame, mentions: pd.DataFrame, kb: dict | None = None) -> pd.DataFrame:
    kb = kb or load_kb()
    max_d = cfg("graph", "max_pair_chars")
    n_inc = len(df)
    dfreq = mentions.drop_duplicates(["incident_id", "type", "name"]).groupby(["type", "name"]).size().to_dict()
    texts = dict(zip(df.id, df.text))
    spans_cache: dict[int, list] = {}
    agg: dict[tuple, dict] = {}
    for (inc, sent), g in mentions.groupby(["incident_id", "sent"]):
        ents = list(g.itertuples())
        if len(ents) < 2:
            continue
        cands: dict[tuple, tuple[int, object, object]] = {}
        for a in ents:
            for b in ents:
                rel = SCHEMA.get((a.type, b.type))
                if not rel or (a.type, a.name) == (b.type, b.name):
                    continue
                dist = max(0, max(a.start, b.start) - min(a.end, b.end))
                if dist > max_d:
                    continue
                key = (a.type, a.name, a.start, b.type)       # nearest tail of each type per head mention
                if a.type in NEAREST_ONLY_HEADS:
                    if key not in cands or dist < cands[key][0]:
                        cands[key] = (dist, a, b)
                else:
                    cands[(a.type, a.name, a.start, b.type, b.name, b.start)] = (dist, a, b)
        for dist, a, b in cands.values():
            k = (a.type, a.name, SCHEMA[(a.type, b.type)], b.type, b.name)
            d = agg.setdefault(k, {"incidents": set(), "evidence": []})
            if inc not in d["incidents"]:
                d["incidents"].add(inc)
                if len(d["evidence"]) < EVIDENCE_MAX:
                    spans = spans_cache.setdefault(inc, sentence_spans(texts[inc]))
                    s0, s1 = spans[sent]
                    d["evidence"].append({"incident": int(inc), "sentence": texts[inc][s0:s1][:300]})
    rows = []
    for (ht, hn, rel, tt, tn), d in agg.items():
        support = len(d["incidents"])
        lift = support * n_inc / (dfreq.get((ht, hn), 1) * dfreq.get((tt, tn), 1))
        rare_head = ht in NEAREST_ONLY_HEADS
        if support < (cfg("graph", "min_support_cve") if rare_head else cfg("graph", "min_support")):
            continue
        if not rare_head and lift < (1.3 if tt in LOW_INFORMATION else 1.0):
            continue
        rows.append({"head_type": ht, "head": hn, "relation": rel, "tail_type": tt, "tail": tn, "source": "corpus",
                     "support": support, "lift": round(lift, 3), "evidence": json.dumps(d["evidence"], ensure_ascii=False)})
    rel_df = pd.DataFrame(rows)
    rel_df = verify_against_kb(rel_df, kb)
    return pd.concat([rel_df, kb_relations(kb)], ignore_index=True).sort_values(
        ["relation", "support"], ascending=[True, False]).reset_index(drop=True)


# ----------------------------------------------------------------------------- 3. reference KB
def kb_nicknames(kb: dict) -> dict[str, list[str]]:
    out = defaultdict(list)
    for cve, d in kb["cves"].items():
        out[d["name"]].append(cve)
    return dict(out)


def verify_against_kb(rel: pd.DataFrame, kb: dict) -> pd.DataFrame:
    """Status of each corpus claim that the reference KB can judge. Everything else is 'unverified' (corpus only)."""
    nick = kb_nicknames(kb)
    status, note = [], []
    for r in rel.itertuples():
        s, n = "unverified", ""
        if r.relation == "AFFECTS" and r.head_type == "CVE" and r.head in kb["cves"]:
            aff = kb["cves"][r.head]["affects"]
            s = "verified" if r.tail in aff else "differs_from_kb"      # corpus names another product: weaker than a contradiction
            n = f"KB: {r.head} ({kb['cves'][r.head]['name']}) affects {', '.join(aff)}"
        elif r.relation == "ALIAS_OF" and r.head in nick:
            s = "verified" if r.tail in nick[r.head] else "contradicted_by_kb"
            n = f"KB: {r.head} = {', '.join(nick[r.head])}"
        status.append(s); note.append(n)
    rel = rel.copy()
    rel["status"], rel["kb_note"] = status, note
    return rel


def kb_relations(kb: dict) -> pd.DataFrame:
    rows = []
    for cve, d in kb["cves"].items():
        for name in d["affects"]:
            tt, tn = resolve_name(name)
            rows.append({"head_type": "CVE", "head": cve, "relation": "AFFECTS", "tail_type": tt, "tail": tn, "source": "reference_kb",
                         "support": 0, "lift": 0.0, "status": "reference", "kb_note": d["summary"],
                         "evidence": json.dumps([{"incident": None, "sentence": d["summary"]}], ensure_ascii=False)})
        rows.append({"head_type": "VULN_NAME", "head": d["name"], "relation": "ALIAS_OF", "tail_type": "CVE", "tail": cve,
                     "source": "reference_kb", "support": 0, "lift": 0.0, "status": "reference", "kb_note": d["summary"],
                     "evidence": json.dumps([{"incident": None, "sentence": d["summary"]}], ensure_ascii=False)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. graph
def knn_adjacency(E: np.ndarray, k: int | None = None, thr: float | None = None) -> sp.csr_matrix:
    """Semantic similarity graph between incidents (cosine on L2-normalised embeddings, k nearest, threshold, symmetrised)."""
    k, thr = k or cfg("graph", "knn_k"), cfg("graph", "knn_threshold") if thr is None else thr
    S = E @ E.T
    np.fill_diagonal(S, -1)
    n = len(E)
    idx = np.argpartition(-S, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in idx[i]:
            if S[i, j] >= thr:
                rows.append(i); cols.append(j); vals.append(S[i, j])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A.maximum(A.T)


def entity_adjacency(mentions: pd.DataFrame, n: int, max_df: int = 60, exclude_types=("TARGET",)) -> sp.csr_matrix:
    """Incidents linked when they share a (non-generic, non-frequent) entity; weight = min(#shared, 3)/3."""
    m = mentions[~mentions.type.isin(exclude_types)].drop_duplicates(["incident_id", "type", "name"])
    vocab = {k: i for i, k in enumerate(m.groupby(["type", "name"]).size().index)}
    cols = [vocab[(t, nme)] for t, nme in zip(m.type, m.name)]
    B = sp.csr_matrix((np.ones(len(m)), (m.incident_id.values, cols)), shape=(n, len(vocab)))
    keep = np.where(np.asarray(B.sum(0)).ravel() <= max_df)[0]
    A = (B[:, keep] @ B[:, keep].T).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.data = np.minimum(A.data, 3) / 3.0
    return A


def build_graph(df: pd.DataFrame, mentions: pd.DataFrame, relations: pd.DataFrame, A_sim: sp.csr_matrix | None = None) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    kb = load_kb()
    dfreq = mentions.drop_duplicates(["incident_id", "type", "name"]).groupby(["type", "name"]).size().to_dict()
    for r in df.itertuples():
        G.add_node(f"INC:{r.id}", type="INCIDENT", label=r.question[:90], category=r.category, incident_id=int(r.id), split=r.split)
    cnt = mentions.groupby(["incident_id", "type", "name"]).size()
    for (inc, t, name), c in cnt.items():
        e = nid(t, name)
        if e not in G:
            G.add_node(e, type=t, label=entity_label(t, name), name=name, n_incidents=int(dfreq.get((t, name), 0)),
                       low_information=t in LOW_INFORMATION)
        G.add_edge(f"INC:{inc}", e, key="MENTIONS", relation="MENTIONS", source="corpus", weight=float(c), support=int(c))
    for r in relations.itertuples():
        a, b = nid(r.head_type, r.head), nid(r.tail_type, r.tail)
        for x, t, n in ((a, r.head_type, r.head), (b, r.tail_type, r.tail)):
            if x not in G:
                G.add_node(x, type=t, label=entity_label(t, n), name=n, n_incidents=int(dfreq.get((t, n), 0)),
                           low_information=t in LOW_INFORMATION, from_reference_kb=True)
        G.add_edge(a, b, key=f"{r.relation}|{r.source}", relation=r.relation, source=r.source, support=int(r.support), lift=float(r.lift),
                   weight=float(max(r.support, 1)), status=r.status, kb_note=r.kb_note, evidence=r.evidence)
    for cve, d in kb["cves"].items():                       # KB metadata on CVE nodes (name, summary, weakness)
        n = nid("CVE", cve)
        if n in G:
            G.nodes[n].update(kb_name=d["name"], kb_summary=d["summary"], kb_weakness=d["weakness"])
    if A_sim is not None:
        A = sp.triu(A_sim).tocoo()
        for i, j, v in zip(A.row, A.col, A.data):
            G.add_edge(f"INC:{i}", f"INC:{j}", key="SIMILAR_TO", relation="SIMILAR_TO", source="embedding", weight=float(v), support=0)
    return G


# ----------------------------------------------------------------------------- views / helpers used by analysis, RAG and UI
def semantic_edges(G: nx.MultiDiGraph):
    """Yield (head, tail, data) for entity->entity semantic relations."""
    for a, b, d in G.edges(data=True):
        if d["relation"] in SEMANTIC_RELATIONS:
            yield a, b, d


def entity_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse parallel semantic edges into a DiGraph of entities (weight = max support), used for centrality / layout."""
    H = nx.DiGraph()
    for x, d in G.nodes(data=True):
        if d["type"] != "INCIDENT":
            H.add_node(x, **d)
    for a, b, d in semantic_edges(G):
        w = max(d.get("support", 0), 1)
        if H.has_edge(a, b):
            H[a][b]["weight"] = max(H[a][b]["weight"], w)
            H[a][b]["relations"].add(d["relation"])
        else:
            H.add_edge(a, b, weight=w, relations={d["relation"]})
    for _, _, d in H.edges(data=True):
        d["relations"] = sorted(d["relations"])
    return H


def edge_evidence(d: dict) -> list[dict]:
    ev = d.get("evidence")
    return json.loads(ev) if isinstance(ev, str) and ev else []
