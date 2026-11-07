"""Question answering: vector RAG vs hybrid RAG vs Graph-RAG, with abstention and a grounding check.

Pipeline (course Chap. 4 Graph-RAG slide): question -> entity linking (NER) -> retrieval (chunks) + graph retrieval (relations,
incidents, paths) -> fusion / reasoning -> evidence sufficiency gate -> generation (extractive by default, Claude optional) -> grounding check.

Modes (the ablation used in the evaluation):
  vector : dense embedding retrieval only                              (classic RAG)
  hybrid : dense + lexical TF-IDF, rank fusion                         (stronger classic RAG)
  graph  : hybrid + knowledge-graph facts, entity-driven candidates, graph-aware re-ranking, KB verification  (Graph-RAG)
Every answer carries its evidence (facts with provenance, passages with incident ids) so it can be audited.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..config import cfg
from ..ner import extract_entities
from ..textutil import split_sentences, tokenize
from .graph_store import GraphStore
from .retrieval import ChunkIndex

_INC_REF = re.compile(r"(?:incident|inc|#)\s*#?(\d+)", re.I)
_RX = {
    "similar": re.compile(r"\b(similar|similaire|ressembl|comparable|like incident)\w*", re.I),
    "vulns": re.compile(r"\b(vulnerabilit|vuln[eé]rabilit|cves?\b|cwes?\b|faille|flaws?)\w*", re.I),
    "affected": re.compile(r"\b(affect|impact|concern|vulnerable|which (?:systems?|software|products?)|quel(?:s|le|les)? (?:logiciels?|syst[eè]mes?|produits?))\w*", re.I),
    "incidents": re.compile(r"\b(incidents?|records?|related to|linked to|about|mention|discuss|li[ée]s? (?:à|au|aux))\b", re.I),
    "targets": re.compile(r"\b(target|cible|victims?|attack(?:s|ed)? what|what.*(?:attack|target))\w*", re.I),
    "defense": re.compile(r"\b(defen[cs]e|mitigat|prevent|protect|counter|remediat|se prot[eé]ger|contre|pr[eé]venir|d[eé]fendre|detect)\w*", re.I),
    "define": re.compile(r"\b(what is|what are|explain|describe|how does|how do|characteristic|qu'est|c'est quoi|d[eé]cris|explique|comment)\b", re.I),
    "relation": re.compile(r"\b(relationship|relation|related|link|connect|between|lien|rapport)\w*", re.I),
}
_CONTENT_STOP = ENGLISH_STOP_WORDS | {"attack", "attacks", "affect", "affects", "incident", "incidents", "related", "similar", "give", "list",
                                       "show", "tell", "does", "work", "works", "used", "use", "using", "vulnerability", "vulnerabilities",
                                       "software", "systems", "system", "target", "targets", "characteristics", "relationship", "quelle",
                                       "quels", "quelles", "est", "les", "des", "que", "qui", "une", "dans", "pour", "sont", "avec"}
_BAD_SENT = re.compile(r"\b(maybe|perhaps|probably refers|the user|I need to|I should|I'll|let me|I think|I recall|hmm|okay,|alright,|wait,)\b", re.I)
_BULLET = re.compile(r"^[\s\-\*•#>]+")


def _clean_sentence(x: str) -> str:
    return re.sub(r"\*\*|__", "", _BULLET.sub("", x)).strip()


STRUCTURED = {"vulns_of_software", "software_of_cve", "describe_cve", "incidents_of_entity", "targets_of_attack", "mitigations",
              "relation_between"}


@dataclass
class Fact:
    text: str
    head: str
    relation: str
    tail: str
    support: int
    source: str            # corpus | reference_kb
    status: str            # verified | differs_from_kb | contradicted_by_kb | unverified | reference
    evidence: list = field(default_factory=list)   # [{incident, sentence}]


@dataclass
class Passage:
    incident_id: int
    chunk_id: int
    score: float
    question: str
    sentences: list[str]
    text: str


@dataclass
class QAResult:
    question: str
    mode: str
    intent: str
    entities: list[str]
    answer: str
    abstained: bool
    reason: str
    facts: list[Fact]
    passages: list[Passage]
    incidents: list[int]            # ranked incident ids supporting the answer
    confidence: str
    grounding: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ============================================================================ generators
class ExtractiveGenerator:
    """Offline, deterministic, hallucination-free: the answer is assembled ONLY from retrieved facts and sentences (all cited)."""
    name = "extractive"

    STATUS = {"verified": "verified by the reference KB", "differs_from_kb": "differs from the reference KB",
              "contradicted_by_kb": "CONTRADICTED by the reference KB", "unverified": "corpus only, unverified", "reference": "reference KB"}

    def generate(self, question, intent, facts: list[Fact], passages: list[Passage], extra: dict) -> str:
        lines = []
        if extra.get("headline"):
            lines.append(extra["headline"])
        for f in facts:
            cite = ", ".join(f"#{e['incident']}" for e in f.evidence if e.get("incident") is not None)
            corp = f"corpus, {f.support} incident(s){': ' + cite if cite else ''}"
            origin = "reference KB" if f.source == "reference_kb" else (corp + " + reference KB" if f.source == "corpus+reference_kb" else corp)
            lines.append(f"- {f.text} [{origin}; {self.STATUS.get(f.status, f.status)}]")
        if passages:
            lines.append("\nSupporting passages:" if facts else "")
            for p in passages[:3]:
                for s in p.sentences[:2]:
                    lines.append(f"- \"{s}\" [#{p.incident_id}]")
        return "\n".join(l for l in lines if l is not None).strip()


class ClaudeGenerator:
    """Optional LLM generation (needs ANTHROPIC_API_KEY). The prompt contains ONLY retrieved evidence and asks for citations/abstention.
    Not exercised in this delivery's tests (no API key available)."""
    name = "claude"

    def __init__(self, model: str | None = None):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model or cfg("llm", "claude_model")

    def generate(self, question, intent, facts, passages, extra):
        ev = "\n".join(f"FACT: {f.text} (source={f.source}, status={f.status}, incidents={[e.get('incident') for e in f.evidence]})" for f in facts)
        ps = "\n".join(f"PASSAGE [#{p.incident_id}]: {' '.join(p.sentences)[:900]}" for p in passages[:4])
        prompt = ("You answer cybersecurity questions using ONLY the evidence below. Cite incident ids like [#12]. "
                  "If a fact is flagged differs_from_kb or contradicted_by_kb, say so. If the evidence does not answer the question, "
                  "say you cannot answer from the available data.\n\n" + ev + "\n" + ps + f"\n\nQUESTION: {question}")
        r = self.client.messages.create(model=self.model, max_tokens=500, messages=[{"role": "user", "content": prompt}])
        return r.content[0].text


def make_generator(kind: str | None = None):
    kind = kind or cfg("llm", "provider")
    if kind == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeGenerator()
    return ExtractiveGenerator()


# ============================================================================ engine
class QAEngine:
    def __init__(self, corpus: pd.DataFrame, index: ChunkIndex, store: GraphStore, embeddings: np.ndarray | None = None, generator=None):
        self.df, self.index, self.store, self.E = corpus, index, store, embeddings
        self.gen = generator or ExtractiveGenerator()
        self.vocab = set(index.vec.vocabulary_)
        self._labels: dict[str, list[str]] | None = None
        self.k_rrf = cfg("rag", "rrf_k")

    # ---------------------------------------------------------------- entity linking
    def label_index(self) -> dict[str, list[str]]:
        if self._labels is None:
            self._labels = {}
            if hasattr(self.store, "G"):
                for n, d in self.store.G.nodes(data=True):
                    if d["type"] not in ("INCIDENT", "IP", "HASH", "MITRE_TECHNIQUE") and len(d["label"]) > 2:
                        self._labels.setdefault(d["label"].lower(), []).append(n)
        return self._labels

    def link_entities(self, question: str) -> list[str]:
        ids = [f"{e.type}:{e.name}" for e in extract_entities(question)]
        low = question.lower()
        for label, nodes in self.label_index().items():          # case-insensitive fallback ("windows", "kerberos")
            if re.search(r"(?<![\w])" + re.escape(label) + r"(?![\w])", low):
                ids += nodes
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i); out.append(i)
        return out

    # ---------------------------------------------------------------- intent
    def detect_intent(self, q: str, ents: list[str]) -> tuple[str, int | None]:
        m = _INC_REF.search(q)
        if _RX["similar"].search(q) and m and int(m.group(1)) < len(self.df):
            return "similar_incidents", int(m.group(1))
        types = [e.split(":", 1)[0] for e in ents]
        has = lambda *ts: any(t in ts for t in types)
        if len(ents) >= 2 and _RX["relation"].search(q):
            return "relation_between", None
        if has("CVE", "VULN_NAME"):
            return ("software_of_cve" if _RX["affected"].search(q) else "describe_cve"), None
        if has("SOFTWARE", "TECHNOLOGY") and _RX["vulns"].search(q):
            return "vulns_of_software", None
        if ents and _RX["incidents"].search(q) and not _RX["define"].search(q):
            return "incidents_of_entity", None
        if has("ATTACK", "MALWARE", "THREAT_ACTOR") and _RX["targets"].search(q):
            return "targets_of_attack", None
        if has("ATTACK", "VULN_NAME", "MALWARE") and _RX["defense"].search(q):
            return "mitigations", None
        if ents and _RX["define"].search(q):
            return "describe_entity", None
        return "general", None

    # ---------------------------------------------------------------- out-of-vocabulary check (abstention)
    def oov_terms(self, q: str) -> list[str]:
        toks = [t for t in tokenize(q) if len(t) > 2 and t not in _CONTENT_STOP]
        return [t for t in toks if t not in self.vocab]

    # ---------------------------------------------------------------- retrieval
    def _retrieve(self, q: str, mode: str, k: int):
        """Returns ranked [(incident_id, score, best_chunk_row, lex, dense)] (one entry per incident)."""
        hits = self.index.hybrid(q, 60) if mode != "vector" else [(i, s, 0.0, s) for i, s in self.index.dense(q, 60)]
        best: dict[int, tuple] = {}
        for row, score, lex, dn in hits:
            inc = self.index.incident_of(row)
            if inc not in best or score > best[inc][1]:
                best[inc] = (inc, score, row, lex, dn)
        return sorted(best.values(), key=lambda x: -x[1])

    def _passage(self, q: str, inc: int, row: int, score: float, qvec, focus: tuple = ()) -> Passage:
        text = self.index.chunk_text(row)
        sents = [x for x in split_sentences(text) if len(x) > 40 and not _BAD_SENT.search(x)] or split_sentences(text)
        if len(sents) > 1:
            sc = (self.index.vec.transform(sents) @ qvec.T).toarray().ravel()
            if focus:                                       # prefer sentences that mention the question's own entities
                sc = sc + np.array([1.0 if any(f in x.lower() for f in focus) else 0.0 for x in sents])
            order = sorted(np.argsort(-sc)[:2])
            sents = [sents[i] for i in order]
        return Passage(inc, int(self.index.chunks.chunk_id.iloc[row]), float(score), self.df.question[inc],
                       [_clean_sentence(x)[:420] for x in sents], text)

    # ---------------------------------------------------------------- graph facts
    def _fact(self, r: dict, text: str) -> Fact:
        return Fact(text=text, head=r["head"], relation=r["relation"], tail=r["tail"], support=int(r["support"] or 0), source=r["source"],
                    status=r.get("status") or "unverified", evidence=(r.get("evidence") or [])[:3])

    def _lab(self, eid: str) -> str:
        e = self.store.entity(eid)
        return e["label"] if e else eid.split(":", 1)[1]

    def _facts(self, intent: str, ents: list[str]) -> tuple[list[Fact], dict]:
        st, facts, extra = self.store, [], {}
        by_type = lambda *ts: [e for e in ents if e.split(":", 1)[0] in ts]
        if intent == "vulns_of_software":
            for e in by_type("SOFTWARE", "TECHNOLOGY"):
                for r in st.relations(e, "AFFECTS", "in", 12):
                    meta = st.entity(r["head"]) or {}
                    nick = f" ({meta['kb_name']})" if meta.get("kb_name") else ""
                    facts.append(self._fact(r, f"{r['other_label']}{nick} affects {self._lab(e)}"))
        elif intent in ("software_of_cve", "describe_cve"):
            for e in by_type("CVE", "VULN_NAME"):
                meta = st.entity(e) or {}
                if meta.get("kb_summary"):
                    extra["headline"] = f"{meta['label']} ({meta.get('kb_name', '')}): {meta['kb_summary']}"
                for r in st.relations(e, "AFFECTS", "out", 12):
                    facts.append(self._fact(r, f"{self._lab(e)} affects {r['other_label']}"))
                for r in st.relations(e, "ALIAS_OF", "both", 6):
                    facts.append(self._fact(r, f"{self._lab(r['head'])} is the name of {self._lab(r['tail'])}"))
        elif intent == "targets_of_attack":
            for e in by_type("ATTACK", "MALWARE", "THREAT_ACTOR"):
                for r in st.relations(e, "TARGETS", "out", 8):
                    facts.append(self._fact(r, f"{self._lab(e)} targets {r['other_label']}"))
        elif intent == "mitigations":
            for e in by_type("ATTACK", "VULN_NAME", "MALWARE"):
                for r in st.relations(e, "MITIGATES", "in", 8):
                    facts.append(self._fact(r, f"{r['other_label']} mitigates {self._lab(e)}"))
        elif intent == "relation_between" and len(ents) >= 2:
            a, b = ents[0], ents[1]
            for r in st.relations(a, None, "both", 100):
                if r["other"] == b:
                    facts.append(self._fact(r, f"{self._lab(r['head'])} {r['relation']} {self._lab(r['tail'])}"))
            if not facts:
                path = st.path_between(a, b)
                if path:
                    extra["headline"] = "Indirect connection in the knowledge graph: " + " -> ".join(
                        f"{s['from']} -[{'/'.join(s['relations'])}]-> {s['to']}" for s in path)
        elif intent in ("describe_entity", "general") and ents:
            for e in ents[:2]:
                for r in st.relations(e, None, "both", 4):
                    t = f"{self._lab(r['head'])} {r['relation']} {self._lab(r['tail'])}"
                    facts.append(self._fact(r, t))
        elif intent == "incidents_of_entity":
            for e in ents[:2]:
                inc = st.incidents_of(e, 200)
                extra.setdefault("entity_incidents", []).extend(i for i, _ in inc)
                if inc:
                    top = "; ".join(f"#{i} {self.df.question[i][:70]}" for i, _ in inc[:5])
                    extra["headline"] = (f"{self._lab(e)} is mentioned in {len(inc)} incident record(s) of the knowledge base. "
                                         f"Most relevant: {top}.")
        return self._merge(facts), extra

    @staticmethod
    def _merge(facts: list[Fact]) -> list[Fact]:
        """One line per (head, relation, tail): corpus and reference-KB provenance are combined, KB verdict wins."""
        out: dict[tuple, Fact] = {}
        for f in facts:
            k = (f.head, f.relation, f.tail)
            if k not in out:
                out[k] = f
                continue
            g = out[k]
            g.support = max(g.support, f.support)
            g.evidence = (g.evidence + [e for e in f.evidence if e not in g.evidence])[:3]
            if g.source != f.source:
                g.source = "corpus+reference_kb"
            if f.status in ("verified", "differs_from_kb", "contradicted_by_kb") or g.status in ("reference", "unverified"):
                g.status = f.status if f.status not in ("reference", "unverified") else g.status
        return list(out.values())

    # ---------------------------------------------------------------- answer
    def answer(self, question: str, mode: str = "graph", k: int | None = None) -> QAResult:
        k = k or cfg("rag", "top_k")
        ents = self.link_entities(question)
        intent, inc_ref = self.detect_intent(question, ents)
        qvec = self.index.vec.transform([question])
        labels = [self._lab(e) for e in ents]

        if intent == "similar_incidents":
            sims = self.store.similar_incidents(inc_ref, k) if mode == "graph" or self.E is None else self._cos_similar(inc_ref, k)
            passages = [Passage(j, -1, float(s), self.df.question[j], [self.df.question[j]], self.df.answer[j][:400]) for j, s in sims]
            ans = f"Incidents most similar to #{inc_ref} ({self.df.question[inc_ref][:90]}...):\n" + "\n".join(
                f"- #{j} (similarity {s:.2f}, {self.df.category[j]}): {self.df.question[j][:110]}" for j, s in sims)
            return QAResult(question, mode, intent, labels, ans, False, "", [], passages, [j for j, _ in sims], "high")

        ranked = self._retrieve(question, mode, k * 4)
        best_lex = max((r[3] for r in ranked[:k]), default=0.0)
        best_dense = max((r[4] for r in ranked[:k]), default=0.0)
        facts, extra = (self._facts(intent, ents) if mode == "graph" else ([], {}))

        # ---- graph-aware re-ranking and entity-driven candidates (graph mode only)
        if mode == "graph" and ents:
            ent_inc = {e: dict(self.store.incidents_of(e, 400)) for e in ents if self.store.has_entity(e)}
            scores = {inc: sc / max(ranked[0][1], 1e-9) for inc, sc, *_ in ranked} if ranked else {}
            rows = {inc: row for inc, _, row, *_ in ranked}
            for e, incs in ent_inc.items():
                for inc in incs:
                    scores[inc] = scores.get(inc, 0.0) + 0.6 / max(len(ent_inc), 1)
                    if inc not in rows:
                        rows[inc] = int(self.index.chunks.index[self.index.chunks.incident_id == inc][0])
            ranked = [(inc, s, rows[inc], 0.0, 0.0) for inc, s in sorted(scores.items(), key=lambda x: -x[1])]
            pools = [set(v) for v in ent_inc.values() if v]        # focus: sources must actually mention the question's entities
            if pools:
                inter = set.intersection(*pools) if len(pools) > 1 else set()
                pool = inter or set.union(*pools)
                focused = [r for r in ranked if r[0] in pool]
                ranked = focused or ranked
            if extra.get("entity_incidents"):                      # "which incidents mention X": the graph answers exhaustively
                ranked = [(inc, 1.0 + c, rows.get(inc, 0), 0.0, 0.0)
                          for inc, c in sorted(ent_inc.get(ents[0], {}).items(), key=lambda x: -x[1])] or ranked

        top = ranked[:k]
        focus = tuple(self._lab(e).lower() for e in ents if not e.startswith("TARGET:") and self.store.has_entity(e))
        passages = [self._passage(question, inc, row, sc, qvec, focus) for inc, sc, row, *_ in top]

        # ---- evidence sufficiency gate (abstention)
        oov = self.oov_terms(question)
        reason = ""
        structured = intent in STRUCTURED
        known_ents = [e for e in ents if self.store.has_entity(e)]
        if not facts and not extra.get("entity_incidents"):
            if oov and (structured or len(oov) >= max(1, len([t for t in tokenize(question) if len(t) > 2 and t not in _CONTENT_STOP]) // 2)):
                reason = f"no evidence in the knowledge base about: {', '.join(oov)}"
            elif best_dense < cfg("rag", "min_dense_score") and best_lex < cfg("rag", "min_lexical_score"):
                reason = "retrieved passages are not relevant enough to the question"
            elif structured and not known_ents:
                reason = "the entities of the question are absent from the knowledge graph"
            elif structured and mode == "graph" and intent in ("vulns_of_software", "software_of_cve", "targets_of_attack", "mitigations"):
                reason = "the knowledge graph holds no such relation for this entity"
        elif oov and structured and not facts and not extra.get("entity_incidents"):
            reason = f"no evidence in the knowledge base about: {', '.join(oov)}"
        # "Which CVE/vulnerability affects <X>?" where X is not a known software/technology and some term is unknown:
        # the generic fallback would return unrelated passages, so refuse (found by the failures U03, P11)
        if (not reason and oov and _RX["vulns"].search(question) and _RX["affected"].search(question)
                and not any(e.split(":", 1)[0] in ("SOFTWARE", "TECHNOLOGY", "CVE", "VULN_NAME") for e in ents)):
            reason = f"the product asked about is unknown to the knowledge base ({', '.join(oov)})"
        if not reason and not facts and oov:
            # an unknown PROPER NOUN (capitalised / contains a digit) is a specific thing we know nothing about, even when a generic
            # entity of the question ("worm") exists (found by failure U04: "the Conficker worm")
            words = re.findall(r"[A-Za-z0-9][\w\-\.]*", question)
            proper = [w for w in words if w.lower() in oov and (w[0].isupper() or any(c.isdigit() for c in w))]
            if proper:
                reason = f"unknown name(s) not present in the knowledge base: {', '.join(proper)}"
        if reason:
            msg = f"I cannot answer from the available data: {reason}."
            if known_ents:
                msg += " Entities recognised: " + ", ".join(self._lab(e) for e in known_ents) + "."
            return QAResult(question, mode, intent, labels, msg, True, reason, [], [], [], "none")

        text = self.gen.generate(question, intent, facts, passages, extra)
        conf = ("high" if (facts and any(f.status in ("verified", "reference") for f in facts)) or extra.get("entity_incidents")
                else ("medium" if facts or best_lex > 0.3 else "low"))
        res = QAResult(question, mode, intent, labels, text, False, "", facts, passages, [p.incident_id for p in passages], conf)
        res.grounding = verify_grounding(res, self.df)
        return res

    def _cos_similar(self, i: int, k: int):
        s = self.E @ self.E[i]
        s[i] = -1
        idx = np.argsort(-s)[:k]
        return [(int(j), float(s[j])) for j in idx]


# ============================================================================ grounding check
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\*\*|__", "", s)).strip().lower()


def verify_grounding(res: QAResult, corpus: pd.DataFrame) -> dict:
    """Machine check that every cited claim exists in the source: passage sentences must be substrings of the cited incident's answer,
    and fact evidence sentences must be substrings of the cited incident's text. KB-sourced facts are 'reference' (not corpus-checkable)."""
    checked = ok = 0
    for p in res.passages:
        if p.chunk_id < 0:
            continue
        src = _norm(corpus.answer[p.incident_id])
        for s in p.sentences:
            checked += 1
            ok += _norm(s) in src
    for f in res.facts:
        if f.source == "reference_kb":
            continue
        for e in f.evidence:
            if e.get("incident") is None:
                continue
            checked += 1
            ok += _norm(e["sentence"]) in _norm(corpus.text[e["incident"]])
    return {"claims_checked": checked, "claims_found_in_source": ok, "grounded_fraction": round(ok / checked, 4) if checked else None}
