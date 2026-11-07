"""Named-entity recognition for cybersecurity text: regex (structured identifiers) + gazetteers (names), with normalisation.

Why rule-based (decision justified in docs/PROJECT_DEFENSE_GUIDE.md):
* there is no annotated NER data for this corpus, so a supervised/fine-tuned NER model cannot be trained honestly;
* generic pretrained NER models (PER/ORG/LOC) do not know CVE, malware, ATT&CK, attack types;
* rules are transparent, offline, extendable (lexicon.py) and *measurable* (see evaluate_ner + annotations/ner_gold.json).
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from functools import lru_cache

from .lexicon import LEXICON, MITRE_NAMES, MITRE_TACTICS

ENTITY_TYPES = ["CVE", "CWE", "MITRE_TECHNIQUE", "TACTIC", "IP", "HASH", "MALWARE", "THREAT_ACTOR", "SOFTWARE", "TOOL",
                "TECHNOLOGY", "ATTACK", "VULN_NAME", "DEFENSE", "TARGET"]

_REGEX_ENT = {
    "CVE": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
    "CWE": re.compile(r"\bCWE-\d{1,4}\b", re.I),
    "MITRE_TECHNIQUE": re.compile(r"\bT\d{4}(?:\.\d{3})?\b"),
    "IP": re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)"),
    "HASH": re.compile(r"\b(?=[a-f0-9]*[a-f])(?=[a-f0-9]*\d)(?:[a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})\b", re.I),
}
_URL = re.compile(r"https?://\S+|www\.\S+")
_TACTIC_ANCHOR = re.compile(r"tactics?\b", re.I)
_VERSION_CONTEXT = re.compile(r"(?<![a-z])(?:version|ver\.?|v|release|build|curl|openssl|php|python|java|linux|kernel|windows|firefox|chrome|"
                              r"since|before|prior to|fixed in|up to|through)\s*:?\s*$", re.I)


@dataclass(frozen=True)
class Entity:
    type: str
    name: str
    start: int
    end: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.type, self.name)


@lru_cache(maxsize=1)
def _compiled():
    """One combined alternation per entity type, with one named group per canonical name (fast: single pass per type)."""
    out = {}
    for etype, d in LEXICON.items():
        names = sorted(d, key=len, reverse=True)
        parts = [f"(?P<g{i}>{d[n]})" for i, n in enumerate(names)]
        suffix = r"(?:s|es|ed|ing)?" if etype in ("ATTACK", "DEFENSE") else (r"" if etype == "TOOL" else r"(?:s|es)?")   # inflections (heap spraying)
        out[etype] = (re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")" + suffix + r"(?![\w])", re.I), names)
    return out


def _valid_ip(text: str, start: int, s: str) -> bool:
    try:
        ipaddress.IPv4Address(s)
    except ValueError:
        return False
    octets = [int(x) for x in s.split(".")]
    if max(octets) <= 9 and len(set(octets)) > 1:      # "3.6.5.2 Classification..." = section number, not an address
        return False                                   # (1.1.1.1, 8.8.8.8, 0.0.0.0 are kept: repeated octets)
    return not _VERSION_CONTEXT.search(text[max(0, start - 14):start])


def extract_entities(text: str) -> list[Entity]:
    """All entity mentions (with offsets), normalised to canonical names, de-duplicated by (type, name, start)."""
    found: dict[tuple, Entity] = {}

    def add(e: Entity):
        found.setdefault((e.type, e.name, e.start), e)

    masked = _URL.sub(lambda m: " " * len(m.group(0)), text)     # URL schemes/paths are not mentions of HTTPS, GitHub...
    for etype, rx in _REGEX_ENT.items():
        for m in rx.finditer(masked if etype == "HASH" else text):
            s = m.group(0)
            if etype == "IP" and not _valid_ip(text, m.start(), s):
                continue
            name = s.upper() if etype in ("CVE", "CWE", "MITRE_TECHNIQUE") else s.lower()
            add(Entity(etype, name, m.start(), m.end()))
    for etype, (rx, names) in _compiled().items():
        for m in rx.finditer(masked):
            g = m.lastgroup
            if g:
                add(Entity(etype, names[int(g[1:])], m.start(), m.end()))
    for m in _TACTIC_ANCHOR.finditer(text):               # ATT&CK tactic names only right after the word "Tactic(s)"
        window = text[m.end(): m.end() + 110]
        for t in MITRE_TACTICS:
            i = window.find(t)
            if i >= 0:
                add(Entity("TACTIC", t, m.end() + i, m.end() + i + len(t)))
    ents = sorted(found.values(), key=lambda e: (e.start, e.type))
    if any(e.type == "ATTACK" and e.name in ("reflected XSS", "stored XSS", "DOM XSS") for e in ents):
        ents = [e for e in ents if not (e.type == "ATTACK" and e.name == "XSS")]   # generic XSS is redundant next to a specific variant
    # drop a generic mention strictly contained in a more specific one of the same type ("XSS" inside "reflected XSS")
    keep = []
    for e in ents:
        if any(o is not e and o.type == e.type and o.start <= e.start and o.end >= e.end and (o.end - o.start) > (e.end - e.start)
               for o in ents):
            continue
        keep.append(e)
    return keep


def entity_set(text: str) -> set[tuple[str, str]]:
    return {e.key for e in extract_entities(text)}


def entity_label(etype: str, name: str) -> str:
    """Display label; MITRE ids are shown with their technique name when known."""
    if etype == "MITRE_TECHNIQUE" and name in MITRE_NAMES:
        return f"{name} {MITRE_NAMES[name]}"
    return name


# ----------------------------------------------------------------------------- evaluation
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate_ner(gold: list[dict], types: list[str] | None = None) -> dict:
    """Entity-level (type + canonical name, per passage) micro P/R/F1, overall and per type.
    `gold` = [{"text": str, "entities": [[type, name], ...]}]."""
    tot = {"tp": 0, "fp": 0, "fn": 0}
    per: dict[str, dict] = {}
    errors = []
    for g in gold:
        gold_set = {(t, n) for t, n in g["entities"]}
        pred = entity_set(g["text"])
        if types:
            gold_set = {x for x in gold_set if x[0] in types}
            pred = {x for x in pred if x[0] in types}
        for t in {x[0] for x in gold_set | pred}:
            gs, ps = {x for x in gold_set if x[0] == t}, {x for x in pred if x[0] == t}
            d = per.setdefault(t, {"tp": 0, "fp": 0, "fn": 0})
            d["tp"] += len(gs & ps); d["fp"] += len(ps - gs); d["fn"] += len(gs - ps)
        for x in pred - gold_set:
            errors.append({"kind": "false_positive", "entity": list(x), "text": g["text"][:160]})
        for x in gold_set - pred:
            errors.append({"kind": "false_negative", "entity": list(x), "text": g["text"][:160]})
        for k in tot:
            tot[k] += sum(per_t[k] for per_t in [{"tp": len(gold_set & pred), "fp": len(pred - gold_set), "fn": len(gold_set - pred)}])
    return {"overall": prf(**tot), "per_type": {t: prf(**d) for t, d in sorted(per.items())}, "n_passages": len(gold),
            "errors_sample": errors[:40]}
