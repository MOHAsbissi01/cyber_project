"""Incident taxonomy, weak (keyword) labels, and the independent gold-label set.

The dataset has NO labels. Two label sources exist and are always reported separately:
* WEAK / silver labels: keyword rules (below). Cheap, noisy, and circular with keyword-based models.
* GOLD labels: a stratified sample annotated independently of the rules (annotations/gold_labels.csv).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# (regex, weight). Prototype rules kept for continuity (they are the silver baseline), ransomware is now a sub-type of malware.
CATEGORY_PATTERNS = {
    "xss": (r"\bxss\b|cross[- ]site scripting|dom clobbering|content security policy|\bcsp\b", 1.0),
    "injection": (r"sql injection|sqli|command injection|ldap injection|\bxxe\b|\bssti\b|template injection|\bssrf\b", 1.0),
    "csrf_session": (r"\bcsrf\b|cross[- ]site request forgery|session (hijack|fixation)|cookie", 1.0),
    "phishing": (r"phish|spear|social engineering|pretext|smishing|vishing", 1.0),
    "malware": (r"malware|trojan|botnet|\brat\b|rootkit|keylogger|spyware|\bworm\b|dropper|command and control|\bc2\b", 1.0),
    "ransomware": (r"ransomware|lockbit|wannacry|\bryuk\b|extortion", 1.2),
    "vulnerability": (r"cve-\d|vulnerab|exploit|buffer overflow|zero[- ]day|remote code execution|privilege escalation|\bpatch", 0.6),
    "network_attack": (r"ddos|denial of service|port scan|\bmitm\b|man-in-the-middle|arp spoof|dns (spoof|poison|tunnel)|firewall|intrusion", 1.0),
    "crypto": (r"cryptograph|encryption|\btls\b|\bssl\b|\bhash(es|ing)?\b|\brsa\b|\baes\b|certificate|\bpki\b", 1.0),
    "incident_response": (r"incident response|forensic|\bsiem\b|playbook|threat hunting|mitre|att&ck|\bsoc\b|log analysis|containment", 1.0),
    "access_control": (r"authenticat|authoriz|oauth|\bjwt\b|password|\bmfa\b|\biam\b|access control|credential", 1.0),
}
MERGE = {"ransomware": "malware"}   # fewer than 20 rows: merged into its parent class (documented decision)
CLASSES = sorted(set(CATEGORY_PATTERNS) - set(MERGE) | {"other"})
_CAT_RE = {k: (re.compile(p, re.I), w) for k, (p, w) in CATEGORY_PATTERNS.items()}

CLASS_DEFINITIONS = {
    "xss": "Cross-site scripting and client-side script injection / CSP / DOM issues.",
    "injection": "Server-side injection: SQLi, command/LDAP/template injection, XXE, SSRF.",
    "csrf_session": "CSRF, session hijacking/fixation, cookie and session handling.",
    "phishing": "Phishing and social engineering.",
    "malware": "Malware families and behaviour: trojans, botnets, rootkits, RATs, ransomware, C2.",
    "vulnerability": "Memory-safety bugs, exploitation techniques, CVEs, privilege escalation, patching.",
    "network_attack": "DDoS, MITM, scanning, DNS/ARP attacks, firewall/IDS topics.",
    "crypto": "Cryptography, TLS/PKI, hashing, encryption attacks.",
    "incident_response": "IR, forensics, SIEM/SOC, threat hunting, MITRE-based detection.",
    "access_control": "Authentication, authorisation, passwords, OAuth/JWT, IAM.",
    "other": "None of the above is the main topic.",
}


def weak_label(question: str, answer: str) -> str:
    best, best_s = "other", 0.0
    for cat, (rx, w) in _CAT_RE.items():
        s = w * (3 * len(rx.findall(question)) + len(rx.findall(answer[:800])))
        if s > best_s:
            best, best_s = cat, s
    return MERGE.get(best, best)


def weak_labels(df: pd.DataFrame) -> pd.Series:
    return pd.Series([weak_label(q, a) for q, a in zip(df.question, df.answer)], index=df.index, name="category")


# ----------------------------------------------------------------------------- gold labels
def export_annotation_sample(df: pd.DataFrame, per_class: int, out: Path, seed: int = 0, pool_split: str = "test") -> pd.DataFrame:
    """Stratified sample (by weak label) drawn from the TEST split; rare classes are taken entirely up to `per_class`."""
    pool = df[df.split == pool_split]
    rng = np.random.RandomState(seed)
    parts = []
    for cat, g in pool.groupby("category"):
        take = min(per_class, len(g))
        parts.append(g.iloc[rng.permutation(len(g))[:take]])
    s = pd.concat(parts).sort_values("id")
    sample = pd.DataFrame({"id": s.id, "question": s.question, "answer_snippet": s.answer.str[:220],
                           "weak_label": s.category, "claude_label": "", "human_label": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, encoding="utf-8")
    return sample


def load_gold(csv_path: Path) -> pd.DataFrame | None:
    """Gold labels: `human_label` wins over `claude_label` (silver annotator); `gold_source` records which one was used."""
    if not Path(csv_path).exists():
        return None
    g = pd.read_csv(csv_path, encoding="utf-8").fillna("")
    h, c = g.get("human_label", pd.Series("", index=g.index)), g["claude_label"]
    g["gold_label"] = np.where(h.astype(str).str.len() > 0, h, c)
    g["gold_source"] = np.where(h.astype(str).str.len() > 0, "human", "claude-annotator")
    g = g[g.gold_label != ""]
    bad = set(g.gold_label) - set(CLASSES)
    if bad:
        raise ValueError(f"Unknown gold labels: {bad}. Allowed: {CLASSES}")
    return g[["id", "gold_label", "gold_source", "weak_label"]]
