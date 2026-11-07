"""Chargement des donnees, pretraitement, labels faibles, NER (regex + gazetteers), embeddings."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

HF_DATASET = "AlicanKiraz0/Cybersecurity-Dataset-v1"

# --------------------------------------------------------------------------- data
_Q_COLS = ["user", "question", "instruction", "prompt", "input"]
_A_COLS = ["assistant", "answer", "response", "output"]


def clean_text(t: str) -> str:
    t = re.sub(r"<think>.*?</think>", " ", str(t), flags=re.S)  # raisonnement parasite
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                # blocs de code
    return re.sub(r"\s+", " ", t).strip()


def load_data(source="sample", limit=None, seed=0) -> pd.DataFrame:
    """source: 'sample' (synthetique), 'hf' (Hugging Face), ou chemin .csv/.parquet/.jsonl"""
    if source == "sample":
        from sample_data import make_sample
        df = make_sample(n=400, seed=seed)
    elif source == "hf":
        from datasets import load_dataset
        df = load_dataset(HF_DATASET, split="train").to_pandas()
    else:
        p = Path(source)
        df = (pd.read_parquet(p) if p.suffix == ".parquet"
              else pd.read_json(p, lines=True) if p.suffix in (".jsonl", ".json")
              else pd.read_csv(p))
    qc = next(c for c in _Q_COLS if c in df.columns)
    ac = next(c for c in _A_COLS if c in df.columns)
    df = df[[qc, ac]].rename(columns={qc: "question", ac: "answer"}).dropna()
    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)
    df = df[(df.question.str.len() > 10) & (df.answer.str.len() > 20)]
    df = df.drop_duplicates("question").reset_index(drop=True)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=seed).reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df["text"] = df["question"] + " " + df["answer"].str[:1500]
    return df


# --------------------------------------------------------------------------- weak labels
# (regex, poids). Le label vient surtout de la QUESTION (poids x3) pour limiter la fuite vers le texte de la reponse.
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
_CAT_RE = {k: (re.compile(p, re.I), w) for k, (p, w) in CATEGORY_PATTERNS.items()}


def weak_label(question: str, answer: str) -> str:
    best, best_s = "other", 0.0
    for cat, (rx, w) in _CAT_RE.items():
        s = w * (3 * len(rx.findall(question)) + len(rx.findall(answer[:800])))
        if s > best_s:
            best, best_s = cat, s
    return best


# --------------------------------------------------------------------------- NER
_REGEX_ENT = {
    "CVE": re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I),
    "CWE": re.compile(r"\bCWE-\d{1,4}\b", re.I),
    "MITRE_TECHNIQUE": re.compile(r"\bT\d{4}(?:\.\d{3})?\b"),
    "IP": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "HASH": re.compile(r"\b(?=[a-f0-9]*[a-f])(?=[a-f0-9]*\d)[a-f0-9]{32,64}\b", re.I),
}

# canonical -> alias regex
_GAZ = {
    "MALWARE": {
        "wannacry": "wannacry|wanacry", "emotet": "emotet", "trickbot": "trickbot", "mirai": "mirai",
        "stuxnet": "stuxnet", "lockbit": "lockbit", "ryuk": "ryuk", "notpetya": "notpetya|petya",
        "cobalt strike": "cobalt strike", "mimikatz": "mimikatz", "qakbot": "qakbot|qbot",
        "zeus": r"zeus(?:\s+trojan)?", "conti": r"conti\b", "revil": "revil|sodinokibi", "agent tesla": "agent tesla",
    },
    "SOFTWARE": {
        "apache": r"apache(?:\s+http)?", "nginx": "nginx", "openssl": "openssl", "log4j": "log4j|log4shell",
        "wordpress": "wordpress", "windows": r"windows(?:\s+\d+|\s+server)?", "linux": "linux",
        "chrome": "chrome", "firefox": "firefox", "mysql": "mysql", "postgresql": "postgres(?:ql)?",
        "jenkins": "jenkins", "docker": "docker", "kubernetes": "kubernetes|k8s", "exchange": "exchange server|ms exchange",
        "sharepoint": "sharepoint", "openssh": "openssh", "tomcat": "tomcat", "php": r"\bphp\b",
        "java": r"\bjava\b", "nodejs": r"node\.?js", "jquery": "jquery", "react": r"\breact\b", "angular": "angular",
        "android": "android", "vmware": "vmware", "citrix": "citrix", "fortinet": "fortinet|fortigate",
        "cisco": "cisco", "office": r"ms office|microsoft office",
    },
    "ATTACK": {
        "reflected xss": "reflected xss|reflected cross-site", "stored xss": "stored xss|persistent xss",
        "dom xss": "dom[- ]based xss|dom xss", "xss": r"\bxss\b|cross[- ]site scripting",
        "sql injection": "sql injection|sqli", "command injection": "command injection|os command",
        "csrf": r"\bcsrf\b|cross[- ]site request forgery", "ssrf": r"\bssrf\b|server[- ]side request forgery",
        "xxe": r"\bxxe\b|xml external entit", "phishing": "phishing|spear[- ]phishing|smishing|vishing",
        "ddos": r"\bddos\b|denial[- ]of[- ]service", "mitm": r"\bmitm\b|man[- ]in[- ]the[- ]middle",
        "ransomware": "ransomware", "brute force": "brute[- ]force|credential stuffing|password spraying",
        "rce": r"\brce\b|remote code execution", "privilege escalation": "privilege escalation",
        "buffer overflow": "buffer overflow", "directory traversal": "directory traversal|path traversal",
        "session hijacking": "session hijack", "dom clobbering": "dom clobbering", "clickjacking": "clickjacking",
        "dns spoofing": "dns (?:spoof|poison)", "zero-day": "zero[- ]day", "social engineering": "social engineering",
        "botnet": "botnet", "keylogger": "keylogger", "lateral movement": "lateral movement",
    },
    "TARGET": {
        "browser": r"\bbrowsers?\b", "web application": r"web app(?:lication)?s?", "api": r"\bapis?\b",
        "database": r"\bdatabases?\b|\bdbms?\b", "server": r"\bservers?\b", "endpoint": r"\bendpoints?\b|workstation",
        "network": r"\bnetworks?\b", "cloud": r"\bcloud\b|\baws\b|\bazure\b", "session": r"\bsessions?\b",
        "cookie": r"\bcookies?\b", "email": r"\be-?mails?\b", "credentials": r"credentials?|passwords?",
        "iot": r"\biot\b|embedded device", "active directory": r"active directory", "mobile app": r"mobile app",
        "dns": r"\bdns\b", "firewall": r"firewalls?", "user": r"\busers?\b|employees?|victims?",
    },
}
_GAZ_RE = {t: {c: re.compile(r"\b(?:%s)\b" % a, re.I) for c, a in d.items()} for t, d in _GAZ.items()}
# types utilises pour decrire une "relation" entre deux entites
ENTITY_TYPES = ["CVE", "CWE", "MITRE_TECHNIQUE", "IP", "HASH", "MALWARE", "SOFTWARE", "ATTACK", "TARGET"]


def _valid_ip(s):
    return all(int(x) <= 255 for x in s.split("."))


def extract_entities(text: str):
    """Retourne une liste unique de (TYPE, nom_canonique)."""
    out = set()
    for t, rx in _REGEX_ENT.items():
        for m in rx.findall(text):
            if t == "IP" and not _valid_ip(m):
                continue
            out.add((t, m.upper() if t in ("CVE", "CWE", "MITRE_TECHNIQUE") else m.lower()))
    for t, d in _GAZ_RE.items():
        for canon, rx in d.items():
            if rx.search(text):
                out.add((t, canon))
    # Evite "xss" quand une variante precise est deja presente
    for variant in ("reflected xss", "stored xss", "dom xss"):
        if ("ATTACK", variant) in out:
            out.discard(("ATTACK", "xss"))
    return sorted(out)


# --------------------------------------------------------------------------- embeddings
class Embedder:
    """kind='tfidf' (TF-IDF + SVD/LSA, hors-ligne) ou 'st' (sentence-transformers, ex. all-MiniLM-L6-v2)."""

    def __init__(self, kind="tfidf", name="sentence-transformers/all-MiniLM-L6-v2", dim=128):
        self.kind, self.name, self.dim = kind, name, dim

    def fit_transform(self, texts):
        if self.kind == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
            return self.encode(texts)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english", max_features=50000)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dim, X.shape[1] - 1), random_state=0)
        return self._norm(self.svd.fit_transform(X))

    def encode(self, texts):
        if self.kind == "st":
            return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))
        return self._norm(self.svd.transform(self.vec.transform(list(texts))))

    @staticmethod
    def _norm(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
