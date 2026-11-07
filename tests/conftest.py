"""Shared fixtures. Unit tests use a tiny hand-written corpus (never used for metrics); integration tests need `python -m cyberintel.pipeline all`."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINI = [
    "CVE-2021-44228 (Log4Shell) is a remote code execution flaw in Log4j. Attackers exploit it against Apache servers.",
    "Mimikatz dumps credentials from memory on a Windows host. Pass-the-hash then enables lateral movement in Active Directory.",
    "Phishing emails target users and steal credentials. Phishing is a social engineering attack against employees.",
    "ASLR mitigates ROP attacks. DEP also mitigates ROP, and CFG mitigates ROP on Windows.",
    "SQL injection targets a database behind a web application. A WAF mitigates SQL injection and input validation mitigates SQL injection.",
    "SQL injection targets the database of a web application. Parameterized queries mitigate SQL injection.",
    "Phishing targets email accounts. Phishing steals credentials from users.",
    "Dirty COW (CVE-2016-5195) is a Linux kernel race condition that allows privilege escalation.",
    "ASLR mitigates ROP chains because gadget addresses are randomised. Attackers need an information leak against ASLR.",
]


@pytest.fixture(scope="session")
def mini_df():
    df = pd.DataFrame({"id": range(len(MINI)), "question": [f"Explain topic {i}" for i in range(len(MINI))], "answer": MINI})
    df["text"] = df.question + " " + df.answer
    df["category"] = ["vulnerability", "malware", "phishing", "vulnerability", "injection", "injection", "phishing", "vulnerability", "vulnerability"]
    df["split"] = ["train"] * 5 + ["val", "test", "test", "train"]
    return df


@pytest.fixture(scope="session")
def has_artifacts():
    from cyberintel.pipeline import art
    from cyberintel.service import REQUIRED
    return all(art(f).exists() for f in REQUIRED)


@pytest.fixture(scope="session")
def svc(has_artifacts):
    if not has_artifacts:
        pytest.skip("artifacts not built: run `python -m cyberintel.pipeline all`")
    from cyberintel.service import CyberIntelService
    return CyberIntelService()
