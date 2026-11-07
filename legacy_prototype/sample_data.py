"""Mini-jeu de donnees SYNTHETIQUE (pour tester le pipeline hors-ligne). Ne remplace pas le vrai dataset."""
import random

import pandas as pd

TOPICS = {
    "xss": ([("How does a {a} attack work against a {t}?", "A {a} attack injects a script that runs in the {t}. A {s} deployment without output encoding is exposed."),
             ("Explain how Content Security Policy mitigates {a} in {s}.", "A strict CSP limits script sources and reduces the impact of {a} on the {t}.")],
            ["reflected xss", "stored xss", "dom-based xss"], ["browser", "web application", "session"]),
    "injection": ([("What is {a} and how can it hit a {t} running {s}?", "{a} lets an attacker alter queries sent to the {t}. {s} is often affected when input is not validated, see CWE-89."),
                   ("How to prevent {a} in {s} applications?", "Use parameterized queries and validate input to protect the {t} from {a}.")],
                  ["SQL injection", "command injection", "XXE", "SSRF"], ["database", "server", "api"]),
    "phishing": ([("How do attackers run a {a} campaign against a {t}?", "A {a} campaign sends crafted e-mail to the {t}; victims enter credentials on a fake {s} login page."),
                  ("What are the signs of {a}?", "Sender spoofing, urgency and fake {s} portals are typical signs of {a} aimed at the {t}.")],
                 ["phishing", "spear-phishing", "social engineering"], ["email", "credentials", "user"]),
    "malware": ([("How does {m} malware operate on a {t}?", "{m} is a trojan that persists on the {t} and contacts a command and control server at 10.20.30.40 using {s}."),
                 ("How can {m} be detected on {s} hosts?", "Detect {m} with EDR rules and network monitoring; it uses lateral movement against each {t}.")],
                ["emotet", "trickbot", "mirai", "agent tesla"], ["endpoint", "network", "iot"]),
    "ransomware": ([("What happened in the {m} ransomware outbreak?", "{m} ransomware encrypts files on every {t} and demands payment; it spread through an {s} flaw (CVE-2017-0144)."),
                    ("How should an organisation respond to {m} ransomware?", "Isolate the {t}, restore from backups and report the {m} extortion to authorities.")],
                   ["wannacry", "lockbit", "ryuk", "revil"], ["server", "endpoint", "network"]),
    "vulnerability": ([("Which vulnerability affects {s}?", "{c} affects {s} and allows remote code execution against the {t}; patch immediately."),
                       ("Explain {c} in {s}.", "{c} is a critical vulnerability in {s}; exploit code targets the {t} (CWE-787).")],
                      ["rce", "privilege escalation", "buffer overflow"], ["server", "api", "cloud"]),
    "network_attack": ([("How does a {a} attack disrupt a {t}?", "A {a} overwhelms the {t}; a firewall and rate limiting on {s} help."),
                        ("How to detect {a} on {s} infrastructure?", "Monitor flows and use IDS signatures against {a} aimed at the {t}.")],
                       ["ddos", "mitm", "dns spoofing"], ["network", "dns", "firewall"]),
    "crypto": ([("Why is {s} TLS configuration important?", "Weak cipher suites in {s} let attackers break encryption between the client and the {t}."),
                ("Explain certificate validation and PKI for a {t}.", "PKI validates certificates so the {t} can trust the peer; hashing with SHA-256 protects integrity.")],
               ["zero-day"], ["server", "api", "browser"]),
    "incident_response": ([("What are the phases of incident response after {a}?", "Containment, eradication and recovery follow detection of {a}; analysts correlate SIEM logs on the {t} (T1059)."),
                           ("How do SOC teams hunt for {a}?", "Threat hunting uses MITRE ATT&CK technique T1566 and forensic log analysis on the {t}.")],
                          ["lateral movement", "ransomware", "phishing"], ["endpoint", "active directory", "cloud"]),
    "access_control": ([("How to secure authentication against {a} on a {t}?", "Enable MFA, rate limiting and strong password policy to stop {a} against the {t} behind {s}."),
                        ("What is OAuth and JWT misuse risk for an {t}?", "Poorly validated JWT tokens give attackers access to the {t} in {s}.")],
                       ["brute force", "session hijacking"], ["api", "credentials", "user"]),
}
SOFT = ["apache", "nginx", "openssl", "log4j", "wordpress", "windows", "linux", "jenkins", "tomcat", "docker",
        "kubernetes", "exchange server", "sharepoint", "openssh", "citrix", "vmware"]
CVES = ["CVE-2021-44228", "CVE-2017-0144", "CVE-2014-0160", "CVE-2019-0708", "CVE-2020-1472", "CVE-2022-22965", "CVE-2021-26855"]
SOFT_CVE = dict(zip(SOFT, [CVES[i % len(CVES)] for i in range(len(SOFT))]))


def make_sample(n=400, seed=0):
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        cat = rnd.choice(list(TOPICS))
        qs, attacks, targets = TOPICS[cat]
        q, a = rnd.choice(qs)
        s = rnd.choice(SOFT)
        kw = dict(a=rnd.choice(attacks), t=rnd.choice(targets), s=s, m=rnd.choice(attacks), c=SOFT_CVE[s])
        extra = rnd.choice(["", " Defenders should monitor logs.", " Apply least privilege and patch regularly.",
                            " Attackers often automate this at scale."])
        rows.append({"user": q.format(**kw) + f" (case {i})", "assistant": a.format(**kw) + extra})
    return pd.DataFrame(rows)
