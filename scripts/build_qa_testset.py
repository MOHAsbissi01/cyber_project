"""Builds annotations/qa_testset.json (hand-written QA test set) and sanity-checks it against the raw corpus.

Relevance of an incident to a question is defined HERE, by a regex over the corpus text (field 'text' = question+answer, or 'question' only
when the question field defines the topic) - independent of the NER/graph used by the system. Answer keywords: the answer is correct if EVERY group
has at least one of its alternatives (case-insensitive substring). 'abstain': the corpus has no evidence; the right behaviour is to refuse.
The set was written after inspecting the corpus (see PROJECT_DEFENSE_GUIDE.md). split: 'dev' = used while building the abstention logic,
'test' = never used for tuning.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cyberintel.pipeline import load_corpus  # noqa: E402

Q = []


def q(id, question, kind, split="test", rel=None, field="text", keys=None, abstain=False, intent=None, kb_only=False):
    Q.append({"kb_only": kb_only, "id": id, "question": question, "kind": kind, "split": split, "relevant_regex": rel, "relevance_field": field,
              "answer_keywords": keys or [], "expect_abstain": abstain, "expected_intent": intent})


# --- A. which vulnerabilities affect a software (graph facts; some answers exist only in the reference KB)
q("A01", "Which vulnerabilities affect log4j?", "software->vulnerability", "dev", r"log4j|log4shell", keys=[["CVE-2021-44228"]], intent="vulns_of_software")
q("A02", "Which CVE affects Docker containers?", "software->vulnerability", rel=r"CVE-2019-5736", keys=[["CVE-2019-5736"]], intent="vulns_of_software")
q("A03", "Which vulnerability affects Kubernetes?", "software->vulnerability", rel=r"CVE-2018-1002105", keys=[["CVE-2018-1002105"]], intent="vulns_of_software", kb_only=True)  # corpus row was an unfinished <think> dump, dropped by cleaning
q("A04", "Which vulnerabilities affect Firefox?", "software->vulnerability", rel=r"CVE-2016-5273|CVE-2017-5123", keys=[["CVE-2016-5273", "CVE-2017-5123"]], intent="vulns_of_software")
q("A05", "Which vulnerabilities affect Chrome?", "software->vulnerability", rel=r"CVE-2019-5786|CVE-2020-6463", keys=[["CVE-2019-5786", "CVE-2020-6463"]], intent="vulns_of_software")
q("A06", "Which CVE affects Internet Explorer?", "software->vulnerability", rel=r"CVE-2014-6332|CVE-2016-0189", keys=[["CVE-2014-6332", "CVE-2016-0189"]], intent="vulns_of_software")
q("A07", "Which CVE affects OpenSSL?", "software->vulnerability", rel=r"openssl", keys=[["CVE-2014-0160"]], intent="vulns_of_software")
q("A08", "Which vulnerability affects PHP?", "software->vulnerability", rel=r"CVE-2018-19374|CVE-2016-8357", keys=[["CVE-2018-19374", "CVE-2016-8357"]], intent="vulns_of_software")
# --- B. which systems are affected by a vulnerability
q("B01", "Which software does CVE-2016-5195 affect?", "vulnerability->software", rel=r"CVE-2016-5195", keys=[["Linux"]], intent="software_of_cve")
q("B02", "Which systems are affected by CVE-2017-0144?", "vulnerability->software", rel=r"CVE-2017-0144", keys=[["Windows", "SMB"]], intent="software_of_cve")
q("B03", "Which software does CVE-2021-44228 affect?", "vulnerability->software", "dev", rel=r"CVE-2021-44228", keys=[["Log4j"]], intent="software_of_cve")
q("B04", "What does CVE-2019-5736 affect?", "vulnerability->software", rel=r"CVE-2019-5736", keys=[["Docker", "runC"]], intent="software_of_cve")
q("B05", "Which products are affected by CVE-2020-1472?", "vulnerability->software", rel=r"CVE-2020-1472|zerologon", keys=[["Windows", "Netlogon", "Active Directory"]], intent="software_of_cve")
q("B06", "Which software is affected by CVE-2018-1002105?", "vulnerability->software", rel=r"CVE-2018-1002105", keys=[["Kubernetes"]], intent="software_of_cve", kb_only=True)
q("B07", "What is Heartbleed?", "vulnerability->software", rel=r"heartbleed", keys=[["OpenSSL"]], intent="describe_cve")
q("B08", "What is the CVE of Dirty COW?", "vulnerability->software", rel=r"dirty\s?cow", keys=[["CVE-2016-5195"]], intent="describe_cve")
# --- C. incidents related to a malware / tool / actor (relevance = text mentions it)
q("C01", "Which incidents are related to Mimikatz?", "incidents-of-entity", rel=r"mimikatz", intent="incidents_of_entity")
q("C02", "Which incidents mention Stuxnet?", "incidents-of-entity", rel=r"stuxnet", intent="incidents_of_entity")
q("C03", "Which incidents are related to Emotet?", "incidents-of-entity", rel=r"emotet", intent="incidents_of_entity")
q("C04", "Which incidents discuss Kerberoasting?", "incidents-of-entity", rel=r"kerberoast", intent="incidents_of_entity")
q("C05", "Which incidents are about Spectre?", "incidents-of-entity", rel=r"spectre", intent="incidents_of_entity")
q("C06", "Which incidents are related to Cobalt Strike?", "incidents-of-entity", rel=r"cobalt strike", intent="incidents_of_entity")
q("C07", "Which incidents mention APT28?", "incidents-of-entity", rel=r"apt\s?28|sofacy|fancy bear", intent="incidents_of_entity")
q("C08", "Which incidents are related to Metasploit?", "incidents-of-entity", rel=r"metasploit", intent="incidents_of_entity")
# --- D. targets of an attack (keywords from domain knowledge)
q("D01", "What does SQL injection target?", "attack->targets", rel=r"sql ?injection|sqli", field="question", keys=[["database", "web application"]], intent="targets_of_attack")
q("D02", "What does phishing target?", "attack->targets", rel=r"phishing", field="question", keys=[["email", "credentials", "user"]], intent="targets_of_attack")
q("D03", "What does a botnet target?", "attack->targets", rel=r"botnet", field="question", keys=[["IoT", "server", "endpoint", "network"]], intent="targets_of_attack")
q("D04", "What do rootkits target?", "attack->targets", rel=r"rootkit", field="question", keys=[["kernel", "UEFI", "firmware", "boot", "Intel"]], intent="targets_of_attack")
q("D05", "What does XSS target?", "attack->targets", rel=r"\bxss\b|cross-site scripting", field="question", keys=[["browser", "web application", "user"]], intent="targets_of_attack")
q("D06", "What does Kerberoasting target?", "attack->targets", rel=r"kerberoast", keys=[["Kerberos", "Active Directory", "credentials", "service"]], intent="targets_of_attack")
# --- E. mitigations
q("E01", "How can ROP attacks be mitigated?", "defence", rel=r"\brop\b|return-oriented", field="question", keys=[["ASLR", "DEP", "CFI", "CFG", "canary", "shadow stack", "Control Flow"]], intent="mitigations")
q("E02", "How do you defend against CSRF?", "defence", rel=r"csrf|cross-site request forgery", field="question", keys=[["SameSite", "token"]], intent="mitigations")
q("E03", "How can SQL injection be prevented?", "defence", rel=r"sql ?injection|sqli", field="question", keys=[["parameteriz", "prepared statement", "input validation", "WAF", "sanitiz"]], intent="mitigations")
q("E04", "What mitigates cross-site scripting?", "defence", rel=r"\bxss\b|cross-site scripting", field="question", keys=[["CSP", "Content Security Policy", "encod", "sanitiz", "escap"]], intent="mitigations")
# --- F. relationship between two entities
q("F01", "What is the relationship between ASLR and ROP?", "relationship", rel=r"aslr", field="question", keys=[["mitigat"]], intent="relation_between")
q("F02", "What is the relationship between Mimikatz and Kerberos?", "relationship", rel=r"mimikatz", keys=[["Kerberos"]], intent="relation_between")
# --- G. similar incidents
q("G01", "Which incidents are similar to incident 3?", "similar", rel=r"\bxss\b|cross-site scripting", field="question", intent="similar_incidents")
# --- H. characteristics / definition (answer keywords + topical relevance)
q("H01", "What is Kerberoasting?", "characteristics", rel=r"kerberoast", keys=[["service ticket", "SPN", "service principal", "TGS", "offline", "crack"]], intent="describe_entity")
q("H02", "What is a watering hole attack?", "characteristics", rel=r"watering hole", field="question", keys=[["website", "compromis"]])
q("H03", "How does ASLR work?", "characteristics", rel=r"\baslr\b", field="question", keys=[["random"]])
q("H04", "What is session fixation?", "characteristics", rel=r"session fixation", field="question", keys=[["session id", "session identifier", "session token", "known"]])
q("H05", "What are the characteristics of ransomware?", "characteristics", rel=r"ransomware", field="question", keys=[["encrypt"]])
q("H06", "Explain Spectre.", "characteristics", rel=r"spectre", keys=[["speculat"]])
# --- I. unanswerable: the corpus has no evidence (the system must abstain)
q("U01", "Which vulnerability affects Zorbatron-9000 firmware?", "unanswerable", "dev", abstain=True)
q("U02", "What is the capital of France?", "unanswerable", "dev", abstain=True)
q("U03", "Which CVE affects SAP HANA 2.0?", "unanswerable", abstain=True, rel=r"sap hana")
q("U04", "Which incidents are related to the Conficker worm?", "unanswerable", abstain=True, rel=r"conficker")
q("U05", "Which CVE affects Siemens S7-1500 PLCs?", "unanswerable", abstain=True, rel=r"s7-1500")
q("U06", "Who is the CEO of Microsoft?", "unanswerable", abstain=True)
q("U07", "Which vulnerability affects Oracle WebLogic 12c?", "unanswerable", abstain=True, rel=r"weblogic")
q("U08", "What is the recipe for chocolate cake?", "unanswerable", abstain=True)

# --- P. PARAPHRASE set: written in one go, WITHOUT consulting the system's outputs, with phrasings and a language (French) that the
#        intent rules were not designed around. Reported separately from the sets above (which share their phrasing patterns with the router).
q("P01", "Tell me about known flaws in Windows.", "paraphrase:software->vulnerability", "paraphrase", rel=r"CVE-2017-0144|CVE-2020-1472|CVE-2021-34527|CVE-2019-0708", keys=[["CVE-2017-0144", "CVE-2019-0708", "CVE-2020-1472", "CVE-2021-34527"]])
q("P02", "Are there any CVEs known for Docker?", "paraphrase:software->vulnerability", "paraphrase", rel=r"CVE-2019-5736", keys=[["CVE-2019-5736"]])
q("P03", "Quelle faille touche Log4j ?", "paraphrase:software->vulnerability", "paraphrase", rel=r"log4j|log4shell", keys=[["CVE-2021-44228"]])
q("P04", "Which operating systems does the Dirty COW bug impact?", "paraphrase:vulnerability->software", "paraphrase", rel=r"dirty\s?cow", keys=[["Linux"]])
q("P05", "What is EternalBlue's CVE identifier?", "paraphrase:vulnerability->software", "paraphrase", rel=r"eternalblue", keys=[["CVE-2017-0144"]])
q("P06", "Give me incidents that talk about the Stuxnet worm.", "paraphrase:incidents-of-entity", "paraphrase", rel=r"stuxnet")
q("P07", "What kinds of assets do rootkits go after?", "paraphrase:attack->targets", "paraphrase", rel=r"rootkit", field="question", keys=[["kernel", "UEFI", "firmware", "boot", "Intel"]])
q("P08", "Which defences stop return-oriented programming?", "paraphrase:defence", "paraphrase", rel=r"\brop\b|return-oriented", field="question", keys=[["ASLR", "DEP", "CFI", "CFG", "canary", "shadow stack", "Control Flow"]])
q("P09", "How does Kerberoasting relate to Active Directory?", "paraphrase:relationship", "paraphrase", rel=r"kerberoast", keys=[["Active Directory"]])
q("P10", "Describe how Spectre attacks work.", "paraphrase:characteristics", "paraphrase", rel=r"spectre", keys=[["speculat"]])
q("P11", "Which CVE affects Zyxel firewalls?", "paraphrase:unanswerable", "paraphrase", abstain=True, rel=r"zyxel")
q("P12", "Quelle est la capitale de l'Allemagne ?", "paraphrase:unanswerable", "paraphrase", abstain=True)
q("P13", "Which incidents mention the NotPetya wiper?", "paraphrase:incidents-of-entity", "paraphrase", rel=r"notpetya|petya")
q("P14", "What does ransomware go after?", "paraphrase:attack->targets", "paraphrase", rel=r"ransomware", field="question", keys=[["email", "network", "server", "endpoint", "credentials", "user", "file", "data"]])
q("P15", "Explain what a watering-hole campaign is.", "paraphrase:characteristics", "paraphrase", rel=r"watering hole", field="question", keys=[["website", "compromis"]])
q("P16", "Which products does CVE-2019-0708 impact?", "paraphrase:vulnerability->software", "paraphrase", rel=r"CVE-2019-0708", keys=[["Windows", "RDP"]])
q("P17", "Which vulnerabilities affect Apache Struts?", "paraphrase:software->vulnerability", "paraphrase", keys=[["CVE-2017-5638"]], kb_only=False, rel=r"struts")
q("P18", "Which CVE affects Cisco IOS XE?", "paraphrase:unanswerable", "paraphrase", abstain=True, rel=r"ios xe")

if __name__ == "__main__":
    df = load_corpus()
    ok = True
    for x in Q:
        if x["relevant_regex"]:
            col = df.question if x["relevance_field"] == "question" else df.text
            n = int(col.str.contains(x["relevant_regex"], case=False, regex=True).sum())
            x["n_relevant"] = n
            flag = "" if (n == 0) == (x["expect_abstain"] or x["kb_only"]) else "   <-- CHECK"
            ok &= not flag
            print(f"{x['id']} relevant={n:4d} {flag}  {x['question']}")
        else:
            print(f"{x['id']} {'(abstain)' if x['expect_abstain'] else ''} {x['question']}")
    out = Path(__file__).resolve().parent.parent / "annotations" / "qa_testset.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"_comment": __doc__, "questions": Q}, f, indent=1, ensure_ascii=False)
    print(len(Q), "questions ->", out, "| all relevance checks consistent:", ok)
