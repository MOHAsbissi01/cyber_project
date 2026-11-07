import pytest

from cyberintel.ner import evaluate_ner, extract_entities, prf


def ents(text):
    return {(e.type, e.name) for e in extract_entities(text)}


def test_structured_identifiers():
    e = ents("CVE-2021-44228 and cwe-79, technique T1055.012 seen from 185.220.101.4 with hash 44d88612fea8a8f36de82e1278abb02f.")
    assert {("CVE", "CVE-2021-44228"), ("CWE", "CWE-79"), ("MITRE_TECHNIQUE", "T1055.012"), ("IP", "185.220.101.4"),
            ("HASH", "44d88612fea8a8f36de82e1278abb02f")} <= e


def test_version_numbers_and_section_numbers_are_not_ips():
    assert not any(t == "IP" for t, _ in ents("See section 3.6.5.2 Classification Algorithms. Fixed in version 7.29.0.1 of curl."))
    assert ("IP", "8.8.8.8") in ents("The public resolver 8.8.8.8 is widely used.")


def test_case_sensitive_ambiguous_words():
    assert ("SOFTWARE", "Windows") in ents("Attackers target Windows hosts.")
    assert ("SOFTWARE", "Windows") not in ents("The time windows overlap.")
    assert ("SOFTWARE", "Microsoft Exchange") not in ents("They use a key exchange protocol.")
    assert ("SOFTWARE", "Java") in ents("Java deserialisation")
    assert ("SOFTWARE", "Java") not in ents("JavaScript runs in browsers")


def test_inflections_and_variants():
    e = ents("Attackers use heap spraying and escalate privileges, then inject code into an existing process.")
    assert {("ATTACK", "heap spray"), ("ATTACK", "privilege escalation"), ("ATTACK", "process injection")} <= e


def test_generic_xss_dropped_next_to_specific_variant():
    e = ents("Stored Cross-Site Scripting (XSS) is dangerous.")
    assert ("ATTACK", "stored XSS") in e and ("ATTACK", "XSS") not in e


def test_urls_are_masked_for_gazetteers_and_hashes():
    e = ents("see https://github.com/x/curl/commit/3e95b6a80941a2d713f31785e14c50b5a04a70d0 and process monitoring")
    assert not any(t in ("HASH", "TECHNOLOGY") for t, _ in e) and ("TOOL", "Process Monitor") not in e


def test_prf_and_entity_level_evaluation():
    assert prf(2, 1, 1)["f1"] == pytest.approx(2 / 3, abs=1e-4)
    gold = [{"text": "Mimikatz on Windows", "entities": [["MALWARE", "Mimikatz"], ["SOFTWARE", "Windows"], ["TOOL", "Ghidra"]]}]
    r = evaluate_ner(gold)
    assert (r["overall"]["tp"], r["overall"]["fp"], r["overall"]["fn"]) == (2, 0, 1)
    assert r["per_type"]["TOOL"]["recall"] == 0.0
