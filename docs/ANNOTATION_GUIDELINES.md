# Annotation guidelines and status of the hand-made evaluation data

All evaluation data lives in `annotations/`. **The annotator for everything below is Claude (an AI assistant), not a human.** It was blind to
the system's outputs and (for the gold labels) to the weak labels. **The student should review it** — see "How to review" — because a professor
may ask who labelled the data. Do not describe these files as "human-annotated" unless the student has reviewed them.

## 1. `gold_labels.csv` — independent topic labels (183 records)

* **Sampling:** stratified by weak label, drawn only from the **test split** (≤ 20 per class, all of a class if it has fewer) so that models trained on
  weak labels are evaluated on records they never saw. `phishing` has only 1 gold example (the test split contains 2): its metrics are not meaningful.
* **Task:** assign ONE topic = the *primary subject of the question* (not of incidental words in the answer).
* **Columns:** `claude_label` (filled), `human_label` (empty, for the student). The loader uses `human_label` when present, otherwise `claude_label`.

| label | choose it when the question is mainly about… |
|---|---|
| `xss` | cross-site scripting, DOM/mutation XSS, CSP for XSS, client-side script injection (incl. CSS injection) |
| `injection` | SQLi, XXE, SSRF, LFI/RFI, template injection, insecure deserialisation, HTTP parameter pollution with injection |
| `csrf_session` | CSRF, SameSite/cookie attributes, session fixation/hijacking, session tokens |
| `phishing` | phishing and social engineering (incl. real-time phishing against 2FA) |
| `malware` | malware families and behaviour: ransomware, rootkits, trojans, cryptojacking, C2, evasion/anti-analysis, packers |
| `vulnerability` | memory corruption, exploitation techniques (ROP, ASLR bypass, heap), kernel/container/VM escape, Spectre/Meltdown, DLL hijacking, fuzzing |
| `network_attack` | DNS/ARP spoofing, MITM/BGP hijack, DDoS, Wi-Fi/5G attacks, covert network channels |
| `crypto` | cryptanalysis, side channels on ciphers, padding oracles, hashing/rainbow tables, TLS attacks, cold-boot key recovery |
| `incident_response` | IR, forensics, threat hunting, threat intelligence, SIEM/SOAR, detection engineering for APTs |
| `access_control` | authentication/authorisation, OAuth/JWT/SAML, Kerberos attacks, password spraying/credential stuffing, zero trust, IAM |
| `other` | none of the above is the main subject (risk scoring, IoT/automotive protocols, voice synthesis, formal methods…) |

Judgement calls (documented because they are arguable): DLL side-loading → `vulnerability`; Kerberos ticket forgery detection → `incident_response`;
YARA rule generation from malware → `malware`; DPA/fault attacks on keys → `crypto`; Zero Trust → `access_control`; watering hole → `other`.

Measured consequence: the keyword rules agree with these labels on **67.2 %** of the 183 records.

## 2. `ner_gold.json` — entity annotations (70 + 45 passages)

* Random two-sentence passages (150–430 characters) from the answers. Entity types = all lexicon types except `TARGET` (generic nouns).
* Names use the canonical form of `cyberintel/lexicon.py`. Entities that are **not** in the lexicon are still annotated (they count as false negatives).
* Sets: `dev` (pids 0–34; the lexicon was tuned on their errors → *not* a generalisation estimate), `test` (35–69; **optimistic**: the lexicon was
  extended after the annotator had read all 70 passages), `final` (100–144; drawn and annotated **after** the lexicon was frozen → the only unbiased estimate).
* `ner_precision_audit.json`: 137 random extractor mentions (per type) judged correct/incorrect from ±50 characters of context.

## 3. `qa_testset.json` — 69 hand-written questions (`scripts/build_qa_testset.py`)

* 51 "main" questions (dev + test) and 18 "paraphrase" questions written without consulting the system's outputs (different phrasings, 2 in French).
* Relevance of an incident = regex over the raw corpus (field `text`, or `question` when the question field defines the topic), independent of the system's NER.
* Answer correctness = every keyword group has ≥ 1 alternative in the answer. `expect_abstain` = the corpus has no evidence (the regex finds 0 documents).
* Limitations: written by the system's author, small, keyword matching is lenient for some "target" questions; see `reports/qa_eval*.json`.

## How to review (≈ 1 hour)

1. Open `annotations/gold_labels.csv`; for each row read `question` and fill `human_label` **only where you disagree** with `claude_label`.
2. Run `python -m cyberintel.pipeline classify` — the table then states `labels_source`.
3. Spot-check 20 passages of `ner_gold.json` and 20 rows of `ner_precision_audit.json`; fix errors and re-run `python -m cyberintel.pipeline ner`.
