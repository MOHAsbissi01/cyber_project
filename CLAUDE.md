# CLAUDE.md — Project brief for Claude Code (VS Code)

You are helping a student (ESPRIT, 5ERP-BI) finish **Subject S2: Cybersecurity – Text Analysis and Question-Answering**
for the module *Advanced Learning for Text and Graph Data* (BD-45, 100% graded on the project). Your job: continue from
the existing working prototype and **prepare every deliverable** listed below. Work step by step, run the code, and
report real numbers — never invent results. The student writes to you in English or French; answer in their language.

## 1. The assignment (source of truth)
Goal: exploit cybersecurity text to detect threats, classify incidents, and build a QA system.
- **NLP**: classify incidents (phishing, malware, vulnerability, …); embeddings; named-entity extraction (IP, malware, CVE, software…).
- **Graphs**: entity–incident graph linking attacks, vectors and targets; **GNN/GTN** to find patterns/relations between incidents.
- **Graph-RAG / QA**: answer e.g. "Which vulnerability affects this software?" and "Which incidents are similar?".
- **Expected deliverables**: (1) graph of incidents and entities with clusters / important relations;
  (2) a model that can classify and analyse incident texts, answer security questions, and identify relations between threats and targets.
- **Dataset**: Hugging Face `AlicanKiraz0/Cybersecurity-Dataset-v1` (columns `system`, `user`, `assistant`; 2,411 rows; apache-2.0; **no labels**).
- **Learning outcomes to cover** (AA1–AA8): define NLP/NLU/NLG; explain NLP role; apply preprocessing; analyse NLP→Transformers→LLM
  evolution; experiment with Transformers (classification/generation); **compare LLMs**; apply **GNN/GTN**; build **RAG and Graph-RAG** pipelines
  (course uses **Neo4j**). Module references: blog.jcharistech.com "building a text knowledge graph in python"; CS685 UMass.

## 2. Current state (rebuilt as the `cyberintel/` package; audit in `docs/AUDIT_REPORT.md`)
Run from this folder (Windows, Python 3.13, venv already created):
```bash
pip install -r requirements.txt
python scripts/setup_nltk.py                       # one-time WordNet + POS tagger into ./nltk_data
python -m cyberintel.pipeline all                  # corpus, ner, topics, classify, kg, rag_index, gnn, linkpred, qa_eval -> artifacts/ + reports/
python -m cyberintel.transformers_exp minilm       # optional (downloads ~90 MB); distilbert = ~45 min on CPU; then re-run `pipeline classify`
streamlit run app/main.py                          # 6-view UI;   python -m pytest tests -q   # 57 tests
```
The original flat prototype is kept untouched in `legacy_prototype/` (never cite its synthetic outputs). Read `PROJECT_DEFENSE_GUIDE.md` for what/why/evidence.

| Area | Files | Notes |
|---|---|---|
| Config | `config.toml`, `cyberintel/config.py` | all paths/hyper-parameters, seed 42 |
| Data | `data.py`, `textutil.py`, `preprocessing.py` | cleaning (unclosed `<think>` rows dropped, sentence-level reasoning removal), one stratified split in the corpus, chunking, NLTK steps |
| Labels | `labels.py`, `annotations/gold_labels.csv` | weak keyword labels + 183 gold labels (annotator = AI, **student review pending**) |
| NER / KG | `ner.py`, `lexicon.py`, `kg.py`, `resources/kb.json` | 15 entity types, sentence-level relations with evidence, reference KB (21 CVEs) fact-checking |
| ML | `classify.py`, `transformers_exp.py`, `gnn.py`, `linkpred.py`, `graph_analysis.py` | TF-IDF/MiniLM/DistilBERT; GCN/SAGE/GAT/GTN (pure torch); link prediction; Louvain |
| RAG | `rag/retrieval.py`, `rag/graph_store.py`, `rag/qa.py`, `rag/evaluation.py` | hybrid retrieval, Graph-RAG, abstention, grounding check, NetworkX + Neo4j stores |
| Backend / UI | `service.py`, `app/` | UI only reads the service and `reports/` |
| Evaluation data | `annotations/` | gold labels, NER gold (dev/test/final) + precision audit, 69-question QA set |

### Headline results (details and caveats: `reports/`, `PROJECT_DEFENSE_GUIDE.md` section 1)
Keyword rules agree with gold on 67.2 % · TF-IDF+LogReg macro-F1 0.646 weak / 0.644 gold (MiniLM 0.636/0.589, DistilBERT 0.601/0.523) ·
NER on a fresh sample P 0.97 / R 0.59 · KG 2,710 nodes, 378 relations with evidence · graph models bring no gain over text features (see `reports/gnn.json`) ·
QA answer correctness vector/hybrid/graph: 0.56/0.67/1.00 (main) and 0.40/0.60/0.93 (paraphrase), before the rule fixes.

## 3. Known problems / honest limitations
1. Dataset = LLM-generated concept Q&A (not incidents), no labels, some wrong CVE explanations; 201 rows with unfinished `<think>` are dropped.
2. Gold labels / NER gold / QA set were written by an AI assistant and by the system's author: the student must review `annotations/` (see `docs/ANNOTATION_GUIDELINES.md`).
3. NER recall is 0.59 on a fresh sample (closed vocabulary); relation-label precision was not measured.
4. GNN/GTN: no measurable gain over text features (homophily ~0.52); link-prediction graph is tiny.
5. **Neo4j code was never run against a live server** (`docs/NEO4J.md`); **Claude generation never run** (no API key); PEFT not done.
6. QA test sets are small; the numbers before the rule fixes (`reports/qa_eval_before_fixes.json`) are the unbiased ones.
7. UI tested with AppTest, a live-server probe and headless-browser screenshots; not reviewed by a human user.

## 4. TODO status (original list) and what remains
Done: set-up, gold subset (annotator = AI), better NER + evaluation, Transformer experiments + LLM comparison (documented), GNN/GTN + link prediction, Graph-RAG with hand-written QA set,
visuals, Streamlit demo, audit/architecture/compliance docs, defence guide.
Remaining for the student: (a) review/override `annotations/gold_labels.csv` (`human_label`) and spot-check `ner_gold.json`; (b) run Neo4j (`docs/NEO4J.md`) and, if a key is available,
Claude generation; (c) optional notebook walkthrough (not produced: avoids unexplained notebooks) and the slide deck / written report (the defence guide and `docs/` are the source material).

## 5. Working rules
- Keep code runnable offline by default; heavy/optional features behind flags. Don't commit data or API keys.
- After each change run the pipeline (or the relevant script) and state what actually happened, including failures.
- Don't cite synthetic-data metrics. Flag any weak-label caveat next to every classification number.
- Prefer small, reviewable edits; update `README.md` and this file's "Current state" when results change.
- Report before-fix AND after-fix numbers when a rule was changed because of a test failure; never quote tuned dev scores as generalisation.
