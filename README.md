# CyberIntel — cybersecurity text analysis, knowledge graph, graph ML and Graph-RAG

ESPRIT 5ERP-BI · module *Advanced Learning for Text and Graph Data* (BD-45) · subject **S2: Cybersecurity – Text Analysis and Question-Answering**.

A complete pipeline and a web app that: classify security texts, extract entities (CVE, malware, software, attacks, defences…), build an **entity–incident knowledge graph** with
evidence for every relation, find clusters and similar incidents, apply **GNN / Graph-Transformer** models, and answer security questions with **RAG vs Graph-RAG** — refusing when the data has no answer.

> **Read this first.** The dataset (`AlicanKiraz0/Cybersecurity-Dataset-v1`) is **LLM-generated Q&A about security concepts, not real incident reports, and it has no labels.**
> "Incident" therefore means "Q&A record"; topic categories are keyword-based *weak labels* (they agree with an independent gold subset on 67 % of records), and every classification result is reported on both.
> The oral-defence file is **[PROJECT_DEFENSE_GUIDE.md](PROJECT_DEFENSE_GUIDE.md)** (what was done, why, evidence, limits, Q&A bank).

## Quick start (Windows, Python 3.13)

```powershell
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python scripts\setup_nltk.py            # one-time: WordNet + POS tagger into ./nltk_data
# put the dataset at data/train.parquet (or add --source hf to the next command to download it)
venv\Scripts\python -m cyberintel.pipeline all       # builds artifacts/ and reports/ (offline; ~20-30 min, mostly GNN training)
venv\Scripts\streamlit run app/main.py               # http://localhost:8501
venv\Scripts\python -m pytest tests -q               # 57 tests
```
Optional (downloads models; results are merged by `pipeline classify`):
`python -m cyberintel.transformers_exp minilm` (≈ 90 MB, 2–3 min) · `python -m cyberintel.transformers_exp distilbert` (≈ 270 MB, ≈ 45 min on CPU).

## The app (6 views, all connected to the real backend)

| View | What you can do |
|---|---|
| **Dashboard** | dataset and graph statistics, topic distribution, strongest relations, fact-check summary, data caveat |
| **Incident analysis** | paste a text → predicted topic with confidence, entities by type, graph context with verification badges, related incidents, preprocessing steps |
| **Knowledge graph** | search an entity → relations with evidence sentences, interactive neighbourhood, global graph filtered by type/support, clusters, all relations table |
| **Question answering** | chat; Graph-RAG, hybrid RAG, vector RAG or side-by-side comparison; facts with provenance, passages, grounding; refusal when unanswerable |
| **Model evaluation** | classification (weak vs gold, confusion matrices, preprocessing ablation), NER, GNN/GTN, link prediction, QA, graph statistics — all read from `reports/` |
| **Dataset explorer** | quality findings, class/split distribution, filterable record browser |

### Screenshots (`docs/screenshots/`, captured from the running app with headless Edge)
[Dashboard](docs/screenshots/01_dashboard.png) · [Incident analysis](docs/screenshots/02_incident_analysis.png) · [Knowledge-graph explorer](docs/screenshots/03_knowledge_graph_explorer.png) ·
[Graph-RAG answer](docs/screenshots/04_question_answering_graph_rag.png) · [Refusal on an unanswerable question](docs/screenshots/05_question_answering_abstention.png) ·
[Model evaluation](docs/screenshots/06_model_evaluation.png) · [Dataset explorer](docs/screenshots/07_dataset_explorer.png)

## Results at a glance (sources: `reports/`; details and caveats in the defence guide)

| Topic | Result |
|---|---|
| Keyword rules vs gold labels | 67.2 % agreement (κ 0.63) on 183 records |
| Topic classification (macro-F1, weak test / gold) | TF-IDF + LogReg **0.646 / 0.644** · MiniLM + LogReg 0.636 / 0.589 · DistilBERT 0.601 / 0.523 |
| NER, fresh annotated sample (58 entities) | precision **0.97**, recall **0.59** (precision audit 133/137) |
| Knowledge graph | 2,710 nodes, 25,970 edges, 378 semantic relations with evidence; 12 incident clusters |
| Reference-KB fact check of corpus CVE claims | 14 verified · 5 differ · 2 contradicted |
| GNN / Graph Transformer (node classification, minilm features) | MLP 0.642 · GCN 0.582 · GTN 0.653 (test macro-F1, 5 seeds): SAGE/GTN ≈ MLP within seed noise, GCN/GAT clearly lower: no significant gain from the graph |
| Link prediction (entity relations, AUC) | structure heuristics 0.50–0.53 · features only 0.78 · GCN-AE 0.84 |
| QA answer correctness, vector / hybrid / Graph-RAG | main set (51 q): 0.56 / 0.67 / **1.00** · paraphrase set (18 q): 0.40 / 0.60 / **0.93** (before the rule fixes; see the guide) |

## Repository layout

```
config.toml              all paths and hyper-parameters            requirements.txt   pinned dependencies
cyberintel/              the package: data, labels, ner, kg, classify, gnn, linkpred, rag/, service, pipeline
app/                     Streamlit UI (main.py, ui.py, views/)     tests/             57 pytest tests (unit + integration + UI)
annotations/             gold labels, NER gold, QA test set        reports/           metrics, figures, GraphML + Neo4j exports
docs/                    audit, architecture, course compliance, LLM comparison, annotation guidelines, Neo4j, screenshots/
scripts/                 build_qa_testset.py, neo4j_import.py, setup_nltk.py
legacy_prototype/        the original prototype, untouched (see its README_LEGACY.md)
PROJECT_DEFENSE_GUIDE.md the file to read before the oral defence
```

## Documentation index

* [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) — Phase-1 audit of the prototype (status of every component, evidence).
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — layers, diagram, design decisions.
* [`docs/COURSE_COMPLIANCE.md`](docs/COURSE_COMPLIANCE.md) — every course requirement → implementation → evidence → status.
* [`docs/LLM_COMPARISON.md`](docs/LLM_COMPARISON.md) — AA6: model families and the measured comparison.
* [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md) — how the hand-made evaluation data was produced and how to review it.
* [`docs/NEO4J.md`](docs/NEO4J.md) — Neo4j status and how to run it.

## Limitations (full list in the defence guide §6)

Data is LLM-written concept Q&A (some CVE explanations are wrong; the KB check flags what it can). Gold labels and NER gold were produced by an AI annotator and await the student's review.
NER recall is limited by its closed vocabulary. Graph models bring no measurable gain over text features on this data. **Neo4j code is written but was not run against a live server;
Claude generation was not run (no API key).** QA test sets are small and written by the system's author.
