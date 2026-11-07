# Phase 1 — Audit report of the existing prototype

*Scope: everything in the project folder as received (flat prototype: `cyberlib.py`, `graph_tools.py`, `qa.py`,
`run_pipeline.py`, `app.py`, `transformer_finetune.py`, `sample_data.py`, `outputs*/`, `class_course/`).
Nothing was modified during the audit. Every number below was measured on the real data (`data/train.parquet`)
by re-running the prototype or by probing the data; the probe scripts are reproducible.*

## 1. Academic requirements extracted from `class_course/`

| Source | Requirement | Course ID |
|---|---|---|
| Fiche module (AA1–AA8) | Define NLP/NLU/NLG; role of NLP in the pipeline; classical preprocessing + linguistic analysis; NLP → Transformers → LLM evolution; experiment with Transformers (classification/generation); **compare LLMs**; **GNN/GTN on graph data**; **RAG and Graph-RAG pipelines** | AA1…AA8 |
| Chap. 1 (NLP) | NLU tasks (classification, **NER**, parsing, topic modelling/LDA); preprocessing (tokenisation, stop words, stemming, lemmatisation, POS, chunking); vectorisation (BoW, **TF-IDF**, Word2Vec/GloVe, contextual); **cosine similarity**; **text → graph** construction (nodes, edges, weights, NetworkX/Gephi/Neo4j) | AA1–AA3 |
| Chap. 2 (Transformers) | RNN/LSTM limits; self-/multi-head attention; encoder/decoder/enc-dec families; BERT; **Graph Transformer Networks (attention over neighbours) vs classical GNN** | AA4, AA5, AA7 |
| Chap. 3 (LLMs) | LLM definition, fine-tuning/PEFT, prompt engineering, evaluation (BLEU/ROUGE/EM/human), limits (hallucination, bias, context window) | AA4, AA6 |
| Chap. 4 (RAG) | Knowledge base, chunking, retriever, integration layer, generator; **evaluation: faithfulness, relevance, Precision@k, Recall@k**; **Graph-RAG pipeline: NER + relation extraction → graph → Neo4j → vector AND graph retrieval → fusion → LLM conditioning**; RAG vs Graph-RAG comparison table | AA8 |
| Fiche de route | S7 = validation; deliverables are homework + project (100 % of the grade) | — |

## 2. Dataset facts (measured)

* `AlicanKiraz0/Cybersecurity-Dataset-v1`, 2,411 rows × (`system`, `user`, `assistant`), no labels, one identical `system` prompt.
* **It is a Q&A corpus of security *concepts* (LLM-generated answers), not a corpus of incident reports.** The word "incident"
  in the prototype is a misnomer: a record is a (question, answer) pair. Only 69 rows contain a CVE id, 31 an IP, 1 a CWE.
* 354 answers contain `<think>` but only 188 contain `</think>` → the prototype's regex `<think>.*?</think>` leaves **211 cleaned rows
  still containing leaked chain-of-thought** ("Okay, the user wants…", "Alright, so I need to…"). Literal `\n\n` sequences also remain in answers.
* Answers average 3,034 chars but the prototype keeps only the first 1,500 (`text = question + answer[:1500]`) for NER/embeddings.
* 15 duplicate questions, 1 duplicate answer.

**Consequence:** the dataset supports classification of *topics*, a concept-level knowledge graph, and Q&A grounded in explanatory passages.
It supports named-entity-rich threat-intel questions (CVE→software) only marginally. This must be stated, not hidden.

## 3. Component-by-component status

Legend: **COMPLETE** · **PARTIAL** (works, needs improvement) · **INCORRECT** (methodological/technical problem) · **MISSING** · **UNNECESSARY**.

| # | Component | Status | Evidence / reasoning | Decision |
|---|---|---|---|---|
| 1 | Directory structure, config | PARTIAL | Flat scripts, hard-coded paths (`data/train.parquet` in `app.py`), no config, no tests, no `.gitignore`, unpinned `requirements.txt`, no git | **Rebuild** as package |
| 2 | Data loading | PARTIAL | Works; handles several column names. Dedupes on question only; no split; no data-quality report | Refactor |
| 3 | Cleaning | INCORRECT | `<think>` regex leaves 211 rows with chain-of-thought; `\n` literals; `answer[:1500]` truncation halves the text NER sees | **Fix** |
| 4 | Weak labels (11 keyword classes) | INCORRECT (as documented) | README says label comes "mostly from the question". Measured: label = question-only label in **71 %** of rows but = **answer-only** label in **85 %**. A classifier trained on the answer therefore largely re-learns the keyword rules → F1 0.54 measures rule reconstruction, not threat understanding. No gold labels exist | **Keep as silver baseline, add independent gold set, report both** |
| 5 | Classification | PARTIAL | Only TF-IDF+LogReg. Vectoriser fitted on all data (unsupervised leak). No confusion matrix, no validation set, no model comparison | Extend |
| 6 | Split consistency | INCORRECT | LogReg: stratified 80/20; GCN/MLP: random 60/20/20 with a different permutation; DistilBERT script: unstratified 80/20; retrieval eval uses `te[:200]` of yet another split → **the reported numbers are not comparable** | **One global split** |
| 7 | Embeddings | PARTIAL | TF-IDF+SVD works; `st` path never run. `min_df=2` drops rare entities (e.g. `log4j` not in vocabulary → retrieval fails). Fitted on all data | Extend (lexical + dense) |
| 8 | NER | PARTIAL | Regex + gazetteers run. No evaluation. `3.6.5.2` (a version string) typed as IP. TARGET (generic words: user, network, server) = **67 %** of all mentions. 992/2,394 rows have no non-TARGET entity. No MITRE technique names, no CVE→product knowledge, no alias normalisation table. 1 CWE found | **Improve + evaluate** |
| 9 | Knowledge graph | PARTIAL | Typed nodes OK. But relation names are **assigned by type pair** (any ATTACK–TARGET pair → `TARGETS`), derived from co-occurrence; no provenance; undirected `nx.Graph` while relations are directional. Edge mix: 13,127 `SIMILAR_TO`, 6,882 `MENTIONS`, **only 377 semantic edges (AFFECTS = 3, ENABLES = 2, USES = 12)** | **Rebuild** (directed, provenance, evidence incidents) |
| 10 | Communities (Louvain) | PARTIAL | Works (18 communities, purity 0.26–0.81). Run on the mixed graph, so similarity edges dominate | Keep, improve description |
| 11 | GCN (pure torch) | PARTIAL | Correct 2-layer Kipf GCN. **GCN 0.47 < MLP 0.52 < LogReg 0.54**, single seed, different split, no analysis | **Investigate** (homophily, k sweep, seeds) |
| 12 | GTN / attention on graphs | MISSING | Explicit in course (Chap. 2, AA7) | **Implement** + compare |
| 13 | Link prediction | MISSING | Graph-ML task that actually uses structure | **Implement** |
| 14 | Transformers (AA5) | MISSING | `transformer_finetune.py` never executed, inconsistent split, no val set | **Run + compare** |
| 15 | LLM comparison (AA6) | MISSING | — | **Write** (documented comparison) |
| 16 | Vector RAG | PARTIAL | Works for topical queries; fails on rare tokens | Improve (hybrid) |
| 17 | Graph-RAG | INCORRECT | Seed + expansion heuristic; "Which vulnerability affects log4j?" returns an unrelated *Cyber Kill Chain* passage (see `outputs_real/qa_demo.md`). **No abstention**, no grounding check. "Generation" is the first 2 sentences of the top passage; `--llm` untested | **Rebuild** |
| 18 | Neo4j | MISSING | Only CSV + Cypher text; no driver, no query layer. Course Chap. 4 requires Neo4j | **Implement** (Cypher store + in-memory twin) |
| 19 | Retrieval/QA evaluation | INCORRECT | hit@5 uses the *weak label* as relevance (circular: label and entities share the same keywords); 0.825 vs 0.835 is within noise; no gold QA set | **Hand-written QA set (≥30)** |
| 20 | Backend / service layer | MISSING | `app.py` recomputes everything and trains a classifier at start-up | **Create** `service.py` + cached artifacts |
| 21 | Frontend (Streamlit) | PARTIAL | Tested with `streamlit.testing`: loads, no exception, 4 tabs work. Missing: dashboard, evaluation page, dataset explorer; French only; recomputation on start | Rebuild as multi-page app |
| 22 | Tests | MISSING | none | **Add** pytest |
| 23 | Documentation | PARTIAL | README useful but contains claims that must be corrected | Rewrite |
| 24 | `sample_data.py`, `outputs/` (synthetic, scores ≈ 1.0) | UNNECESSARY (for results) | Template-generated; scores meaningless | Keep **only** as unit-test fixture; never cite |
| 25 | `outputs_real/` | PARTIAL | Reproduced exactly in this audit (macro-F1 0.5404, GCN 0.4694, hit@5 0.825/0.835). Honest baseline | Keep as "prototype baseline" |

## 4. What works / what to reuse / what to rebuild

**Works (reproduced):** end-to-end pipeline in ~70 s; GraphML/Neo4j-CSV export; Louvain; pure-torch GCN; Streamlit app starts and all 4 tabs execute without exception.

**Reuse:** category regexes (as the *silver* baseline), the gazetteer idea, `Embedder`, `build_graph` co-occurrence/lift logic, `train_node_classifier`, the Streamlit tab ideas.

**Refactor:** loading/cleaning, split handling, graph construction (directed + provenance), exports.

**Rebuild:** Graph-RAG retrieval + answering (abstention, grounding), evaluation protocol, UI architecture.

**Missing and to implement:** gold labels, NER evaluation, Transformer experiments, GTN, link prediction, Neo4j layer, QA test set, service layer, tests, LLM comparison, course-compliance checklist, defence document.

## 5. Methodological risks to avoid in the rebuild

1. Never present agreement-with-rules as ground-truth accuracy → always report **weak-label test** *and* **gold** metrics, side by side.
2. One deterministic stratified split (train/val/test) shared by every model; unsupervised fitting (TF-IDF, LSA) on **train only** for classification experiments.
3. Multiple seeds, mean ± std, for neural models.
4. QA relevance defined independently of the extractors used by the system.
5. A QA system must be able to **abstain** when the corpus has no evidence.
