# Course compliance checklist (`class_course/` → implementation → evidence)

Status legend: **Done** (implemented and tested/measured) · **Partial** (implemented, with a stated gap) · **Documented** (conceptual deliverable, no code needed) · **Not done** (with reason).
Evidence files are in `reports/` (regenerate with `python -m cyberintel.pipeline all`) or `tests/`.

## Learning outcomes (module sheet, AA1–AA8)

| AA | Requirement | Where implemented | Evidence | Status |
|---|---|---|---|---|
| AA1 | Define NLP, NLU, NLG | `PROJECT_DEFENSE_GUIDE.md` §3 maps them to the project: **NLU** = classification, NER, relation extraction, topic modelling; **NLG** = grounded answer generation | the guide; UI shows NLU outputs (Incident analysis) and NLG output (Question answering) | Documented |
| AA2 | Role of NLP in the processing chain | `docs/ARCHITECTURE.md` (pipeline diagram: ingestion → preprocessing → vectorisation → modelling → evaluation/deployment, as in course Chap. 1 p.16) | architecture diagram; `pipeline.py` stages | Documented + Done |
| AA3 | Classical preprocessing and linguistic analysis | `preprocessing.py` (normalisation, tokenisation, stop words, Porter stemming, WordNet lemmatisation, POS tagging), `data.py` (cleaning), `ner.py`, `stage_topics` (LDA) | `reports/classification.json::preprocessing_ablation` (effect measured), `reports/topics.json`, UI "Preprocessing steps" expander, `tests/test_data.py` | Done |
| AA4 | Analyse NLP → Transformers → LLM evolution | measured ladder TF-IDF → LSA/MLP → MiniLM → DistilBERT; written analysis in `docs/LLM_COMPARISON.md` §1, §4 | `reports/classification.json`, `reports/distilbert.json`, `reports/minilm_logreg.json` | Done |
| AA5 | Experiment with Transformers (classification, generation) | classification: `transformers_exp.py` (MiniLM embeddings + LogReg; DistilBERT fine-tuning with val-based epoch selection, class-weighted loss). Generation: `rag/qa.py::ClaudeGenerator` | DistilBERT run: 44.6 min on CPU, test macro-F1 0.601; **generation experiment not run** (no API key) | **Partial** (classification done; generative LLM untested) |
| AA6 | Compare existing LLMs (capabilities, architecture, use cases) | `docs/LLM_COMPARISON.md`: family comparison table + measured comparison of the models that could run | table in the doc; encoders measured, decoder-only LLMs documented only | **Partial** (documented comparison; generative LLMs not benchmarked) |
| AA7 | NLP tasks on graph data with GNN/GTN | `gnn.py`: GCN, GraphSAGE, GAT, **Graph Transformer (GTN)** with multi-head attention over neighbours, edge-type bias, Laplacian positional encoding; `linkpred.py`; `graph_analysis.py` (Louvain, PageRank) | `reports/gnn.json` (5 seeds, mean ± std, weak + gold), `reports/linkpred.json`, `reports/graph_stats.json`, `tests/test_gnn.py` | Done |
| AA8 | RAG and Graph-RAG pipelines on text and graph data | `rag/retrieval.py`, `rag/graph_store.py`, `rag/qa.py`, `rag/evaluation.py` | `reports/qa_eval.json` (vector vs hybrid vs graph, 69 questions, bootstrap CIs), `tests/test_rag.py`, UI "Question answering" | Done (Neo4j layer: see below) |

## Chapter 1 — NLP and graphs

| Course item | Implementation | Status |
|---|---|---|
| Structured / semi-structured / unstructured data | the Q&A parquet is semi-structured (3 columns) with unstructured text; the graph is structured output | Documented |
| Preprocessing: tokenisation, stop words, stemming, lemmatisation, POS | `preprocessing.py`; ablation measured | Done |
| Not used from the course: chunking of syntax (SN/SV), coreference, Levenshtein, Word2Vec/GloVe | not needed for the tasks; TF-IDF + LSA + contextual embeddings cover vectorisation | Not done (justified) |
| Vectorisation: BoW/TF-IDF, contextual embeddings | `embeddings.py` (TF-IDF + SVD/LSA, sentence-transformers) | Done |
| Cosine similarity | `embeddings.cosine_topk`, kNN graph, dense retrieval | Done |
| NLU: classification | `classify.py` | Done |
| NLU: NER | `ner.py` (regex + gazetteers), entity-level P/R/F1 on hand-annotated passages | Done (recall limited, measured) |
| NLU: topic modelling (LDA) | `stage_topics` | Done |
| Text → graph (nodes, edges, weights, NetworkX/Gephi/Neo4j) | `kg.py`; GraphML export for Gephi; Neo4j export | Done |
| Graph types: knowledge graph, similarity graph, co-occurrence graph, heterogeneous/weighted | KG (typed, directed), kNN similarity graph, shared-entity graph; Incident + 14 entity types | Done |
| NLG pipeline / evaluation (BLEU, ROUGE, BERTScore) | extractive generator; correctness by keyword matching and grounding check instead of BLEU/ROUGE | **Partial** (BLEU/ROUGE not applicable to extractive answers) |

## Chapter 2 — Transformers for text and graphs

| Course item | Implementation | Status |
|---|---|---|
| RNN/LSTM limits vs Transformers | `LLM_COMPARISON.md` §4 | Documented |
| Self-attention, multi-head attention | `gnn.py::GraphTransformerLayer` (Q·K/√d, softmax, 4 heads) written from scratch; unit-tested softmax normalisation | Done |
| Positional encoding | Laplacian-eigenvector positional encoding for graphs (ablation) | Done |
| BERT-type model for text | DistilBERT fine-tuned; MiniLM embeddings | Done |
| GTN vs classical GNN (attention over neighbours, heterogeneous graphs) | GTN with learned edge-type bias vs GCN/SAGE/GAT, same protocol | Done |

## Chapter 3 — LLMs

| Course item | Implementation | Status |
|---|---|---|
| Definition, scaling, limits (hallucination, bias, context window, cost) | `LLM_COMPARISON.md` §4; each limit tied to something observed in this project | Documented |
| Fine-tuning | DistilBERT | Done |
| PEFT (LoRA, adapters) | – | Not done (no GPU; dataset too small to justify) |
| Prompt engineering (zero/few-shot, CoT, system prompt) | `ClaudeGenerator` prompt: instruction + context + input + output indicator (course Chap. 4 p.12), "answer only from evidence, cite, abstain" | **Partial** (designed, not executed) |
| Evaluation of LLMs | accuracy/macro-F1, answer correctness, grounding, abstention; no perplexity/BLEU | Partial |

## Chapter 4 — RAG and Graph-RAG

| Course item | Implementation | Status |
|---|---|---|
| Knowledge base, chunking | `data.make_chunks` (sentence-aligned ≈ 700 characters) | Done |
| Retriever (embeddings, similarity search) | `rag/retrieval.py`: dense (MiniLM) + lexical (TF-IDF) + Reciprocal Rank Fusion; numpy cosine (no FAISS needed at 11k chunks) | Done |
| Integration layer / prompt assembly | `rag/qa.py::QAEngine.answer` (entity linking → retrieval → graph facts → gate → generator) | Done |
| Generator | extractive (default) / Claude (optional, untested) | Done / Partial |
| Optional ranker | graph-aware re-ranking (entity match boost) | Done |
| Evaluation: precision@k, recall@k, relevance, faithfulness | `rag/evaluation.py`: hit@5, P@5, capped R@5, MRR, answer correctness, abstention, grounding; bootstrap CIs | Done |
| Graph-RAG pipeline: NER → relation extraction → graph → embeddings → indexing → graph+vector retrieval → fusion → conditioning | all steps implemented; entity embeddings = mean LSA embedding of mentioning incidents (used in link prediction), TransE/Node2Vec not implemented | Done / Partial |
| Neo4j for the graph | `Neo4jStore` (Cypher), `scripts/neo4j_import.py`, `docker-compose.yml`, CSV + `import.cypher` | **Partial**: written and statically checked, **not executed on a live server** (see `docs/NEO4J.md`) |
| RAG vs Graph-RAG comparison (course table p.21) | measured on 51 + 18 questions; by question type | Done |

## Project deliverables (subject S2)

| Deliverable | Where | Status |
|---|---|---|
| (1) Graph of incidents and entities with clusters and important relations | `reports/export/incident_graph.graphml` (Gephi), `reports/export/neo4j/`, UI "Knowledge graph" (entity explorer, global graph, clusters, relations table), `reports/figures/entity_graph.png` | Done |
| (2a) Model that classifies and analyses incident texts | `classify.py` + UI "Incident analysis" | Done |
| (2b) Answers security questions | `rag/qa.py` + UI "Question answering" | Done |
| (2c) Identifies relations between threats and targets | `kg.py` relations `TARGETS`, `MITIGATES`, `AFFECTS`…; link prediction; UI relations table | Done |
| Extra file for the oral defence | `PROJECT_DEFENSE_GUIDE.md` | Done |

## Sustainable-development links named in the project brief (SDG 4, 5, 8, 9, 10)
Quality education (4: an explainable teaching tool for security concepts), industry/innovation/infrastructure (9: defensive security tooling), decent work (8) and reduced inequalities (5, 10: open, offline, reproducible tooling usable without paid APIs). These links are stated for the report, not measured.
