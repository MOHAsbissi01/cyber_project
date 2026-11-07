# AA6 — Comparing language models: capabilities, architecture, use cases

Scope of the comparison: **what was measured in this project** is in section 2 (real runs). Section 1 is a documented, source-based comparison of model
families (course Chap. 2–3); it contains **no benchmark numbers I did not measure**. Section 3 states what could not be tested.

## 1. Model families (architecture → what it is good for → used here?)

| Family (examples) | Architecture (course Chap. 2) | Pre-training objective | Strengths | Limits | Typical use | In this project |
|---|---|---|---|---|---|---|
| **Encoder-only** — BERT, RoBERTa, ALBERT, DistilBERT, MiniLM, domain-adapted encoders (e.g. SecureBERT, CySecBERT: not tested) | Transformer *encoder*, bidirectional self-attention | masked-language modelling (BERT); distillation (DistilBERT, MiniLM) | rich contextual embeddings; cheap to fine-tune; fast inference | cannot generate free text; fixed context (typically 512 tokens) | classification, NER, semantic search, re-ranking | **DistilBERT** fine-tuned; **all-MiniLM-L6-v2** embeddings |
| **Decoder-only (causal)** — GPT family, LLaMA, Mistral, Claude | Transformer *decoder*, causal (masked) attention | next-token prediction at very large scale | free-form generation, instruction following, few-shot/zero-shot, long context, tool use | hallucination; cost/latency; closed weights for some; context window; bias (course Chap. 3 limits) | assistants, summarisation, QA, code, agents, RAG generator | optional generator (`ClaudeGenerator`, **not run**: no API key) |
| **Encoder–decoder** — T5, BART, mBART | encoder + decoder with cross-attention | denoising / text-to-text | summarisation, translation, controlled text-to-text | smaller ecosystem than decoder-only today | seq2seq tasks | not used |
| **Graph Transformers** — GTN, Graph-BERT | attention restricted to / biased by graph structure | task-specific | heterogeneous graphs, attention weights per relation type | need a graph and labels; sensitive to graph quality | node/link tasks on graphs | **own implementation** (`gnn.py::GTN`) |
| **Classical baselines** — TF-IDF + linear models, LSA | no learned context | supervised | strong on small data, transparent, instant | no semantics beyond the vocabulary | topic classification, lexical retrieval | **TF-IDF + LogReg/SVM/NB**, **LSA + MLP** |

Selection criteria used to choose what to run: dataset size (2.2k records → fine-tuning a large model is not justified), hardware (CPU only → models ≤ 70M
parameters), offline reproducibility (no paid API required for any reported number), and course coverage (BERT-type encoder for AA5, graph transformer for AA7).

## 2. Measured comparison on this dataset (topic classification, 11 weak classes; same split for all)

Numbers come from `reports/classification.json` (regenerate with the pipeline). *Weak* = agreement with keyword rules; *gold* = 183 independent labels on test-split records.

| Model | Parameters / features | Training cost (CPU) | val macro-F1 | test macro-F1 (weak) | **gold macro-F1** | gold accuracy |
|---|---|---|---|---|---|---|
| majority class | – | – | – | 0.043 | – | 0.126 |
| TF-IDF + LogReg | 60k sparse features | seconds | 0.605 | 0.646 | **0.644** | 0.705 |
| TF-IDF + linear SVM | 60k sparse features | seconds | 0.601 | 0.670 | 0.641 | 0.716 |
| TF-IDF + ComplementNB | 60k sparse features | seconds | 0.500 | 0.559 | 0.579 | 0.689 |
| LSA (128-d) + MLP | 128 dense | seconds | 0.529 | 0.589 | 0.622 | 0.678 |
| MiniLM embeddings + LogReg | 22M params, frozen, 384-d | 2.4 min (encoding) | 0.589 | 0.636 | 0.589 | 0.650 |
| DistilBERT fine-tuned (3 epochs, max 160 tokens) | 66M params | **44.6 min** | 0.602 | 0.601 | 0.523 | 0.585 |

**Reading (honest):**
* No model clearly beats TF-IDF + LogReg. With ~1.5k training examples, a vocabulary-driven task and labels that *come from keywords*, lexical models are hard to beat;
  fine-tuning a 66M-parameter model on this little data, for 3 epochs on CPU, **did not pay off** (and costs ≈ 45 minutes against seconds).
* The gold subset has 183 records; a difference below ≈ 0.05 macro-F1 is within noise. The ranking TF-IDF ≥ MiniLM ≥ DistilBERT on gold is suggestive, not proven.
* The weak-label test score and the gold score are close for TF-IDF (0.646 vs 0.644), but this hides different errors: gold F1 is very low on `other` (0.24) — a class the keyword rules define as "no keyword found".
* We did **not** compare generative LLMs (GPT/LLaMA/Claude) on classification or QA: no API key and no GPU. See section 3.

## 3. What was not tested (and why)

| Item | Reason | How to run it |
|---|---|---|
| Claude (or any LLM) as the answer generator | no `ANTHROPIC_API_KEY` | `set ANTHROPIC_API_KEY=...`, sidebar toggle "Claude answer generation" or `config.toml: [llm] provider = "claude"`; compare against the extractive generator on `annotations/qa_testset.json` |
| Zero/few-shot LLM topic classification | same | prompt with the class definitions of `docs/ANNOTATION_GUIDELINES.md` |
| Domain-adapted encoders (SecureBERT/CySecBERT) | extra downloads, not needed to answer the course question | `python -m cyberintel.transformers_exp` with `--model` changed |
| LoRA / PEFT fine-tuning (course Chap. 3) | no GPU; dataset too small to justify | `peft` library on a Colab GPU |

## 4. Concepts the course asks to be able to explain (short answers)

* **RNN/LSTM → Transformer:** sequential processing and vanishing gradients vs parallel self-attention with direct long-range access (cost: quadratic in sequence length).
* **Encoder vs decoder vs enc-dec:** understanding (BERT) vs generation (GPT) vs sequence-to-sequence (T5/BART).
* **Fine-tuning vs RAG vs prompting:** fine-tuning changes weights (used here for DistilBERT); RAG injects fresh external knowledge at inference (the QA system); prompting steers an LLM without training.
* **Why RAG here:** a 2.2k-record corpus changes nothing in a model's weights, but must be *citable*; retrieval + graph facts give provenance, which fine-tuning cannot.
* **Limits (Chap. 3) observed in this project:** hallucination — the corpus itself is LLM-generated and contains wrong CVE explanations (flagged by the KB check); bias/quality — answers contain leaked chain-of-thought (cleaned); context window — answers truncated to 160 tokens for DistilBERT.
