# legacy_prototype/ — the original flat prototype, kept untouched for reference

These are byte-identical copies (SHA-256 verified) of the files that existed at the start of the audit. They are **not used** by the new code.
Run them (if you want to compare) from inside this folder: `python run_pipeline.py --source ../data/train.parquet --out outputs_real`.

| Legacy file | Replaced by | Why |
|---|---|---|
| `cyberlib.py` | `cyberintel/data.py`, `labels.py`, `ner.py`, `lexicon.py`, `embeddings.py` | cleaning bugs, truncated text, split not shared, no NER evaluation |
| `graph_tools.py` | `cyberintel/kg.py`, `graph_analysis.py`, `gnn.py` | co-occurrence at document level, undirected, no provenance; GCN only |
| `qa.py` | `cyberintel/rag/` | no abstention/grounding, circular evaluation, no Neo4j layer |
| `run_pipeline.py`, `transformer_finetune.py` | `cyberintel/pipeline.py`, `pipeline_stages.py`, `transformers_exp.py` | incomparable splits, Transformer script never run |
| `app.py` | `app/` (6 views) + `service.py` | recomputed everything at start-up; 4 tabs |
| `sample_data.py`, `outputs_synthetic/` | – | template-generated data, scores ≈ 1.0, meaningless; **never cite** |
| `outputs_real/` | `reports/` | the honest baseline of the prototype: macro-F1 0.54, GCN 0.47 < MLP 0.52, hit@5 0.825 vs 0.835 (audit: `docs/AUDIT_REPORT.md`) |
