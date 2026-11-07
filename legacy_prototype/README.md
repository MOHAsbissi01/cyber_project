# S2 – Cybersécurité : analyse de texte, graphe et Graph-RAG

Module *Advanced Learning for Text and Graph Data* (ESPRIT 5ERP-BI). Dataset : `AlicanKiraz0/Cybersecurity-Dataset-v1`
(colonnes `system`, `user`, `assistant` ; ~2,4k paires Q/R, licence apache-2.0, **sans labels**).

## Lancer
```bash
pip install -r requirements.txt
python run_pipeline.py --source sample            # test hors-ligne, données synthétiques
python run_pipeline.py --source hf                # vrai dataset (téléchargement via `datasets`)
python run_pipeline.py --source hf --embedder st  # embeddings Transformer (sentence-transformers)
python transformer_finetune.py --source hf        # fine-tuning DistilBERT (Colab/GPU conseillé)
```
Sorties dans `outputs/` : `metrics.json`, `classification_report.txt`, `entities.csv`, `incident_graph.graphml`
(Gephi), `entity_graph.png`, `communities.csv`, `incident_clusters.csv`, `top_relations.csv`,
`neo4j/` (CSV + `import.cypher`), `qa_demo.md`, `gcn_hidden_embeddings.npy`.

## Démo (Streamlit)
```bash
pip install -r requirements.txt     # streamlit (+ pyvis, optionnel)
streamlit run app.py
```
Fonctionne hors-ligne (TF-IDF par défaut) sur `data/train.parquet` ; « sample » est proposé dans la barre latérale
(données synthétiques, démonstration uniquement). Au premier chargement l'app construit données, labels faibles, NER,
embeddings, graphe, communautés et `GraphRAG` (mis en cache ensuite).

| Onglet | Contenu |
|---|---|
| Analyse d'incident | catégorie prédite (TF-IDF + LogReg, probabilités), entités par type, k incidents similaires |
| Question-réponse | 5 exemples FR/EN ; RAG vectoriel vs Graph-RAG côte à côte, faits du graphe, sources |
| Graphe | explorateur d'entité (relations, co-occurrence, lift, sous-graphe pyvis ou matplotlib), graphe global, top relations |
| Clusters | tableau des communautés, entités principales et échantillon d'incidents |

Barre latérale : données, embeddings `sentence-transformers` (grisé si non installé), k, génération Claude
(grisée sans `ANTHROPIC_API_KEY`). Les catégories restent des **labels faibles** : la prédiction mesure l'accord avec les règles.

## Correspondance avec le sujet et les acquis (AA)
| Livrable / technique | Fichier | AA |
|---|---|---|
| Nettoyage, labels faibles, classification (TF-IDF+LogReg, DistilBERT) | `cyberlib.py`, `run_pipeline.py`, `transformer_finetune.py` | 3, 5, 6 |
| Embeddings (LSA ou sentence-transformers) | `cyberlib.Embedder` | 3, 5 |
| NER : CVE, CWE, MITRE, IP, hash (regex) ; malware, logiciel, attaque, cible (gazetteers) | `cyberlib.extract_entities` | 3 |
| Graphe incident–entité + relations (TARGETS, AFFECTS, EXPLOITS…) + lift | `graph_tools.build_graph` | 1, 7 |
| Clusters (Louvain) et relations importantes | `detect_communities`, `top_relations` | 7 |
| GNN : GCN 2 couches (torch pur) vs MLP sans graphe | `train_node_classifier` | 7 |
| Graph-RAG vs RAG, export Neo4j | `qa.py`, `export_neo4j` | 8 |

## Limites à assumer dans le rapport
- **Pas de labels** : les catégories sont des *labels faibles* par mots-clés (surtout sur la question). Les scores
  de classification mesurent donc la cohérence avec ces règles, pas une vérité terrain. Idéal : annoter ~150 lignes
  à la main et évaluer dessus.
- Le dataset couvre des thèmes variés (vulnérabilités, malware, crypto, accès…) mais contient des réponses
  bruitées/incohérentes. Sur le vrai jeu (2 394 lignes après nettoyage) le NER trouve peu de CVE (42) et de CWE (1) :
  étendez les listes `_GAZ` selon `entities.csv`. `phishing` et `ransomware` sont très rares (14 et 18 lignes).
- Résultats réels (`--source data/train.parquet`, TF-IDF) : LogReg macro-F1 0,54 ; MLP 0,52 ; GCN 0,47 ;
  hit@5 retrieval : vectoriel 0,825, Graph-RAG 0,835. Le GCN ne bat pas le MLP : à discuter dans le rapport
  (labels faibles bruités, graphe de similarité redondant avec les features).
- Les chiffres obtenus sur `--source sample` (≈1.0) viennent de textes générés par gabarits : **ne pas les citer**.
- `--llm` (génération par API Claude) et `transformer_finetune.py` ne sont pas testés ici (réseau/modèle requis).
