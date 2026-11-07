"""Demo Streamlit S2 : analyse d'incident, QA (RAG vs Graph-RAG), explorateur de graphe, clusters.

Lancement : streamlit run app.py
Tout le calcul reutilise cyberlib.py, graph_tools.py et qa.py (aucune logique dupliquee).
"""
import importlib.util
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import graph_tools as gt
from cyberlib import Embedder, extract_entities, load_data, weak_label
from graph_tools import nid
from qa import GraphRAG

ROOT = Path(__file__).parent
PARQUET = ROOT / "data" / "train.parquet"
HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
HAS_PYVIS = importlib.util.find_spec("pyvis") is not None
HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
SEMANTIC_SKIP = ("MENTIONS", "SIMILAR_TO")  # aretes non semantiques

EXAMPLE_TEXT = ("Our SOC detected a phishing campaign: employees received emails with a fake login page. "
                "Several credentials were stolen and the attacker then exploited CVE-2021-44228 in log4j "
                "to run commands on an Apache server, with traffic to 185.220.101.4.")
EXAMPLE_QUESTIONS = [
    "Quelle vulnérabilité affecte log4j ?",
    "What does ransomware target?",
    "Quels incidents sont similaires à l'incident 3 ?",
    "Which vulnerability affects apache?",
    "How should a SOC respond to phishing?",
]
TYPE_COLORS = {"ATTACK": "#d62728", "TARGET": "#1f77b4", "SOFTWARE": "#2ca02c", "CVE": "#ff7f0e",
               "MALWARE": "#9467bd", "CWE": "#8c564b", "MITRE_TECHNIQUE": "#e377c2", "IP": "#7f7f7f",
               "HASH": "#bcbd22"}


# --------------------------------------------------------------------------- chargement (une seule fois)
@st.cache_resource(show_spinner="Premier chargement : données, NER, graphe, communautés, index…")
def load_all(source: str, embedder_kind: str) -> dict:
    """Construit tous les objets lourds, comme run_pipeline.py (étapes 1 à 3 et 5)."""
    df = load_data(source)
    df["category"] = [weak_label(q, r) for q, r in zip(df.question, df.answer)]
    df["entities"] = [extract_entities(t) for t in df.text]

    # classifieur TF-IDF + LogReg sur la réponse seule (même réglage et même split que run_pipeline.py)
    lab = df.category.astype("category")
    y, names = lab.cat.codes.values, list(lab.cat.categories)
    tr, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0,
                             stratify=y if min(np.bincount(y)) > 1 else None)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    Xa = vec.fit_transform(df.answer)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xa[tr], y[tr])

    # embeddings, graphe, communautés, Graph-RAG
    emb = Embedder(embedder_kind)
    E = emb.fit_transform(df.text.tolist())
    G = gt.build_graph(df, min_cooc=2)
    gt.add_similarity_edges(G, gt.knn_adjacency(E))
    part = gt.detect_communities(G)
    return {"df": df, "vec": vec, "clf": clf, "class_names": names, "emb": emb, "E": E, "G": G,
            "part": part, "rag": GraphRAG(df, G, emb, E),
            "communities": pd.DataFrame(gt.describe_communities(G, part)),
            "top_rel": pd.DataFrame(gt.top_relations(G, 50))}


@st.cache_data(show_spinner="Dessin du graphe global…")
def global_graph_png(_G, _part, key: str) -> bytes:
    """plot_entity_graph écrit un fichier PNG : on le produit une fois et on garde les octets."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "g.png"
        gt.plot_entity_graph(_G, _part, path)
        return path.read_bytes()


# --------------------------------------------------------------------------- utilitaires
def classify_text(app: dict, text: str) -> pd.Series:
    """Probabilités par catégorie (le modèle est entraîné sur des réponses : on lui donne le texte collé)."""
    p = app["clf"].predict_proba(app["vec"].transform([text]))[0]
    return pd.Series(p, index=[app["class_names"][c] for c in app["clf"].classes_]).sort_values(ascending=False)


def group_entities(ents) -> dict:
    out = defaultdict(list)
    for t, n in ents:
        out[t].append(n)
    return dict(out)


def neighbors_table(G, node: str) -> pd.DataFrame:
    """Relations sémantiques d'une entité : relation, voisin, co-occurrence, lift."""
    rows = [{"relation": d["relation"], "voisin": G.nodes[nb]["label"], "type voisin": G.nodes[nb]["type"],
             "co-occurrence": d["cooc"], "lift": d["lift"]}
            for nb, d in G[node].items() if d["relation"] not in SEMANTIC_SKIP and "cooc" in d]
    return pd.DataFrame(rows).sort_values("co-occurrence", ascending=False) if rows else pd.DataFrame()


def ego_graph(G, node: str, max_nb: int = 25):
    """Sous-graphe d'entités autour de `node` (voisins sémantiques les plus forts)."""
    nbs = sorted(((nb, d["weight"]) for nb, d in G[node].items() if d["relation"] not in SEMANTIC_SKIP),
                 key=lambda x: -x[1])[:max_nb]
    H = G.subgraph([node] + [nb for nb, _ in nbs]).copy()
    H.remove_edges_from([(a, b) for a, b, d in list(H.edges(data=True)) if d["relation"] in SEMANTIC_SKIP])
    return H


def draw_matplotlib(H, center: str):
    import matplotlib.pyplot as plt
    import networkx as nx
    pos = nx.spring_layout(H, seed=1, weight="weight")
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(H, pos, alpha=0.4, width=[0.5 + d["weight"] ** 0.5 for *_, d in H.edges(data=True)], ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=[TYPE_COLORS.get(H.nodes[x]["type"], "#333") for x in H],
                           node_size=[700 if x == center else 300 for x in H], ax=ax)
    nx.draw_networkx_labels(H, pos, {x: H.nodes[x]["label"] for x in H}, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(H, pos, {(a, b): d["relation"] for a, b, d in H.edges(data=True)},
                                 font_size=6, ax=ax)
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)


def draw_pyvis(H, center: str):
    from pyvis.network import Network
    net = Network(height="480px", width="100%", cdn_resources="in_line", notebook=False)
    for x in H:
        net.add_node(x, label=H.nodes[x]["label"], title=H.nodes[x]["type"],
                     color=TYPE_COLORS.get(H.nodes[x]["type"], "#333"), size=30 if x == center else 18)
    for a, b, d in H.edges(data=True):
        net.add_edge(a, b, title=f"{d['relation']} (x{d['cooc']}, lift {d['lift']})", value=d["weight"])
    html = net.generate_html()
    if hasattr(st, "iframe"):  # Streamlit récent ; sinon ancienne API
        st.iframe(html, height=500)
    else:
        components.html(html, height=500)


def show_sources(df, ids):
    for i in ids:
        with st.expander(f"Incident #{i} · {df.category[i]} · {df.question[i][:80]}"):
            st.write(df.answer[i][:1500])


# --------------------------------------------------------------------------- onglets
def tab_analysis(app: dict, k: int):
    text = st.text_area("Texte de l'incident", EXAMPLE_TEXT, height=150)
    if not st.button("Analyser", type="primary"):
        return
    if not text.strip():
        st.warning("Colle d'abord un texte d'incident.")
        return
    probs = classify_text(app, text)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Catégorie prédite : {probs.index[0]}")
        st.bar_chart(probs.rename("probabilité"))
        st.caption("⚠️ Catégories = labels faibles (règles par mots-clés) : le modèle mesure l'accord avec ces règles.")
    with c2:
        st.subheader("Entités extraites")
        groups = group_entities(extract_entities(text))
        if not groups:
            st.info("Aucune entité détectée (NER regex + gazetteers).")
        for t, names in groups.items():
            st.markdown(f"**{t}** : " + ", ".join(f"`{n}`" for n in names))
    st.subheader(f"{k} incidents les plus similaires")
    df = app["df"]
    hits = app["rag"].vector_search(text, k)
    st.dataframe(pd.DataFrame([{"id": i, "catégorie": df.category[i], "question": df.question[i][:140],
                                "score": round(s, 3)} for i, s in hits]), hide_index=True, width="stretch")


def tab_qa(app: dict, k: int, use_llm: bool):
    st.write("Exemples (cliquer pour remplir) :")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked = False
    for c, q in zip(cols, EXAMPLE_QUESTIONS):
        if c.button(q, key=f"ex_{q}", width="stretch"):
            st.session_state["question"], clicked = q, True  # un exemple cliqué lance aussi la réponse
    question = st.text_input("Question", key="question")
    if not (st.button("Répondre", type="primary") or clicked):
        return
    if not question.strip():
        st.warning("Écris une question (ou clique sur un exemple).")
        return
    left, right = st.columns(2)
    for col, mode, title in ((left, "vector", "RAG vectoriel"), (right, "graph", "Graph-RAG")):
        with col:
            st.subheader(title)
            try:
                r = app["rag"].answer(question, mode, k=k, use_llm=use_llm)
            except Exception as e:  # ex. clé API invalide, réseau coupé
                st.error(f"Erreur pendant la réponse : {e}")
                continue
            st.write(r["answer"] or "_Pas de réponse._")
            st.markdown("**Entités détectées :** " + (", ".join(f"`{t}:{n}`" for t, n in r["entities"]) or "aucune"))
            if r["facts"]:
                with st.expander(f"Faits du graphe ({len(r['facts'])})", expanded=True):
                    st.markdown("\n".join(f"- {f}" for f in r["facts"]))
            elif mode == "graph":
                st.caption("Aucun fait du graphe pour cette question.")
            show_sources(app["df"], r["sources"])


def tab_graph(app: dict):
    G = app["G"]
    ents = sorted((d["type"], d["label"]) for _, d in G.nodes(data=True) if d["type"] != "INCIDENT")
    types = sorted({t for t, _ in ents})
    c1, c2 = st.columns([1, 2])
    t = c1.selectbox("Type d'entité", types, index=types.index("SOFTWARE") if "SOFTWARE" in types else 0)
    names = [n for tt, n in ents if tt == t]
    name = c2.selectbox("Entité", names)
    node = nid(t, name) if name else None
    if node not in G:
        st.warning("Entité absente du graphe.")
    else:
        n_inc = sum(1 for nb in G.neighbors(node) if nb.startswith("INC:"))
        st.write(f"**{name}** ({t}) : mentionnée dans {n_inc} incidents.")
        tbl = neighbors_table(G, node)
        if tbl.empty:
            st.info("Cette entité n'a aucune relation sémantique (co-occurrence < 2 incidents).")
        else:
            st.dataframe(tbl, hide_index=True, width="stretch")
            H = ego_graph(G, node)
            (draw_pyvis if HAS_PYVIS else draw_matplotlib)(H, node)
            if not HAS_PYVIS:
                st.caption("pyvis non installé : rendu matplotlib (pip install pyvis pour la version interactive).")
    st.divider()
    st.subheader("Graphe global des entités")
    st.image(global_graph_png(G, app["part"], f"{len(G)}-{G.number_of_edges()}"))
    st.subheader("Relations les plus importantes")
    if app["top_rel"].empty:
        st.info("Aucune relation au-dessus du seuil.")
    else:
        st.dataframe(app["top_rel"], hide_index=True, width="stretch")


def tab_clusters(app: dict):
    comm, G, part, df = app["communities"], app["G"], app["part"], app["df"]
    st.dataframe(comm, hide_index=True, width="stretch")
    st.caption("Pureté = part de la catégorie dominante (label faible) dans la communauté.")
    c = st.selectbox("Communauté", comm["community"].tolist(),
                     format_func=lambda c: f"#{c} · {comm.set_index('community').loc[c, 'dominant_category']} "
                                           f"({comm.set_index('community').loc[c, 'n_incidents']} incidents)")
    nodes = [n for n, ci in part.items() if ci == c]
    ents = sorted((n for n in nodes if not n.startswith("INC:")), key=lambda n: -G.degree(n))[:12]
    inc = [int(n[4:]) for n in nodes if n.startswith("INC:")]
    st.markdown("**Entités principales :** " + (", ".join(f"{G.nodes[n]['label']} [{G.nodes[n]['type']}]" for n in ents) or "aucune"))
    if not inc:
        st.info("Cette communauté ne contient aucun incident.")
        return
    sample = np.random.default_rng(0).choice(inc, size=min(10, len(inc)), replace=False)
    st.dataframe(pd.DataFrame([{"id": int(i), "catégorie": df.category[i], "question": df.question[i][:150]}
                               for i in sample]), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- application
def main():
    st.set_page_config(page_title="S2 Cybersécurité – Graph-RAG", layout="wide")
    st.title("🛡️ S2 – Analyse de textes de cybersécurité & Graph-RAG")

    with st.sidebar:
        st.header("Paramètres")
        source = st.radio("Données", ["data/train.parquet", "sample (synthétique)"])
        use_st = st.checkbox("Embeddings sentence-transformers", value=False, disabled=not HAS_ST,
                             help="Désactivé si sentence-transformers n'est pas installé (sinon TF-IDF + SVD, hors-ligne).")
        k = st.slider("Nombre de résultats k", 1, 10, 5)
        use_llm = st.checkbox("Génération Claude", value=False, disabled=not HAS_KEY,
                              help="Nécessite la variable d'environnement ANTHROPIC_API_KEY.")
        if source.startswith("sample"):
            st.warning("Données synthétiques : démonstration uniquement, résultats non significatifs.")

    src = "sample" if source.startswith("sample") else str(PARQUET)
    if src != "sample" and not PARQUET.exists():
        st.error(f"Fichier introuvable : {PARQUET}. Télécharge le dataset (voir README) ou choisis « sample ».")
        st.stop()
    try:
        app = load_all(src, "st" if use_st else "tfidf")
    except Exception as e:
        st.error(f"Chargement impossible : {e}")
        st.stop()
    st.sidebar.success(f"{len(app['df'])} incidents · {app['G'].number_of_nodes()} nœuds · {len(app['communities'])} communautés")

    t1, t2, t3, t4 = st.tabs(["Analyse d'incident", "Question-réponse", "Graphe", "Clusters"])
    with t1:
        tab_analysis(app, k)
    with t2:
        tab_qa(app, k, use_llm)
    with t3:
        tab_graph(app)
    with t4:
        tab_clusters(app)


main()
