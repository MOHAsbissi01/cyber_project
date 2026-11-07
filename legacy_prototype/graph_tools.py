"""Graphe incidents-entites, communautes, similarite, GCN (torch pur), exports (GraphML, Neo4j, PNG)."""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

# relation semantique selon le couple de types (ordre alphabetique des types)
RELATIONS = {
    ("ATTACK", "TARGET"): "TARGETS",
    ("MALWARE", "TARGET"): "TARGETS",
    ("ATTACK", "SOFTWARE"): "EXPLOITS",
    ("CVE", "SOFTWARE"): "AFFECTS",
    ("ATTACK", "CVE"): "ENABLES",
    ("CWE", "CVE"): "CLASSIFIES",
    ("ATTACK", "CWE"): "ENABLES",
    ("ATTACK", "MALWARE"): "USES",
    ("MALWARE", "SOFTWARE"): "EXPLOITS",
    ("CVE", "MALWARE"): "EXPLOITED_BY",
    ("ATTACK", "MITRE_TECHNIQUE"): "MAPS_TO",
}


def rel_name(t1, t2):
    return RELATIONS.get(tuple(sorted((t1, t2))))


def nid(t, name):
    return f"{t}:{name}"


# --------------------------------------------------------------------------- graphe
def build_graph(df, min_cooc=2):
    """Noeuds: Incident + entites typees. Aretes: MENTIONS (incident-entite) et relations
    semantiques entite-entite deduites de la co-occurrence (>= min_cooc incidents), avec lift."""
    G = nx.Graph()
    n = len(df)
    ent_df = Counter()
    cooc = Counter()
    for r in df.itertuples():
        inc = f"INC:{r.id}"
        G.add_node(inc, type="INCIDENT", label=r.question[:80], category=r.category, incident_id=int(r.id))
        for t, name in r.entities:
            e = nid(t, name)
            if e not in G:
                G.add_node(e, type=t, label=name)
            G.add_edge(inc, e, relation="MENTIONS", weight=1.0)
            ent_df[e] += 1
        for i, (t1, n1) in enumerate(r.entities):
            for t2, n2 in r.entities[i + 1:]:
                if rel_name(t1, t2):
                    cooc[tuple(sorted((nid(t1, n1), nid(t2, n2))))] += 1
    for (a, b), c in cooc.items():
        if c >= min_cooc:
            lift = c * n / (ent_df[a] * ent_df[b])
            G.add_edge(a, b, relation=rel_name(G.nodes[a]["type"], G.nodes[b]["type"]),
                       weight=float(c), cooc=int(c), lift=round(lift, 3))
    return G


def entity_subgraph(G):
    return G.subgraph([x for x, d in G.nodes(data=True) if d["type"] != "INCIDENT"]).copy()


def knn_adjacency(E, k=8, thr=0.25):
    """Graphe de similarite semantique incident-incident (cosinus, E normalise)."""
    S = E @ E.T
    np.fill_diagonal(S, -1)
    n = len(E)
    rows, cols, vals = [], [], []
    idx = np.argpartition(-S, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    for i in range(n):
        for j in idx[i]:
            if S[i, j] >= thr:
                rows.append(i); cols.append(j); vals.append(S[i, j])
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    return A.maximum(A.T)


def entity_adjacency(df, max_df=60):
    """Incidents relies s'ils partagent une entite (entites rares/discriminantes seulement)."""
    vocab = {}
    rows, cols = [], []
    for r in df.itertuples():
        for e in r.entities:
            rows.append(r.id); cols.append(vocab.setdefault(e, len(vocab)))
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(df), len(vocab)))
    keep = np.where(np.asarray(B.sum(0)).ravel() <= max_df)[0]
    B = B[:, keep]
    A = (B @ B.T).tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.data = np.minimum(A.data, 3) / 3.0
    return A


def add_similarity_edges(G, A):
    A = sp.triu(A).tocoo()
    for i, j, v in zip(A.row, A.col, A.data):
        G.add_edge(f"INC:{i}", f"INC:{j}", relation="SIMILAR_TO", weight=float(v))


# --------------------------------------------------------------------------- communautes
def detect_communities(G, seed=0):
    """Louvain sur le graphe complet (ponderations). Renvoie {noeud: id_communaute}."""
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
    comms = sorted(comms, key=len, reverse=True)
    return {node: ci for ci, c in enumerate(comms) for node in c}


def describe_communities(G, part, top=6):
    by = defaultdict(list)
    for node, c in part.items():
        by[c].append(node)
    rows = []
    for c, nodes in sorted(by.items(), key=lambda x: -len(x[1])):
        inc = [x for x in nodes if G.nodes[x]["type"] == "INCIDENT"]
        ents = Counter(G.nodes[x]["label"] + f" [{G.nodes[x]['type']}]" for x in nodes if G.nodes[x]["type"] != "INCIDENT"
                       for _ in range(G.degree(x)))
        cats = Counter(G.nodes[x]["category"] for x in inc)
        rows.append({"community": c, "n_incidents": len(inc), "n_entities": len(nodes) - len(inc),
                     "dominant_category": cats.most_common(1)[0][0] if cats else "-",
                     "category_purity": round(cats.most_common(1)[0][1] / len(inc), 3) if inc else 0,
                     "top_entities": "; ".join(k for k, _ in ents.most_common(top))})
    return rows


def top_relations(G, k=30, min_lift=1.0):
    """Relations menaces->cibles/logiciels les plus importantes (poids = co-occurrence)."""
    out = []
    for a, b, d in G.edges(data=True):
        if d["relation"] not in ("MENTIONS", "SIMILAR_TO") and d.get("lift", 0) >= min_lift:
            out.append({"source": G.nodes[a]["label"], "source_type": G.nodes[a]["type"], "relation": d["relation"],
                        "target": G.nodes[b]["label"], "target_type": G.nodes[b]["type"],
                        "cooccurrence": d["cooc"], "lift": d["lift"]})
    return sorted(out, key=lambda r: (-r["cooccurrence"], -r["lift"]))[:k]


# --------------------------------------------------------------------------- GCN (torch pur)
def _norm_adj(A):
    A = A + sp.eye(A.shape[0])
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    return sp.diags(d) @ A @ sp.diags(d)


def _to_torch_sparse(A):
    import torch
    A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), A.shape).coalesce()


def train_node_classifier(X, A, y, n_classes, use_graph=True, hidden=64, epochs=300, lr=0.01, wd=5e-4,
                          dropout=0.5, seed=0, split=(0.6, 0.2, 0.2)):
    """GCN a 2 couches (Kipf & Welling) ; use_graph=False -> MLP (baseline sans graphe).
    Retourne metriques test + embeddings caches + predictions."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    a, b = int(split[0] * n), int((split[0] + split[1]) * n)
    tr, va, te = perm[:a], perm[a:b], perm[b:]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    An = _to_torch_sparse(_norm_adj(A)) if use_graph else None

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1, s.l2, s.dp = nn.Linear(X.shape[1], hidden), nn.Linear(hidden, n_classes), nn.Dropout(dropout)

        def forward(s, x):
            x = s.dp(x)
            h = s.l1(x)
            h = torch.sparse.mm(An, h) if use_graph else h
            h = s.dp(torch.relu(h))
            o = s.l2(h)
            o = torch.sparse.mm(An, o) if use_graph else o
            return o, h

    m = Net()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    best, best_state = -1, None
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        out, _ = m(Xt)
        nn.functional.cross_entropy(out[tr], yt[tr]).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(Xt)[0].argmax(1)
            v = (pred[va] == yt[va]).float().mean().item()
        if v > best:
            best, best_state = v, {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        out, h = m(Xt)
    pred = out.argmax(1).numpy()
    return {"val_acc": round(best, 4), "test_acc": round(accuracy_score(y[te], pred[te]), 4),
            "test_macro_f1": round(f1_score(y[te], pred[te], average="macro"), 4),
            "hidden": h.numpy(), "pred": pred, "test_idx": te}


# --------------------------------------------------------------------------- exports
def export_graphml(G, path):
    H = G.copy()
    for _, d in H.nodes(data=True):
        for k, v in list(d.items()):
            d[k] = str(v) if not isinstance(v, (int, float, str)) else v
    nx.write_graphml(H, path)


def export_neo4j(G, outdir):
    """CSV + script Cypher pour importer le graphe dans Neo4j (Chapitre 4 du module)."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "type", "label", "category"])
        for x, d in G.nodes(data=True):
            w.writerow([x, d["type"], d.get("label", ""), d.get("category", "")])
    with open(out / "edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["source", "target", "relation", "weight"])
        for a, b, d in G.edges(data=True):
            w.writerow([a, b, d["relation"], d.get("weight", 1)])
    (out / "import.cypher").write_text(
        "// Copier nodes.csv et edges.csv dans le dossier import/ de Neo4j puis executer :\n"
        "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;\n"
        "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS r\n"
        "MERGE (n:Node {id:r.id}) SET n.type=r.type, n.label=r.label, n.category=r.category;\n"
        "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS r\n"
        "MATCH (a:Node {id:r.source}), (b:Node {id:r.target})\n"
        "MERGE (a)-[e:REL {relation:r.relation}]->(b) SET e.weight=toFloat(r.weight);\n"
        "// Exemple : vulnerabilites d'un logiciel\n"
        "// MATCH (s:Node {id:'SOFTWARE:log4j'})-[e:REL {relation:'AFFECTS'}]-(c:Node {type:'CVE'}) RETURN c.label, e.weight;\n",
        encoding="utf-8")


def plot_entity_graph(G, part, path, max_nodes=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E = entity_subgraph(G)
    E.remove_edges_from([(a, b) for a, b, d in E.edges(data=True) if d["relation"] == "MENTIONS"])
    keep = sorted(E.nodes, key=lambda x: -sum(d.get("weight", 1) for _, _, d in E.edges(x, data=True)))[:max_nodes]
    H = E.subgraph(keep)
    pos = nx.spring_layout(H, seed=1, k=1.2 / math.sqrt(max(len(H), 1)), weight="weight")
    colors = {"ATTACK": "#d62728", "TARGET": "#1f77b4", "SOFTWARE": "#2ca02c", "CVE": "#ff7f0e",
              "MALWARE": "#9467bd", "CWE": "#8c564b", "MITRE_TECHNIQUE": "#e377c2", "IP": "#7f7f7f", "HASH": "#bcbd22"}
    fig, ax = plt.subplots(figsize=(14, 10))
    w = [0.5 + d["weight"] ** 0.5 for _, _, d in H.edges(data=True)]
    nx.draw_networkx_edges(H, pos, width=w, alpha=0.35, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_color=[colors.get(H.nodes[x]["type"], "#333") for x in H],
                           node_size=[120 + 60 * H.degree(x) for x in H], ax=ax)
    nx.draw_networkx_labels(H, pos, {x: H.nodes[x]["label"] for x in H}, font_size=8, ax=ax)
    for t, c in colors.items():
        ax.scatter([], [], c=c, label=t)
    ax.legend(loc="lower left"); ax.set_axis_off()
    ax.set_title("Graphe entites : attaques, cibles, logiciels, CVE (taille = degre, couleur = type)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
