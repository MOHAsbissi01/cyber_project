"""Link prediction on the entity knowledge graph (AA7, relationship discovery): can we recover held-out entity-entity relations?

Positives : corpus-derived semantic relations between entities (unordered pairs), randomly split 70/10/20 into train/val/test edges.
Negatives : random entity pairs of a TYPE-COMPATIBLE kind (a pair whose types appear together in the relation schema), never a known edge.
            Using type-compatible negatives avoids a trivially easy task (e.g. a CVE-CVE pair can never be a relation).
Methods   : classical structure heuristics (common neighbours, Jaccard, Adamic-Adar, preferential attachment) on the TRAIN graph,
            a feature-only baseline (cosine of node features: no graph), and a GCN auto-encoder (graph + features).
Metrics   : ROC-AUC and Average Precision on the held-out edges, mean +- std over several random splits.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from .kg import SCHEMA, SEMANTIC_RELATIONS

TYPES = ["ATTACK", "DEFENSE", "SOFTWARE", "TECHNOLOGY", "TARGET", "TOOL", "MALWARE", "CVE", "VULN_NAME", "MITRE_TECHNIQUE", "TACTIC",
         "THREAT_ACTOR", "IP", "HASH", "CWE"]
_COMPAT = {frozenset(k) for k in SCHEMA}


def entity_features(G: nx.MultiDiGraph, nodes: list[str], mentions: pd.DataFrame, E: np.ndarray) -> np.ndarray:
    """[type one-hot | log(1+#incidents) | mean LSA embedding of the incidents that mention the entity]."""
    inc_of = mentions.drop_duplicates(["incident_id", "type", "name"]).groupby(["type", "name"]).incident_id.apply(list).to_dict()
    X = np.zeros((len(nodes), len(TYPES) + 1 + E.shape[1]), dtype=np.float32)
    for i, n in enumerate(nodes):
        d = G.nodes[n]
        X[i, TYPES.index(d["type"])] = 1
        incs = inc_of.get((d["type"], d["name"]), [])
        X[i, len(TYPES)] = np.log1p(len(incs))
        if incs:
            v = E[incs].mean(0)
            X[i, len(TYPES) + 1:] = v / (np.linalg.norm(v) + 1e-9)
    return X


def positive_pairs(G: nx.MultiDiGraph) -> list[tuple[str, str]]:
    ps = set()
    for a, b, d in G.edges(data=True):
        if d["relation"] in SEMANTIC_RELATIONS and d["source"] == "corpus":
            ps.add(tuple(sorted((a, b))))
    return sorted(ps)


def sample_negatives(nodes, types, positives: set, n: int, rng) -> list[tuple[int, int]]:
    out, seen = [], set()
    while len(out) < n:
        i, j = rng.randint(len(nodes)), rng.randint(len(nodes))
        if i == j or frozenset((types[i], types[j])) not in _COMPAT:
            continue
        key = tuple(sorted((nodes[i], nodes[j])))
        if key in positives or key in seen:
            continue
        seen.add(key); out.append((i, j))
    return out


def _gcn_ae(X, edges_train, n, seed, hidden=64, epochs=200, lr=0.01, use_graph=True):
    torch.manual_seed(seed)
    A = torch.zeros(n, n)
    if use_graph:
        for i, j in edges_train:
            A[i, j] = A[j, i] = 1
    A = A + torch.eye(n)
    d = A.sum(1).pow(-0.5)
    An = d[:, None] * A * d[None, :]
    Xt = torch.tensor(X)
    l1, l2 = nn.Linear(X.shape[1], hidden), nn.Linear(hidden, hidden)
    opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=lr, weight_decay=5e-4)
    pos = torch.tensor(edges_train)
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        z = An @ l2(F.relu(An @ l1(Xt)))
        neg = torch.tensor(rng.randint(0, n, size=(len(edges_train), 2)))
        s_pos = (z[pos[:, 0]] * z[pos[:, 1]]).sum(1)
        s_neg = (z[neg[:, 0]] * z[neg[:, 1]]).sum(1)
        loss = F.binary_cross_entropy_with_logits(torch.cat([s_pos, s_neg]), torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (An @ l2(F.relu(An @ l1(Xt)))).numpy()


def run_link_prediction(G, mentions, E, seeds=(0, 1, 2, 3, 4)) -> dict:
    pairs = positive_pairs(G)
    nodes = sorted({x for p in pairs for x in p})
    idx = {n: i for i, n in enumerate(nodes)}
    types = [G.nodes[n]["type"] for n in nodes]
    X = entity_features(G, nodes, mentions, E)
    pos_all = [(idx[a], idx[b]) for a, b in pairs]
    pos_keys = set(pairs)
    res: dict[str, list] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(pos_all))
        n_te, n_va = int(0.2 * len(perm)), int(0.1 * len(perm))
        te_p = [pos_all[i] for i in perm[:n_te]]
        va_p = [pos_all[i] for i in perm[n_te:n_te + n_va]]
        tr_p = [pos_all[i] for i in perm[n_te + n_va:]]
        te_n = sample_negatives(nodes, types, pos_keys, len(te_p), rng)
        T = nx.Graph()
        T.add_nodes_from(range(len(nodes)))
        T.add_edges_from(tr_p)
        cand = te_p + te_n
        y = np.array([1] * len(te_p) + [0] * len(te_n))

        def score(fn):
            return np.array([fn(i, j) for i, j in cand], dtype=float)

        deg = dict(T.degree())
        methods = {
            "common_neighbours": score(lambda i, j: len(list(nx.common_neighbors(T, i, j)))),
            "jaccard": score(lambda i, j: next(nx.jaccard_coefficient(T, [(i, j)]))[2]),
            "adamic_adar": score(lambda i, j: next(nx.adamic_adar_index(T, [(i, j)]))[2]),
            "preferential_attachment": score(lambda i, j: deg[i] * deg[j]),
            "feature_cosine_(no_graph)": score(lambda i, j: float(X[i] @ X[j] / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]) + 1e-9))),
        }
        for name, use_graph in (("gcn_autoencoder", True), ("mlp_autoencoder_(no_graph)", False)):
            Z = _gcn_ae(X, tr_p, len(nodes), seed, use_graph=use_graph)
            methods[name] = score(lambda i, j, Z=Z: float(Z[i] @ Z[j]))
        for name, s in methods.items():
            res.setdefault(name, []).append((roc_auc_score(y, s), average_precision_score(y, s)))
    out = {"n_entities": len(nodes), "n_positive_edges": len(pos_all), "seeds": list(seeds),
           "protocol": "70/10/20 random edge split, type-compatible negatives (1:1), metrics on held-out test edges", "methods": {}}
    for name, v in res.items():
        a = np.array(v)
        out["methods"][name] = {"auc": {"mean": round(float(a[:, 0].mean()), 4), "std": round(float(a[:, 0].std()), 4)},
                                "ap": {"mean": round(float(a[:, 1].mean()), 4), "std": round(float(a[:, 1].std()), 4)}}
    return out
