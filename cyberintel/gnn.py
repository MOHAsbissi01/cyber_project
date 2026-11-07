"""Graph neural networks on the incident graph (AA7): GCN, GraphSAGE, GAT, Graph Transformer (GTN) vs an MLP baseline.

Task: transductive node classification of incidents into topics. The graph links incidents that are semantically similar
(kNN on embeddings) and/or that share specific entities. Labels of val/test nodes are never used for training.

Why pure torch (decision): torch_geometric is not needed at this scale (2.2k nodes -> dense N x N attention is cheap), has no
reliable Windows/py3.13 wheels, and writing the layers ourselves lets us explain every line at the defence.
All models share: same features, same split, class-weighted loss, model selection on val macro-F1, several seeds.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------------------------------------------------------- graph helpers (dense, N is small)
def dense(A: sp.spmatrix) -> torch.Tensor:
    return torch.tensor(A.toarray(), dtype=torch.float32)


def gcn_norm(A: torch.Tensor) -> torch.Tensor:
    """D^-1/2 (A + I) D^-1/2 (Kipf & Welling)."""
    A = A + torch.eye(A.shape[0])
    d = A.sum(1).pow(-0.5)
    return d[:, None] * A * d[None, :]


def mean_norm(A: torch.Tensor) -> torch.Tensor:
    d = A.sum(1, keepdim=True).clamp(min=1e-9)
    return A / d


def laplacian_pe(A: sp.spmatrix, k: int = 8) -> np.ndarray:
    """First k non-trivial eigenvectors of the normalised Laplacian = positional encoding for graph transformers."""
    from scipy.sparse.linalg import eigsh
    A = A.maximum(A.T).astype(float)
    d = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    L = sp.eye(A.shape[0]) - dinv @ A @ dinv
    vals, vecs = eigsh(L, k=k + 1, which="SM", tol=1e-3, v0=np.ones(A.shape[0]) / math.sqrt(A.shape[0]))
    return vecs[:, np.argsort(vals)[1:k + 1]].astype(np.float32)


# ----------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(s, din, hid, nc, dropout):
        super().__init__()
        s.l1, s.l2, s.dp = nn.Linear(din, hid), nn.Linear(hid, nc), nn.Dropout(dropout)

    def forward(s, x, g):
        h = F.relu(s.l1(s.dp(x)))
        return s.l2(s.dp(h)), h


class GCN(nn.Module):
    def __init__(s, din, hid, nc, dropout):
        super().__init__()
        s.l1, s.l2, s.dp = nn.Linear(din, hid), nn.Linear(hid, nc), nn.Dropout(dropout)

    def forward(s, x, g):
        h = F.relu(g["gcn"] @ s.l1(s.dp(x)))
        return g["gcn"] @ s.l2(s.dp(h)), h


class SAGE(nn.Module):
    """GraphSAGE with the mean aggregator: h_i = W [x_i || mean_{j in N(i)} x_j]."""

    def __init__(s, din, hid, nc, dropout):
        super().__init__()
        s.l1, s.l2, s.dp = nn.Linear(2 * din, hid), nn.Linear(2 * hid, nc), nn.Dropout(dropout)

    def forward(s, x, g):
        x = s.dp(x)
        h = F.relu(s.l1(torch.cat([x, g["mean"] @ x], 1)))
        h = s.dp(h)
        return s.l2(torch.cat([h, g["mean"] @ h], 1)), h


def edge_softmax(score: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of edge scores (E, H) over the incoming edges of each destination node (scatter implementation)."""
    h = score.shape[1]
    mx = torch.full((n, h), float("-inf")).scatter_reduce(0, dst[:, None].expand(-1, h), score, "amax", include_self=False)
    ex = (score - mx[dst]).exp()
    den = torch.zeros(n, h).index_add_(0, dst, ex)
    return ex / den[dst].clamp(min=1e-16)


class GATLayer(nn.Module):
    """Graph attention (Velickovic et al.): e_ij = LeakyReLU(a_dst . h_i + a_src . h_j), softmax over the neighbours j of i."""

    def __init__(s, din, dout, heads, dropout, concat=True):
        super().__init__()
        s.h, s.d, s.concat, s.p = heads, dout, concat, dropout
        s.W = nn.Linear(din, heads * dout, bias=False)
        s.a_src, s.a_dst = nn.Parameter(torch.empty(heads, dout)), nn.Parameter(torch.empty(heads, dout))
        nn.init.xavier_uniform_(s.a_src); nn.init.xavier_uniform_(s.a_dst)

    def forward(s, x, g):
        N, (dst, src) = x.shape[0], (g["dst"], g["src"])
        h = s.W(x).view(N, s.h, s.d)
        e = F.leaky_relu((h * s.a_dst).sum(-1)[dst] + (h * s.a_src).sum(-1)[src], 0.2)          # E, H
        a = F.dropout(edge_softmax(e, dst, N), s.p, s.training)
        out = torch.zeros(N, s.h, s.d).index_add_(0, dst, a[:, :, None] * h[src])
        return out.reshape(N, -1) if s.concat else out.mean(1)


class GAT(nn.Module):
    def __init__(s, din, hid, nc, dropout, heads=4):
        super().__init__()
        s.g1, s.g2, s.dp = GATLayer(din, hid // heads, heads, dropout), GATLayer(hid, nc, 1, dropout, concat=False), nn.Dropout(dropout)

    def forward(s, x, g):
        h = F.elu(s.g1(s.dp(x), g))
        return s.g2(s.dp(h), g), h


class GraphTransformerLayer(nn.Module):
    """Self-attention restricted to graph neighbours (+ self). The attention score of an edge gets a learnable bias per EDGE TYPE
    (similarity edge vs shared-entity edge): the model learns how much each relation type should count (heterogeneous graph).
    Implemented on the edge list (scatter softmax): same maths as masked dense attention, ~100x cheaper on a sparse graph."""

    def __init__(s, d, heads, dropout, n_edge_types):
        super().__init__()
        s.h, s.dh, s.p = heads, d // heads, dropout
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        s.n1, s.n2, s.dp = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)
        s.edge_bias = nn.Parameter(torch.zeros(n_edge_types, heads))

    def forward(s, x, g):
        N, (dst, src) = x.shape[0], (g["dst"], g["src"])
        q, k, v = s.qkv(x).view(N, 3, s.h, s.dh).unbind(1)
        sc = (q[dst] * k[src]).sum(-1) / math.sqrt(s.dh) + g["etype"] @ s.edge_bias                # E, H
        a = F.dropout(edge_softmax(sc, dst, N), s.p, s.training)
        out = torch.zeros(N, s.h, s.dh).index_add_(0, dst, a[:, :, None] * v[src]).reshape(N, -1)
        x = s.n1(x + s.dp(s.o(out)))
        return s.n2(x + s.dp(s.ff(x)))


class GTN(nn.Module):
    def __init__(s, din, hid, nc, dropout, layers=2, heads=4, n_edge_types=2, pe_dim=0):
        super().__init__()
        s.inp, s.pe_dim = nn.Linear(din + pe_dim, hid), pe_dim
        s.layers = nn.ModuleList(GraphTransformerLayer(hid, heads, dropout, n_edge_types) for _ in range(layers))
        s.out, s.dp = nn.Linear(hid, nc), nn.Dropout(dropout)

    def forward(s, x, g):
        if s.pe_dim:
            pe = g["pe"] * (torch.randint(0, 2, (1, s.pe_dim)) * 2 - 1 if s.training else 1)   # random sign flip (eigenvector sign ambiguity)
            x = torch.cat([x, pe], 1)
        h = s.dp(F.relu(s.inp(s.dp(x))))
        for layer in s.layers:
            h = layer(h, g)
        return s.out(s.dp(h)), h


MODELS = {"mlp": MLP, "gcn": GCN, "sage": SAGE, "gat": GAT, "gtn": GTN}


# ----------------------------------------------------------------------------- training
def build_graph_inputs(A_sim: sp.spmatrix | None, A_ent: sp.spmatrix | None, n: int, pe_k: int = 0) -> dict:
    mats = [m for m in (A_sim, A_ent) if m is not None]
    A = sum(mats) if mats else sp.csr_matrix((n, n))
    Ad = dense(A)
    mask = (Ad > 0) | (torch.eye(n) > 0)                       # neighbours + self
    dst, src = mask.nonzero(as_tuple=True)                     # attention of node `dst` over its neighbour `src`
    etype = torch.stack([(dense(m)[dst, src] > 0).float() if m is not None else torch.zeros(len(dst)) for m in (A_sim, A_ent)], 1)
    g = {"gcn": gcn_norm(Ad), "mean": mean_norm((Ad > 0).float()), "dst": dst, "src": src, "etype": etype}
    if pe_k:
        g["pe"] = torch.tensor(laplacian_pe(A, pe_k))
    return g


def train_node_classifier(model_name: str, X: np.ndarray, y: np.ndarray, split: np.ndarray, graph: dict, n_classes: int, seed: int,
                          hidden: int = 64, epochs: int = 200, lr: float = 0.01, wd: float = 5e-4, dropout: float = 0.5,
                          pe_dim: int = 0, patience: int = 40) -> dict:
    """Returns predictions for all nodes plus val/test metrics. Model selection = best val macro-F1."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    tr, va = [torch.tensor(np.where(split == s)[0]) for s in ("train", "val")]
    cnt = np.bincount(y[split == "train"], minlength=n_classes).astype(float)
    w = torch.tensor((cnt.sum() / (n_classes * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)
    kw = dict(din=X.shape[1], hid=hidden, nc=n_classes, dropout=dropout)
    if model_name == "gtn":
        kw["pe_dim"] = pe_dim
    model = MODELS[model_name](**kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out, _ = model(Xt, graph)
        F.cross_entropy(out[tr], yt[tr], weight=w).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xt, graph)[0][va].argmax(1).numpy()
        f = f1_score(y[va.numpy()], pv, average="macro", zero_division=0)
        if f > best:
            best, state, bad = f, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        out, h = model(Xt, graph)
    pred = out.argmax(1).numpy()
    return {"pred": pred, "hidden": h.numpy(), "val_macro_f1": float(best), "epochs_run": ep + 1,
            "attn_edge_bias": [l.edge_bias.detach().numpy().round(3).tolist() for l in getattr(model, "layers", [])]}


def label_propagation(A: sp.spmatrix, y: np.ndarray, split: np.ndarray, n_classes: int, alpha: float = 0.9, iters: int = 30) -> np.ndarray:
    """Graph-only baseline (no text features): propagate TRAIN labels along the graph, F <- alpha*S F + (1-alpha) Y."""
    A = A.maximum(A.T).astype(float)
    d = np.asarray(A.sum(1)).ravel()
    S = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9))) @ A @ sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    Y = np.zeros((len(y), n_classes))
    tr = split == "train"
    Y[np.where(tr)[0], y[tr]] = 1
    Fm = Y.copy()
    for _ in range(iters):
        Fm = alpha * (S @ Fm) + (1 - alpha) * Y
    pred = Fm.argmax(1)
    pred[(Fm.sum(1) == 0)] = np.bincount(y[tr]).argmax()      # isolated nodes: majority class
    return pred


def summarize(preds_by_seed: list[np.ndarray], y: np.ndarray, split: np.ndarray, gold_idx: np.ndarray | None, y_gold: np.ndarray | None) -> dict:
    te = split == "test"
    rows = []
    for p in preds_by_seed:
        r = {"test_macro_f1": f1_score(y[te], p[te], average="macro", zero_division=0), "test_acc": accuracy_score(y[te], p[te])}
        if gold_idx is not None and len(gold_idx):
            r["gold_macro_f1"] = f1_score(y_gold, p[gold_idx], average="macro", zero_division=0)
            r["gold_acc"] = accuracy_score(y_gold, p[gold_idx])
        rows.append(r)
    d = pd.DataFrame(rows)
    return {k: {"mean": round(float(d[k].mean()), 4), "std": round(float(d[k].std(ddof=0)), 4)} for k in d.columns} | {"n_seeds": len(rows)}
