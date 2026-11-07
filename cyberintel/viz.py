"""Static figures for reports (matplotlib, Agg backend). The UI draws interactive versions itself."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

TYPE_COLORS = {"ATTACK": "#d95f02", "TARGET": "#9aa5b1", "SOFTWARE": "#1b9e77", "CVE": "#e7298a", "MALWARE": "#7570b3",
               "CWE": "#a6761d", "MITRE_TECHNIQUE": "#e6ab02", "TACTIC": "#b8860b", "IP": "#666666", "HASH": "#999999",
               "THREAT_ACTOR": "#b2182b", "TOOL": "#2166ac", "TECHNOLOGY": "#4393c3", "VULN_NAME": "#f781bf", "DEFENSE": "#33a02c",
               "INCIDENT": "#cccccc"}


def plot_confusion(cm, classes, path: Path, title: str, normalize: bool = True):
    cm = np.array(cm, dtype=float)
    disp = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if disp[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_entity_graph(H: nx.DiGraph, path: Path, max_nodes: int = 55, title: str = "Entity graph"):
    """H: collapsed entity DiGraph. Shows the highest-degree entities; colour = type, size = degree, arrows = relation direction."""
    keep = sorted(H.nodes, key=lambda x: -H.degree(x, weight="weight"))[:max_nodes]
    S = H.subgraph(keep)
    pos = nx.spring_layout(S.to_undirected(), seed=1, k=1.4 / math.sqrt(max(len(S), 1)), weight="weight")
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(S, pos, width=[0.5 + d["weight"] ** 0.5 * 0.5 for *_, d in S.edges(data=True)], alpha=0.35, ax=ax,
                           arrows=True, arrowsize=8, edge_color="#555")
    nx.draw_networkx_nodes(S, pos, node_color=[TYPE_COLORS.get(S.nodes[x]["type"], "#333") for x in S],
                           node_size=[150 + 70 * S.degree(x) for x in S], ax=ax, edgecolors="white")
    nx.draw_networkx_labels(S, pos, {x: S.nodes[x]["label"] for x in S}, font_size=7.5, ax=ax)
    for t in sorted({S.nodes[x]["type"] for x in S}):
        ax.scatter([], [], c=TYPE_COLORS.get(t, "#333"), label=t)
    ax.legend(loc="lower left", fontsize=8); ax.set_axis_off(); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_embedding_2d(Z, labels, path: Path, title: str, max_legend: int = 14):
    fig, ax = plt.subplots(figsize=(9, 7))
    labs = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, l in enumerate(labs):
        m = np.array(labels) == l
        ax.scatter(Z[m, 0], Z[m, 1], s=8, color=cmap(i % 20), label=str(l), alpha=0.75)
    ax.legend(markerscale=2, fontsize=7, ncol=2 if len(labs) > 8 else 1, loc="best") if len(labs) <= max_legend else None
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_bar(names, values, path: Path, title: str, ylabel: str, errs=None, ylim=(0, 1)):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, values, yerr=errs, color="#4393c3", capsize=3)
    for i, v in enumerate(values):
        ax.text(i, v + (0.01 if errs is None else (errs[i] or 0) + 0.01), f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
