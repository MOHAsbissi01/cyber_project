import numpy as np
import pytest
import scipy.sparse as sp
import torch

from cyberintel import gnn as G


@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(0)
    n, c = 120, 3
    y = rng.randint(0, c, n)
    X = np.hstack([np.eye(c)[y] * 1.5, rng.randn(n, 5) * 0.3])        # separable features
    A = sp.random(n, n, density=0.04, random_state=0, format="csr")
    A = ((A + A.T) > 0).astype(float)
    split = np.array(["train"] * 70 + ["val"] * 20 + ["test"] * 30)
    return X, y, A, split, c


def test_edge_softmax_sums_to_one_per_destination():
    dst = torch.tensor([0, 0, 1, 1, 1, 2])
    p = G.edge_softmax(torch.randn(6, 4), dst, 3)
    sums = torch.zeros(3, 4).index_add_(0, dst, p)
    assert torch.allclose(sums, torch.ones(3, 4), atol=1e-5)


@pytest.mark.parametrize("name", ["mlp", "gcn", "sage", "gat", "gtn"])
def test_models_train_and_beat_chance(name, toy):
    X, y, A, split, c = toy
    g = G.build_graph_inputs(A, A, len(y))
    r = G.train_node_classifier(name, X, y, split, g, c, seed=0, epochs=60, hidden=16)
    assert r["pred"].shape == (len(y),) and r["hidden"].shape[0] == len(y)
    assert (r["pred"][split == "test"] == y[split == "test"]).mean() > 0.6


def test_label_propagation_uses_only_train_labels(toy):
    X, y, A, split, c = toy
    p = G.label_propagation(A, y, split, c)
    assert p.shape == y.shape
    y2 = y.copy()
    y2[split == "test"] = (y2[split == "test"] + 1) % c           # changing TEST labels must not change the predictions
    assert (G.label_propagation(A, y2, split, c) == p).all()


def test_laplacian_pe_shape(toy):
    _, _, A, _, _ = toy
    assert G.laplacian_pe(A, 4).shape == (A.shape[0], 4)
