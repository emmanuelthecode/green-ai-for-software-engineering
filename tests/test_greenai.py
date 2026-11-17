import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from greenai import carbon, data  # noqa: E402
from greenai.compress import QuantisedMLP, magnitude_prune, structured_prune  # noqa: E402
from greenai.energy import EnergyTracker  # noqa: E402
from greenai.mlp import MLP, softmax, train  # noqa: E402


@pytest.fixture(scope="module")
def blobs():
    """Three well-separated Gaussian clusters - any working model should ace it."""
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((3, 20)) * 4
    y = rng.integers(0, 3, 600)
    X = centres[y] + rng.standard_normal((600, 20))
    return X[:400], y[:400], X[400:], y[400:]


def numeric_grad(model, X, y, param, idx, eps=1e-6, **kw):
    old = param[idx]
    param[idx] = old + eps
    lp, _ = model.gradients(X, y, **kw)
    param[idx] = old - eps
    lm, _ = model.gradients(X, y, **kw)
    param[idx] = old
    return (lp - lm) / (2 * eps)


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    X, y = rng.standard_normal((8, 6)), rng.integers(0, 3, 8)
    m = MLP([6, 5, 3], seed=0, dtype=np.float64)
    _, grads = m.gradients(X, y)
    for p, g in zip(m.W + m.b, grads):
        idx = tuple(rng.integers(0, s) for s in p.shape)
        assert np.isclose(g[idx], numeric_grad(m, X, y, p, idx), atol=1e-6)


def test_distillation_and_lora_gradients():
    rng = np.random.default_rng(2)
    X, y = rng.standard_normal((8, 6)), rng.integers(0, 3, 8)
    tp = softmax(rng.standard_normal((8, 3)))
    m = MLP([6, 5, 3], seed=0, dtype=np.float64)
    m.add_lora(rank=2, seed=0)
    A, B = m.lora[0]
    B[...] = rng.standard_normal(B.shape) * 0.1      # make the adapter non-trivial
    _, grads = m.gradients(X, y, teacher_probs=tp)
    for p, g in zip([A, B, m.W[-1], m.b[-1]], grads):
        idx = tuple(rng.integers(0, s) for s in p.shape)
        assert np.isclose(g[idx], numeric_grad(m, X, y, p, idx, teacher_probs=tp), atol=1e-6)


def test_training_learns(blobs):
    Xtr, ytr, Xte, yte = blobs
    m = MLP([20, 16, 3], seed=0)
    train(m, Xtr, ytr, epochs=10)
    assert m.accuracy(Xte, yte) > 0.95


def test_lora_only_trains_adapters_and_head(blobs):
    Xtr, ytr, _, _ = blobs
    m = MLP([20, 16, 3], seed=0)
    frozen_before = m.W[0].copy()
    m.add_lora(rank=2)
    assert m.n_trainable() == 20 * 2 + 2 * 16 + 16 * 3 + 3
    train(m, Xtr, ytr, epochs=2)
    assert np.array_equal(m.W[0], frozen_before)


def test_pruning(blobs):
    Xtr, ytr, Xte, yte = blobs
    m = MLP([20, 32, 3], seed=0)
    train(m, Xtr, ytr, epochs=10)

    p = magnitude_prune(m, 0.5)
    assert abs((p.W[0] == 0).mean() - 0.5) < 0.02
    train(p, Xtr, ytr, epochs=2)
    assert abs((p.W[0] == 0).mean() - 0.5) < 0.02        # zeros stay zero

    s = structured_prune(m, 0.25)
    assert s.W[0].shape == (20, 8) and s.W[1].shape == (8, 3)
    assert s.flops_per_sample() < m.flops_per_sample()


def test_quantisation_is_close(blobs):
    Xtr, ytr, Xte, yte = blobs
    m = MLP([20, 16, 3], seed=0)
    train(m, Xtr, ytr, epochs=10)
    q = QuantisedMLP(m)
    assert q.size_bytes() < m.size_bytes() / 2.5
    assert np.abs(q.forward(Xte) - m.forward(Xte)).max() < 0.5
    assert q.accuracy(Xte, yte) >= m.accuracy(Xte, yte) - 0.02


def test_early_stopping_stops(blobs):
    Xtr, ytr, Xte, yte = blobs
    m = MLP([20, 16, 3], seed=0)
    ran = train(m, Xtr, ytr, epochs=100, val=(Xte, yte), patience=2)
    assert ran < 100


def test_energy_tracker():
    with EnergyTracker("t") as m:
        sum(i * i for i in range(200_000))
    assert m.wall_s > 0 and m.energy_j > 0 and m.co2_g(100) > 0


def test_carbon_scheduling_picks_cleanest_window():
    profile = np.array([5.0] * 24)
    profile[3:5] = 1.0
    start, avg = carbon.best_window(profile, 2)
    assert (start, avg) == (3, 1.0)
    out = carbon.compare(1.0, profile, hours=2, naive_start=10)
    assert np.isclose(out["saving_pct"], 80.0)


def test_tokens_split_identifiers():
    toks = data.tokens("def parseHttpHeader(raw_value): pass")
    for t in ("parsehttpheader", "parse", "http", "header", "raw", "value", "def"):
        assert t in toks
