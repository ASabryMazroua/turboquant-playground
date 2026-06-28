"""Unit tests for turbo_kv.qjl — the 1-bit QJL residual (TurboQuant-prod, M6)."""
import math

import pytest

torch = pytest.importorskip("torch")

from turbo_kv import rotations as R  # noqa: E402
from turbo_kv.qjl import (  # noqa: E402
    QJLSketch,
    logits_mse,
    logits_prod,
)


def test_qjl_large_m_approximates_inner_product():
    # With many projections the sign-sketch estimator converges to the true q·r.
    torch.manual_seed(0)
    d = 32
    q = torch.randn(4, d, dtype=torch.float64)
    r = torch.randn(5, d, dtype=torch.float64)
    sk = QJLSketch(d, m=20000, seed=1)
    signs, norm = sk.sketch(r)
    est = sk.estimate_matrix(q, signs, norm)         # [4, 5]
    true = q @ r.t()
    rel = (est - true).abs().mean() / true.abs().mean()
    assert rel < 0.1


def test_qjl_estimator_is_unbiased_on_average():
    # A single d-row sketch is noisy, but the estimator is unbiased: averaging
    # over independent sketches converges to the true inner product.
    torch.manual_seed(0)
    d = 64
    q = torch.randn(1, d, dtype=torch.float64)
    r = torch.randn(1, d, dtype=torch.float64)
    true = float(q @ r.t())
    ests = []
    for seed in range(600):
        sk = QJLSketch(d, m=d, seed=seed)
        signs, norm = sk.sketch(r)
        ests.append(float(sk.estimate_matrix(q, signs, norm)))
    mean_est = sum(ests) / len(ests)
    assert abs(mean_est - true) < 0.1 * (abs(true) + 1.0)


def test_prod_residual_is_unbiased_in_expectation():
    # MSE-only IP estimate is biased (shrunk ‖k̂‖). A single 1-bit sketch is
    # unbiased but noisy; averaging the QJL-prod estimate over many independent
    # sketches converges to an unbiased estimate, so its |bias| < MSE-only's.
    torch.manual_seed(0)
    d, n = 64, 512
    K = torch.randn(n, d, dtype=torch.float64)
    K[:, 0] *= 15.0  # outlier channel
    qq = torch.randn(96, d, dtype=torch.float64)
    rot = R.make_rotation("rht", d, seed=0, dtype=torch.float64)
    Rk, Rq = rot.rotate(K), rot.rotate(qq)
    exact = Rq @ Rk.t()

    bits = 3.0
    bias_mse = (logits_mse(Rq, Rk, bits, axis="token") - exact).mean().abs()
    acc = torch.zeros_like(exact)
    n_sketch = 96
    for seed in range(n_sketch):
        acc += logits_prod(Rq, Rk, bits, QJLSketch(d, m=d, seed=seed), axis="token")
    bias_prod = (acc / n_sketch - exact).mean().abs()
    assert bias_prod < bias_mse


def test_sketch_bits_per_value():
    assert QJLSketch(64, m=64).sketch_bits_per_value() == 1.0
    assert QJLSketch(64, m=32).sketch_bits_per_value() == 0.5
