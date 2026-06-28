"""Unit tests for the M11 direct large-m QJL key sketch (turbo_kv.qjl additions)."""
import math

import pytest

torch = pytest.importorskip("torch")

from turbo_kv.qjl import (  # noqa: E402
    QJLSketch,
    direct_bits_per_value,
    encode_key_direct,
    logits_direct,
)


def _ip_rmse(Rq, Rk, m, n_outliers, seed=0):
    sk = QJLSketch(Rk.shape[-1], m=m, seed=seed)
    signs, norm, oi, ov = encode_key_direct(Rk, sk, n_outliers=n_outliers)
    est = logits_direct(Rq, signs, norm, sk, out_idx=oi, out_val=ov)
    true = Rq @ Rk.t()
    return float(torch.sqrt(((est - true) ** 2).mean()))


def test_direct_rmse_decreases_with_m():
    # Variance of the unbiased sign-sketch estimator scales as 1/m, so the IP
    # RMSE of logits_direct (no outliers) must fall monotonically as m grows.
    torch.manual_seed(0)
    d = 64
    Rq = torch.randn(8, d, dtype=torch.float64)
    Rk = torch.randn(16, d, dtype=torch.float64)
    rmses = [_ip_rmse(Rq, Rk, m, n_outliers=0, seed=1) for m in (64, 128, 256, 512)]
    for lo, hi in zip(rmses[1:], rmses[:-1]):
        assert lo < hi


def test_outliers_reduce_error():
    # A key with a couple of extreme coordinates inflates the sketch variance;
    # keeping them exact (n_outliers=8) lowers the IP RMSE at the same m.
    torch.manual_seed(0)
    d = 64
    Rq = torch.randn(8, d, dtype=torch.float64)
    Rk = torch.randn(16, d, dtype=torch.float64)
    Rk[:, 3] *= 12.0
    Rk[:, 17] *= 9.0
    err_no = _ip_rmse(Rq, Rk, m=128, n_outliers=0, seed=2)
    err_out = _ip_rmse(Rq, Rk, m=128, n_outliers=8, seed=2)
    assert err_out < err_no


def test_direct_is_unbiased_on_average():
    # A single sketch is noisy but the estimator is unbiased: averaging
    # logits_direct over many independent sketch seeds approaches the true IP.
    torch.manual_seed(0)
    d = 64
    Rq = torch.randn(1, d, dtype=torch.float64)
    Rk = torch.randn(1, d, dtype=torch.float64)
    true = float(Rq @ Rk.t())
    ests = []
    for seed in range(400):
        sk = QJLSketch(d, m=d, seed=seed)
        signs, norm, oi, ov = encode_key_direct(Rk, sk, n_outliers=0)
        ests.append(float(logits_direct(Rq, signs, norm, sk, out_idx=oi, out_val=ov)))
    mean_est = sum(ests) / len(ests)
    assert abs(mean_est - true) < 0.1 * (abs(true) + 1.0)


def test_outlier_term_matches_exact_contribution():
    # With outliers, logits_direct = (sketch estimate of dense part) + exact
    # outlier IP. Subtracting the no-outlier sketch estimate must recover the
    # exact Rq·Rk_outliers term (independent of the sketch noise on the dense
    # part, since both share the same sketch/signs are recomputed on dense).
    torch.manual_seed(0)
    d = 64
    Rq = torch.randn(4, d, dtype=torch.float64)
    Rk = torch.randn(6, d, dtype=torch.float64)
    Rk[:, 5] *= 10.0
    sk = QJLSketch(d, m=256, seed=3)
    signs, norm, oi, ov = encode_key_direct(Rk, sk, n_outliers=8)
    with_out = logits_direct(Rq, signs, norm, sk, out_idx=oi, out_val=ov)
    dense_only = sk.estimate_matrix(Rq, signs, norm)
    # Build the exact outlier matrix from the returned indices/values.
    outlier_mat = torch.zeros(Rk.shape[0], d, dtype=Rq.dtype)
    outlier_mat.scatter_(-1, oi.to(torch.long), ov.to(Rq.dtype))
    exact_out_term = Rq @ outlier_mat.t()
    assert torch.allclose(with_out - dense_only, exact_out_term, atol=1e-4)


def test_direct_bits_per_value_formula():
    # m sign bits + 16-bit norm + outliers (16-bit value + ceil(log2 d) index),
    # all amortized over head_dim coordinates.
    d = 64
    idx_bits = math.ceil(math.log2(d))  # 6 for d=64

    sk256 = QJLSketch(d, m=256)
    expect0 = 256 / d + 16.0 / d
    assert abs(direct_bits_per_value(sk256, head_dim=d, n_outliers=0) - expect0) < 1e-9

    expect8 = 512 / d + 16.0 / d + 8 * (16.0 + idx_bits) / d
    sk512 = QJLSketch(d, m=512)
    assert abs(direct_bits_per_value(sk512, head_dim=d, n_outliers=8) - expect8) < 1e-9
