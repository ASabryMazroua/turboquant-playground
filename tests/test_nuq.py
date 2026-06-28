"""Unit tests for turbo_kv.quantizers NUQ (M12: non-uniform quantization)."""
import pytest

torch = pytest.importorskip("torch")

from turbo_kv import quantizers as Q  # noqa: E402


def _heavy_tailed(n, d, seed=0):
    """Heavy-tailed samples (Student-t mixed with a spiky Gaussian core)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.distributions.StudentT(df=2.5).sample((n, d))  # heavy tails
    core = torch.randn(n, d, generator=g) * 0.2              # dense mode
    mask = (torch.rand(n, d, generator=g) < 0.85).float()
    return mask * core + (1 - mask) * t


def test_nuq_beats_uniform_heavy_tailed():
    # KVQuant premise: fitted levels beat a uniform grid on heavy-tailed data.
    x = _heavy_tailed(512, 64)
    bits = 3
    mse_uniform = Q.quantization_error(x, bits, axis="token")["mse"]
    mse_nuq = Q.quantization_error_nuq(x, bits, axis="token", method="kmeans")["mse"]
    assert mse_nuq < mse_uniform


def test_kmeans_monotone_per_iteration():
    # Lloyd iterations cannot increase the reconstruction MSE.
    x = _heavy_tailed(256, 32, seed=1)
    prev = None
    for iters in range(0, 6):
        mse = Q.quantization_error_nuq(x, 3, axis="token", method="kmeans", iters=iters)["mse"]
        if prev is not None:
            assert mse <= prev + 1e-6
        prev = mse


@pytest.mark.parametrize("axis", ["token", "channel"])
def test_nuq_uses_at_most_L_levels_per_group(axis):
    x = _heavy_tailed(128, 48, seed=2)
    bits = 3
    L = Q.num_levels(bits)
    x_hat = Q.fake_quantize_nuq(x, bits, axis=axis, method="kmeans")
    assert x_hat.shape == x.shape
    if axis == "token":
        groups = x_hat            # each row is a group
    else:
        groups = x_hat.t()        # each coordinate (column) is a group
    for row in groups:
        assert torch.unique(row).numel() <= L


@pytest.mark.parametrize("axis", ["token", "channel"])
def test_quantile_method_finite_and_shaped(axis):
    x = _heavy_tailed(96, 40, seed=3)
    x_hat = Q.fake_quantize_nuq(x, 3.5, axis=axis, method="quantile")
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_fit_levels_broadcast_shapes():
    x = torch.randn(20, 16)
    L = Q.num_levels(3)
    lv_tok = Q.fit_nuq_levels(x, 3, axis="token", method="quantile")
    lv_ch = Q.fit_nuq_levels(x, 3, axis="channel", method="quantile")
    assert lv_tok.shape == (20, 1, L)   # one codebook per token row
    assert lv_ch.shape == (16, L)       # one codebook per coordinate
    # Both must broadcast against x[..., None] -> [20, 16, L].
    assert (x[..., None] - lv_tok).shape == (20, 16, L)
    assert (x[..., None] - lv_ch).shape == (20, 16, L)
