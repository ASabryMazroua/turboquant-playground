"""Unit tests for turbo_kv.quantizers (M2: per-coordinate scalar quant)."""
import math

import pytest

torch = pytest.importorskip("torch")

from turbo_kv import quantizers as Q  # noqa: E402


def test_num_levels_and_effective_bits():
    assert Q.num_levels(4) == 16
    assert Q.num_levels(3) == 8
    assert Q.num_levels(2.5) == 6  # round(2**2.5)=round(5.657)=6
    assert Q.num_levels(3.5) == 11  # round(2**3.5)=round(11.31)=11
    assert abs(Q.effective_bits(4) - 4.0) < 1e-9
    assert abs(Q.effective_bits(3) - 3.0) < 1e-9


@pytest.mark.parametrize("bits", [2.5, 3, 3.5, 4])
@pytest.mark.parametrize("symmetric", [False, True])
def test_roundtrip_shape_and_finite(bits, symmetric):
    x = torch.randn(128, 64)
    x_hat = Q.fake_quantize_per_coord(x, bits, symmetric=symmetric)
    assert x_hat.shape == x.shape
    assert torch.isfinite(x_hat).all()


def test_error_decreases_with_more_bits():
    x = torch.randn(512, 64)
    errs = [Q.quantization_error(x, b)["mse"] for b in [2.5, 3, 3.5, 4]]
    # More bits ⇒ smaller (or equal) MSE, monotonically.
    for lo, hi in zip(errs, errs[1:]):
        assert hi <= lo + 1e-9


def test_reconstruction_within_quant_step():
    # Affine min/max quantizer: |x - x̂| ≤ scale/2 per coordinate.
    torch.manual_seed(0)
    x = torch.randn(256, 32)
    bits = 4
    x_hat = Q.fake_quantize_per_coord(x, bits, symmetric=False)
    L = Q.num_levels(bits)
    rng = x.amax(0) - x.amin(0)
    step = rng / (L - 1)
    assert (x_hat - x).abs().max(0).values.le(step / 2 + 1e-5).all()


def test_higher_bits_better_cosine():
    x = torch.randn(512, 64)
    c_lo = Q.quantization_error(x, 2.5)["cosine"]
    c_hi = Q.quantization_error(x, 4)["cosine"]
    assert c_hi >= c_lo


def test_per_token_axis_scales_per_row():
    # Per-token quant gives each row its own scale → a row with a huge outlier
    # does not corrupt the resolution of other rows.
    x = torch.randn(64, 32)
    x[0] *= 100.0  # outlier token
    x_hat = Q.fake_quantize(x, 4, axis="token")
    assert x_hat.shape == x.shape
    # The well-behaved rows keep low relative error regardless of the outlier row.
    rel = ((x_hat[1:] - x[1:]).abs().mean() / x[1:].abs().mean()).item()
    assert rel < 0.1


def test_rotation_helps_per_token_with_channel_outlier():
    # The TurboQuant premise: with a CHANNEL outlier and PER-TOKEN quant, an
    # orthogonal (Hadamard) rotation spreads the outlier across coordinates and
    # cuts the per-token quantization error. Per-channel quant is unaffected.
    from turbo_kv import rotations as R

    torch.manual_seed(0)
    d = 64
    x = torch.randn(256, d) * 0.1
    x[:, 3] += 8.0  # dominant channel outlier
    rot = R.make_rotation("rht", d, seed=0, dtype=torch.float32)
    xr = rot.rotate(x)

    e_none = Q.quantization_error(x, 4, axis="token")["mse"]
    e_rot = Q.quantization_error(xr, 4, axis="token")["mse"]
    assert e_rot < e_none  # rotation reduces per-token error

    # Per-channel quant already isolates the outlier channel; rotation does not help.
    c_none = Q.quantization_error(x, 4, axis="channel")["mse"]
    c_rot = Q.quantization_error(xr, 4, axis="channel")["mse"]
    assert c_rot >= c_none * 0.5  # not a large improvement (typically worse)
