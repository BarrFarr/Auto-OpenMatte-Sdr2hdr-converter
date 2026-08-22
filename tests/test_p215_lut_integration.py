"""P2.15: Production LUT integration tests.

Tests:
1. LUT construction
2. LUT range
3. Black branch
4. Low-end bridge
5. Top-end behavior
6. PCHIP-vs-LUT accuracy
7. Curve-specific LUT (per-shot)
8. Production integration (apply_shot_transform uses LUT)
"""

import numpy as np
import pytest

from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.processing.luminance import (
    _LOG_EPS,
    _LUT_SIZE,
    _PEAK_NITS,
    apply_luminance_curve,
    apply_luminance_curve_reference,
    build_curve_lut,
    estimate_luminance_curve,
)
from auto_openmatte.processing.transform import apply_shot_transform


@pytest.fixture
def realistic_curve():
    """Generate a realistic fitted curve from synthetic overlap data."""
    rng = np.random.default_rng(42)
    sdr = rng.random(100000) * 0.5 + 0.001
    hdr = sdr * 0.002 + rng.random(100000) * 0.0001
    config = ColorConfig(luminance_bins=256, low_percentile=1.0, high_percentile=99.9)
    return estimate_luminance_curve(sdr, hdr, config=config)


@pytest.fixture
def lut_data(realistic_curve):
    """Pre-built LUT for the realistic curve."""
    return build_curve_lut(realistic_curve)


# --- 1. LUT Construction ---


class TestLUTConstruction:
    def test_lut_size(self, lut_data):
        assert len(lut_data["lut"]) == _LUT_SIZE

    def test_lut_dtype(self, lut_data):
        assert lut_data["lut"].dtype == np.float64

    def test_lut_keys(self, lut_data):
        expected = {"lut", "curve_start_linear", "curve_max_linear",
                    "curve_start_hdr_linear", "inv_range"}
        assert set(lut_data.keys()) == expected

    def test_lut_values_non_negative(self, lut_data):
        assert np.all(lut_data["lut"] >= 0)

    def test_lut_monotonic(self, lut_data):
        diffs = np.diff(lut_data["lut"])
        assert np.all(diffs >= -1e-15), "LUT must be monotonically non-decreasing"

    def test_lut_memory(self, lut_data):
        assert lut_data["lut"].nbytes == _LUT_SIZE * 8  # float64


# --- 2. LUT Range ---


class TestLUTRange:
    def test_curve_start_positive(self, lut_data):
        assert lut_data["curve_start_linear"] > 0

    def test_curve_max_greater_than_start(self, lut_data):
        assert lut_data["curve_max_linear"] > lut_data["curve_start_linear"]

    def test_inv_range_correct(self, lut_data):
        expected = 1.0 / (lut_data["curve_max_linear"] - lut_data["curve_start_linear"])
        assert abs(lut_data["inv_range"] - expected) < 1e-10

    def test_range_matches_curve(self, realistic_curve, lut_data):
        x_pts = np.array([p[0] for p in realistic_curve])
        expected_start = (10.0**x_pts[0] - _LOG_EPS) / _PEAK_NITS
        expected_max = (10.0**x_pts[-1] - _LOG_EPS) / _PEAK_NITS
        assert abs(lut_data["curve_start_linear"] - expected_start) < 1e-12
        assert abs(lut_data["curve_max_linear"] - expected_max) < 1e-12


# --- 3. Black Branch ---


class TestBlackBranch:
    def test_zero_input(self, realistic_curve, lut_data):
        inp = np.array([0.0])
        result = apply_luminance_curve(inp, realistic_curve, prebuilt_lut=lut_data)
        assert result[0] == 0.0

    def test_negative_input(self, realistic_curve, lut_data):
        inp = np.array([-1e-10, -1.0])
        result = apply_luminance_curve(inp, realistic_curve, prebuilt_lut=lut_data)
        assert np.all(result == 0.0)

    def test_array_of_zeros(self, realistic_curve, lut_data):
        inp = np.zeros(1000)
        result = apply_luminance_curve(inp, realistic_curve, prebuilt_lut=lut_data)
        assert np.all(result == 0.0)


# --- 4. Low-End Bridge ---


class TestLowEndBridge:
    def test_bridge_matches_reference(self, realistic_curve, lut_data):
        """Bridge values must be bit-perfect with reference (not from LUT)."""
        cs = lut_data["curve_start_linear"]
        inputs = np.linspace(1e-10, cs * 0.99, 1000)
        ref = apply_luminance_curve_reference(inputs, realistic_curve)
        lut = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        np.testing.assert_array_equal(ref, lut)

    def test_bridge_monotonic(self, realistic_curve, lut_data):
        cs = lut_data["curve_start_linear"]
        inputs = np.linspace(0, cs, 10000)
        result = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        assert np.all(np.diff(result) >= 0)

    def test_bridge_starts_at_zero(self, realistic_curve, lut_data):
        inp = np.array([1e-15])
        result = apply_luminance_curve(inp, realistic_curve, prebuilt_lut=lut_data)
        assert result[0] < 1e-10  # effectively zero

    def test_bridge_no_white_dots(self, realistic_curve, lut_data):
        """Near-zero inputs must NOT produce bright outputs."""
        inputs = np.array([1e-12, 1e-10, 1e-8, 1e-7, 1e-6])
        result = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        nits = result * _PEAK_NITS
        assert np.all(nits < 0.1), f"White dots detected: {nits}"


# --- 5. Top-End Behavior ---


class TestTopEnd:
    def test_above_curve_max_clamped(self, realistic_curve, lut_data):
        """Values above curve_max_linear should clamp to LUT's last entry."""
        cm = lut_data["curve_max_linear"]
        inputs = np.array([cm, cm * 1.1, cm * 2.0, 1.0])
        result = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        # All above-max values should get the same output as max
        assert np.all(result[1:] == result[0])

    def test_top_end_matches_reference_at_max(self, realistic_curve, lut_data):
        """At exactly curve_max, LUT and reference should be close."""
        cm = lut_data["curve_max_linear"]
        inp = np.array([cm])
        ref = apply_luminance_curve_reference(inp, realistic_curve)
        lut = apply_luminance_curve(inp, realistic_curve, prebuilt_lut=lut_data)
        err_nits = abs(ref[0] - lut[0]) * _PEAK_NITS
        assert err_nits < 0.01


# --- 6. PCHIP vs LUT Accuracy ---


class TestPCHIPvsLUTAccuracy:
    def test_max_error_within_threshold(self, realistic_curve, lut_data):
        cs = lut_data["curve_start_linear"]
        cm = lut_data["curve_max_linear"]
        inputs = np.logspace(np.log10(cs), np.log10(cm), 100000)
        ref = apply_luminance_curve_reference(inputs, realistic_curve)
        lut = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        err_nits = np.abs(ref - lut) * _PEAK_NITS
        assert np.max(err_nits) <= 0.01, f"Max error {np.max(err_nits):.6f} nits > 0.01"

    def test_mean_error_within_threshold(self, realistic_curve, lut_data):
        cs = lut_data["curve_start_linear"]
        cm = lut_data["curve_max_linear"]
        inputs = np.logspace(np.log10(cs), np.log10(cm), 100000)
        ref = apply_luminance_curve_reference(inputs, realistic_curve)
        lut = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        err_nits = np.abs(ref - lut) * _PEAK_NITS
        assert np.mean(err_nits) <= 0.001, f"Mean error {np.mean(err_nits):.8f} > 0.001"

    def test_p999_error(self, realistic_curve, lut_data):
        cs = lut_data["curve_start_linear"]
        cm = lut_data["curve_max_linear"]
        inputs = np.logspace(np.log10(cs), np.log10(cm), 100000)
        ref = apply_luminance_curve_reference(inputs, realistic_curve)
        lut = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        err_nits = np.abs(ref - lut) * _PEAK_NITS
        assert np.percentile(err_nits, 99.9) <= 0.005

    def test_full_range_no_nans(self, realistic_curve, lut_data):
        inputs = np.linspace(0, 1.0, 100000)
        result = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# --- 7. Curve-Specific LUT ---


class TestCurveSpecificLUT:
    def test_different_curves_produce_different_luts(self, realistic_curve):
        lut1 = build_curve_lut(realistic_curve)
        # Modify curve slightly
        curve2 = [[p[0], p[1] * 1.05] for p in realistic_curve]
        lut2 = build_curve_lut(curve2)
        assert not np.array_equal(lut1["lut"], lut2["lut"])

    def test_same_curve_produces_same_lut(self, realistic_curve):
        lut1 = build_curve_lut(realistic_curve)
        lut2 = build_curve_lut(realistic_curve)
        np.testing.assert_array_equal(lut1["lut"], lut2["lut"])

    def test_lut_range_depends_on_curve(self, realistic_curve):
        lut1 = build_curve_lut(realistic_curve)
        # Curve with different range
        curve2 = [[p[0] + 0.1, p[1] + 0.1] for p in realistic_curve]
        lut2 = build_curve_lut(curve2)
        assert lut1["curve_start_linear"] != lut2["curve_start_linear"]


# --- 8. Production Integration ---


class TestProductionIntegration:
    def test_apply_shot_transform_uses_lut(self, realistic_curve, lut_data):
        """apply_shot_transform should accept and use prebuilt_lut."""
        frame = np.random.default_rng(0).random((10, 20, 3)) * 0.5
        transform = ShotTransform(
            shot_id=0, luminance_curve=realistic_curve,
            exposure=1.0, contrast=1.0, saturation=1.0, confidence=0.99,
        )
        # Should not raise
        result = apply_shot_transform(frame, transform, prebuilt_lut=lut_data)
        assert result.shape == frame.shape
        assert not np.any(np.isnan(result))

    def test_transform_without_prebuilt_lut(self, realistic_curve):
        """Transform should work without prebuilt_lut (builds internally)."""
        frame = np.random.default_rng(0).random((10, 20, 3)) * 0.5
        transform = ShotTransform(
            shot_id=0, luminance_curve=realistic_curve,
            exposure=1.0, contrast=1.0, saturation=1.0, confidence=0.99,
        )
        result = apply_shot_transform(frame, transform)
        assert result.shape == frame.shape

    def test_lut_vs_no_lut_same_result(self, realistic_curve, lut_data):
        """With or without prebuilt_lut, production path gives same result."""
        frame = np.random.default_rng(0).random((10, 20, 3)) * 0.5
        transform = ShotTransform(
            shot_id=0, luminance_curve=realistic_curve,
            exposure=1.0, contrast=1.0, saturation=1.0, confidence=0.99,
        )
        r1 = apply_shot_transform(frame, transform, prebuilt_lut=lut_data)
        r2 = apply_shot_transform(frame, transform)  # builds LUT internally
        np.testing.assert_array_equal(r1, r2)

    def test_apply_curve_without_prebuilt_builds_internally(self, realistic_curve, lut_data):
        """apply_luminance_curve without prebuilt_lut still gives LUT result."""
        inputs = np.linspace(0, 0.5, 10000)
        r1 = apply_luminance_curve(inputs, realistic_curve, prebuilt_lut=lut_data)
        r2 = apply_luminance_curve(inputs, realistic_curve)  # None → build internally
        np.testing.assert_array_equal(r1, r2)
