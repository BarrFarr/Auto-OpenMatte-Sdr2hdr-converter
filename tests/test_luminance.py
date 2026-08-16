"""Tests for luminance mapping estimation."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from auto_openmatte.processing.luminance import (
    apply_luminance_curve,
    estimate_luminance_curve,
)
from auto_openmatte.utils.math_utils import fit_monotonic_spline, percentile_bins


class TestPercentileBins:
    """Tests for percentile binning."""

    def test_linear_relationship(self) -> None:
        """Linear data should produce linear bins."""
        rng = np.random.default_rng(42)
        x = rng.random(10000)
        y = x * 2.0 + 0.1  # Linear: y = 2x + 0.1

        centers, medians = percentile_bins(x, y, n_bins=50)
        assert len(centers) > 30  # Most bins should have data

        # Check linearity: medians ≈ 2*centers + 0.1
        expected = centers * 2.0 + 0.1
        np.testing.assert_allclose(medians, expected, atol=0.05)

    def test_few_samples_returns_empty(self) -> None:
        """Very few samples should produce few valid bins."""
        x = np.array([0.5])
        y = np.array([0.7])
        centers, medians = percentile_bins(x, y, n_bins=256)
        # Should be empty or very few bins (< 10 samples per bin required)
        assert len(centers) == 0


class TestMonotonicSpline:
    """Tests for monotonic spline fitting."""

    def test_already_monotonic(self) -> None:
        """Already monotonic data should pass through unchanged."""
        x = np.linspace(0, 1, 20)
        y = x ** 2  # Naturally monotonic
        x_out, y_out = fit_monotonic_spline(x, y)
        np.testing.assert_allclose(y_out, y, atol=1e-10)

    def test_non_monotonic_fixed(self) -> None:
        """Non-monotonic data should be made monotonic."""
        x = np.linspace(0, 1, 10)
        y = np.array([0.0, 0.1, 0.2, 0.15, 0.3, 0.4, 0.35, 0.5, 0.6, 0.7])
        # y[3]=0.15 < y[2]=0.2  and  y[6]=0.35 < y[5]=0.4

        x_out, y_out = fit_monotonic_spline(x, y)
        # Result should be monotonically non-decreasing
        diffs = np.diff(y_out)
        assert np.all(diffs >= 0), f"Non-monotonic: {y_out}"


class TestLuminanceCurve:
    """Tests for luminance curve estimation."""

    def test_linear_mapping(self) -> None:
        """Known linear relationship should produce linear curve."""
        rng = np.random.default_rng(42)
        sdr = rng.random(50000)
        hdr = sdr * 1.5  # Simple linear scaling

        curve = estimate_luminance_curve(sdr, hdr)
        assert len(curve) >= 5

        # Apply curve — should produce values close to 1.5x
        test_input = np.linspace(0.1, 0.9, 50)
        mapped = apply_luminance_curve(test_input, curve)
        expected = test_input * 1.5
        np.testing.assert_allclose(mapped, expected, atol=0.1)

    def test_gamma_mapping(self) -> None:
        """Gamma-like relationship should be captured."""
        rng = np.random.default_rng(42)
        sdr = rng.random(50000)
        hdr = sdr ** 0.5  # Gamma curve

        curve = estimate_luminance_curve(sdr, hdr)
        assert len(curve) >= 5

        # Curve should be monotonic
        for i in range(1, len(curve)):
            assert curve[i][1] >= curve[i - 1][1], "Curve not monotonic"

    def test_too_few_samples(self) -> None:
        """Should return identity-like curve with too few samples."""
        sdr = np.array([0.5])
        hdr = np.array([0.7])
        curve = estimate_luminance_curve(sdr, hdr)
        # Should have at least 2 points and not crash
        assert len(curve) >= 2


class TestApplyLuminanceCurve:
    """Tests for curve application."""

    def test_identity_curve(self) -> None:
        """Identity curve should not change values (in log domain)."""
        # Log-domain identity: log10(x*10000+eps) → log10(x*10000+eps)
        # Build identity curve in log space
        import numpy as np
        eps = 1e-6
        peak = 10000.0
        test_x = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        log_x = np.log10(test_x * peak + eps)
        curve = [[float(lx), float(lx)] for lx in log_x]  # Identity in log
        output = apply_luminance_curve(test_x, curve)
        np.testing.assert_allclose(output, test_x, rtol=0.01)

    def test_scaling_curve(self) -> None:
        """Curve that maps to 2x luminance in log domain."""
        import numpy as np
        eps = 1e-6
        peak = 10000.0
        # Create a curve where output = 2*input in linear
        # In log: log10(2*x*peak + eps) for output
        test_x = np.array([0.01, 0.05, 0.1, 0.2, 0.4])
        log_in = np.log10(test_x * peak + eps)
        log_out = np.log10(test_x * 2.0 * peak + eps)
        curve = [[float(li), float(lo)] for li, lo in zip(log_in, log_out)]
        output = apply_luminance_curve(test_x, curve)
        expected = test_x * 2.0
        np.testing.assert_allclose(output, expected, rtol=0.02)

    def test_empty_curve(self) -> None:
        """Empty curve should return input unchanged."""
        input_vals = np.array([0.3, 0.5, 0.8])
        output = apply_luminance_curve(input_vals, [])
        np.testing.assert_array_equal(output, input_vals)
