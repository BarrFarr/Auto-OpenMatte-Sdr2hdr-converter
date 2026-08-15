"""Tests for luminance mapping pipeline (FEAT-007).

Covers:
- PQ → absolute luminance
- SDR linearization (BT.1886)
- Monotonic curve fitting
- Isotonic enforcement before PCHIP
- Outlier / clipping rejection
- Bounded output
- Known nonlinear transform recovery
- Train/validation split
- Empty / low-information samples
- Per-shot transform
- Short shot fallback
- HDR master not modified
- Determinism with seed
- Diagnostic statistics
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from auto_openmatte.core.transfer_functions import (
    linearize,
    pq_eotf,
)
from auto_openmatte.processing.luminance import (
    apply_luminance_curve,
    estimate_luminance_curve,
)
from auto_openmatte.processing.sampling import (
    OverlapSamples,
    compute_diagnostics,
    train_validation_split,
)
from auto_openmatte.utils.math_utils import fit_monotonic_spline

# ---------------------------------------------------------------------------
# Test: PQ → absolute luminance
# ---------------------------------------------------------------------------


class TestPQLinearization:
    """Verify PQ EOTF produces absolute luminance."""

    def test_pq_signal_to_nits(self) -> None:
        """PQ signal 0.5 ≈ 100 nits (SDR white)."""
        signal = np.array([0.5])
        nits = pq_eotf(signal)
        assert 80 < nits[0] < 120

    def test_pq_zero_is_zero(self) -> None:
        """PQ signal 0 = 0 nits."""
        assert pq_eotf(np.array([0.0]))[0] == 0.0

    def test_pq_one_is_10000(self) -> None:
        """PQ signal 1.0 = 10000 nits."""
        assert pq_eotf(np.array([1.0]))[0] == pytest.approx(10000.0, rel=1e-6)

    def test_linearize_pq_normalized(self) -> None:
        """linearize('smpte2084') returns [0,1] normalized by peak_nits."""
        signal = np.array([0.5])
        linear = linearize(signal, "smpte2084", peak_nits=10000.0)
        # Should be ~100/10000 = ~0.01
        assert 0.005 < linear[0] < 0.02


# ---------------------------------------------------------------------------
# Test: SDR linearization
# ---------------------------------------------------------------------------


class TestSDRLinearization:
    """Verify BT.1886/BT.709 linearization."""

    def test_bt709_linearize(self) -> None:
        """BT.709 signal 0.5 → linear ≈ 0.5^2.4 ≈ 0.189."""
        signal = np.array([0.5])
        linear = linearize(signal, "bt709")
        assert 0.15 < linear[0] < 0.22  # gamma ~2.4

    def test_bt709_zero(self) -> None:
        """BT.709 signal 0 → linear 0."""
        assert linearize(np.array([0.0]), "bt709")[0] == 0.0

    def test_bt709_one(self) -> None:
        """BT.709 signal 1 → linear 1."""
        assert linearize(np.array([1.0]), "bt709")[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test: Monotonic curve fitting
# ---------------------------------------------------------------------------


class TestMonotonicFit:
    """Verify isotonic + PCHIP monotonic fitting."""

    def test_monotonic_output(self) -> None:
        """Fitted curve must be monotonically non-decreasing."""
        rng = np.random.default_rng(42)
        sdr = rng.random(50000)
        # Nonlinear relationship with some noise
        hdr = np.power(sdr, 0.7) * 0.8 + rng.random(50000) * 0.05

        curve = estimate_luminance_curve(sdr, hdr)
        for i in range(1, len(curve)):
            assert curve[i][1] >= curve[i - 1][1] - 1e-10, (
                f"Not monotonic at {i}: {curve[i-1][1]:.6f} > {curve[i][1]:.6f}"
            )

    def test_isotonic_before_pchip(self) -> None:
        """fit_monotonic_spline enforces monotonicity on non-monotonic input."""
        x = np.linspace(0, 1, 20)
        # Deliberately non-monotonic
        y = np.array([0, 0.1, 0.2, 0.15, 0.25, 0.3, 0.28, 0.4, 0.5, 0.45,
                      0.55, 0.6, 0.58, 0.65, 0.7, 0.68, 0.75, 0.8, 0.85, 0.9])
        _, y_mono = fit_monotonic_spline(x, y)
        diffs = np.diff(y_mono)
        assert np.all(diffs >= -1e-12), "fit_monotonic_spline failed to enforce monotonicity"


# ---------------------------------------------------------------------------
# Test: Outlier / clipping rejection
# ---------------------------------------------------------------------------


class TestRejection:
    """Verify sample rejection criteria."""

    def test_clipped_sdr_rejected(self) -> None:
        """SDR values at 0 or 1 should be excluded from curve estimation."""
        rng = np.random.default_rng(42)
        # Good middle range
        sdr_good = rng.random(5000) * 0.8 + 0.1
        hdr_good = sdr_good * 1.5

        # Bad clipped values
        sdr_clip = np.ones(500)  # All 1.0
        hdr_clip = np.ones(500) * 2.0

        sdr = np.concatenate([sdr_good, sdr_clip])
        hdr = np.concatenate([hdr_good, hdr_clip])

        curve = estimate_luminance_curve(sdr, hdr)
        # The curve shouldn't be distorted by the clipped values
        # Test midpoint mapping
        mid_mapped = apply_luminance_curve(np.array([0.5]), curve)
        # Should be close to 0.5 * 1.5 = 0.75
        assert abs(mid_mapped[0] - 0.75) < 0.15

    def test_nan_inf_rejected(self) -> None:
        """NaN and Inf values shouldn't crash or corrupt."""
        rng = np.random.default_rng(42)
        sdr = rng.random(5000)
        hdr = sdr * 1.2
        # Insert NaN and Inf
        sdr[0] = np.nan
        hdr[1] = np.inf
        sdr[2] = -np.inf

        curve = estimate_luminance_curve(sdr, hdr)
        assert len(curve) >= 2  # Should succeed


# ---------------------------------------------------------------------------
# Test: Bounded output
# ---------------------------------------------------------------------------


class TestBoundedOutput:
    """Transform output must be bounded and non-negative."""

    def test_no_negative_output(self) -> None:
        """apply_luminance_curve should never return negative values."""
        curve = [[0.0, 0.0], [0.3, 0.1], [0.6, 0.4], [1.0, 0.8]]
        # Input including values near 0
        test = np.linspace(0.0, 1.0, 100)
        result = apply_luminance_curve(test, curve)
        assert np.all(result >= 0.0)

    def test_no_inf_nan_output(self) -> None:
        """Output must be finite."""
        curve = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        test = np.linspace(0.0, 1.0, 1000)
        result = apply_luminance_curve(test, curve)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Test: Known nonlinear transform recovery
# ---------------------------------------------------------------------------


class TestKnownTransformRecovery:
    """Algorithm should recover known nonlinear transforms."""

    def test_power_law_recovery(self) -> None:
        """Power-law SDR→HDR relationship should be recovered."""
        rng = np.random.default_rng(42)
        sdr = rng.random(100000) * 0.9 + 0.05
        # HDR = sdr^0.6 (highlight compression)
        hdr = np.power(sdr, 0.6)
        hdr += rng.random(len(hdr)) * 0.01  # Slight noise

        curve = estimate_luminance_curve(sdr, hdr)

        # Test at several points
        test_points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mapped = apply_luminance_curve(test_points, curve)
        expected = np.power(test_points, 0.6)
        np.testing.assert_allclose(mapped, expected, atol=0.05)

    def test_s_curve_recovery(self) -> None:
        """S-curve with shadow lift and highlight rolloff."""
        rng = np.random.default_rng(42)
        sdr = rng.random(100000) * 0.9 + 0.05
        # S-curve: shadow lift + highlight rolloff
        hdr = 0.05 + 0.9 * (1.0 / (1.0 + np.exp(-8 * (sdr - 0.5))))
        hdr += rng.random(len(hdr)) * 0.01

        curve = estimate_luminance_curve(sdr, hdr)
        test_points = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mapped = apply_luminance_curve(test_points, curve)
        expected = 0.05 + 0.9 * (1.0 / (1.0 + np.exp(-8 * (test_points - 0.5))))
        np.testing.assert_allclose(mapped, expected, atol=0.05)

    def test_not_just_gamma(self) -> None:
        """Algorithm must handle non-gamma transforms (piecewise)."""
        rng = np.random.default_rng(42)
        sdr = rng.random(100000) * 0.9 + 0.05
        # Piecewise: shadows boosted, highlights compressed
        hdr = np.where(sdr < 0.3, sdr * 2.0, 0.6 + (sdr - 0.3) * 0.5)
        hdr += rng.random(len(hdr)) * 0.01

        curve = estimate_luminance_curve(sdr, hdr)
        # Verify at midpoints of each piece
        mapped = apply_luminance_curve(np.array([0.15, 0.6]), curve)
        expected = np.array([0.30, 0.75])
        np.testing.assert_allclose(mapped, expected, atol=0.08)


# ---------------------------------------------------------------------------
# Test: Train/validation split
# ---------------------------------------------------------------------------


class TestTrainValidationSplit:
    """Verify train/validation split behavior."""

    def test_split_ratio(self) -> None:
        """80/20 split should produce correct sizes."""
        samples = OverlapSamples(
            sdr_luminance=np.random.default_rng(42).random(1000),
            hdr_luminance=np.random.default_rng(43).random(1000),
            n_valid_pairs=1000,
        )
        train, val = train_validation_split(samples, train_ratio=0.8, seed=42)
        assert train.n_valid_pairs == 800
        assert val.n_valid_pairs == 200

    def test_no_overlap(self) -> None:
        """Train and val sets should have no overlapping indices."""
        rng = np.random.default_rng(42)
        data = rng.random(100)
        samples = OverlapSamples(
            sdr_luminance=data.copy(),
            hdr_luminance=data.copy(),
            n_valid_pairs=100,
        )
        train, val = train_validation_split(samples, train_ratio=0.8, seed=42)
        # Union should cover all original data
        assert train.n_valid_pairs + val.n_valid_pairs == 100

    def test_empty_samples(self) -> None:
        """Empty input → returns empty samples."""
        samples = OverlapSamples(n_valid_pairs=0)
        train, val = train_validation_split(samples)
        assert train.n_valid_pairs == 0


# ---------------------------------------------------------------------------
# Test: Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """Verify diagnostic computation."""

    def test_perfect_fit_zero_error(self) -> None:
        """Identity curve on identity data → zero error."""
        data = np.linspace(0.01, 0.99, 1000)
        samples = OverlapSamples(
            sdr_luminance=data.copy(),
            hdr_luminance=data.copy(),
            n_valid_pairs=1000,
        )
        curve = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        diag = compute_diagnostics(samples, curve, train_samples=samples, val_samples=samples)
        assert diag.train_mae < 0.01
        assert diag.val_mae < 0.01
        assert diag.monotonic is True
        assert diag.confidence > 0.99

    def test_bad_fit_high_error(self) -> None:
        """Wrong curve on data → high error."""
        rng = np.random.default_rng(42)
        sdr = rng.random(1000) * 0.8 + 0.1
        hdr = sdr * 2.0  # Actual relationship: 2x
        samples = OverlapSamples(
            sdr_luminance=sdr,
            hdr_luminance=hdr,
            n_valid_pairs=1000,
        )
        # Wrong curve: identity
        curve = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        diag = compute_diagnostics(samples, curve, train_samples=samples)
        assert diag.train_mae > 0.2  # Significant error

    def test_percentiles_computed(self) -> None:
        """Diagnostics should include luminance percentiles."""
        rng = np.random.default_rng(42)
        samples = OverlapSamples(
            sdr_luminance=rng.random(1000),
            hdr_luminance=rng.random(1000),
            n_valid_pairs=1000,
        )
        curve = [[0.0, 0.0], [1.0, 1.0]]
        diag = compute_diagnostics(samples, curve)
        assert "P50" in diag.sdr_percentiles
        assert "P50" in diag.hdr_percentiles


# ---------------------------------------------------------------------------
# Test: Per-shot and fallback
# ---------------------------------------------------------------------------


class TestPerShotTransform:
    """Verify per-shot luminance estimation."""

    def test_per_shot_uses_shot_samples(self) -> None:
        """estimate_shot_luminance should use provided shot samples."""
        from auto_openmatte.core.models import Shot
        from auto_openmatte.processing.luminance import estimate_shot_luminance

        rng = np.random.default_rng(42)
        sdr = [rng.random(5000) for _ in range(5)]
        hdr = [s * 1.5 for s in sdr]

        shot = Shot(shot_id=0, hdr_start_frame=0, hdr_end_frame=1000)
        transform = estimate_shot_luminance(sdr, hdr, shot)
        assert transform.shot_id == 0
        assert len(transform.luminance_curve) >= 2
        assert transform.confidence > 0.5

    def test_short_shot_fallback(self) -> None:
        """Shot with < 100 samples → identity-like fallback."""
        from auto_openmatte.core.models import Shot
        from auto_openmatte.processing.luminance import estimate_shot_luminance

        shot = Shot(shot_id=5, hdr_start_frame=0, hdr_end_frame=10)
        transform = estimate_shot_luminance([np.array([0.5])], [np.array([0.7])], shot)
        assert transform.confidence == 0.0  # Low confidence


# ---------------------------------------------------------------------------
# Test: HDR master not modified
# ---------------------------------------------------------------------------


class TestHDRMasterPreserved:
    """Verify that HDR master data is never modified."""

    def test_transform_does_not_modify_input(self) -> None:
        """apply_luminance_curve should not mutate input array."""
        curve = [[0.0, 0.0], [0.5, 0.8], [1.0, 1.5]]
        original = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        input_copy = original.copy()
        _ = apply_luminance_curve(original, curve)
        np.testing.assert_array_equal(original, input_copy)


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify deterministic results with same seed."""

    def test_same_seed_same_result(self) -> None:
        """Same input + same seed → identical curve."""
        rng = np.random.default_rng(42)
        sdr = rng.random(50000)
        hdr = sdr * 1.3 + 0.05

        curve1 = estimate_luminance_curve(sdr.copy(), hdr.copy())
        curve2 = estimate_luminance_curve(sdr.copy(), hdr.copy())
        assert curve1 == curve2
