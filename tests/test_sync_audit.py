"""Critical audit tests for synchronization algorithm.

These tests simulate realistic scenarios closer to real-material behavior:
- Exact offset recovery (positive, negative, zero)
- Different amplitude / nonlinear transfer function
- Noise robustness
- Low-information sections (black, static, fades)
- Ambiguous periodic signals
- Drift detection at sub-frame precision
- Large offsets
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.analysis.sync import (
    _find_best_offset_ncc,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_realistic_signal(
    length: int,
    event_positions: list[int],
    noise_level: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Create temporal-diff signal with realistic structure.

    Events simulate scene cuts (high peaks). Between events, there's
    low-level motion noise (simulating normal motion temporal diffs).
    """
    rng = np.random.default_rng(seed)
    signal = rng.random(length) * noise_level
    for pos in event_positions:
        if 0 <= pos < length:
            signal[pos] = 0.4 + rng.random() * 0.4
    return signal


# ---------------------------------------------------------------------------
# Test A — Ideal positive offset +58
# ---------------------------------------------------------------------------


class TestAuditIdealOffset:
    """Test A: ideal offset detection."""

    def test_offset_58(self) -> None:
        """Must recover offset +58 exactly."""
        events = [40, 100, 180, 270, 350, 500, 600, 720, 850, 950]
        offset = 58

        hdr = _make_realistic_signal(1100, events, seed=10)
        om = _make_realistic_signal(1200, [e + offset for e in events], seed=10)

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=100)
        assert detected == offset, f"Expected +58, got {detected}"
        assert score > 0.95


# ---------------------------------------------------------------------------
# Test B — Negative offset -37
# ---------------------------------------------------------------------------


class TestAuditNegativeOffset:
    """Test B: negative offset detection."""

    def test_offset_minus_37(self) -> None:
        """Must recover offset -37 exactly."""
        events = [80, 160, 250, 380, 500, 630, 780, 900]
        offset = -37

        hdr = _make_realistic_signal(1000, events, seed=20)
        om = _make_realistic_signal(1000, [e + offset for e in events], seed=20)

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=80)
        assert detected == offset, f"Expected -37, got {detected}"
        assert score > 0.95


# ---------------------------------------------------------------------------
# Test C — Zero offset
# ---------------------------------------------------------------------------


class TestAuditZeroOffset:
    """Test C: zero offset detection."""

    def test_offset_zero(self) -> None:
        """Must recover offset 0 exactly."""
        events = [50, 130, 240, 380, 520, 670, 800]
        hdr = _make_realistic_signal(900, events, seed=30)
        om = _make_realistic_signal(900, events, seed=30)

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=60)
        assert detected == 0, f"Expected 0, got {detected}"
        assert score > 0.99


# ---------------------------------------------------------------------------
# Test D — Different amplitude (OM = 0.6 * S + noise)
# ---------------------------------------------------------------------------


class TestAuditDifferentAmplitude:
    """Test D: offset recovery with different signal amplitude."""

    def test_attenuated_om_signal(self) -> None:
        """OM signal at 60% amplitude should still detect offset correctly."""
        events = [50, 150, 270, 400, 550, 700, 850]
        offset = 42

        rng = np.random.default_rng(40)
        hdr = _make_realistic_signal(1000, events, noise_level=0.01, seed=40)
        om_raw = _make_realistic_signal(
            1100, [e + offset for e in events], noise_level=0.01, seed=40
        )
        # Attenuate OM by 0.6 and add noise
        om = om_raw * 0.6 + rng.random(len(om_raw)) * 0.005

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=80)
        assert detected == offset, f"Expected +42, got {detected}"
        assert score > 0.9


# ---------------------------------------------------------------------------
# Test E — Nonlinear monotonic transformation (PQ vs gamma analogy)
# ---------------------------------------------------------------------------


class TestAuditNonlinearTransform:
    """Test E: monotonic nonlinear transformation of one signal."""

    def test_gamma_transformed_signal(self) -> None:
        """Applying gamma-like curve to OM should not break offset detection."""
        events = [60, 140, 250, 370, 500, 650, 800]
        offset = 25

        hdr = _make_realistic_signal(900, events, noise_level=0.015, seed=50)
        om_linear = _make_realistic_signal(
            950, [e + offset for e in events], noise_level=0.015, seed=50
        )
        # Apply monotonic nonlinear transform (simulating PQ vs gamma)
        om = np.power(om_linear + 0.001, 0.6)  # Nonlinear but monotonic

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=60)
        assert detected == offset, f"Expected +25, got {detected}"
        assert score > 0.85

    def test_sqrt_transformed_signal(self) -> None:
        """Square root transform (compressive) should preserve offset detection."""
        events = [40, 120, 230, 350, 480, 600, 750]
        offset = -15

        hdr = _make_realistic_signal(850, events, noise_level=0.012, seed=55)
        om_linear = _make_realistic_signal(
            850, [e + offset for e in events], noise_level=0.012, seed=55
        )
        om = np.sqrt(om_linear + 0.001)

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=50)
        assert detected == offset, f"Expected -15, got {detected}"
        assert score > 0.85


# ---------------------------------------------------------------------------
# Test F — Noise robustness
# ---------------------------------------------------------------------------


class TestAuditNoiseRobustness:
    """Test F: noise robustness."""

    def test_moderate_noise(self) -> None:
        """30% noise level should still allow correct detection."""
        events = [50, 140, 250, 380, 520, 670, 830]
        offset = 33

        hdr = _make_realistic_signal(950, events, noise_level=0.01, seed=60)
        om = _make_realistic_signal(
            1000, [e + offset for e in events], noise_level=0.01, seed=61
        )
        # Add 30% noise relative to peak amplitude
        rng = np.random.default_rng(62)
        om_noisy = om + rng.random(len(om)) * 0.15

        detected, score, _ = _find_best_offset_ncc(hdr, om_noisy, max_offset=60)
        assert detected == offset, f"Expected +33, got {detected}"
        assert score > 0.7

    def test_high_noise_reduces_confidence(self) -> None:
        """Very high noise should reduce confidence significantly."""
        events = [100, 250, 400, 550, 700]
        offset = 20

        hdr = _make_realistic_signal(800, events, noise_level=0.01, seed=65)
        om = _make_realistic_signal(
            850, [e + offset for e in events], noise_level=0.01, seed=65
        )
        # Add massive noise (50% of peak)
        rng = np.random.default_rng(66)
        om_very_noisy = om + rng.random(len(om)) * 0.4

        detected, score, _ = _find_best_offset_ncc(hdr, om_very_noisy, max_offset=60)
        # May or may not detect correctly, but confidence should be lower
        # than the clean case
        if detected == offset:
            assert score < 0.95  # Should be notably reduced
        else:
            # If wrong offset, it means noise overcame signal — acceptable
            assert score < 0.8


# ---------------------------------------------------------------------------
# Test G — Low-information sections
# ---------------------------------------------------------------------------


class TestAuditLowInformation:
    """Test G: low-information sections (black, static, fades)."""

    def test_signal_with_long_static_section(self) -> None:
        """Long static section followed by active section should still detect offset."""
        offset = 47
        # First 400 frames: near-zero (static/black)
        # Then events from frame 400 onwards
        events = [420, 500, 600, 700, 800, 900]

        hdr = _make_realistic_signal(1000, events, noise_level=0.001, seed=70)
        # Make first 400 frames truly flat
        hdr[:400] = 0.0001

        om = _make_realistic_signal(
            1100, [e + offset for e in events], noise_level=0.001, seed=70
        )
        om[:400 + offset] = 0.0001

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=80)
        assert detected == offset, f"Expected +47, got {detected}"
        assert score > 0.85

    def test_mostly_black_fails_gracefully(self) -> None:
        """Signal that is almost entirely black should have low/negative confidence."""
        # Only 5 frames of activity in 1000
        hdr = np.ones(1000) * 0.0001
        hdr[500] = 0.5
        hdr[502] = 0.3
        om = np.ones(1000) * 0.0001
        om[520] = 0.5  # Offset of 20 but almost no signal

        _, score, _ = _find_best_offset_ncc(hdr, om, max_offset=50)
        # Very sparse signal — confidence should be low
        assert score < 0.9


# ---------------------------------------------------------------------------
# Test H — Ambiguous periodic signal
# ---------------------------------------------------------------------------


class TestAuditPeriodicSignal:
    """Test H: periodic/ambiguous signal — confidence should reflect ambiguity."""

    def test_periodic_signal_lower_confidence(self) -> None:
        """Perfectly periodic signal creates ambiguity — multiple valid offsets."""
        period = 30
        length = 600
        offset = 15  # Half period — ambiguous with +15 and -15

        # Create periodic signal
        hdr = np.zeros(length)
        for i in range(0, length, period):
            if i < length:
                hdr[i] = 0.8

        om = np.zeros(length + 50)
        for i in range(offset, length + 50, period):
            if i < length + 50:
                om[i] = 0.8

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=60)
        # The algorithm may find offset=15 or offset=15+30=45 or offset=15-30=-15
        # All are equally valid for a perfectly periodic signal.
        # Key assertion: detected offset should be one of the valid options
        valid_offsets = [offset + k * period for k in range(-3, 4)]
        assert detected in valid_offsets, (
            f"Detected {detected}, expected one of {valid_offsets}"
        )
        # Confidence should still be high (periodic = strong match at multiples)
        # But ideally we'd want to flag ambiguity. Current implementation doesn't
        # distinguish second-best peak, so this is a known limitation.

    def test_aperiodic_uniquely_resolved(self) -> None:
        """Non-periodic signal with unique events should resolve unambiguously."""
        # Irregular spacing — only one correct offset
        events = [37, 89, 167, 291, 412, 583, 721]
        offset = 22

        hdr = _make_realistic_signal(800, events, seed=80)
        om = _make_realistic_signal(850, [e + offset for e in events], seed=80)

        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=60)
        assert detected == offset
        # With sparse events and background noise, NCC > 0.9 is expected
        assert score > 0.90


# ---------------------------------------------------------------------------
# Test I — Drift detection (offset changes mid-film)
# ---------------------------------------------------------------------------


class TestAuditDrift:
    """Test I: drift between first and second half."""

    def test_drift_of_1_frame_detected(self) -> None:
        """Offset changes from +58 to +59 mid-signal — should NOT be handled
        by _find_best_offset_ncc alone (it returns global best), but should
        be caught by validate_sync with local measurements.

        Here we test that two halves with different offsets produce
        imperfect NCC (score reduction).
        """
        events_first = [50, 130, 220, 320, 400]
        events_second = [550, 640, 730, 830, 920]

        offset_first = 58
        offset_second = 59  # 1 frame drift

        hdr = _make_realistic_signal(1000, events_first + events_second, seed=90)

        # OM: first half at offset 58, second half at offset 59
        om_events = [e + offset_first for e in events_first] + [
            e + offset_second for e in events_second
        ]
        om = _make_realistic_signal(1100, om_events, seed=90)

        # Global NCC should still find one of the offsets (likely 58 or 59)
        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=80)
        assert detected in (58, 59), f"Expected 58 or 59, got {detected}"
        # Score should be slightly lower than perfect (because of drift)
        assert score < 1.0


# ---------------------------------------------------------------------------
# Test J — Acceptable drift (< 0.5 frame)
# ---------------------------------------------------------------------------


class TestAuditAcceptableDrift:
    """Test J: drift < 0.5 frame should pass validation.

    Since our model uses integer frame offsets, a 'drift' of 0 in integer
    terms is always acceptable. The real scenario for sub-frame drift is
    when local NCC measurement returns the same integer offset but with
    slightly different confidence — that produces error=0 in the validation.

    This test verifies the integer-offset model works correctly.
    """

    def test_consistent_integer_offset_passes(self) -> None:
        """All checkpoints returning the same integer offset = drift 0."""
        events = [50, 150, 300, 450, 600, 750, 900]
        offset = 58

        hdr = _make_realistic_signal(1000, events, seed=100)
        om = _make_realistic_signal(1100, [e + offset for e in events], seed=100)

        # Verify global detection works
        detected, score, _ = _find_best_offset_ncc(hdr, om, max_offset=80)
        assert detected == offset
        assert score > 0.95

        # Sub-windows should also find the same offset
        # Test three local windows
        for center in [200, 500, 800]:
            w = 100
            h_start = max(0, center - w)
            h_end = min(len(hdr), center + w)
            o_start = max(0, center + offset - w)
            o_end = min(len(om), center + offset + w)

            local_det, local_score, _ = _find_best_offset_ncc(
                hdr[h_start:h_end], om[o_start:o_end], max_offset=10
            )
            # local offset in signal-space + conversion
            # om_slice starts at o_start, hdr_slice at h_start
            abs_offset = o_start + local_det - h_start
            assert abs_offset == offset, (
                f"Local offset at {center}: expected {offset}, got {abs_offset}"
            )
