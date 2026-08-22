"""Tests for transfer function math (PQ, HLG, BT.1886)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.core.transfer_functions import (
    bt1886_eotf,
    bt1886_oetf,
    delinearize,
    hlg_eotf_inv,
    hlg_oetf,
    linearize,
    pq_eotf,
    pq_oetf,
)


class TestPQ:
    """Tests for PQ (ST 2084) transfer functions."""

    def test_pq_black(self) -> None:
        """PQ signal 0 should map to 0 luminance."""
        result = pq_eotf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_pq_white(self) -> None:
        """PQ signal 1 should map to 10000 cd/m²."""
        result = pq_eotf(np.array([1.0]))
        assert result[0] == pytest.approx(10000.0, rel=1e-4)

    def test_pq_midpoint(self) -> None:
        """PQ signal ~0.508 corresponds to ~100 cd/m² (SDR reference white)."""
        # PQ value for 100 nits is approximately 0.508
        result = pq_eotf(np.array([0.508]))
        assert 90.0 < result[0] < 110.0  # ~100 nits

    def test_pq_round_trip(self) -> None:
        """EOTF followed by OETF should be identity."""
        signal = np.linspace(0.0, 1.0, 100)
        luminance = pq_eotf(signal)
        reconstructed = pq_oetf(luminance)
        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)

    def test_pq_monotonic(self) -> None:
        """PQ EOTF should be strictly monotonically increasing."""
        signal = np.linspace(0.0, 1.0, 1000)
        luminance = pq_eotf(signal)
        diffs = np.diff(luminance)
        assert np.all(diffs >= 0)


class TestHLG:
    """Tests for HLG (ARIB STD-B67) transfer functions."""

    def test_hlg_black(self) -> None:
        """HLG: linear 0 → signal 0."""
        result = hlg_oetf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_hlg_white(self) -> None:
        """HLG: linear 1 → signal 1."""
        result = hlg_oetf(np.array([1.0]))
        assert result[0] == pytest.approx(1.0, abs=1e-4)

    def test_hlg_round_trip(self) -> None:
        """OETF followed by inverse should be identity."""
        linear = np.linspace(0.0, 1.0, 100)
        signal = hlg_oetf(linear)
        reconstructed = hlg_eotf_inv(signal)
        np.testing.assert_allclose(reconstructed, linear, atol=1e-5)

    def test_hlg_monotonic(self) -> None:
        """HLG OETF should be monotonically increasing."""
        linear = np.linspace(0.0, 1.0, 1000)
        signal = hlg_oetf(linear)
        diffs = np.diff(signal)
        assert np.all(diffs >= 0)


class TestBT1886:
    """Tests for BT.1886 transfer functions."""

    def test_bt1886_black(self) -> None:
        """BT.1886: signal 0 → linear 0."""
        result = bt1886_eotf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-10)

    def test_bt1886_white(self) -> None:
        """BT.1886: signal 1 → linear 1."""
        result = bt1886_eotf(np.array([1.0]))
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_bt1886_round_trip(self) -> None:
        """EOTF followed by OETF should be identity."""
        signal = np.linspace(0.0, 1.0, 100)
        linear = bt1886_eotf(signal)
        reconstructed = bt1886_oetf(linear)
        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)


class TestLinearize:
    """Tests for the generic linearize/delinearize interface."""

    def test_linearize_pq(self) -> None:
        """linearize with PQ should produce correct output."""
        signal = np.array([0.0, 0.5, 1.0])
        result = linearize(signal, "smpte2084")
        # Normalized by peak nits, so max should be ~1.0
        assert result[-1] == pytest.approx(1.0, rel=0.01)

    def test_linearize_bt709(self) -> None:
        """linearize with BT.709 should apply gamma."""
        signal = np.array([0.5])
        result = linearize(signal, "bt709")
        # 0.5^2.4 ≈ 0.189
        assert 0.15 < result[0] < 0.22

    def test_round_trip_pq(self) -> None:
        """linearize then delinearize should be identity."""
        signal = np.linspace(0.1, 0.9, 50)
        linear = linearize(signal, "smpte2084")
        reconstructed = delinearize(linear, "smpte2084")
        np.testing.assert_allclose(reconstructed, signal, atol=1e-5)

    def test_round_trip_bt709(self) -> None:
        """Round trip for BT.709."""
        signal = np.linspace(0.01, 1.0, 50)
        linear = linearize(signal, "bt709")
        reconstructed = delinearize(linear, "bt709")
        np.testing.assert_allclose(reconstructed, signal, atol=1e-5)
