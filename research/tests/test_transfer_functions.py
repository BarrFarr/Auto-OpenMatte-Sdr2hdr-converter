"""Tests for transfer function implementations.

Verifies PQ, HLG, and BT.1886 round-trip accuracy and known values.
"""

from __future__ import annotations

import numpy as np
import pytest

from reshaping_research.utils.transfer_functions import (
    pq_eotf,
    pq_oetf,
    hlg_oetf,
    hlg_eotf_inv,
    bt1886_eotf,
    bt1886_oetf,
    linearize,
    delinearize,
    PQ_PEAK_LUMINANCE,
)


class TestPQ:
    """Tests for PQ (ST 2084) transfer functions."""

    def test_pq_eotf_zero(self) -> None:
        """Signal 0 maps to 0 nits."""
        result = pq_eotf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_pq_eotf_one(self) -> None:
        """Signal 1 maps to 10000 nits."""
        result = pq_eotf(np.array([1.0]))
        assert result[0] == pytest.approx(PQ_PEAK_LUMINANCE, rel=1e-4)

    def test_pq_eotf_mid(self) -> None:
        """Signal 0.5 maps to approximately 100 nits (known PQ value)."""
        result = pq_eotf(np.array([0.5]))
        # PQ signal 0.5 ~= 100.7 nits
        assert 90.0 < result[0] < 120.0

    def test_pq_roundtrip(self) -> None:
        """PQ EOTF -> OETF round trip preserves values."""
        signal = np.linspace(0.0, 1.0, 100)
        recovered = pq_oetf(pq_eotf(signal))
        np.testing.assert_allclose(recovered, signal, atol=1e-6)

    def test_pq_oetf_roundtrip(self) -> None:
        """PQ OETF -> EOTF round trip preserves luminance."""
        luminance = np.linspace(0.0, 10000.0, 100)
        recovered = pq_eotf(pq_oetf(luminance))
        np.testing.assert_allclose(recovered, luminance, atol=0.01)

    def test_pq_monotonic(self) -> None:
        """PQ EOTF is monotonically increasing."""
        signal = np.linspace(0.0, 1.0, 1000)
        result = pq_eotf(signal)
        assert np.all(np.diff(result) >= 0)

    def test_pq_clipping(self) -> None:
        """Values outside [0, 1] are clipped."""
        signal = np.array([-0.1, 1.1])
        result = pq_eotf(signal)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] == pytest.approx(PQ_PEAK_LUMINANCE, rel=1e-4)


class TestHLG:
    """Tests for HLG (ARIB STD-B67) transfer functions."""

    def test_hlg_oetf_zero(self) -> None:
        """Linear 0 maps to signal 0."""
        result = hlg_oetf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_hlg_oetf_one(self) -> None:
        """Linear 1 maps to signal 1."""
        result = hlg_oetf(np.array([1.0]))
        assert result[0] == pytest.approx(1.0, abs=1e-4)

    def test_hlg_roundtrip(self) -> None:
        """HLG OETF -> inverse round trip preserves values."""
        linear = np.linspace(0.0, 1.0, 100)
        recovered = hlg_eotf_inv(hlg_oetf(linear))
        np.testing.assert_allclose(recovered, linear, atol=1e-6)

    def test_hlg_monotonic(self) -> None:
        """HLG OETF is monotonically increasing."""
        linear = np.linspace(0.0, 1.0, 1000)
        result = hlg_oetf(linear)
        assert np.all(np.diff(result) >= 0)


class TestBT1886:
    """Tests for BT.1886 transfer functions."""

    def test_bt1886_eotf_zero(self) -> None:
        """Signal 0 maps to 0."""
        result = bt1886_eotf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_bt1886_eotf_one(self) -> None:
        """Signal 1 maps to 1."""
        result = bt1886_eotf(np.array([1.0]))
        assert result[0] == pytest.approx(1.0, abs=1e-6)

    def test_bt1886_gamma(self) -> None:
        """BT.1886 applies gamma 2.4."""
        signal = np.array([0.5])
        result = bt1886_eotf(signal)
        expected = 0.5**2.4
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_bt1886_roundtrip(self) -> None:
        """BT.1886 EOTF -> OETF round trip preserves values."""
        signal = np.linspace(0.0, 1.0, 100)
        recovered = bt1886_oetf(bt1886_eotf(signal))
        np.testing.assert_allclose(recovered, signal, atol=1e-6)


class TestLinearizeDelinearize:
    """Tests for the linearize/delinearize utility functions."""

    def test_linearize_pq(self) -> None:
        """PQ linearize produces normalized [0, 1] output."""
        signal = np.linspace(0.0, 1.0, 50)
        result = linearize(signal, "pq")
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)

    def test_linearize_bt1886(self) -> None:
        """BT.1886 linearize produces [0, 1] output."""
        signal = np.linspace(0.0, 1.0, 50)
        result = linearize(signal, "bt1886")
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0 + 1e-6)

    def test_roundtrip_pq(self) -> None:
        """linearize -> delinearize round trip for PQ."""
        signal = np.linspace(0.01, 0.99, 50)
        linear = linearize(signal, "pq")
        recovered = delinearize(linear, "pq")
        np.testing.assert_allclose(recovered, signal, atol=1e-5)

    def test_roundtrip_bt1886(self) -> None:
        """linearize -> delinearize round trip for BT.1886."""
        signal = np.linspace(0.01, 0.99, 50)
        linear = linearize(signal, "bt1886")
        recovered = delinearize(linear, "bt1886")
        np.testing.assert_allclose(recovered, signal, atol=1e-6)

    def test_roundtrip_hlg(self) -> None:
        """linearize -> delinearize round trip for HLG."""
        signal = np.linspace(0.01, 0.99, 50)
        linear = linearize(signal, "hlg")
        recovered = delinearize(linear, "hlg")
        np.testing.assert_allclose(recovered, signal, atol=1e-6)
