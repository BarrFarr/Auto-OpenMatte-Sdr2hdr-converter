"""P2.17: PQ OETF sqrt-LUT integration tests."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import auto_openmatte.core.transfer_functions as transfer_functions
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import (
    _PQ_OETF_LUT_SIZE,
    _get_pq_oetf_sqrt_lut,
    delinearize,
    pq_eotf,
    pq_oetf,
    pq_oetf_sqrt_lut,
)
from auto_openmatte.processing.transform import apply_shot_transform


def _nits_error(reference_signal: np.ndarray, candidate_signal: np.ndarray) -> np.ndarray:
    return np.abs(pq_eotf(reference_signal) - pq_eotf(candidate_signal))


class TestP217LUTConstruction:
    def test_lut_has_65536_float64_entries(self) -> None:
        lut = _get_pq_oetf_sqrt_lut()
        assert len(lut) == 65536
        assert len(lut) == _PQ_OETF_LUT_SIZE
        assert lut.dtype == np.float64
        assert lut.nbytes == 65536 * 8

    def test_lut_is_cached_singleton(self) -> None:
        _get_pq_oetf_sqrt_lut.cache_clear()
        first = _get_pq_oetf_sqrt_lut()
        second = _get_pq_oetf_sqrt_lut()
        assert first is second
        assert _get_pq_oetf_sqrt_lut.cache_info().misses == 1
        assert _get_pq_oetf_sqrt_lut.cache_info().hits >= 1

    def test_lut_is_monotonic(self) -> None:
        lut = _get_pq_oetf_sqrt_lut()
        assert np.all(np.diff(lut) >= 0.0)

    def test_lut_boundaries(self) -> None:
        values = pq_oetf_sqrt_lut(np.array([0.0, 10000.0]))
        reference = pq_oetf(np.array([0.0, 10000.0]))
        np.testing.assert_array_equal(values, reference)


class TestP217NumericalAccuracy:
    def test_dense_million_point_grid(self) -> None:
        normalized = np.linspace(0.0, 1.0, 1_000_001)
        luminance = normalized * 10000.0
        reference = pq_oetf(luminance)
        candidate = pq_oetf_sqrt_lut(luminance)
        pq_error = np.abs(reference - candidate)
        nits_error = _nits_error(reference, candidate)

        assert np.max(pq_error) <= 1.0e-7
        assert np.mean(pq_error) <= 1.0e-8
        assert np.sqrt(np.mean(pq_error**2)) <= 2.0e-8
        assert np.max(nits_error) <= 1.0e-4
        assert np.percentile(nits_error, 99.9) <= 1.0e-4
        assert np.percentile(nits_error, 99.99) <= 1.0e-4

    @pytest.mark.parametrize(
        "lower,upper",
        [(0.0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0)],
    )
    def test_dense_range(self, lower: float, upper: float) -> None:
        normalized = np.linspace(lower, upper, 250_001)
        luminance = normalized * 10000.0
        reference = pq_oetf(luminance)
        candidate = pq_oetf_sqrt_lut(luminance)
        nits_error = _nits_error(reference, candidate)
        assert np.max(nits_error) <= 1.0e-4

    def test_required_boundary_values(self) -> None:
        normalized = np.array(
            [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4,
             1e-3, 0.01, 0.1, 0.5, 1.0],
            dtype=np.float64,
        )
        luminance = normalized * 10000.0
        reference = pq_oetf(luminance)
        candidate = pq_oetf_sqrt_lut(luminance)
        nits_error = _nits_error(reference, candidate)
        assert np.max(nits_error) <= 1.0e-4
        assert np.all(np.diff(candidate) >= 0.0)

    def test_clipping_matches_reference(self) -> None:
        luminance = np.array([-100.0, 0.0, 10000.0, 11000.0])
        np.testing.assert_allclose(
            pq_oetf_sqrt_lut(luminance), pq_oetf(luminance), atol=1.0e-7
        )

    def test_low_end_has_no_warning_or_discontinuity(self) -> None:
        luminance = np.linspace(0.0, 100.0, 100_001)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = pq_oetf_sqrt_lut(luminance)
        assert not caught
        assert np.all(np.diff(output) >= 0.0)
        reference = pq_oetf(luminance)
        assert np.max(np.abs(output - reference)) <= 1.0e-6
        assert np.count_nonzero(np.diff(output) == 0.0) == 0


class TestP217ProductionPath:
    def test_delinearize_uses_lut_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0
        original = transfer_functions.pq_oetf_sqrt_lut

        def spy(luminance: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return original(luminance)

        monkeypatch.setattr(transfer_functions, "pq_oetf_sqrt_lut", spy)
        result = delinearize(np.array([0.0, 0.5, 1.0]), "smpte2084")
        assert calls == 1
        assert result.shape == (3,)

    def test_full_transform_matches_reference_pq(self) -> None:
        rng = np.random.default_rng(217)
        frame = rng.random((24, 32, 3), dtype=np.float64)
        transform = ShotTransform(shot_id=217)
        transform_module = __import__(
            "auto_openmatte.processing.transform", fromlist=["delinearize"]
        )
        production_delinearize = transform_module.delinearize

        def reference_delinearize(
            linear: np.ndarray,
            transfer: str,
            peak_nits: float = 10000.0,
        ) -> np.ndarray:
            if transfer.lower().replace("-", "").replace("_", "") in (
                "smpte2084", "pq", "st2084"
            ):
                return pq_oetf(linear * peak_nits)
            return production_delinearize(linear, transfer, peak_nits)

        transform_module.delinearize = reference_delinearize
        try:
            reference = apply_shot_transform(frame, transform)
        finally:
            transform_module.delinearize = production_delinearize
        candidate = apply_shot_transform(frame, transform)

        rgb_error = np.abs(reference - candidate)
        assert np.max(rgb_error) <= 1.0e-7
        assert np.mean(rgb_error) <= 1.0e-8

    def test_different_peak_nits_preserves_reference_scale(self) -> None:
        linear = np.linspace(0.0, 1.0, 10001)
        for peak_nits in (1000.0, 4000.0, 10000.0):
            reference = pq_oetf(linear * peak_nits)
            candidate = delinearize(linear, "smpte2084", peak_nits=peak_nits)
            np.testing.assert_allclose(candidate, reference, atol=1.0e-7)
