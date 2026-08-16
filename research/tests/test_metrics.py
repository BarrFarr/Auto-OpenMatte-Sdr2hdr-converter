"""Tests for quality metrics (luminance, color, spatial, temporal)."""

from __future__ import annotations

import numpy as np
import pytest

from reshaping_research.metrics.luminance_metrics import (
    luminance_mae,
    luminance_rmse,
    percentile_errors,
    segmented_errors,
)
from reshaping_research.metrics.color_metrics import (
    delta_e_2000,
    delta_e_ictcp,
    chroma_error,
    hue_error,
)
from reshaping_research.metrics.spatial_metrics import (
    gradient_continuity,
    edge_preservation,
    seam_error,
)
from reshaping_research.metrics.temporal_metrics import (
    parameter_variance,
    flicker_risk,
)
from reshaping_research.utils.color_spaces import linear_to_lab_d65


class TestLuminanceMetrics:
    """Tests for luminance error metrics."""

    def test_mae_identical(self) -> None:
        """MAE is zero for identical inputs."""
        values = np.array([100.0, 200.0, 300.0])
        assert luminance_mae(values, values) == pytest.approx(0.0)

    def test_mae_known_value(self) -> None:
        """MAE computes correctly for known offset."""
        pred = np.array([110.0, 210.0, 310.0])
        ref = np.array([100.0, 200.0, 300.0])
        assert luminance_mae(pred, ref) == pytest.approx(10.0)

    def test_rmse_identical(self) -> None:
        """RMSE is zero for identical inputs."""
        values = np.array([100.0, 200.0, 300.0])
        assert luminance_rmse(values, values) == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        """RMSE computes correctly for known values."""
        pred = np.array([110.0, 210.0, 310.0])
        ref = np.array([100.0, 200.0, 300.0])
        # All errors are 10, so RMSE = 10
        assert luminance_rmse(pred, ref) == pytest.approx(10.0)

    def test_rmse_ge_mae(self) -> None:
        """RMSE is always >= MAE."""
        rng = np.random.default_rng(42)
        pred = rng.uniform(0, 1000, 100)
        ref = rng.uniform(0, 1000, 100)
        assert luminance_rmse(pred, ref) >= luminance_mae(pred, ref)

    def test_percentile_errors_ordering(self) -> None:
        """Higher percentiles have higher or equal errors."""
        rng = np.random.default_rng(42)
        pred = rng.uniform(0, 500, 1000)
        ref = rng.uniform(0, 500, 1000)
        errors = percentile_errors(pred, ref, (50.0, 90.0, 99.0))
        assert errors["P50"] <= errors["P90"]
        assert errors["P90"] <= errors["P99"]

    def test_segmented_errors_structure(self) -> None:
        """Segmented errors returns valid dataclass."""
        pred = np.array([10.0, 100.0, 500.0])
        ref = np.array([15.0, 110.0, 480.0])
        result = segmented_errors(pred, ref)
        assert result.shadow_mae >= 0.0
        assert result.midtone_mae >= 0.0
        assert result.highlight_mae >= 0.0

    def test_segmented_errors_shadow_only(self) -> None:
        """Shadow segment computes correctly when all values are dark."""
        pred = np.array([10.0, 20.0, 30.0])
        ref = np.array([12.0, 22.0, 32.0])
        result = segmented_errors(pred, ref)
        assert result.shadow_mae == pytest.approx(2.0)
        assert result.midtone_mae == pytest.approx(0.0)
        assert result.highlight_mae == pytest.approx(0.0)


class TestColorMetrics:
    """Tests for color difference metrics."""

    def test_delta_e_2000_identical(self) -> None:
        """DeltaE2000 is zero for identical colors."""
        lab = np.array([[50.0, 10.0, -20.0]])
        result = delta_e_2000(lab, lab)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_delta_e_2000_positive(self) -> None:
        """DeltaE2000 is positive for different colors."""
        lab1 = np.array([[50.0, 10.0, -20.0]])
        lab2 = np.array([[55.0, 15.0, -15.0]])
        result = delta_e_2000(lab1, lab2)
        assert result[0] > 0.0

    def test_delta_e_2000_symmetric(self) -> None:
        """DeltaE2000 is approximately symmetric."""
        lab1 = np.array([[50.0, 10.0, -20.0]])
        lab2 = np.array([[55.0, 15.0, -15.0]])
        d1 = delta_e_2000(lab1, lab2)
        d2 = delta_e_2000(lab2, lab1)
        # DeltaE2000 is not perfectly symmetric but should be close
        assert d1[0] == pytest.approx(d2[0], rel=0.1)

    def test_delta_e_ictcp_identical(self) -> None:
        """DeltaE ICtCp is zero for identical values."""
        rgb = np.array([[100.0, 100.0, 100.0]])
        result = delta_e_ictcp(rgb, rgb)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_delta_e_ictcp_positive(self) -> None:
        """DeltaE ICtCp is positive for different values."""
        rgb1 = np.array([[100.0, 100.0, 100.0]])
        rgb2 = np.array([[200.0, 100.0, 100.0]])
        result = delta_e_ictcp(rgb1, rgb2)
        assert result[0] > 0.0

    def test_chroma_error_identical(self) -> None:
        """Chroma error is zero for identical inputs."""
        rgb = np.array([[500.0, 300.0, 200.0]])
        result = chroma_error(rgb, rgb)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_hue_error_identical(self) -> None:
        """Hue error is zero for identical inputs."""
        rgb = np.array([[500.0, 300.0, 200.0]])
        result = hue_error(rgb, rgb)
        assert result[0] == pytest.approx(0.0, abs=1e-3)

    def test_hue_error_range(self) -> None:
        """Hue error is within [-180, 180] degrees."""
        rng = np.random.default_rng(42)
        rgb1 = rng.uniform(10, 1000, (50, 3))
        rgb2 = rng.uniform(10, 1000, (50, 3))
        result = hue_error(rgb1, rgb2)
        assert np.all(result >= -180.0)
        assert np.all(result <= 180.0)


class TestSpatialMetrics:
    """Tests for spatial/seam metrics."""

    def test_gradient_continuity_smooth(self) -> None:
        """Smooth image has low gradient discontinuity."""
        # Create a smooth vertical gradient
        img = np.zeros((64, 32, 3), dtype=np.float64)
        y_vals = np.linspace(10.0, 200.0, 64)
        for i in range(64):
            img[i, :, :] = y_vals[i]

        result = gradient_continuity(img, seam_row=32)
        assert result < 1.0  # Very low for smooth gradient

    def test_gradient_continuity_discontinuous(self) -> None:
        """Discontinuous image has high gradient discontinuity."""
        img = np.zeros((64, 32, 3), dtype=np.float64)
        img[:32, :, :] = 50.0  # Dark top
        img[32:, :, :] = 500.0  # Bright bottom

        result = gradient_continuity(img, seam_row=32)
        assert result > 100.0  # Large discontinuity

    def test_edge_preservation_identical(self) -> None:
        """Edge preservation is 1.0 for identical images."""
        img = np.random.default_rng(42).uniform(10, 500, (64, 32, 3))
        result = edge_preservation(img, img, seam_row=32)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_seam_error_returns_dataclass(self) -> None:
        """seam_error returns valid SeamMetrics."""
        img = np.random.default_rng(42).uniform(10, 500, (64, 32, 3))
        result = seam_error(img, seam_row=32)
        assert hasattr(result, "luminance_gradient_diff")
        assert hasattr(result, "chroma_gradient_diff")
        assert hasattr(result, "continuity_score")
        assert 0.0 <= result.continuity_score <= 1.0


class TestTemporalMetrics:
    """Tests for temporal stability metrics."""

    def test_parameter_variance_constant(self) -> None:
        """Constant parameters have zero variance."""
        params = [1.0, 1.0, 1.0, 1.0, 1.0]
        assert parameter_variance(params) == pytest.approx(0.0)

    def test_parameter_variance_positive(self) -> None:
        """Varying parameters have positive variance."""
        params = [1.0, 1.1, 0.9, 1.2, 0.8]
        assert parameter_variance(params) > 0.0

    def test_flicker_risk_stable(self) -> None:
        """Stable parameters have zero flicker risk."""
        params = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = flicker_risk(params)
        assert result.flicker_risk == pytest.approx(0.0)
        assert result.coefficient_of_variation == pytest.approx(0.0)

    def test_flicker_risk_unstable(self) -> None:
        """Highly varying parameters have high flicker risk."""
        params = [1.0, 2.0, 0.5, 2.5, 0.1]
        result = flicker_risk(params, threshold=0.05)
        assert result.flicker_risk > 0.5
        assert result.max_frame_delta > 0.0

    def test_flicker_risk_single_value(self) -> None:
        """Single value produces zero risk."""
        result = flicker_risk([1.0])
        assert result.flicker_risk == pytest.approx(0.0)
        assert result.mean == pytest.approx(1.0)
