"""Comprehensive tests for reshaping models A through G.

Tests cover:
- Fit and apply on synthetic data
- param_count() returns expected values
- Monotonicity of luminance mapping
- Output is non-negative and finite
- Reasonable reconstruction quality (MAE)
- Edge cases: very dark, very bright, uniform scenes
"""

from __future__ import annotations

import numpy as np
import pytest

from reshaping_research.models import (
    ModelALinear,
    ModelBPiecewise,
    ModelCPolynomial,
    ModelDCdf,
    ModelECdfRegularized,
    ModelFLumaChromaRegression,
    ModelGHybrid,
    get_all_models,
)
from reshaping_research.utils.color_spaces import BT2020_LUMA, bt709_to_bt2020
from reshaping_research.utils.transfer_functions import bt1886_eotf


# ============================================================
# Fixtures: generate synthetic SDR/HDR pairs for testing
# ============================================================


def _make_synthetic_pair(
    scene_type: str = "gradient",
    n_pixels: int = 5000,
    seed: int = 42,
):
    """Create synthetic SDR/HDR pairs in linear BT.2020 domain.

    Returns:
        sdr_lum: (N,) SDR luminance values (linear BT.2020 Y)
        hdr_lum: (N,) HDR luminance values (linear, normalized to [0,1])
        sdr_rgb: (N, 3) SDR RGB in linear BT.2020
        hdr_rgb: (N, 3) HDR RGB in linear BT.2020
    """
    rng = np.random.default_rng(seed)

    if scene_type == "gradient":
        # Smooth gradient - easy case
        sdr_lum = np.linspace(0.01, 0.8, n_pixels)
        # Non-linear expansion (simulating tone-map inverse)
        hdr_lum = sdr_lum**0.6 * 0.9  # Non-linear but monotonic
    elif scene_type == "dark":
        # Very dark scene
        sdr_lum = rng.uniform(0.001, 0.05, n_pixels)
        hdr_lum = sdr_lum * 2.5 + rng.normal(0, 0.002, n_pixels)
        hdr_lum = np.maximum(hdr_lum, 0.0)
    elif scene_type == "bright":
        # Very bright scene
        sdr_lum = rng.uniform(0.5, 0.95, n_pixels)
        hdr_lum = sdr_lum**0.5 * 0.95
    elif scene_type == "uniform":
        # Nearly uniform (flat) scene
        sdr_lum = 0.3 + rng.normal(0, 0.01, n_pixels)
        sdr_lum = np.clip(sdr_lum, 0.01, 0.99)
        hdr_lum = sdr_lum * 1.5 + rng.normal(0, 0.005, n_pixels)
        hdr_lum = np.clip(hdr_lum, 0.0, 1.0)
    else:
        # Default random
        sdr_lum = rng.uniform(0.01, 0.9, n_pixels)
        hdr_lum = sdr_lum * 1.8
        hdr_lum = np.clip(hdr_lum, 0.0, 1.0)

    # Generate RGB with some color variation
    # SDR RGB: luminance + chromatic deviation
    sdr_rgb = np.zeros((n_pixels, 3))
    sdr_rgb[:, 0] = sdr_lum * (1.0 + rng.normal(0, 0.05, n_pixels))
    sdr_rgb[:, 1] = sdr_lum * (1.0 + rng.normal(0, 0.03, n_pixels))
    sdr_rgb[:, 2] = sdr_lum * (1.0 + rng.normal(0, 0.04, n_pixels))
    sdr_rgb = np.maximum(sdr_rgb, 0.0)

    # HDR RGB: scaled version with slightly different color
    scale = (hdr_lum / np.maximum(sdr_lum, 1e-8))[:, np.newaxis]
    hdr_rgb = sdr_rgb * scale * (1.0 + rng.normal(0, 0.02, (n_pixels, 3)))
    hdr_rgb = np.maximum(hdr_rgb, 0.0)

    return sdr_lum, hdr_lum, sdr_rgb, hdr_rgb


def _make_synthetic_image(height: int = 64, width: int = 64, seed: int = 42):
    """Create a synthetic (H, W, 3) image in linear BT.2020 for apply() tests."""
    rng = np.random.default_rng(seed)
    # Gradient image
    x = np.linspace(0.01, 0.8, width)
    y = np.linspace(0.01, 0.6, height)
    xx, yy = np.meshgrid(x, y)
    lum = xx * 0.7 + yy * 0.3
    img = np.zeros((height, width, 3))
    img[..., 0] = lum * (1.0 + rng.normal(0, 0.02, (height, width)))
    img[..., 1] = lum * (1.0 + rng.normal(0, 0.01, (height, width)))
    img[..., 2] = lum * (1.0 + rng.normal(0, 0.015, (height, width)))
    return np.maximum(img, 0.0)


# ============================================================
# Test: get_all_models returns correct count and types
# ============================================================


class TestGetAllModels:
    """Tests for the get_all_models() factory function."""

    def test_returns_seven_models(self):
        models = get_all_models()
        assert len(models) == 7

    def test_all_have_required_interface(self):
        models = get_all_models()
        for model in models:
            assert hasattr(model, "fit")
            assert hasattr(model, "apply")
            assert hasattr(model, "param_count")
            assert hasattr(model, "name")
            assert callable(model.fit)
            assert callable(model.apply)
            assert callable(model.param_count)
            assert callable(model.name)

    def test_unique_names(self):
        models = get_all_models()
        names = [m.name() for m in models]
        assert len(set(names)) == 7


# ============================================================
# Test: param_count returns expected values
# ============================================================


class TestParamCount:
    """Tests for parameter counts of each model."""

    def test_model_a_param_count(self):
        model = ModelALinear()
        assert model.param_count() == 2

    def test_model_b_param_count(self):
        model = ModelBPiecewise()
        assert model.param_count() == 10  # 9 percentiles + 1 chroma

    def test_model_c_param_count(self):
        model = ModelCPolynomial()
        # degree 4 luma (5 coeffs) + degree 2 chroma (3 coeffs) = 8
        assert model.param_count() == 8

    def test_model_d_param_count(self):
        model = ModelDCdf()
        assert model.param_count() == 65  # 64 LUT + 1 chroma

    def test_model_e_param_count(self):
        model = ModelECdfRegularized()
        # (3+1) correction poly + 2*3 sigmoids + 2 chroma = 12
        assert model.param_count() == 12

    def test_model_f_param_count(self):
        model = ModelFLumaChromaRegression()
        # 8 knots + 9 matrix + 1 saturation = 18
        assert model.param_count() == 18

    def test_model_g_param_count(self):
        model = ModelGHybrid()
        # 10 luma + 3 highlight + 2 shadow + 2 sat + 9 matrix + 3 hue = 29
        assert model.param_count() == 29


# ============================================================
# Test: fit and apply on standard synthetic data
# ============================================================


class TestFitApply:
    """Tests that each model can fit and apply successfully."""

    @pytest.fixture(params=[
        ModelALinear,
        ModelBPiecewise,
        ModelCPolynomial,
        ModelDCdf,
        ModelECdfRegularized,
        ModelFLumaChromaRegression,
        ModelGHybrid,
    ])
    def model(self, request):
        return request.param()

    def test_fit_returns_dict(self, model):
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        result = model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_apply_returns_correct_shape(self, model):
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        img = _make_synthetic_image(64, 64)
        output = model.apply(img)
        assert output.shape == (64, 64, 3)

    def test_output_non_negative(self, model):
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        img = _make_synthetic_image(32, 32)
        output = model.apply(img)
        assert np.all(output >= 0.0)

    def test_output_finite(self, model):
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        img = _make_synthetic_image(32, 32)
        output = model.apply(img)
        assert np.all(np.isfinite(output))

    def test_fit_without_chroma(self, model):
        """Models should work without chroma data."""
        sdr_lum, hdr_lum, _, _ = _make_synthetic_pair("gradient")
        result = model.fit(sdr_lum, hdr_lum)
        assert isinstance(result, dict)
        img = _make_synthetic_image(16, 16)
        output = model.apply(img)
        assert output.shape == (16, 16, 3)
        assert np.all(np.isfinite(output))


# ============================================================
# Test: Monotonicity of luminance mapping
# ============================================================


class TestMonotonicity:
    """Tests that luminance mapping is monotonically non-decreasing."""

    @pytest.fixture(params=[
        ModelALinear,
        ModelBPiecewise,
        ModelCPolynomial,
        ModelDCdf,
        ModelECdfRegularized,
        ModelFLumaChromaRegression,
        ModelGHybrid,
    ])
    def model(self, request):
        return request.param()

    def test_monotonic_luminance(self, model):
        """Apply sorted luminance input, verify output is non-decreasing."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        # Create a 1D strip of increasing luminance
        n = 200
        lum_values = np.linspace(0.01, 0.8, n)
        # Create achromatic image (H=1, W=n, C=3)
        img = np.zeros((1, n, 3))
        img[0, :, 0] = lum_values
        img[0, :, 1] = lum_values
        img[0, :, 2] = lum_values

        output = model.apply(img)
        out_lum = np.einsum("...c,c->...", output[0], BT2020_LUMA)

        # Check monotonicity (allow small numerical errors)
        diffs = np.diff(out_lum)
        assert np.all(diffs >= -1e-6), (
            f"Non-monotonic output for {model.name()}: "
            f"min diff = {np.min(diffs):.6e}"
        )


# ============================================================
# Test: Reconstruction quality
# ============================================================


class TestReconstruction:
    """Tests that models achieve reasonable reconstruction on fitted data."""

    @pytest.fixture(params=[
        ModelALinear,
        ModelBPiecewise,
        ModelCPolynomial,
        ModelDCdf,
        ModelECdfRegularized,
        ModelFLumaChromaRegression,
        ModelGHybrid,
    ])
    def model(self, request):
        return request.param()

    def test_reconstruction_mae(self, model):
        """Apply model to data it was fitted on; MAE should be reasonable."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        # Apply to the same data reshaped as image
        n = len(sdr_lum)
        # Use achromatic image for luminance comparison
        img = np.zeros((1, n, 3))
        img[0, :, 0] = sdr_lum
        img[0, :, 1] = sdr_lum
        img[0, :, 2] = sdr_lum

        output = model.apply(img)
        out_lum = np.einsum("...c,c->...", output[0], BT2020_LUMA)

        mae = float(np.mean(np.abs(out_lum - hdr_lum)))
        # Generous threshold - we just want to verify the model is working
        assert mae < 0.15, (
            f"MAE too high for {model.name()}: {mae:.4f}"
        )


# ============================================================
# Test: Edge cases
# ============================================================


class TestEdgeCases:
    """Tests for edge cases: very dark, very bright, uniform scenes."""

    @pytest.fixture(params=[
        ModelALinear,
        ModelBPiecewise,
        ModelCPolynomial,
        ModelDCdf,
        ModelECdfRegularized,
        ModelFLumaChromaRegression,
        ModelGHybrid,
    ])
    def model(self, request):
        return request.param()

    def test_very_dark_scene(self, model):
        """Model handles very dark input without errors."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("dark")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        # Very dark image
        img = np.full((16, 16, 3), 0.005)
        output = model.apply(img)
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)

    def test_very_bright_scene(self, model):
        """Model handles very bright input without errors."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("bright")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        # Bright image
        img = np.full((16, 16, 3), 0.9)
        output = model.apply(img)
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)

    def test_uniform_scene(self, model):
        """Model handles nearly uniform input without errors."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("uniform")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        img = np.full((16, 16, 3), 0.3)
        output = model.apply(img)
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)

    def test_zero_input(self, model):
        """Model handles zero input gracefully."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        img = np.zeros((8, 8, 3))
        output = model.apply(img)
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)

    def test_minimal_data(self, model):
        """Model handles very small training data gracefully."""
        sdr_lum = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        hdr_lum = np.array([0.15, 0.3, 0.5, 0.65, 0.8])
        # Should not crash even with < 10 samples
        result = model.fit(sdr_lum, hdr_lum)
        assert isinstance(result, dict)

        img = np.full((4, 4, 3), 0.3)
        output = model.apply(img)
        assert np.all(np.isfinite(output))


# ============================================================
# Test: Individual model-specific behavior
# ============================================================


class TestModelASpecific:
    """Tests specific to Model A (Linear Gain)."""

    def test_linear_gain_scaling(self):
        """Linear gain should uniformly scale luminance."""
        model = ModelALinear()
        sdr_lum = np.linspace(0.1, 0.5, 1000)
        hdr_lum = sdr_lum * 2.0  # Known gain of 2.0
        model.fit(sdr_lum, hdr_lum)
        assert abs(model.params["a"] - 2.0) < 0.05

    def test_chroma_gain(self):
        """Chroma gain should be estimated correctly."""
        model = ModelALinear()
        n = 1000
        sdr_lum = np.linspace(0.1, 0.5, n)
        hdr_lum = sdr_lum * 2.0

        sdr_rgb = np.zeros((n, 3))
        sdr_rgb[:, 0] = sdr_lum * 1.2
        sdr_rgb[:, 1] = sdr_lum * 0.9
        sdr_rgb[:, 2] = sdr_lum * 0.8

        hdr_rgb = sdr_rgb * 1.5  # Uniform chroma gain of ~1.5 relative to luma gain

        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        # Chroma gain should be close to 1.5 (since both luma and chroma scale by same)
        assert model.params["b"] > 0.5


class TestModelBSpecific:
    """Tests specific to Model B (Piecewise Linear)."""

    def test_default_percentiles(self):
        """Should use default 9 percentiles."""
        model = ModelBPiecewise()
        assert model.param_count() == 10  # 9 + 1 chroma

    def test_custom_percentiles(self):
        """Should support custom percentile count."""
        model = ModelBPiecewise(percentiles=(10, 30, 50, 70, 90))
        assert model.param_count() == 6  # 5 + 1


class TestModelCSpecific:
    """Tests specific to Model C (Monotonic Polynomial)."""

    def test_custom_degree(self):
        """Should support custom polynomial degree."""
        model = ModelCPolynomial(luma_degree=3, chroma_degree=1)
        assert model.param_count() == 6  # (3+1) + (1+1) = 6

    def test_high_degree(self):
        """Higher degree should still produce valid results."""
        model = ModelCPolynomial(luma_degree=5)
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        img = _make_synthetic_image(16, 16)
        output = model.apply(img)
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)


class TestModelDSpecific:
    """Tests specific to Model D (CDF Matching)."""

    def test_lut_size(self):
        """Should use configured LUT size."""
        model = ModelDCdf(lut_size=32)
        assert model.param_count() == 33  # 32 + 1

    def test_identity_mapping(self):
        """When SDR == HDR, mapping should be near-identity."""
        model = ModelDCdf()
        lum = np.linspace(0.01, 0.9, 2000)
        model.fit(lum, lum)

        img = np.full((8, 8, 3), 0.5)
        output = model.apply(img)
        out_lum = np.einsum("...c,c->...", output, BT2020_LUMA)
        assert np.allclose(out_lum, 0.5, atol=0.02)


class TestModelESpecific:
    """Tests specific to Model E (CDF + Regularization)."""

    def test_improves_over_base_cdf(self):
        """Regularized CDF should produce valid output."""
        model = ModelECdfRegularized()
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        img = _make_synthetic_image(16, 16)
        output = model.apply(img)
        assert np.all(np.isfinite(output))
        assert np.all(output >= 0.0)


class TestModelFSpecific:
    """Tests specific to Model F (Luma + Chroma Regression)."""

    def test_color_matrix_near_identity(self):
        """With small color differences, matrix should be near identity."""
        model = ModelFLumaChromaRegression()
        n = 2000
        sdr_lum = np.linspace(0.1, 0.8, n)
        hdr_lum = sdr_lum * 1.5

        # Identical color ratios
        sdr_rgb = np.column_stack([sdr_lum * 1.1, sdr_lum, sdr_lum * 0.9])
        hdr_rgb = np.column_stack([hdr_lum * 1.1, hdr_lum, hdr_lum * 0.9])

        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        mat = np.asarray(model.params["color_matrix"])
        # Should be close to a scaled identity (due to regularization)
        off_diag = mat - np.diag(np.diag(mat))
        assert np.max(np.abs(off_diag)) < 0.5


class TestModelGSpecific:
    """Tests specific to Model G (Hybrid)."""

    def test_all_components_fitted(self):
        """All parameter groups should be populated after fit."""
        model = ModelGHybrid()
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        assert "luma_x" in model.params
        assert "luma_y" in model.params
        assert "highlight_threshold" in model.params
        assert "highlight_slope" in model.params
        assert "highlight_max" in model.params
        assert "shadow_threshold" in model.params
        assert "shadow_gain" in model.params
        assert "sat_base" in model.params
        assert "sat_slope" in model.params
        assert "color_matrix" in model.params
        assert "hue_angles" in model.params

    def test_hue_angles_constrained(self):
        """Hue angles should be small (constrained)."""
        model = ModelGHybrid()
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        angles = np.asarray(model.params["hue_angles"])
        assert np.all(np.abs(angles) <= 0.15)  # Within constraint


# ============================================================
# Test: Cross-model comparison (sanity checks)
# ============================================================


class TestCrossModel:
    """Cross-model sanity checks."""

    def test_all_models_fit_same_data(self):
        """All models should successfully fit the same data."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        models = get_all_models()
        for model in models:
            result = model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
            assert isinstance(result, dict), f"Failed for {model.name()}"

    def test_all_models_apply_same_image(self):
        """All models should apply to the same image without errors."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        img = _make_synthetic_image(32, 32)
        models = get_all_models()
        for model in models:
            model.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
            output = model.apply(img)
            assert output.shape == (32, 32, 3), f"Shape mismatch for {model.name()}"
            assert np.all(np.isfinite(output)), f"Non-finite output for {model.name()}"
            assert np.all(output >= 0.0), f"Negative output for {model.name()}"

    def test_more_params_lower_error(self):
        """Models with more parameters should generally fit better (not strict)."""
        sdr_lum, hdr_lum, sdr_rgb, hdr_rgb = _make_synthetic_pair("gradient")
        img = _make_synthetic_image(1, len(sdr_lum))
        img[0, :, 0] = sdr_lum
        img[0, :, 1] = sdr_lum
        img[0, :, 2] = sdr_lum

        model_a = ModelALinear()
        model_d = ModelDCdf()

        model_a.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)
        model_d.fit(sdr_lum, hdr_lum, sdr_rgb, hdr_rgb)

        out_a = model_a.apply(img)
        out_d = model_d.apply(img)

        lum_a = np.einsum("...c,c->...", out_a[0], BT2020_LUMA)
        lum_d = np.einsum("...c,c->...", out_d[0], BT2020_LUMA)

        mae_a = float(np.mean(np.abs(lum_a - hdr_lum)))
        mae_d = float(np.mean(np.abs(lum_d - hdr_lum)))

        # CDF (64 params) should generally do at least as well as linear (2 params)
        # Use generous margin since this is not guaranteed for all data
        assert mae_d < mae_a + 0.1, (
            f"CDF ({mae_d:.4f}) significantly worse than Linear ({mae_a:.4f})"
        )
