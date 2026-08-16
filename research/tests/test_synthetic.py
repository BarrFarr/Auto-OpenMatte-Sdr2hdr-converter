"""Tests for synthetic scene generation and SDR simulation."""

from __future__ import annotations

import numpy as np
import pytest

from reshaping_research.synthetic.scene_generator import generate_scene, SceneType
from reshaping_research.synthetic.sdr_simulator import simulate_sdr
from reshaping_research.synthetic.open_matte_simulator import create_open_matte_pair


class TestSceneGenerator:
    """Tests for the synthetic scene generator."""

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_scene_shape(self, scene_type: SceneType) -> None:
        """All scene types produce the correct output shape."""
        scene = generate_scene(scene_type, height=128, width=128)
        assert scene.shape == (128, 128, 3)

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_scene_non_negative(self, scene_type: SceneType) -> None:
        """All scene values are non-negative (linear light)."""
        scene = generate_scene(scene_type, height=128, width=128)
        assert np.all(scene >= 0.0)

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_scene_deterministic(self, scene_type: SceneType) -> None:
        """Same seed produces identical scenes."""
        scene1 = generate_scene(scene_type, height=64, width=64, seed=123)
        scene2 = generate_scene(scene_type, height=64, width=64, seed=123)
        np.testing.assert_array_equal(scene1, scene2)

    def test_neutral_luminance_range(self) -> None:
        """Neutral scene has moderate luminance range."""
        scene = generate_scene(SceneType.NEUTRAL, height=256, width=256)
        max_lum = np.max(scene)
        assert max_lum < 1000.0, f"Neutral scene too bright: {max_lum}"
        assert max_lum > 100.0, f"Neutral scene too dim: {max_lum}"

    def test_high_key_brightness(self) -> None:
        """High-key scene reaches high luminance values."""
        scene = generate_scene(SceneType.HIGH_KEY, height=256, width=256)
        max_lum = np.max(scene)
        assert max_lum > 2000.0, f"High-key not bright enough: {max_lum}"

    def test_low_key_darkness(self) -> None:
        """Low-key scene is predominantly dark."""
        scene = generate_scene(SceneType.LOW_KEY, height=256, width=256)
        # Most pixels should be under 50 nits
        lum = np.mean(scene, axis=-1)
        dark_fraction = np.mean(lum < 50.0)
        assert dark_fraction > 0.5, f"Low-key not dark enough: {dark_fraction:.2%} dark"

    def test_difficult_extreme_range(self) -> None:
        """Difficult scene has extreme dynamic range."""
        scene = generate_scene(SceneType.DIFFICULT, height=256, width=256)
        max_lum = np.max(scene)
        min_lum = np.min(scene[scene > 0])
        assert max_lum > 5000.0, f"Difficult scene not bright enough: {max_lum}"
        assert min_lum < 10.0, f"Difficult scene not dark enough: {min_lum}"

    def test_different_seeds_different_scenes(self) -> None:
        """Different seeds produce different scenes."""
        scene1 = generate_scene(SceneType.NEUTRAL, height=64, width=64, seed=1)
        scene2 = generate_scene(SceneType.NEUTRAL, height=64, width=64, seed=2)
        assert not np.array_equal(scene1, scene2)


class TestSDRSimulator:
    """Tests for the SDR simulation pipeline."""

    def test_sdr_output_range(self) -> None:
        """SDR output is in [0, 1] range."""
        hdr = generate_scene(SceneType.NEUTRAL, height=64, width=64)
        sdr = simulate_sdr(hdr)
        assert np.all(sdr >= 0.0)
        assert np.all(sdr <= 1.0)

    def test_sdr_output_shape(self) -> None:
        """SDR output has same shape as input."""
        hdr = generate_scene(SceneType.HIGH_KEY, height=64, width=128)
        sdr = simulate_sdr(hdr)
        assert sdr.shape == hdr.shape

    def test_sdr_not_all_zero(self) -> None:
        """SDR output contains non-zero values for typical scene."""
        hdr = generate_scene(SceneType.NEUTRAL, height=64, width=64)
        sdr = simulate_sdr(hdr)
        assert np.mean(sdr) > 0.01

    def test_sdr_not_all_one(self) -> None:
        """SDR output is not fully clipped for moderate scenes."""
        hdr = generate_scene(SceneType.NEUTRAL, height=64, width=64)
        sdr = simulate_sdr(hdr)
        assert np.mean(sdr) < 0.99

    def test_sdr_tone_mapping_compresses(self) -> None:
        """Tone mapping compresses HDR dynamic range."""
        hdr = generate_scene(SceneType.HIGH_KEY, height=64, width=64)
        sdr = simulate_sdr(hdr)
        # HDR has wide range (> 2000 nits), SDR should be in [0, 1]
        hdr_range = np.max(hdr) - np.min(hdr)
        sdr_range = np.max(sdr) - np.min(sdr)
        assert sdr_range < hdr_range

    @pytest.mark.parametrize("scene_type", list(SceneType))
    def test_sdr_all_scene_types(self, scene_type: SceneType) -> None:
        """SDR simulation works for all scene types without errors."""
        hdr = generate_scene(scene_type, height=64, width=64)
        sdr = simulate_sdr(hdr)
        assert sdr.shape == (64, 64, 3)
        assert np.all(np.isfinite(sdr))


class TestOpenMatteSimulator:
    """Tests for the open matte pair generation."""

    def test_open_matte_dimensions(self) -> None:
        """Open matte SDR is taller than HDR by 2x extension_rows."""
        hdr = generate_scene(SceneType.NEUTRAL, height=128, width=128)
        pair = create_open_matte_pair(hdr, extension_rows=32)

        assert pair.hdr_center.shape == (128, 128, 3)
        assert pair.sdr_wide.shape == (192, 128, 3)  # 128 + 2*32

    def test_open_matte_overlap_mask(self) -> None:
        """Overlap mask correctly identifies the HDR region."""
        hdr = generate_scene(SceneType.NEUTRAL, height=100, width=80)
        pair = create_open_matte_pair(hdr, extension_rows=24)

        assert pair.overlap_mask.shape == (148, 80)  # 100 + 2*24
        # Check overlap region
        assert np.all(pair.overlap_mask[24:124, :])
        assert not np.any(pair.overlap_mask[:24, :])
        assert not np.any(pair.overlap_mask[124:, :])

    def test_open_matte_seam_positions(self) -> None:
        """Seam positions are at the correct rows."""
        hdr = generate_scene(SceneType.NEUTRAL, height=100, width=80)
        pair = create_open_matte_pair(hdr, extension_rows=24)

        assert pair.seam_top_row == 24
        assert pair.seam_bottom_row == 124
        assert pair.hdr_offset_top == 24
        assert pair.hdr_offset_bottom == 124

    def test_open_matte_sdr_valid(self) -> None:
        """SDR wide frame has valid values."""
        hdr = generate_scene(SceneType.MIXED, height=128, width=128)
        pair = create_open_matte_pair(hdr, extension_rows=32)

        assert np.all(pair.sdr_wide >= 0.0)
        assert np.all(pair.sdr_wide <= 1.0)
        assert np.all(np.isfinite(pair.sdr_wide))

    def test_open_matte_deterministic(self) -> None:
        """Same inputs produce identical outputs."""
        hdr = generate_scene(SceneType.NEUTRAL, height=64, width=64)
        pair1 = create_open_matte_pair(hdr, extension_rows=16, seed=42)
        pair2 = create_open_matte_pair(hdr, extension_rows=16, seed=42)
        np.testing.assert_array_equal(pair1.sdr_wide, pair2.sdr_wide)
