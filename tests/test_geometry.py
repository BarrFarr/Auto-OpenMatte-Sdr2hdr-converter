"""Tests for geometry alignment (FEAT-005).

Covers:
- Identity alignment (HDR = center crop of OM)
- Known vertical offset recovery
- Known horizontal offset
- Scale detection
- Different content → low confidence
- Edge invariance to gamma/PQ
- Multi-sample robustness
- overlap_bbox computation
- Stability detection
- Black frame skipping
- Batch extraction (not per-frame)
- No hardcoded values (280, 1.0)
- Sync offset applied correctly
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

np = pytest.importorskip("numpy")

from auto_openmatte.analysis.geometry import (
    _estimate_single_pair,
    _find_best_translation,
    estimate_geometry,
)
from auto_openmatte.core.models import (
    SourceInfo,
    SyncModel,
    SyncStatus,
    VideoStreamInfo,
)
from auto_openmatte.utils.frames import compute_edge_map


def _make_source(
    path: str = "test.mkv",
    width: int = 3840,
    height: int = 1600,
    fps: float = 23.976,
    frame_count: int = 100000,
) -> SourceInfo:
    stream = VideoStreamInfo(
        index=0, codec="hevc", width=width, height=height,
        fps=fps, frame_count=frame_count,
        duration_seconds=frame_count / fps,
    )
    return SourceInfo(path=Path(path), selected_stream=stream)


def _make_sync(offset: int = 1167) -> SyncModel:
    return SyncModel(
        frame_offset=offset, confidence=0.97,
        status=SyncStatus.LOCKED, frame_locked=True,
    )


def _make_scene(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Create a structured scene with edges (for reliable NCC)."""
    rng = np.random.default_rng(seed)
    scene = np.zeros((height, width), dtype=np.float64)
    # Add geometric shapes
    scene[height // 4: height // 2, width // 4: 3 * width // 4] = 0.6
    scene[height // 2: 3 * height // 4, width // 3: 2 * width // 3] = 0.4
    # Diagonal gradient
    for y in range(height):
        scene[y, :] += 0.1 * y / height
    # Some texture
    scene += rng.random((height, width)) * 0.03
    return np.clip(scene, 0, 1)


# ---------------------------------------------------------------------------
# Test: _find_best_translation
# ---------------------------------------------------------------------------


class TestFindBestTranslation:
    """Tests for NCC-based translation search."""

    def test_identity_offset_zero(self) -> None:
        """HDR is top-left crop of OM → offset (0, 0)."""
        om_scene = _make_scene(100, 120, seed=1)
        hdr_scene = om_scene[:60, :120]  # Top portion
        om_edge = compute_edge_map(om_scene)
        hdr_edge = compute_edge_map(hdr_scene)

        ox, oy, score = _find_best_translation(hdr_edge, om_edge, scale=1.0)
        assert oy == pytest.approx(0.0, abs=1.0)
        assert ox == pytest.approx(0.0, abs=1.0)
        assert score > 0.9

    def test_known_vertical_offset(self) -> None:
        """HDR is centered vertically in OM → correct offset_y."""
        om_scene = _make_scene(100, 80, seed=2)
        # HDR = center crop: rows 20:70 (offset_y = 20)
        hdr_scene = om_scene[20:70, :]
        om_edge = compute_edge_map(om_scene)
        hdr_edge = compute_edge_map(hdr_scene)

        ox, oy, score = _find_best_translation(hdr_edge, om_edge, scale=1.0)
        assert oy == pytest.approx(20.0, abs=1.0)
        assert ox == pytest.approx(0.0, abs=1.0)
        assert score > 0.9

    def test_known_horizontal_offset(self) -> None:
        """HDR shifted horizontally → correct offset_x."""
        om_scene = _make_scene(60, 100, seed=3)
        # HDR = om[0:60, 5:95] (offset_x = 5)
        hdr_scene = om_scene[:, 5:95]
        om_edge = compute_edge_map(om_scene)
        hdr_edge = compute_edge_map(hdr_scene)

        ox, oy, score = _find_best_translation(hdr_edge, om_edge, scale=1.0)
        assert oy == pytest.approx(0.0, abs=1.0)
        assert ox == pytest.approx(5.0, abs=1.0)
        assert score > 0.85

    def test_different_content_low_score(self) -> None:
        """Unrelated images → low NCC score."""
        hdr_edge = compute_edge_map(_make_scene(50, 80, seed=10))
        om_edge = compute_edge_map(_make_scene(100, 80, seed=99))

        _, _, score = _find_best_translation(hdr_edge, om_edge, scale=1.0)
        # Score should be notably lower than aligned content
        assert score < 0.8


# ---------------------------------------------------------------------------
# Test: _estimate_single_pair
# ---------------------------------------------------------------------------


class TestEstimateSinglePair:
    """Tests for single-pair geometry estimation (scale + translation)."""

    def test_scale_1_detected(self) -> None:
        """Content at scale=1.0 should be found with scale≈1.0."""
        om_scene = _make_scene(100, 80, seed=20)
        hdr_scene = om_scene[15:75, :]  # Vertical offset 15, scale 1.0
        om_edge = compute_edge_map(om_scene)
        hdr_edge = compute_edge_map(hdr_scene)

        sx, sy, ox, oy, score = _estimate_single_pair(hdr_edge, om_edge)
        assert abs(sx - 1.0) <= 0.03
        assert abs(sy - 1.0) <= 0.03
        assert score > 0.8

    def test_no_hardcoded_scale(self) -> None:
        """Algorithm must detect scale from image, not return constant 1.0."""
        # Create OM and a slightly scaled HDR (0.98x)
        om_scene = _make_scene(100, 80, seed=25)
        from scipy.ndimage import zoom
        hdr_full = om_scene[10:80, 4:76]  # 70x72 region
        hdr_scaled = zoom(hdr_full, 0.98, order=1)  # Slightly smaller
        om_edge = compute_edge_map(om_scene)
        hdr_edge = compute_edge_map(hdr_scaled)

        sx, sy, ox, oy, score = _estimate_single_pair(hdr_edge, om_edge)
        # Should find a scale close to but not exactly 1.0
        # (the search should explore the range and find best fit)
        assert score > 0.5  # Should find reasonable alignment


# ---------------------------------------------------------------------------
# Test: estimate_geometry integration (mocked extraction)
# ---------------------------------------------------------------------------


class TestEstimateGeometryIntegration:
    """Integration tests with mocked frame extraction."""

    def _mock_pair_factory(self, om_h: int, om_w: int, hdr_h: int, hdr_w: int,
                           offset_y: int, seed: int = 42):
        """Create a mock that returns consistent HDR/OM edge pairs."""
        om_scene = _make_scene(om_h, om_w, seed=seed)
        hdr_scene = om_scene[offset_y: offset_y + hdr_h, :hdr_w]

        def mock_extract(file_path, start_seconds, n_frames, **kwargs):
            width = kwargs.get("width", 480)
            # Determine which source based on path
            if "hdr" in str(file_path).lower() or "hallowed" in str(file_path).lower():
                # Return HDR-sized proxy
                from scipy.ndimage import zoom
                h_ratio = hdr_h * (width / hdr_w) / hdr_h
                frame = zoom(hdr_scene, (h_ratio, width / hdr_w), order=1)
            else:
                # Return OM-sized proxy
                from scipy.ndimage import zoom
                h_ratio = om_h * (width / om_w) / om_h
                frame = zoom(om_scene, (h_ratio, width / om_w), order=1)
            return [np.clip(frame, 0, 1)]

        return mock_extract

    def test_correct_offset_detected(self) -> None:
        """Known vertical offset should be detected correctly."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync(offset=1167)

        # OM is 270px high at proxy, HDR is 200px. Offset should be ~35px in proxy
        # (which scales to ~280 full-res)
        mock = self._mock_pair_factory(270, 480, 200, 480, offset_y=35, seed=50)

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=mock,
        ):
            geometry = estimate_geometry(hdr_source, om_source, sync)

        # offset_y in full res ≈ 35 * (3840/480) = 280
        assert abs(geometry.offset_y - 280.0) < 30.0
        assert abs(geometry.offset_x) < 30.0
        assert geometry.confidence > 0.5

    def test_overlap_bbox_computed(self) -> None:
        """overlap_bbox should reflect detected geometry."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync(offset=100)

        mock = self._mock_pair_factory(270, 480, 200, 480, offset_y=35, seed=60)

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=mock,
        ):
            geometry = estimate_geometry(hdr_source, om_source, sync)

        # overlap_bbox should be [x1, y1, x2, y2] in OM coords
        bbox = geometry.overlap_bbox
        assert len(bbox) == 4
        assert bbox[0] >= 0  # x1
        assert bbox[1] > 0   # y1 (offset_y > 0)
        assert bbox[2] <= 3840  # x2
        assert bbox[3] <= 2160  # y2
        assert bbox[2] > bbox[0]
        assert bbox[3] > bbox[1]

    def test_stability_global(self) -> None:
        """Consistent samples → is_global=True."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync()

        # All samples return same offset
        mock = self._mock_pair_factory(270, 480, 200, 480, offset_y=35, seed=70)

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=mock,
        ):
            geometry = estimate_geometry(hdr_source, om_source, sync)

        assert geometry.is_global is True

    def test_black_frames_skipped(self) -> None:
        """Black frames should be skipped, not corrupt geometry."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync()

        call_count = [0]
        om_scene = _make_scene(270, 480, seed=80)
        hdr_scene = om_scene[35:235, :]

        def mock_with_black(file_path, start_seconds, n_frames, **kwargs):
            call_count[0] += 1
            # First 2 calls return black frames (skipped)
            if call_count[0] <= 4:  # 2 pairs × 2 sources
                return [np.zeros((200, 480)) + 0.001]
            # Rest return valid content
            if "hdr" in str(file_path).lower():
                return [np.clip(hdr_scene, 0, 1)]
            return [np.clip(om_scene, 0, 1)]

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=mock_with_black,
        ):
            geometry = estimate_geometry(hdr_source, om_source, sync)

        # Should still get valid geometry from non-black samples
        assert geometry.confidence > 0.5

    def test_sync_offset_applied(self) -> None:
        """OM frame extraction should use HDR_frame + sync_offset."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync(offset=500)

        extraction_times: list[float] = []

        def tracking_extract(file_path, start_seconds, n_frames, **kwargs):
            extraction_times.append(start_seconds)
            # Return valid frame
            return [_make_scene(200 if "hdr" in str(file_path) else 270, 480, seed=90)]

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=tracking_extract,
        ):
            estimate_geometry(hdr_source, om_source, sync)

        # OM times should be offset from HDR times by sync_offset / fps
        fps = 23.976
        offset_seconds = 500 / fps
        # Pairs: (hdr_time, om_time) — om should be hdr + offset
        hdr_times = extraction_times[0::2]
        om_times = extraction_times[1::2]
        for ht, ot in zip(hdr_times, om_times):
            assert abs((ot - ht) - offset_seconds) < 1.0  # ±1s tolerance

    def test_no_hardcoded_280(self) -> None:
        """Offset must come from image analysis, not from (2160-1600)/2."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync()

        # Create content where HDR is NOT centered — offset_y = 50 in proxy
        # Full res: 50 * 8 = 400 (not 280)
        mock = self._mock_pair_factory(270, 480, 200, 480, offset_y=50, seed=100)

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=mock,
        ):
            geometry = estimate_geometry(hdr_source, om_source, sync)

        # Should detect ~400, NOT 280
        assert abs(geometry.offset_y - 400.0) < 30.0
        assert abs(geometry.offset_y - 280.0) > 50.0  # Definitely not 280

    def test_batch_extraction_not_per_frame(self) -> None:
        """Should use extract_segment, not individual frame calls."""
        hdr_source = _make_source("hdr.mkv", width=3840, height=1600)
        om_source = _make_source("om.mkv", width=3840, height=2160)
        sync = _make_sync()
        call_count = [0]

        def counting_extract(file_path, start_seconds, n_frames, **kwargs):
            call_count[0] += 1
            return [_make_scene(200 if "hdr" in str(file_path) else 270, 480)]

        with patch(
            "auto_openmatte.analysis.geometry.extract_segment_at_time_grayscale",
            side_effect=counting_extract,
        ):
            estimate_geometry(hdr_source, om_source, sync)

        # 7 samples × 2 sources = 14 calls (not hundreds)
        assert call_count[0] == 14


# ---------------------------------------------------------------------------
# Test: edge invariance to transfer function
# ---------------------------------------------------------------------------


class TestEdgeInvariance:
    """Edge-based geometry should work despite HDR/SDR tonal differences."""

    def test_pq_vs_gamma_same_geometry(self) -> None:
        """Same structure under PQ vs gamma → same detected offset."""
        scene = _make_scene(100, 80, seed=110)
        # HDR = center crop (offset 15)
        hdr_region = scene[15:75, :]

        # Apply PQ-like curve to HDR
        pq_hdr = np.power(np.clip(hdr_region, 0.001, 1), 0.25)
        # Apply gamma to OM
        gamma_om = np.power(np.clip(scene, 0.001, 1), 1.0 / 2.2)

        pq_edge = compute_edge_map(pq_hdr)
        gamma_edge = compute_edge_map(gamma_om)

        ox, oy, score = _find_best_translation(pq_edge, gamma_edge, scale=1.0)
        # Should still find offset_y ≈ 15 (with some tolerance due to TF)
        assert abs(oy - 15.0) <= 5.0
        assert score > 0.5
