"""Tests for project file I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from auto_openmatte.core.models import (
    ColorPrimaries,
    GeometryModel,
    HDRFormat,
    HDRMetadata,
    ProjectData,
    Shot,
    ShotTransform,
    SourceInfo,
    SourceRole,
    SyncModel,
    SyncStatus,
    TransferFunction,
    VideoStreamInfo,
)
from auto_openmatte.core.project import load_project, save_project


@pytest.fixture
def sample_project(tmp_path: Path) -> tuple[ProjectData, Path]:
    """Create a sample project for testing."""
    project = ProjectData()
    project.version = "1.0"

    # HDR source
    hdr_stream = VideoStreamInfo(
        index=0, codec="hevc", width=3840, height=1600,
        fps=23.976, bit_depth=10,
        transfer=TransferFunction.PQ,
        color_primaries=ColorPrimaries.BT2020,
    )
    project.hdr_source = SourceInfo(
        path=Path("/input/hdr.mkv"),
        container="matroska",
        video_streams=[hdr_stream],
        selected_stream=hdr_stream,
        hdr_metadata=HDRMetadata(format=HDRFormat.HDR10, max_cll=1000, max_fall=400),
        role=SourceRole.HDR_REFERENCE,
    )

    # OM source
    om_stream = VideoStreamInfo(
        index=0, codec="hevc", width=3840, height=2160,
        fps=23.976, bit_depth=8,
        transfer=TransferFunction.BT709,
        color_primaries=ColorPrimaries.BT709,
    )
    project.openmatte_source = SourceInfo(
        path=Path("/input/om.mkv"),
        container="matroska",
        video_streams=[om_stream],
        selected_stream=om_stream,
        hdr_metadata=HDRMetadata(),
        role=SourceRole.OPEN_MATTE,
    )

    # Sync
    project.sync_model = SyncModel(
        frame_offset=58,
        confidence=0.998,
        status=SyncStatus.LOCKED,
        frame_locked=True,
        drift_frames=0.0,
        offset_seconds=58 / 23.976,
    )

    # Geometry
    project.global_geometry = GeometryModel(
        scale_x=1.0,
        scale_y=1.0,
        offset_x=0.0,
        offset_y=280.0,
        overlap_bbox=[0.0, 280.0, 3840.0, 1880.0],
        confidence=0.97,
    )

    # Shots
    project.shots = [
        Shot(shot_id=1, hdr_start_frame=0, hdr_end_frame=100,
             om_start_frame=58, om_end_frame=158, duration_frames=100),
        Shot(shot_id=2, hdr_start_frame=100, hdr_end_frame=300,
             om_start_frame=158, om_end_frame=358, duration_frames=200),
    ]

    # Transforms
    project.transforms = [
        ShotTransform(
            shot_id=1,
            luminance_curve=[[0.0, 0.0], [0.5, 0.6], [1.0, 1.2]],
            exposure=1.1,
            contrast=1.05,
            saturation=0.98,
            confidence=0.95,
        ),
    ]

    project.analysis_complete = True
    project.ready_for_render = True

    project_path = tmp_path / "project.json"
    return project, project_path


class TestProjectSaveLoad:
    """Tests for project serialization round-trip."""

    def test_save_creates_file(self, sample_project: tuple[ProjectData, Path]) -> None:
        """save_project should create a JSON file."""
        project, path = sample_project
        save_project(project, path)
        assert path.exists()

        # Verify it's valid JSON
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == "1.0"

    def test_round_trip(self, sample_project: tuple[ProjectData, Path]) -> None:
        """Save then load should preserve data."""
        project, path = sample_project
        save_project(project, path)
        loaded = load_project(path)

        assert loaded.version == project.version
        assert loaded.sync_model.frame_offset == 58
        assert loaded.sync_model.status == SyncStatus.LOCKED
        assert loaded.sync_model.frame_locked is True
        assert loaded.global_geometry.offset_y == 280.0
        assert len(loaded.shots) == 2
        assert loaded.shots[0].shot_id == 1
        assert loaded.analysis_complete is True
        assert loaded.ready_for_render is True

    def test_hdr_source_preserved(self, sample_project: tuple[ProjectData, Path]) -> None:
        """HDR source metadata should survive round-trip."""
        project, path = sample_project
        save_project(project, path)
        loaded = load_project(path)

        assert loaded.hdr_source is not None
        assert loaded.hdr_source.selected_stream is not None
        assert loaded.hdr_source.selected_stream.width == 3840
        assert loaded.hdr_source.selected_stream.height == 1600
        assert loaded.hdr_source.hdr_metadata.format == HDRFormat.HDR10
        assert loaded.hdr_source.hdr_metadata.max_cll == 1000

    def test_transforms_preserved(self, sample_project: tuple[ProjectData, Path]) -> None:
        """Transforms should survive round-trip."""
        project, path = sample_project
        save_project(project, path)
        loaded = load_project(path)

        assert len(loaded.transforms) == 1
        t = loaded.transforms[0]
        assert t.shot_id == 1
        assert t.luminance_curve == [[0.0, 0.0], [0.5, 0.6], [1.0, 1.2]]
        assert t.exposure == pytest.approx(1.1)
        assert t.confidence == pytest.approx(0.95)

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        """Loading nonexistent file should raise ProjectError."""
        from auto_openmatte.core.exceptions import ProjectError
        with pytest.raises(ProjectError):
            load_project(tmp_path / "nonexistent.json")
