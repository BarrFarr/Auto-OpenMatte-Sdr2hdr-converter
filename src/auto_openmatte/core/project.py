"""Project file I/O — serialize/deserialize ProjectData to JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from auto_openmatte.core.exceptions import ProjectError
from auto_openmatte.core.models import (
    ColorPrimaries,
    FrameRateType,
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


def _serialize_path(obj: Any) -> Any:
    """Custom serialization for non-JSON types."""
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_project(project: ProjectData, path: Path) -> None:
    """Save project data to JSON file."""
    try:
        data = asdict(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_serialize_path)
    except (OSError, TypeError) as e:
        raise ProjectError(f"Failed to save project to {path}: {e}") from e


def _rebuild_stream(data: dict[str, Any]) -> VideoStreamInfo:
    """Rebuild VideoStreamInfo from dict."""
    data["color_primaries"] = ColorPrimaries(data.get("color_primaries", "unknown"))
    data["transfer"] = TransferFunction(data.get("transfer", "unknown"))
    data["frame_rate_type"] = FrameRateType(data.get("frame_rate_type", "CFR"))
    return VideoStreamInfo(**data)


def _rebuild_source(data: dict[str, Any] | None) -> SourceInfo | None:
    """Rebuild SourceInfo from dict."""
    if data is None:
        return None
    source = SourceInfo(path=Path(data["path"]))
    source.container = data.get("container", "")
    source.role = SourceRole(data.get("role", "UNDETERMINED"))

    streams_data = data.get("video_streams", [])
    source.video_streams = [_rebuild_stream(s) for s in streams_data]

    sel = data.get("selected_stream")
    source.selected_stream = _rebuild_stream(sel) if sel else None

    hdr = data.get("hdr_metadata", {})
    source.hdr_metadata = HDRMetadata(
        format=HDRFormat(hdr.get("format", "SDR")),
        mastering_display=hdr.get("mastering_display"),
        max_cll=hdr.get("max_cll"),
        max_fall=hdr.get("max_fall"),
        side_data=hdr.get("side_data", []),
    )
    return source


def _rebuild_sync(data: dict[str, Any]) -> SyncModel:
    """Rebuild SyncModel from dict."""
    return SyncModel(
        frame_offset=data.get("frame_offset", 0),
        confidence=data.get("confidence", 0.0),
        status=SyncStatus(data.get("status", "NOT_RUN")),
        mean_error_frames=data.get("mean_error_frames", 0.0),
        max_error_frames=data.get("max_error_frames", 0.0),
        drift_frames=data.get("drift_frames", 0.0),
        offset_seconds=data.get("offset_seconds", 0.0),
        frame_locked=data.get("frame_locked", False),
        method=data.get("method", "image_based_frame_offset"),
        checkpoints=data.get("checkpoints", []),
    )


def _rebuild_geometry(data: dict[str, Any]) -> GeometryModel:
    """Rebuild GeometryModel from dict."""
    return GeometryModel(
        scale_x=data.get("scale_x", 1.0),
        scale_y=data.get("scale_y", 1.0),
        offset_x=data.get("offset_x", 0.0),
        offset_y=data.get("offset_y", 0.0),
        overlap_bbox=data.get("overlap_bbox", [0.0, 0.0, 0.0, 0.0]),
        confidence=data.get("confidence", 0.0),
        is_global=data.get("is_global", True),
        shot_id=data.get("shot_id"),
    )


def _rebuild_shot(data: dict[str, Any]) -> Shot:
    """Rebuild Shot from dict."""
    return Shot(
        shot_id=data["shot_id"],
        hdr_start_frame=data.get("hdr_start_frame", 0),
        hdr_end_frame=data.get("hdr_end_frame", 0),
        om_start_frame=data.get("om_start_frame", 0),
        om_end_frame=data.get("om_end_frame", 0),
        duration_frames=data.get("duration_frames", 0),
        cut_type=data.get("cut_type", "hard"),
        confidence=data.get("confidence", 1.0),
    )


def _rebuild_transform(data: dict[str, Any]) -> ShotTransform:
    """Rebuild ShotTransform from dict."""
    geom_data = data.get("geometry")
    geometry = _rebuild_geometry(geom_data) if geom_data else None
    return ShotTransform(
        shot_id=data["shot_id"],
        luminance_curve=data.get("luminance_curve", []),
        exposure=data.get("exposure", 1.0),
        contrast=data.get("contrast", 1.0),
        color_matrix=data.get("color_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        saturation=data.get("saturation", 1.0),
        confidence=data.get("confidence", 0.0),
        geometry=geometry,
    )


def load_project(path: Path) -> ProjectData:
    """Load project data from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ProjectError(f"Failed to load project from {path}: {e}") from e

    project = ProjectData()
    project.version = data.get("version", "1.0")
    project.hdr_source = _rebuild_source(data.get("hdr_source"))
    project.openmatte_source = _rebuild_source(data.get("openmatte_source"))
    project.sync_model = _rebuild_sync(data.get("sync_model", {}))
    project.global_geometry = _rebuild_geometry(data.get("global_geometry", {}))
    project.shots = [_rebuild_shot(s) for s in data.get("shots", [])]
    project.transforms = [_rebuild_transform(t) for t in data.get("transforms", [])]
    project.analysis_complete = data.get("analysis_complete", False)
    project.ready_for_render = data.get("ready_for_render", False)
    project.warnings = data.get("warnings", [])
    return project
