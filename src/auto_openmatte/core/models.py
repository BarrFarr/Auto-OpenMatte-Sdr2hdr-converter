"""Data models for the Auto Open-Matte pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SourceInfo:
    """Video source metadata extracted from ffprobe/mediainfo."""

    path: str
    container: str
    codec: str
    profile: str
    level: str
    width: int
    height: int
    fps: float
    nominal_fps: str
    vfr_cfr: Literal["VFR", "CFR", "unknown"]
    duration: float
    frame_count: int
    pixel_format: str
    bit_depth: int
    chroma_subsampling: str
    color_range: str
    color_primaries: str
    transfer_characteristics: str
    matrix_coefficients: str
    # HDR metadata (may be empty/None for SDR sources)
    mastering_display: str
    max_cll: int
    max_fall: int
    # Legacy aliases for backward compatibility
    aspect_ratio: str = ""
    color_space: str = ""
    color_transfer: str = ""


@dataclass
class HDRMetadata:
    """HDR-specific metadata for a source."""

    mastering_display: str
    max_cll: int
    max_fall: int
    hdr_format: Literal["HDR10", "HLG"]


@dataclass
class SyncResult:
    """Result of temporal synchronization between HDR and Open Matte sources."""

    method: str
    offset: float
    scale: float
    confidence: float
    mean_error: float
    max_error: float
    status: Literal["success", "failed", "low_confidence"]
    segments: list[dict[str, float]] = field(default_factory=list)


@dataclass
class ShotInfo:
    """Information about a detected shot/scene boundary."""

    shot_id: int
    start: float
    end: float
    duration: float
    transition_type: Literal["cut", "dissolve", "fade", "unknown"]


@dataclass
class GeometryResult:
    """Result of geometric alignment between HDR and Open Matte frames."""

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    confidence: float


@dataclass
class LuminanceMappingResult:
    """Result of luminance curve fitting between sources."""

    curve_points: list[tuple[float, float]]
    method: str
    residual_error: float
    confidence: float
    is_monotonic: bool


@dataclass
class ColorTransformResult:
    """Result of color transform analysis between sources."""

    whitepoint_shift: tuple[float, float]
    color_matrix: list[list[float]]
    saturation_factor: float
    confidence: float


@dataclass
class ShotTransform:
    """Complete transform parameters for a single shot."""

    shot_id: int
    hdr_start: float
    hdr_end: float
    om_start: float
    om_end: float
    geometry: GeometryResult
    luminance_curve: LuminanceMappingResult
    color_transform: ColorTransformResult
    confidence: float


@dataclass
class ProjectFile:
    """Top-level project file storing all analysis state."""

    sources: dict[str, SourceInfo]
    sync_result: SyncResult | None
    shots: list[ShotInfo]
    transforms: list[ShotTransform]
    status: Literal["initialized", "analyzed", "previewed", "rendered"]
