"""Data models for the Auto OpenMatte pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class HDRFormat(Enum):
    """Detected HDR format."""

    SDR = "SDR"
    HDR10 = "HDR10"
    HDR10_PLUS = "HDR10+"
    HLG = "HLG"
    DOLBY_VISION = "Dolby Vision"
    UNKNOWN = "Unknown"


class TransferFunction(Enum):
    """Transfer function / EOTF."""

    BT709 = "bt709"
    BT1886 = "bt1886"
    PQ = "smpte2084"
    HLG = "arib-std-b67"
    GAMMA22 = "gamma22"
    GAMMA28 = "gamma28"
    LINEAR = "linear"
    UNKNOWN = "unknown"


class ColorPrimaries(Enum):
    """Color primaries."""

    BT709 = "bt709"
    BT2020 = "bt2020"
    DCI_P3 = "smpte432"
    UNKNOWN = "unknown"


class FrameRateType(Enum):
    """Frame rate type."""

    CFR = "CFR"
    VFR = "VFR"


class SyncStatus(Enum):
    """Synchronization status."""

    LOCKED = "LOCKED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"


class SourceRole(Enum):
    """Role of a source in the pipeline."""

    HDR_REFERENCE = "HDR_REFERENCE"
    OPEN_MATTE = "OPEN_MATTE"
    UNDETERMINED = "UNDETERMINED"


@dataclass
class VideoStreamInfo:
    """Information about a single video stream."""

    index: int
    codec: str
    profile: str | None = None
    level: int | None = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    fps_rational: str = ""
    frame_rate_type: FrameRateType = FrameRateType.CFR
    duration_seconds: float = 0.0
    frame_count: int | None = None
    pix_fmt: str = ""
    bit_depth: int = 8
    chroma_subsampling: str = ""
    color_range: str = ""
    color_primaries: ColorPrimaries = ColorPrimaries.UNKNOWN
    transfer: TransferFunction = TransferFunction.UNKNOWN
    matrix_coefficients: str = ""
    is_default: bool = False


@dataclass
class HDRMetadata:
    """HDR-specific metadata."""

    format: HDRFormat = HDRFormat.SDR
    mastering_display: str | None = None
    max_cll: int | None = None
    max_fall: int | None = None
    # Raw side data from ffprobe
    side_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SourceInfo:
    """Complete information about a source file."""

    path: Path
    container: str = ""
    video_streams: list[VideoStreamInfo] = field(default_factory=list)
    selected_stream: VideoStreamInfo | None = None
    hdr_metadata: HDRMetadata = field(default_factory=HDRMetadata)
    role: SourceRole = SourceRole.UNDETERMINED


@dataclass
class SyncModel:
    """Synchronization model between HDR and Open Matte.

    Primary model: frame_offset (integer frames).
    Fallback for VFR: timestamp-based with offset_seconds.
    """

    # Primary: frame-based offset (OM_frame = HDR_frame + frame_offset)
    frame_offset: int = 0
    # Confidence of the synchronization (0.0 - 1.0)
    confidence: float = 0.0
    # Status
    status: SyncStatus = SyncStatus.NOT_RUN
    # Validation results
    mean_error_frames: float = 0.0
    max_error_frames: float = 0.0
    drift_frames: float = 0.0
    # Fallback: timestamp-based (for VFR)
    offset_seconds: float = 0.0
    # Whether frame-index mapping is valid
    frame_locked: bool = False
    # Method used
    method: str = "image_based_frame_offset"
    # Validation checkpoints: list of (hdr_frame, expected_om_frame, actual_similarity)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeometryModel:
    """Geometric relationship between HDR and Open Matte."""

    # Scale factors to map HDR coordinates into Open Matte space
    scale_x: float = 1.0
    scale_y: float = 1.0
    # Offset of HDR origin within Open Matte frame (in OM pixels)
    offset_x: float = 0.0
    offset_y: float = 0.0
    # Bounding box of overlapping region in OM coordinates [x1, y1, x2, y2]
    overlap_bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    # Confidence of the alignment
    confidence: float = 0.0
    # Whether this is a global or per-shot geometry
    is_global: bool = True
    # Shot ID if per-shot (None for global)
    shot_id: int | None = None


@dataclass
class Shot:
    """A detected shot (scene) on the HDR timeline."""

    shot_id: int
    # HDR frame range
    hdr_start_frame: int = 0
    hdr_end_frame: int = 0
    # Mapped Open Matte frame range
    om_start_frame: int = 0
    om_end_frame: int = 0
    # Duration in frames
    duration_frames: int = 0
    # Cut type
    cut_type: str = "hard"  # "hard", "fade", "dissolve"
    # Confidence of cut detection
    confidence: float = 1.0


@dataclass
class ShotTransform:
    """Per-shot transformation parameters."""

    shot_id: int
    # Luminance curve control points: list of (sdr_value, hdr_value) pairs
    luminance_curve: list[list[float]] = field(default_factory=list)
    # Exposure adjustment (multiplicative in linear space)
    exposure: float = 1.0
    # Contrast adjustment
    contrast: float = 1.0
    # 3x3 color correction matrix (row-major, in linear RGB)
    color_matrix: list[list[float]] = field(
        default_factory=lambda: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    # Saturation multiplier
    saturation: float = 1.0
    # Confidence
    confidence: float = 0.0
    # Geometry override (None = use global)
    geometry: GeometryModel | None = None


@dataclass
class ProjectData:
    """Complete project data, serialized to project.json."""

    # Version
    version: str = "1.0"
    # Sources
    hdr_source: SourceInfo | None = None
    openmatte_source: SourceInfo | None = None
    # Synchronization
    sync_model: SyncModel = field(default_factory=SyncModel)
    # Geometry
    global_geometry: GeometryModel = field(default_factory=GeometryModel)
    # Shots
    shots: list[Shot] = field(default_factory=list)
    # Per-shot transforms
    transforms: list[ShotTransform] = field(default_factory=list)
    # Pipeline status
    analysis_complete: bool = False
    ready_for_render: bool = False
    # Warnings and issues
    warnings: list[str] = field(default_factory=list)
