"""Global configuration and default thresholds."""

from dataclasses import dataclass, field


@dataclass
class SyncConfig:
    """Configuration for synchronization stage."""

    # Search range in seconds for global offset detection
    search_range_seconds: float = 120.0
    # Number of sample points for offset validation across the film
    validation_points: int = 20
    # Minimum confidence to accept synchronization
    min_confidence: float = 0.95
    # Maximum allowed drift in frames before reporting DRIFT
    max_drift_frames: float = 0.5
    # Proxy width for sync analysis
    proxy_width: int = 480


@dataclass
class ShotConfig:
    """Configuration for shot detection."""

    # Threshold multiplier for adaptive shot detection (mean + k*std)
    threshold_multiplier: float = 3.0
    # Minimum shot duration in frames
    min_shot_frames: int = 6
    # Whether to detect fades and dissolves
    detect_transitions: bool = True


@dataclass
class GeometryConfig:
    """Configuration for geometric alignment."""

    # Minimum confidence for geometry alignment
    min_confidence: float = 0.95
    # Number of feature points to detect
    max_features: int = 5000
    # Whether to allow per-shot geometry (vs global only)
    allow_per_shot: bool = True
    # Tolerance for geometry stability check (pixels)
    stability_tolerance: float = 2.0


@dataclass
class ColorConfig:
    """Configuration for luminance/color matching."""

    # Number of sample frames per shot for analysis
    samples_per_shot: int = 30
    # Minimum confidence for color transform
    min_confidence: float = 0.90
    # Percentile range for outlier rejection
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    # Number of bins for luminance curve estimation
    luminance_bins: int = 512
    # Feather width in pixels for HDR/OM boundary blend
    feather_width: int = 4


@dataclass
class RenderConfig:
    """Configuration for final render."""

    # Output codec
    codec: str = "libx265"
    # CRF value
    crf: int = 16
    # Pixel format
    pix_fmt: str = "yuv420p10le"
    # Additional FFmpeg encoder params
    encoder_params: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration combining all sub-configs."""

    sync: SyncConfig = field(default_factory=SyncConfig)
    shots: ShotConfig = field(default_factory=ShotConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    # Debug flags
    debug: bool = False
    debug_sync: bool = False
    # Output directory for analysis artifacts
    output_dir: str = "./project"
