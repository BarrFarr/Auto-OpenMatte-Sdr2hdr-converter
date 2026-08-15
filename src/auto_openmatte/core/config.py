"""Pipeline configuration constants and defaults."""

# Synchronization
SYNC_SEARCH_RANGE: int = 120
SYNC_CONFIDENCE_THRESHOLD: float = 0.95

# Sampling
SAMPLES_PER_SHOT: int = 30

# Confidence thresholds
ALIGNMENT_CONFIDENCE_THRESHOLD: float = 0.95
COLOR_CONFIDENCE_THRESHOLD: float = 0.90

# Composition
BLEND_WIDTH_DEFAULT: int = 4

# Proxy generation
PROXY_WIDTH: int = 480

# Supported HDR formats
SUPPORTED_HDR_FORMATS: list[str] = ["HDR10", "HLG"]
