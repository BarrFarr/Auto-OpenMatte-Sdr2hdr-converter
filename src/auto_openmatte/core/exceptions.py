"""Custom exception hierarchy for Auto OpenMatte."""


class AutoOpenMatteError(Exception):
    """Base exception for all Auto OpenMatte errors."""


class InspectionError(AutoOpenMatteError):
    """Error during source inspection / ffprobe."""


class HDRDetectionError(AutoOpenMatteError):
    """Cannot determine HDR status of a source."""


class StreamSelectionError(AutoOpenMatteError):
    """Cannot select appropriate video stream."""


class SynchronizationError(AutoOpenMatteError):
    """Synchronization failed or confidence too low."""


class SyncDriftError(SynchronizationError):
    """Frame offset drifts across the timeline."""


class ShotDetectionError(AutoOpenMatteError):
    """Error during shot boundary detection."""


class GeometryError(AutoOpenMatteError):
    """Error during geometric alignment."""


class LuminanceError(AutoOpenMatteError):
    """Error during luminance mapping estimation."""


class ColorError(AutoOpenMatteError):
    """Error during color correction estimation."""


class CompositionError(AutoOpenMatteError):
    """Error during final composition."""


class RenderError(AutoOpenMatteError):
    """Error during video rendering."""


class ProjectError(AutoOpenMatteError):
    """Error reading/writing project file."""


class UnsupportedHDRError(AutoOpenMatteError):
    """Detected HDR format is not supported by the current processing pipeline."""
