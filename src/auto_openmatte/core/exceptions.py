"""Custom exceptions for the Auto Open-Matte pipeline."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""


class SyncFailedError(PipelineError):
    """Raised when temporal synchronization fails."""


class LowConfidenceError(PipelineError):
    """Raised when a pipeline step produces results below confidence threshold."""


class UnsupportedHDRFormatError(PipelineError):
    """Raised when an unsupported HDR format is encountered."""


class InvalidSourceError(PipelineError):
    """Raised when a source video is invalid or cannot be processed."""
