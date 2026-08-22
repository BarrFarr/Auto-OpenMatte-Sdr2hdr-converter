"""Replaceable contract for the existing native GPU RGB resample stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ResampleRequest:
    """Geometry and format contract for one device-side resample operation."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    source_format: str = "RGB32F"
    target_format: str = "RGB32F"
    method: str = "INTER_LINEAR"

    def __post_init__(self) -> None:
        for name in (
            "source_width",
            "source_height",
            "target_width",
            "target_height",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.source_format or not self.target_format:
            raise ValueError("source_format and target_format are required")
        if not self.method:
            raise ValueError("method is required")

    @property
    def source_geometry(self) -> tuple[int, int]:
        """Return source geometry as ``(width, height)``."""
        return int(self.source_width), int(self.source_height)

    @property
    def target_geometry(self) -> tuple[int, int]:
        """Return target geometry as ``(width, height)``."""
        return int(self.target_width), int(self.target_height)

    @property
    def resample_active(self) -> bool:
        """Whether this request requires an actual resize kernel."""
        return self.source_geometry != self.target_geometry

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly output contract."""
        return {
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "source_format": self.source_format,
            "target_format": self.target_format,
            "method": self.method,
            "resample_active": self.resample_active,
        }


class ResampleBackend(Protocol):
    """Device-side backend that can replace the current GPU implementation."""

    def process(self, rgb: Any, request: ResampleRequest, stream: Any) -> Any:
        """Process one device RGB frame on the supplied stream."""
        ...
