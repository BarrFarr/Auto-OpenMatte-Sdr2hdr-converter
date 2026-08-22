"""Neutral contracts shared by concrete media/compute backends.

The contracts in this module intentionally describe data and lifecycle rather
than a vendor API.  A frame may be backed by host memory, device memory, or an
external/shared allocation; callers must not assume a pointer, array library,
or codec-specific surface type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol


class MemoryDomain(str, Enum):
    """Where a frame allocation is resident."""

    HOST = "host"
    DEVICE = "device"
    UNIFIED = "unified"
    EXTERNAL = "external"


class Ownership(str, Enum):
    """Lifetime ownership of a frame allocation or view."""

    OWNED = "owned"
    BORROWED = "borrowed"
    SHARED = "shared"


@dataclass(frozen=True)
class FrameTimestamps:
    """Frame timing values expressed in the declared ``time_base``."""

    pts: int | float | None = None
    dts: int | float | None = None
    duration: int | float | None = None
    time_base: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.time_base is not None:
            numerator, denominator = (int(value) for value in self.time_base)
            if numerator <= 0 or denominator <= 0:
                raise ValueError("time_base must contain positive numerator and denominator")
            object.__setattr__(self, "time_base", (numerator, denominator))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pts": self.pts,
            "dts": self.dts,
            "duration": self.duration,
            "time_base": list(self.time_base) if self.time_base is not None else None,
        }


@dataclass
class FrameMemory:
    """Opaque memory handle used by a :class:`Frame`.

    ``payload`` is deliberately opaque.  A concrete backend may expose an
    array-like view, a set of planes, or another representation, but this
    contract does not import or require any vendor memory type.
    """

    payload: Any
    domain: MemoryDomain
    nbytes: int
    device: str | int | None = None
    ownership: Ownership = Ownership.BORROWED
    release_callback: Callable[[object | None], None] | None = None
    _released: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.domain = MemoryDomain(self.domain)
        self.ownership = Ownership(self.ownership)
        self.nbytes = int(self.nbytes)
        if self.nbytes < 0:
            raise ValueError("nbytes must be non-negative")

    @property
    def released(self) -> bool:
        return self._released

    def view(self) -> Any:
        """Return the opaque backend-native view without changing ownership."""
        if self._released:
            raise RuntimeError("frame memory has already been released")
        return self.payload

    def release(self, context: object | None = None) -> None:
        """Release the allocation once, optionally using an opaque context."""
        if self._released:
            return
        if self.release_callback is not None:
            self.release_callback(context)
        self._released = True

    def _invalidate(self) -> None:
        """Invalidate a borrowed view after its owning batch is released."""
        self._released = True


@dataclass
class Frame:
    """Portable description of one video frame and its memory lifetime."""

    width: int
    height: int
    pixel_format: str
    bit_depth: int
    color_range: str
    primaries: str
    transfer: str
    matrix: str
    timestamps: FrameTimestamps
    memory_domain: MemoryDomain
    backend: str
    device: str | int | None
    ownership: Ownership
    memory: FrameMemory
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.bit_depth = int(self.bit_depth)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame width and height must be positive")
        if self.bit_depth <= 0:
            raise ValueError("frame bit_depth must be positive")
        for name in (
            "pixel_format",
            "color_range",
            "primaries",
            "transfer",
            "matrix",
            "backend",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"frame {name} is required")
        self.memory_domain = MemoryDomain(self.memory_domain)
        self.ownership = Ownership(self.ownership)
        if self.memory.domain != self.memory_domain:
            raise ValueError("frame memory_domain must match its memory handle")

    @property
    def released(self) -> bool:
        return self.memory.released

    def view(self) -> Any:
        """Return the backend-native view for a compute operation."""
        return self.memory.view()

    def release(self, context: object | None = None) -> None:
        """Release the frame according to its backend-owned lifetime contract."""
        self.memory.release(context)

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "bit_depth": self.bit_depth,
            "color_range": self.color_range,
            "primaries": self.primaries,
            "transfer": self.transfer,
            "matrix": self.matrix,
            "timestamps": self.timestamps.as_dict(),
            "memory_domain": self.memory_domain.value,
            "backend": self.backend,
            "device": self.device,
            "ownership": self.ownership.value,
            "memory_bytes": self.memory.nbytes,
            "metadata": dict(self.metadata),
        }


@dataclass
class FrameBatch:
    """Neutral grouping for a synchronized multi-stream decode result.

    The V5 pipeline consumes an HDR/OM pair.  Single-stream backends can return
    a one-frame batch or expose :class:`Frame` directly; the common contract
    does not prescribe a particular number of streams.
    """

    frames: tuple[Frame, ...]
    sequence: int
    readiness: object | None = None
    release_callback: Callable[[object | None], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _released: bool = field(default=False, init=False, repr=False)
    _release_credit: Callable[[], None] | None = field(default=None, init=False, repr=False)
    _credit_returned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.frames = tuple(self.frames)
        if not self.frames:
            raise ValueError("FrameBatch requires at least one frame")
        self.sequence = int(self.sequence)

    @property
    def released(self) -> bool:
        return self._released

    def wait_ready(self, context: object | None = None) -> None:
        """Wait for producer work through a backend-supplied opaque callback."""
        waiter = self.metadata.get("wait_ready")
        if waiter is not None:
            waiter(context)

    def attach_release_credit(self, callback: Callable[[], None]) -> None:
        """Attach one bounded-capacity credit returned after release."""
        if self._released:
            raise RuntimeError("cannot attach a credit after batch release")
        self._release_credit = callback

    def release(self, context: object | None = None) -> None:
        if self._released:
            return

        # A batch-level callback owns all member lifetimes.  Without one,
        # release each member through its own frame contract instead.
        first_error: BaseException | None = None
        if self.release_callback is not None:
            try:
                self.release_callback(context)
            except BaseException as exc:
                first_error = exc
        else:
            for frame in self.frames:
                try:
                    frame.release(context)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    elif hasattr(first_error, "add_note"):
                        first_error.add_note(
                            f"frame release also failed: {type(exc).__name__}: {exc}"
                        )
        if first_error is not None:
            # Keep the batch retryable and do not return capacity while a
            # backend-owned member may still be in flight.
            raise first_error

        for frame in self.frames:
            frame.memory._invalidate()
        if self._release_credit is not None and not self._credit_returned:
            self._release_credit()
            self._credit_returned = True
        self._released = True


@dataclass(frozen=True)
class BackendCapabilities:
    """Declared capabilities of one concrete backend implementation."""

    backend: str
    supported_codecs: tuple[str, ...] = ()
    supported_pixel_formats: tuple[str, ...] = ()
    bit_depths: tuple[int, ...] = ()
    hardware_decode: bool = False
    hardware_encode: bool = False
    resampling: bool = False
    zero_copy: bool = False
    device_memory_type: str = MemoryDomain.HOST.value

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend name is required")
        object.__setattr__(self, "supported_codecs", tuple(str(value) for value in self.supported_codecs))
        object.__setattr__(self, "supported_pixel_formats", tuple(str(value) for value in self.supported_pixel_formats))
        object.__setattr__(self, "bit_depths", tuple(sorted({int(value) for value in self.bit_depths})))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "supported_codecs": list(self.supported_codecs),
            "supported_pixel_formats": list(self.supported_pixel_formats),
            "bit_depths": list(self.bit_depths),
            "hardware_decode": bool(self.hardware_decode),
            "hardware_encode": bool(self.hardware_encode),
            "resampling": bool(self.resampling),
            "zero_copy": bool(self.zero_copy),
            "device_memory_type": self.device_memory_type,
        }


class ComputeBackend(Protocol):
    """Numeric operations required by fitting and rendering."""

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def xp(self) -> Any:
        ...

    def asarray(self, value: Any) -> Any:
        ...

    def tohost(self, value: Any) -> Any:
        ...

    def blur(self, image: Any, sigma: float) -> Any:
        ...

    def luminance(self, rgb: Any) -> Any:
        ...

    def matrix(self, transform: Any, rgb: Any) -> Any:
        ...

    def pq_oetf(self, nits: Any) -> Any:
        ...

    def pq_eotf(self, signal: Any) -> Any:
        ...

    def to_ictcp(self, rgb_nits: Any) -> Any:
        ...

    def from_ictcp(self, ictcp: Any) -> Any:
        ...

    def to_pq16(self, normalized: Any, out: Any = None, stream: object = None, blocking: bool = True) -> Any:
        ...

    def percentile(self, values: Any, q: float) -> float:
        ...


class DecoderBackend(Protocol):
    """Decode lifecycle for one stream or a synchronized frame batch."""

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    @property
    def count(self) -> int:
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        ...

    @property
    def lifecycle(self) -> dict[str, Any]:
        ...

    def open(self) -> None:
        ...

    def read(self) -> Frame | FrameBatch | None:
        ...

    def close(self) -> None:
        ...


class ResampleBackend(Protocol):
    """Replaceable device- or host-memory resampling operation."""

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def process(self, frame: Any, request: "ResampleRequest", stream: object = None) -> Any:
        ...


@dataclass(frozen=True)
class ResampleRequest:
    """Geometry and format contract for one resample operation."""

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
        return int(self.source_width), int(self.source_height)

    @property
    def target_geometry(self) -> tuple[int, int]:
        return int(self.target_width), int(self.target_height)

    @property
    def resample_active(self) -> bool:
        return self.source_geometry != self.target_geometry

    def as_dict(self) -> dict[str, Any]:
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


class EncoderBackend(Protocol):
    """Ordered encoder lifecycle consuming backend-neutral frames."""

    @property
    def capabilities(self) -> BackendCapabilities:
        ...

    def open(self) -> None:
        ...

    def write(self, frame: Frame, pts: int) -> None:
        ...

    def close(self) -> None:
        ...

    def abort(self) -> None:
        ...


__all__ = [
    "BackendCapabilities",
    "ComputeBackend",
    "DecoderBackend",
    "EncoderBackend",
    "Frame",
    "FrameBatch",
    "FrameMemory",
    "FrameTimestamps",
    "MemoryDomain",
    "Ownership",
    "ResampleBackend",
    "ResampleRequest",
]
