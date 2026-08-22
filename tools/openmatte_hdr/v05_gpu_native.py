"""V5 phase-1 native CUDA source and frozen V4 math adapter.

The native bridge owns FFmpeg/NVDEC frames and a bounded CUDA working-RGB
ring.  This module exposes those device buffers as CuPy arrays without copying
RGB through host memory.  The V4 fitting equations remain inherited unchanged;
only the source adapter and the temporary host output boundary are V5-specific.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import v05_streaming as v05

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_BRIDGE = WORKSPACE / "dev" / "v05-native" / "v5_gpu_bridge.dll"
DEFAULT_CUDA_BIN = Path(
    os.environ.get(
        "CUDA_PATH",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9",
    )
) / "bin"

HDR_MODE = 0
SDR_MODE = 1

_DIAGNOSTIC_LOCK = threading.Lock()
_DIAGNOSTIC_EVENTS: list[dict[str, Any]] = []


def diagnostic_marker(marker: str, **fields: Any) -> dict[str, Any]:
    """Emit a flushed, machine-readable phase marker without changing the pipeline."""

    event = {
        "marker": str(marker),
        "monotonic_seconds": time.perf_counter(),
        "thread": threading.current_thread().name,
        **fields,
    }
    with _DIAGNOSTIC_LOCK:
        _DIAGNOSTIC_EVENTS.append(dict(event))
    print(
        f"[V05_MARKER] {marker} "
        f"{json.dumps(event, default=str, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )
    return event


def diagnostic_exception(stage: str, error: BaseException, sequence: int | None = None) -> dict[str, Any]:
    """Record and print the complete traceback at the first failing stage."""

    traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return diagnostic_marker(
        "FIRST_FAILURE",
        stage=str(stage),
        sequence=None if sequence is None else int(sequence),
        exception_type=type(error).__name__,
        exception=str(error),
        traceback=traceback_text,
    )


def diagnostic_events() -> list[dict[str, Any]]:
    with _DIAGNOSTIC_LOCK:
        return [dict(event) for event in _DIAGNOSTIC_EVENTS]


class NativeBridgeError(v05.V05Error):
    """Raised when the native FFmpeg/CUDA bridge rejects an operation."""


class _V5GpuFrame(ctypes.Structure):
    _fields_ = [
        ("device_ptr", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("pitch_bytes", ctypes.c_int),
        ("slot", ctypes.c_int),
        ("software_format", ctypes.c_int),
        ("sequence", ctypes.c_uint64),
    ]


def _error_text(buffer: ctypes.Array[ctypes.c_char]) -> str:
    text = bytes(buffer.value).decode("utf-8", errors="replace").strip()
    return text or "native V5 bridge returned an unspecified error"


def _stream_pointer(stream: Any | None, cupy: Any) -> int:
    if stream is None:
        stream = cupy.cuda.get_current_stream()
    return int(getattr(stream, "ptr", stream))


def _fps_fraction(config: Any) -> tuple[int, int]:
    text = str(config.fps_text)
    try:
        value = Fraction(text)
    except ValueError:
        value = Fraction(float(config.fps)).limit_denominator(1000000)
    return int(value.numerator), int(value.denominator)


@dataclass
class NativeGpuFrame:
    """A borrowed CuPy view whose bridge slot must be released explicitly."""

    decoder: "NativeDecoder" = field(repr=False)
    array: Any
    sequence: int
    slot: int
    software_format: int
    released: bool = False

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    def release(self, stream: Any | None = None) -> None:
        if self.released:
            return
        self.decoder.release(self.slot, self.sequence, stream)
        self.released = True


@dataclass
class NativeGpuPair:
    """A paired HDR/OM CUDA frame retained until the render stream is done."""

    sequence: int
    hdr: NativeGpuFrame
    sdr: NativeGpuFrame
    ready_event: Any
    surface_credit: Any | None = field(default=None, repr=False)
    credit_returned: bool = field(default=False, repr=False)

    @property
    def fully_released(self) -> bool:
        return bool(self.hdr.released and self.sdr.released)

    def release(self, stream: Any | None = None) -> None:
        first_error: BaseException | None = None
        for name, frame in (("sdr", self.sdr), ("hdr", self.hdr)):
            try:
                frame.release(stream)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                elif hasattr(first_error, "add_note"):
                    first_error.add_note(
                        f"{name} frame release also failed: {type(exc).__name__}: {exc}"
                    )
        if first_error is not None:
            raise first_error

    def return_surface_credit(self) -> bool:
        if self.surface_credit is None or self.credit_returned:
            return False
        self.surface_credit.release()
        self.credit_returned = True
        return True


class NativeDecoder:
    """ctypes owner for one FFmpeg decoder and its CUDA working-RGB ring."""

    def __init__(
        self,
        path: Path,
        decoder_name: str,
        source_mode: int,
        start_frame: int,
        fps_num: int,
        fps_den: int,
        *,
        ffmpeg_bin: Path,
        bridge_path: Path = DEFAULT_BRIDGE,
        device_index: int = 0,
        ring_size: int = 4,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        try:
            import cupy
        except Exception as exc:  # pragma: no cover - environment-specific
            raise NativeBridgeError(f"CuPy is required for V5 native decode: {exc}") from exc

        self.cupy = cupy
        self.path = Path(path)
        self.decoder_name = str(decoder_name)
        self.source_mode = int(source_mode)
        self.device_index = int(device_index)
        self._dll_handles: list[Any] = []
        self._closed = False
        self.ring_size = int(ring_size)
        self._outstanding: dict[int, int] = {}
        self._acquired_count = 0
        self._released_count = 0
        self._max_in_flight = 0
        self._error_capacity = 4096
        self._error = ctypes.create_string_buffer(self._error_capacity)

        bridge_path = Path(bridge_path).resolve()
        ffmpeg_bin = Path(ffmpeg_bin).resolve()
        if not bridge_path.is_file():
            raise NativeBridgeError(f"V5 native bridge DLL is missing: {bridge_path}")
        for directory in (ffmpeg_bin, DEFAULT_CUDA_BIN):
            if directory.is_dir() and hasattr(os, "add_dll_directory"):
                self._dll_handles.append(os.add_dll_directory(str(directory)))

        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        self._dll = loader(str(bridge_path))
        self._configure_abi()
        self._handle = self._dll.v5_decoder_open(
            str(self.path).encode("utf-8"),
            self.decoder_name.encode("ascii"),
            self.source_mode,
            int(start_frame),
            int(fps_num),
            int(fps_den),
            self.device_index,
            int(ring_size),
            self._error,
            self._error_capacity,
        )
        if not self._handle:
            raise NativeBridgeError(
                f"v5_decoder_open({self.decoder_name}, {self.path}) failed: {_error_text(self._error)}"
            )
        self.resample_active = False
        self.source_width = int(self._dll.v5_decoder_width(self._handle))
        self.source_height = int(self._dll.v5_decoder_height(self._handle))
        if target_size is not None:
            self._apply_target_size(target_size)
        self.width = int(self._dll.v5_decoder_width(self._handle))
        self.height = int(self._dll.v5_decoder_height(self._handle))
        self.software_format = int(self._dll.v5_decoder_software_format(self._handle))
        self.frame_bytes = int(self._dll.v5_decoder_frame_bytes(self._handle))
        diagnostic_marker(
            "HDR_DECODER_OK" if self.source_mode == HDR_MODE else "OM_DECODER_OK",
            path=str(self.path),
            decoder=self.decoder_name,
            source_mode="HDR" if self.source_mode == HDR_MODE else "OM",
            device_index=self.device_index,
            width=self.width,
            height=self.height,
            software_format_at_open=self.software_format,
            frame_bytes=self.frame_bytes,
        )

    def _apply_target_size(self, target_size: tuple[int, int]) -> None:
        """Request the emitted working-RGB geometry from the bridge.

        Requesting the decoded size is a no-op inside the bridge, so callers can
        pass the geometry unconditionally without changing behaviour for sources
        that already match their target.
        """

        width, height = (int(value) for value in target_size)
        if (width, height) == (self.source_width, self.source_height):
            self.resample_active = False
            return
        entry = getattr(self._dll, "v5_decoder_set_target_size", None)
        if entry is None:
            raise NativeBridgeError(
                "this bridge build cannot resample the HDR source; decoded "
                f"{self.source_width}x{self.source_height} but {width}x{height} was requested"
            )
        entry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
        ]
        entry.restype = ctypes.c_int
        status = int(entry(self._handle, width, height, self._error, self._error_capacity))
        if status != 0:
            raise NativeBridgeError(
                f"v5_decoder_set_target_size({width}x{height}) failed: {_error_text(self._error)}"
            )
        probe = getattr(self._dll, "v5_decoder_resample_active", None)
        if probe is not None:
            probe.argtypes = [ctypes.c_void_p]
            probe.restype = ctypes.c_int
            self.resample_active = bool(int(probe(self._handle)))
        else:
            self.resample_active = True

    def _configure_abi(self) -> None:
        c_char_pointer = ctypes.POINTER(ctypes.c_char)
        self._dll.v5_decoder_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            c_char_pointer,
            ctypes.c_int,
        ]
        self._dll.v5_decoder_open.restype = ctypes.c_void_p
        self._dll.v5_decoder_next.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(_V5GpuFrame),
            c_char_pointer,
            ctypes.c_int,
        ]
        self._dll.v5_decoder_next.restype = ctypes.c_int
        self._dll.v5_decoder_release.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            c_char_pointer,
            ctypes.c_int,
        ]
        self._dll.v5_decoder_release.restype = ctypes.c_int
        for name in (
            "v5_decoder_width",
            "v5_decoder_height",
            "v5_decoder_software_format",
        ):
            getattr(self._dll, name).argtypes = [ctypes.c_void_p]
            getattr(self._dll, name).restype = ctypes.c_int
        self._dll.v5_decoder_frame_bytes.argtypes = [ctypes.c_void_p]
        self._dll.v5_decoder_frame_bytes.restype = ctypes.c_uint64
        self._dll.v5_decoder_close.argtypes = [ctypes.c_void_p]
        self._dll.v5_decoder_close.restype = None

    def _ensure_open(self) -> None:
        if self._closed or not self._handle:
            raise NativeBridgeError("native V5 decoder is closed")

    def next(self, stream: Any | None = None) -> NativeGpuFrame | None:
        self._ensure_open()
        record = _V5GpuFrame()
        result = int(
            self._dll.v5_decoder_next(
                self._handle,
                ctypes.c_uint64(_stream_pointer(stream, self.cupy)),
                ctypes.byref(record),
                self._error,
                self._error_capacity,
            )
        )
        if result == 0:
            return None
        if result < 0:
            raise NativeBridgeError(f"v5_decoder_next failed: {_error_text(self._error)}")
        slot = int(record.slot)
        sequence = int(record.sequence)
        self._outstanding[slot] = sequence
        self._acquired_count += 1
        self._max_in_flight = max(self._max_in_flight, len(self._outstanding))
        try:
            if record.device_ptr == 0 or record.bytes == 0:
                raise NativeBridgeError("native bridge returned an empty CUDA working frame")
            if record.width != self.width or record.height != self.height:
                raise NativeBridgeError(
                    f"native frame geometry changed from {self.width}x{self.height} "
                    f"to {record.width}x{record.height}"
                )
            memory = self.cupy.cuda.UnownedMemory(
                int(record.device_ptr),
                int(record.bytes),
                self,
                device_id=self.device_index,
            )
            pointer = self.cupy.cuda.MemoryPointer(memory, 0)
            array = self.cupy.ndarray(
                (int(record.height), int(record.width), 3),
                dtype=self.cupy.float32,
                memptr=pointer,
            )
            return NativeGpuFrame(
                decoder=self,
                array=array,
                sequence=sequence,
                slot=slot,
                software_format=int(record.software_format),
            )
        except BaseException as original_error:
            try:
                self.release(slot, sequence, stream)
            except BaseException as release_error:
                if hasattr(original_error, "add_note"):
                    original_error.add_note(
                        "Native decoder slot cleanup failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )
            raise

    def release(self, slot: int, sequence: int, stream: Any | None = None) -> None:
        self._ensure_open()
        slot = int(slot)
        sequence = int(sequence)
        expected_sequence = self._outstanding.get(slot)
        if expected_sequence is None:
            raise NativeBridgeError(f"native decoder slot {slot} is not owned by Python")
        if expected_sequence != sequence:
            raise NativeBridgeError(
                f"native decoder ownership mismatch for slot {slot}: "
                f"expected sequence {expected_sequence}, got {sequence}"
            )
        result = int(
            self._dll.v5_decoder_release(
                self._handle,
                slot,
                ctypes.c_uint64(sequence),
                ctypes.c_uint64(_stream_pointer(stream, self.cupy)),
                self._error,
                self._error_capacity,
            )
        )
        if result != 0:
            raise NativeBridgeError(f"v5_decoder_release failed: {_error_text(self._error)}")
        del self._outstanding[slot]
        self._released_count += 1

    @property
    def lifecycle(self) -> dict[str, int]:
        return {
            "ring_size": int(self.ring_size),
            "acquired": int(self._acquired_count),
            "released": int(self._released_count),
            "in_flight": int(len(self._outstanding)),
            "max_in_flight": int(self._max_in_flight),
        }

    def close(self) -> None:
        if self._closed:
            return
        if self._outstanding:
            raise NativeBridgeError(
                f"cannot close native decoder with unreleased slots: {sorted(self._outstanding)}"
            )
        self._dll.v5_decoder_close(self._handle)
        self._handle = None
        self._closed = True
        self._dll_handles.clear()

    def __enter__(self) -> "NativeDecoder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class NativeFrameSource:
    """Sequential HDR/Open Matte pair source backed only by CUDA working RGB."""

    def __init__(
        self,
        config: Any,
        count: int,
        *,
        bridge_path: Path = DEFAULT_BRIDGE,
        ring_size: int = 4,
        device_index: int = 0,
        hdr_decoder: str = "hevc_cuvid",
        om_decoder: str = "h264_cuvid",
    ) -> None:
        self.config = config
        self.count = int(count)
        self.bridge_path = Path(bridge_path)
        self.ring_size = int(ring_size)
        self.device_index = int(device_index)
        self.hdr_decoder = str(hdr_decoder)
        self.om_decoder = str(om_decoder)
        self._last_pair_metadata: dict[str, Any] = {}
        self._pair_hdr_decoder: NativeDecoder | None = None
        self._pair_om_decoder: NativeDecoder | None = None
        self._pair_decode_stream: Any | None = None
        self._pair_sequence = 0
        self._lifecycle_passes: list[dict[str, Any]] = []

    def hdr_target_size(self) -> tuple[int, int]:
        """Working-RGB geometry the HDR master must have for compositing.

        The HDR master fills the overlap rectangle, so that is the geometry the
        renderer and the frozen fitter both expect. When the decoded master
        already has this size, as with the scale-1.0 BR2049 contract, the bridge
        treats the request as a no-op.
        """

        x1, y1, x2, y2 = (int(value) for value in self.config.overlap)
        return x2 - x1, y2 - y1

    def open_pairs(self) -> None:
        """Open both native decoders for bounded decode/render overlap."""

        if self._pair_hdr_decoder is not None or self._pair_om_decoder is not None:
            return
        import cupy

        fps_num, fps_den = _fps_fraction(self.config)
        self._pair_decode_stream = cupy.cuda.Stream(non_blocking=True)
        try:
            self._pair_hdr_decoder = NativeDecoder(
                self.config.hdr_source,
                self.hdr_decoder,
                HDR_MODE,
                int(self.config.shot_start),
                fps_num,
                fps_den,
                ffmpeg_bin=self.config.ffmpeg.parent,
                bridge_path=self.bridge_path,
                device_index=self.device_index,
                ring_size=self.ring_size,
                target_size=self.hdr_target_size(),
            )
            self._pair_om_decoder = NativeDecoder(
                self.config.om_source,
                self.om_decoder,
                SDR_MODE,
                int(self.config.shot_start + self.config.offset_frames),
                fps_num,
                fps_den,
                ffmpeg_bin=self.config.ffmpeg.parent,
                bridge_path=self.bridge_path,
                device_index=self.device_index,
                ring_size=self.ring_size,
            )
            self._validate_geometry(self._pair_hdr_decoder, self._pair_om_decoder)
            self._pair_sequence = 0
            self._last_pair_metadata.update(
                {
                    "seek_mode": "libavformat PTS fast seek with backward keyframe and timestamp drop",
                    "hdr_start_pts_seconds": float(Fraction(self.config.shot_start * fps_den, fps_num)),
                    "om_start_pts_seconds": float(Fraction((self.config.shot_start + self.config.offset_frames) * fps_den, fps_num)),
                    "frame_count": self.count,
                }
            )
        except BaseException as original_error:
            try:
                self.close_pairs()
            except BaseException as cleanup_error:
                diagnostic_exception("open_pairs_cleanup", cleanup_error)
                if hasattr(original_error, "add_note"):
                    original_error.add_note(
                        f"open_pairs cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    def next_pair(self) -> NativeGpuPair | None:
        """Return one retained CUDA pair; caller must release it after GPU use."""

        self.open_pairs()
        if self._pair_hdr_decoder is None or self._pair_om_decoder is None or self._pair_decode_stream is None:
            raise NativeBridgeError("native pair source is not open")
        hdr_frame = self._pair_hdr_decoder.next(self._pair_decode_stream)
        if hdr_frame is None:
            return None
        try:
            om_frame = self._pair_om_decoder.next(self._pair_decode_stream)
        except BaseException as original_error:
            try:
                hdr_frame.release(self._pair_decode_stream)
            except BaseException as release_error:
                diagnostic_exception("hdr_frame_release_after_om_failure", release_error, self._pair_sequence)
                if hasattr(original_error, "add_note"):
                    original_error.add_note(
                        f"HDR frame release failed: {type(release_error).__name__}: {release_error}"
                    )
            raise
        if om_frame is None:
            hdr_frame.release(self._pair_decode_stream)
            raise NativeBridgeError(f"Open Matte source ended early at pair {self._pair_sequence}")
        try:
            import cupy

            ready_event = cupy.cuda.Event()
            ready_event.record(self._pair_decode_stream)
        except BaseException as original_error:
            first_release_error: BaseException | None = None
            for name, frame in (("om", om_frame), ("hdr", hdr_frame)):
                try:
                    frame.release(self._pair_decode_stream)
                except BaseException as release_error:
                    if first_release_error is None:
                        first_release_error = release_error
                    elif hasattr(first_release_error, "add_note"):
                        first_release_error.add_note(
                            f"{name} frame release also failed: "
                            f"{type(release_error).__name__}: {release_error}"
                        )
            if first_release_error is not None and hasattr(original_error, "add_note"):
                original_error.add_note(
                    "Pair cleanup after ready-event failure failed: "
                    f"{type(first_release_error).__name__}: {first_release_error}"
                )
            raise
        if self._pair_sequence == 0:
            self._last_pair_metadata.update(
                {
                    "hdr_software_format": hdr_frame.software_format,
                    "om_software_format": om_frame.software_format,
                }
            )
            diagnostic_marker(
                "FIRST_FRAME_OK",
                stage="render_decode_pair",
                sequence=0,
                hdr_software_format=hdr_frame.software_format,
                om_software_format=om_frame.software_format,
                hdr_device_ptr=int(hdr_frame.array.data.ptr),
                om_device_ptr=int(om_frame.array.data.ptr),
            )
        pair = NativeGpuPair(self._pair_sequence, hdr_frame, om_frame, ready_event)
        self._pair_sequence += 1
        return pair

    def release_pair(self, pair: NativeGpuPair, stream: Any | None = None) -> None:
        pair.release(stream)

    def close_pairs(self) -> None:
        cleanup_error: BaseException | None = None
        if self._pair_decode_stream is not None:
            try:
                self._pair_decode_stream.synchronize()
            except BaseException as exc:
                diagnostic_exception("decode_stream_cleanup", exc)
                cleanup_error = exc
        for name, decoder in (("om_decoder_cleanup", self._pair_om_decoder), ("hdr_decoder_cleanup", self._pair_hdr_decoder)):
            if decoder is None:
                continue
            try:
                decoder.close()
            except BaseException as exc:
                diagnostic_exception(name, exc)
                if cleanup_error is None:
                    cleanup_error = exc
        self._pair_om_decoder = None
        self._pair_hdr_decoder = None
        self._pair_decode_stream = None
        if cleanup_error is not None:
            raise cleanup_error

    def iter_frames(
        self,
        profiler: v05.Profiler | None = None,
        include_hashes: bool = False,
        *,
        pass_name: str | None = None,
    ) -> Iterator[v05.DecodedFrame]:
        del include_hashes  # Native decode intentionally does not hash/copy RGB on host.
        pass_record: dict[str, Any] = {
            "name": str(pass_name or f"pass_{len(self._lifecycle_passes) + 1}"),
            "decoded_pairs": 0,
            "hdr": None,
            "om": None,
        }
        self._lifecycle_passes.append(pass_record)
        fps_num, fps_den = _fps_fraction(self.config)
        hdr_decoder: NativeDecoder | None = None
        om_decoder: NativeDecoder | None = None
        try:
            hdr_decoder = NativeDecoder(
                self.config.hdr_source,
                self.hdr_decoder,
                HDR_MODE,
                int(self.config.shot_start),
                fps_num,
                fps_den,
                ffmpeg_bin=self.config.ffmpeg.parent,
                bridge_path=self.bridge_path,
                device_index=self.device_index,
                ring_size=self.ring_size,
                target_size=self.hdr_target_size(),
            )
            om_decoder = NativeDecoder(
                self.config.om_source,
                self.om_decoder,
                SDR_MODE,
                int(self.config.shot_start + self.config.offset_frames),
                fps_num,
                fps_den,
                ffmpeg_bin=self.config.ffmpeg.parent,
                bridge_path=self.bridge_path,
                device_index=self.device_index,
                ring_size=self.ring_size,
            )
            self._validate_geometry(hdr_decoder, om_decoder)
            for sequence in range(self.count):
                started = time.perf_counter()
                hdr_frame: NativeGpuFrame | None = None
                om_frame: NativeGpuFrame | None = None
                try:
                    hdr_frame = hdr_decoder.next()
                    om_frame = om_decoder.next()
                    if hdr_frame is None or om_frame is None:
                        raise NativeBridgeError(
                            f"native source ended early at pair {sequence}"
                        )
                    if self._last_pair_metadata.get("hdr_software_format") is None:
                        self._last_pair_metadata.update(
                            {
                                "hdr_software_format": hdr_frame.software_format,
                                "om_software_format": om_frame.software_format,
                            }
                        )
                    if sequence == 0:
                        diagnostic_marker(
                            "FIRST_FRAME_OK",
                            stage="fit_decode_pair",
                            sequence=0,
                            hdr_software_format=hdr_frame.software_format,
                            om_software_format=om_frame.software_format,
                            hdr_device_ptr=int(hdr_frame.array.data.ptr),
                            om_device_ptr=int(om_frame.array.data.ptr),
                        )
                    if profiler is not None:
                        profiler.record("native_decode", time.perf_counter() - started)
                        profiler.frame({})
                    pass_record["decoded_pairs"] += 1
                    yield v05.DecodedFrame(
                        sequence=sequence,
                        hdr=hdr_frame.array,
                        sdr=om_frame.array,
                        hashes={},
                    )
                finally:
                    if om_frame is not None:
                        om_frame.release()
                    if hdr_frame is not None:
                        hdr_frame.release()
        finally:
            pass_record["hdr"] = None if hdr_decoder is None else hdr_decoder.lifecycle
            pass_record["om"] = None if om_decoder is None else om_decoder.lifecycle
            if hdr_decoder is not None:
                hdr_decoder.close()
            if om_decoder is not None:
                om_decoder.close()
            self._last_pair_metadata["lifecycle"] = self.lifecycle

    def _validate_geometry(
        self,
        hdr_decoder: NativeDecoder,
        om_decoder: NativeDecoder,
    ) -> None:
        hdr_width, hdr_height = (int(value) for value in self.config.hdr_size)
        om_width, om_height = (int(value) for value in self.config.om_size)
        target_width, target_height = self.hdr_target_size()
        decoded = {
            "hdr": [
                int(getattr(hdr_decoder, "source_width", hdr_decoder.width)),
                int(getattr(hdr_decoder, "source_height", hdr_decoder.height)),
            ],
            "om": [om_decoder.width, om_decoder.height],
        }
        decoded_expected = {
            "hdr": [hdr_width, hdr_height],
            "om": [om_width, om_height],
        }
        if decoded != decoded_expected:
            raise NativeBridgeError(
                "phase-1 native source requires profile geometry to match decoded "
                f"surfaces exactly: expected={decoded_expected}, actual={decoded}"
            )
        actual = {
            "hdr": [hdr_decoder.width, hdr_decoder.height],
            "om": [om_decoder.width, om_decoder.height],
        }
        expected = {
            "hdr": [target_width, target_height],
            "om": [om_width, om_height],
        }
        if actual != expected:
            raise NativeBridgeError(
                "native source must emit the HDR master at the overlap geometry: "
                f"expected={expected}, actual={actual}"
            )
        self._last_pair_metadata = {
            "hdr_decoder": self.hdr_decoder,
            "om_decoder": self.om_decoder,
            "hdr_expected_software_format": "P010LE",
            "om_expected_software_format": "NV12",
            "hdr_software_format": None,
            "om_software_format": None,
            "hdr_geometry": actual["hdr"],
            "om_geometry": actual["om"],
            "hdr_decoded_geometry": decoded["hdr"],
            "hdr_target_geometry": [target_width, target_height],
            "hdr_resample_active": bool(getattr(hdr_decoder, "resample_active", False)),
            "hdr_resample_contract": (
                "pre-PQ INTER_LINEAR on normalized RGB48 codes, native GPU, "
                "no-op when the decoded master already fills the overlap"
            ),
            "ring_size_per_source": self.ring_size,
            "full_frame_cpu_rgb_transport": False,
        }

    @property
    def lifecycle(self) -> dict[str, Any]:
        passes = [dict(record) for record in self._lifecycle_passes]
        total_acquired = sum(
            int((record.get("hdr") or {}).get("acquired", 0))
            + int((record.get("om") or {}).get("acquired", 0))
            for record in passes
        )
        total_released = sum(
            int((record.get("hdr") or {}).get("released", 0))
            + int((record.get("om") or {}).get("released", 0))
            for record in passes
        )
        max_in_flight = max(
            [
                int((record.get(source_name) or {}).get("max_in_flight", 0))
                for record in passes
                for source_name in ("hdr", "om")
            ]
            or [0]
        )
        return {
            "scope": "NativeFrameSource.iter_frames passes",
            "passes": passes,
            "acquired": total_acquired,
            "released": total_released,
            "in_flight": total_acquired - total_released,
            "max_in_flight": max_in_flight,
            "ring_size": int(self.ring_size),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._last_pair_metadata)
        metadata["lifecycle"] = self.lifecycle
        return metadata


class NativeV4GpuTapeShotFitter(v05.V4GpuShotFitter):
    """Legacy V4 feature-tape fitter retained for coefficient comparison."""

    def _resize_device_into(self, source: Any, target: Any, target_size: tuple[int, int]) -> None:
        target_width, target_height = (int(value) for value in target_size)
        source_height, source_width = (int(source.shape[0]), int(source.shape[1]))
        if (source_width, source_height) == (target_width, target_height):
            target[...] = source
            return
        if (
            source_height % target_height != 0
            or source_width % target_width != 0
            or source_height // target_height != source_width // target_width
        ):
            raise NativeBridgeError(
                "native phase-1 fit ingest supports exact integer-area reduction only; "
                f"source={source_width}x{source_height}, target={target_width}x{target_height}"
            )
        factor = source_height // target_height
        reduced = source.reshape(
            target_height,
            factor,
            target_width,
            factor,
            3,
        ).mean(axis=(1, 3), dtype=self.backend.xp.float32)
        target[...] = reduced

    def _iter_batches(
        self,
        source: NativeFrameSource,
        indices: Iterable[int],
        *,
        pass_name: str | None = None,
    ) -> Iterator[tuple[Any, Any, np.ndarray]]:
        if self.fit_config is None:
            raise NativeBridgeError("V4 fit geometry was not initialized")
        wanted = {int(index) for index in indices}
        xp = self.backend.xp
        fit_width, fit_height = (int(value) for value in self.fit_config.om_size)
        hdr_width, hdr_height = (int(value) for value in self.fit_config.hdr_size)
        sdr_batch = xp.empty(
            (self.batch_size, fit_height, fit_width, 3), dtype=xp.float32
        )
        hdr_batch = xp.empty(
            (self.batch_size, hdr_height, hdr_width, 3), dtype=xp.float32
        )
        sequences = np.empty(self.batch_size, dtype=np.int64)
        filled = 0
        iterator = source.iter_frames(
            self.source_profiler,
            include_hashes=False,
            pass_name=pass_name,
        )
        try:
            for frame in iterator:
                if frame.sequence not in wanted:
                    continue
                resize_started = time.perf_counter()
                self._resize_device_into(frame.sdr, sdr_batch[filled], self.fit_config.om_size)
                self._resize_device_into(frame.hdr, hdr_batch[filled], self.fit_config.hdr_size)
                self.profile["fit_resize_seconds"] += time.perf_counter() - resize_started
                self.profile["fit_resize_frames"] += 1
                self.profile["fit_resize_images"] += 2
                self.profile["fit_prepare_frames"] += 1
                sequences[filled] = int(frame.sequence)
                filled += 1
                if filled < self.batch_size:
                    continue
                self.profile["fit_direct_fill_batches"] += 1
                yield sdr_batch[:filled], hdr_batch[:filled], sequences[:filled].copy()
                filled = 0
            if filled:
                self.profile["fit_direct_fill_batches"] += 1
                yield sdr_batch[:filled], hdr_batch[:filled], sequences[:filled].copy()
        finally:
            iterator.close()
        self.profile["fit_batch_path"] = "native_cuda_device_area_batches"
        self.profile["fit_hash_mode"] = "native_no_host_rgb_hash"

    def _upload_batch(self, sdr_device: Any, hdr_device: Any) -> tuple[Any, Any]:
        self.profile["native_device_batches"] = int(
            self.profile.get("native_device_batches", 0)
        ) + 1
        self.profile["native_h2d_calls"] = 0
        return sdr_device, hdr_device


class NativeV4GpuStreamingShotFitter(NativeV4GpuTapeShotFitter):
    """Native V4 fitter with bounded multi-pass reductions and no GPU feature tape."""

    def _initialize_stream_geometry(self, config: Any) -> None:
        height = int(config.om_size[1])
        _, y1, _, y2 = (int(value) for value in config.overlap)
        seam_band = int(self.fit_seam_band)
        if not (0 < y1 < y2 < height) or y2 - seam_band < 0:
            raise v05.V05Error("V4 requires seam edge rows inside the Open Matte frame")
        blur_radius = int(
            max(0.0, 4.0 * float(max(self.fit_chroma_sigma, self.fit_intensity_sigma)) + 0.5)
        )
        top = self._patch_geometry(y1 - 1, y1 + seam_band, height, blur_radius)
        bottom = self._patch_geometry(y2 - seam_band, y2 + 1, height, blur_radius)
        self.geometry = {
            "top": {
                **top,
                "edge": y1 - 1 - top["source_start"],
                "band_start": y1 - top["source_start"],
                "band_end": y1 + seam_band - top["source_start"],
            },
            "bottom": {
                **bottom,
                "edge": y2 - bottom["source_start"],
                "band_start": y2 - seam_band - bottom["source_start"],
                "band_end": y2 - bottom["source_start"],
            },
        }
        self.profile.update(
            {
                "feature_tape_device_bytes": 0,
                "feature_tape_full_frame": False,
                "feature_tape_frames": 0,
                "feature_tape_detail_dtype": "streamed_float32_batch",
                "feature_tape_reference_dtype": "streamed_float32_batch",
                "feature_tape_gain_dtype": "host_float64_exact_rows",
                "feature_tape_top_patch_rows": int(top["source_end"] - top["source_start"]),
                "feature_tape_bottom_patch_rows": int(bottom["source_end"] - bottom["source_start"]),
                "feature_tape_blur_radius": int(blur_radius),
                "streaming_accumulator_contract": {
                    "gain": "exact host rows with median and 5-95 percentile reduction",
                    "spatial": "fixed-size XTX-style numerator/denominator vectors",
                    "intensity": "fixed-size scalar normal-equation accumulators",
                    "candidates": "fixed-size per-model residual and hue accumulators",
                },
                "numerical_parity_contract": {
                    "reference": "NativeV4GpuTapeShotFitter",
                    "gain_reduction": "GPU-produced float64 rows reduced on host with NumPy median/linear percentile",
                    "gain_abs_tolerance_stops": 1.0e-10,
                    "spatial_abs_tolerance": 1.0e-8,
                    "controls_abs_tolerance": 1.0e-10,
                    "candidate_selection": "exact model name and candidate ordering",
                },
            }
        )
        self._sample_vram("stream_geometry_initialized")

    def _gain_stream_pass(
        self,
        source: NativeFrameSource,
        config: Any,
    ) -> dict[str, Any]:
        xp = self.backend.xp
        _, y1, _, y2 = (int(value) for value in config.overlap)
        seam_band = int(self.fit_seam_band)
        low, high = config.detail_ratio_clip
        rows = np.empty(
            (int(source.count), 2, int(config.om_size[0])),
            dtype=np.float64,
        )
        expected_sequence = 0
        started = time.perf_counter()
        for sdr, hdr, sequences in self._iter_batches(
            source,
            range(source.count),
            pass_name="gain",
        ):
            if len(sequences) == 0 or int(sequences[0]) != expected_sequence:
                raise v05.V05Error("V4 gain traversal lost or reordered a decoded frame")
            gpu_started = time.perf_counter()
            base = self._blur(self.backend, sdr, config.base_sigma)
            hdr_y = self.backend.luminance(hdr)
            base_y = self.backend.luminance(base)
            top = xp.median(
                xp.log2(
                    (hdr_y[:, :seam_band] + v05.fastcore.EPS)
                    / (base_y[:, y1 : y1 + seam_band] + v05.fastcore.EPS)
                ),
                axis=1,
            )
            bottom = xp.median(
                xp.log2(
                    (hdr_y[:, -seam_band:] + v05.fastcore.EPS)
                    / (base_y[:, y2 - seam_band : y2] + v05.fastcore.EPS)
                ),
                axis=1,
            )
            self.profile["capture_gpu"] += time.perf_counter() - gpu_started
            compact = self._to_host(
                xp.concatenate((top, bottom), axis=1),
                "gain_rows",
            )
            rows[np.asarray(sequences, dtype=np.int64)] = np.asarray(
                compact,
                dtype=np.float64,
            ).reshape(len(sequences), 2, int(config.om_size[0]))
            expected_sequence = int(sequences[-1]) + 1
            self.profile["capture_batches"] += 1
            self._sample_vram("stream_gain_batch")
            del sdr, hdr, base, hdr_y, base_y, top, bottom, compact
        if expected_sequence != source.count:
            raise v05.V05Error(
                f"V4 gain traversal decoded {expected_sequence} of {source.count} frames"
            )
        self.profile["pass_gain"] = time.perf_counter() - started
        self.profile["capture_seconds"] = self.profile["pass_gain"]
        self.profile["exact_gain_rows_host_bytes"] = int(rows.nbytes)
        return {"rows": rows, "seconds": self.profile["pass_gain"]}

    def _finalize_stream_gain(self, rows: np.ndarray, config: Any) -> dict[str, Any]:
        started = time.perf_counter()
        top_median = np.median(rows[:, 0, :], axis=0)
        bottom_median = np.median(rows[:, 1, :], axis=0)
        top_spread = np.percentile(rows[:, 0, :], 95) - np.percentile(rows[:, 0, :], 5)
        shot_stops = float(np.median(np.concatenate((top_median, bottom_median))))
        limit = config.gain_clamp_stops
        top_smoothed = v05.om.smooth_profile(top_median, config.gain_smooth_sigma)
        bottom_smoothed = v05.om.smooth_profile(bottom_median, config.gain_smooth_sigma)
        top_profile = np.clip(top_smoothed, shot_stops - limit, shot_stops + limit)
        bottom_profile = np.clip(bottom_smoothed, shot_stops - limit, shot_stops + limit)
        self.profile["gain_reduce_seconds"] = time.perf_counter() - started
        return {
            "shot_gain_stops": shot_stops,
            "shot_gain_linear": float(np.exp2(shot_stops)),
            "top_profile_stops": top_profile,
            "bottom_profile_stops": bottom_profile,
            "top_profile_range_stops": [float(top_profile.min()), float(top_profile.max())],
            "bottom_profile_range_stops": [float(bottom_profile.min()), float(bottom_profile.max())],
            "per_frame_top_median_spread_stops": float(top_spread),
            "clamped_fraction": float(
                np.mean(
                    np.abs(np.concatenate((top_smoothed, bottom_smoothed)) - shot_stops)
                    > limit
                )
            ),
            "measurement_seconds": self.profile["capture_seconds"],
            "fit_storage_bytes": int(rows.nbytes),
            "method": "v04 median over all frame seam measurements; exact host gain rows with bounded multi-pass feature reductions",
        }

    def _iter_stream_replay_batches(
        self,
        source: NativeFrameSource,
        indices: list[int],
        pass_name: str,
    ) -> Iterator[dict[str, Any]]:
        xp = self.backend.xp
        top_geometry = self.geometry["top"]
        bottom_geometry = self.geometry["bottom"]
        for sdr, hdr, _sequences in self._iter_batches(
            source,
            indices,
            pass_name=pass_name,
        ):
            base = self._blur(self.backend, sdr, self.fit_config.base_sigma)
            ratio = xp.clip(
                sdr / xp.maximum(base, v05.fastcore.EPS),
                self.fit_config.detail_ratio_clip[0],
                self.fit_config.detail_ratio_clip[1],
            )
            detail = base * xp.power(ratio, self.fit_config.detail_strength)
            top_detail = detail[:, top_geometry["source_start"] : top_geometry["source_end"]]
            bottom_detail = detail[:, bottom_geometry["source_start"] : bottom_geometry["source_end"]]
            top_gain = self.gain_field_device[
                top_geometry["source_start"] : top_geometry["source_end"]
            ]
            bottom_gain = self.gain_field_device[
                bottom_geometry["source_start"] : bottom_geometry["source_end"]
            ]
            top_predicted = xp.clip(top_detail * top_gain[None, ..., None], 0.0, 1.0)
            bottom_predicted = xp.clip(bottom_detail * bottom_gain[None, ..., None], 0.0, 1.0)
            top_ictcp = self.backend.to_ictcp(top_predicted * v05.fastcore.PEAK_NITS)
            bottom_ictcp = self.backend.to_ictcp(bottom_predicted * v05.fastcore.PEAK_NITS)
            top_intensity_full = self._blur(
                self.backend, top_ictcp[..., 0], self.fit_intensity_sigma
            )
            bottom_intensity_full = self._blur(
                self.backend, bottom_ictcp[..., 0], self.fit_intensity_sigma
            )
            top_chroma_full = self._blur(
                self.backend, top_ictcp[..., 1:], self.fit_chroma_sigma
            )
            bottom_chroma_full = self._blur(
                self.backend, bottom_ictcp[..., 1:], self.fit_chroma_sigma
            )
            hdr_ictcp = self.backend.to_ictcp(hdr * v05.fastcore.PEAK_NITS)
            spatial_reference = self._blur(
                self.backend, hdr_ictcp[..., 1:], self.fit_chroma_sigma
            )
            if self.same_reference_sigma:
                fit_reference = spatial_reference
            else:
                fit_reference = self._blur(
                    self.backend, hdr_ictcp[..., 1:], self.fit_intensity_sigma
                )
            self._sample_vram("stream_replay_batch")
            yield {
                "top_intensity": top_intensity_full[
                    :, top_geometry["band_start"] : top_geometry["band_end"]
                ],
                "bottom_intensity": bottom_intensity_full[
                    :, bottom_geometry["band_start"] : bottom_geometry["band_end"]
                ],
                "top_chroma": top_chroma_full[
                    :, top_geometry["band_start"] : top_geometry["band_end"]
                ],
                "bottom_chroma": bottom_chroma_full[
                    :, bottom_geometry["band_start"] : bottom_geometry["band_end"]
                ],
                "top_edge_intensity": top_intensity_full[:, top_geometry["edge"]],
                "bottom_edge_intensity": bottom_intensity_full[:, bottom_geometry["edge"]],
                "top_edge_chroma": top_chroma_full[:, top_geometry["edge"]],
                "bottom_edge_chroma": bottom_chroma_full[:, bottom_geometry["edge"]],
                "top_reference": spatial_reference[:, : self.fit_seam_band],
                "bottom_reference": spatial_reference[:, -self.fit_seam_band :],
                "top_edge_reference": spatial_reference[:, 0],
                "bottom_edge_reference": spatial_reference[:, -1],
                "top_fit_reference": fit_reference[:, : self.fit_seam_band],
                "bottom_fit_reference": fit_reference[:, -self.fit_seam_band :],
                "top_fit_edge_reference": fit_reference[:, 0],
                "bottom_fit_edge_reference": fit_reference[:, -1],
            }
            del sdr, hdr, base, ratio, detail, top_detail, bottom_detail
            del top_predicted, bottom_predicted, top_ictcp, bottom_ictcp
            del top_intensity_full, bottom_intensity_full, top_chroma_full, bottom_chroma_full
            del hdr_ictcp, spatial_reference, fit_reference

    def _spatial_stream_pass(
        self,
        source: NativeFrameSource,
        indices: list[int],
    ) -> dict[str, Any]:
        xp = self.backend.xp
        width = int(self.fit_config.om_size[0])
        totals = {
            name: {
                "num_re": xp.zeros(width, dtype=xp.float64),
                "num_im": xp.zeros(width, dtype=xp.float64),
                "den": xp.zeros(width, dtype=xp.float64),
            }
            for name in ("top", "bottom")
        }
        sum_weight = xp.float64(0.0)
        sum_intensity = xp.float64(0.0)
        sum_intensity_squared = xp.float64(0.0)
        before_sum = xp.float64(0.0)
        before_count = 0
        pass_started = time.perf_counter()
        batch_started = pass_started
        for feature in self._iter_stream_replay_batches(source, indices, "training_spatial"):
            bands = {
                "top": (
                    feature["top_chroma"],
                    feature["top_reference"],
                    feature["top_intensity"],
                ),
                "bottom": (
                    feature["bottom_chroma"],
                    feature["bottom_reference"],
                    feature["bottom_intensity"],
                ),
            }
            for name, (predicted_band, hdr_band, intensity_band) in bands.items():
                p_re, p_im = predicted_band[..., 0], predicted_band[..., 1]
                h_re, h_im = hdr_band[..., 0], hdr_band[..., 1]
                totals[name]["num_re"] += xp.sum(p_re * h_re + p_im * h_im, axis=(0, 1))
                totals[name]["num_im"] += xp.sum(p_re * h_im - p_im * h_re, axis=(0, 1))
                totals[name]["den"] += xp.sum(p_re * p_re + p_im * p_im, axis=(0, 1))
                before_sum += xp.sum(xp.abs(hdr_band - predicted_band))
                before_count += int(hdr_band.size)
                weight = xp.sum(predicted_band * predicted_band, axis=-1)
                valid_weight = xp.where(weight > v05.v2.HUE_CHROMA_FLOOR**2, weight, 0.0)
                sum_weight += xp.sum(valid_weight)
                sum_intensity += xp.sum(valid_weight * intensity_band)
                sum_intensity_squared += xp.sum(valid_weight * intensity_band * intensity_band)
            self.profile["replay_spatial_batches"] += 1
            self.profile["spatial_gpu"] += time.perf_counter() - batch_started
            batch_started = time.perf_counter()
        packed = xp.concatenate(
            (
                xp.stack(
                    (
                        totals["top"]["num_re"],
                        totals["top"]["num_im"],
                        totals["top"]["den"],
                        totals["bottom"]["num_re"],
                        totals["bottom"]["num_im"],
                        totals["bottom"]["den"],
                    ),
                    axis=0,
                ).reshape(-1),
                xp.stack((sum_weight, sum_intensity, sum_intensity_squared, before_sum)),
            )
        )
        host = self._to_host(packed, "final")
        self.profile["pass_replay_spatial"] = time.perf_counter() - pass_started
        return {
            "totals": {
                "top": {
                    "num_re": host[0 * width : 1 * width],
                    "num_im": host[1 * width : 2 * width],
                    "den": host[2 * width : 3 * width],
                },
                "bottom": {
                    "num_re": host[3 * width : 4 * width],
                    "num_im": host[4 * width : 5 * width],
                    "den": host[5 * width : 6 * width],
                },
            },
            "sum_weight": float(host[6 * width]),
            "sum_intensity": float(host[6 * width + 1]),
            "sum_intensity_squared": float(host[6 * width + 2]),
            "before_sum": float(host[6 * width + 3]),
            "before_count": before_count,
        }

    def _intensity_stream_pass(
        self,
        source: NativeFrameSource,
        field_parts: dict[str, tuple[Any, Any]],
        centre: float,
        normalization: float,
        indices: list[int],
    ) -> dict[str, Any]:
        xp = self.backend.xp
        scale_numerator = xp.float64(0.0)
        hue_numerator = xp.float64(0.0)
        denominator = xp.float64(0.0)
        valid_samples = xp.int64(0)
        pass_started = time.perf_counter()
        batch_started = pass_started
        for feature in self._iter_stream_replay_batches(source, indices, "training_intensity"):
            for name, intensity_band, predicted_band, reference_band in (
                ("top", feature["top_intensity"], feature["top_chroma"], feature["top_fit_reference"]),
                ("bottom", feature["bottom_intensity"], feature["bottom_chroma"], feature["bottom_fit_reference"]),
            ):
                spatial = v05.v2.complex_transform(
                    self.backend, predicted_band, *field_parts[name]
                )
                predicted_magnitude = xp.hypot(spatial[..., 0], spatial[..., 1])
                reference_magnitude = xp.hypot(reference_band[..., 0], reference_band[..., 1])
                valid = (predicted_magnitude > v05.v2.HUE_CHROMA_FLOOR) & (
                    reference_magnitude > v05.v2.HUE_CHROMA_FLOOR
                )
                weight = xp.where(valid, predicted_magnitude * predicted_magnitude, 0.0)
                t = v05.v4.normalized_intensity(
                    self.backend, intensity_band, centre, normalization
                )
                log_ratio = xp.log(
                    xp.maximum(reference_magnitude, v05.fastcore.EPS)
                    / xp.maximum(predicted_magnitude, v05.fastcore.EPS)
                )
                phase = xp.arctan2(
                    spatial[..., 0] * reference_band[..., 1]
                    - spatial[..., 1] * reference_band[..., 0],
                    spatial[..., 0] * reference_band[..., 0]
                    + spatial[..., 1] * reference_band[..., 1],
                )
                scale_numerator += xp.sum(weight * t * log_ratio)
                hue_numerator += xp.sum(weight * t * phase)
                denominator += xp.sum(weight * t * t)
                valid_samples += xp.sum(valid)
            self.profile["replay_intensity_batches"] += 1
            self.profile["intensity_gpu"] += time.perf_counter() - batch_started
            batch_started = time.perf_counter()
        host = self._to_host(
            xp.stack((scale_numerator, hue_numerator, denominator, valid_samples)),
            "final",
        )
        self.profile["pass_replay_intensity"] = time.perf_counter() - pass_started
        return {
            "scale_numerator": float(host[0]),
            "hue_numerator": float(host[1]),
            "denominator": float(host[2]),
            "valid_samples": int(round(float(host[3]))),
        }

    def _candidate_stream_pass(
        self,
        source: NativeFrameSource,
        field_parts: dict[str, tuple[Any, Any]],
        controls: dict[str, Any],
        models: list[dict[str, Any]],
        indices: list[int],
    ) -> dict[str, Any]:
        xp = self.backend.xp
        scales = self.backend.asarray(
            np.asarray([model["log_saturation_slope"] for model in models], dtype=np.float32)
        )
        hues = self.backend.asarray(
            np.asarray([model["hue_slope_radians"] for model in models], dtype=np.float32)
        )
        self.profile["h2d_calls"] += 2
        self._sample_vram("candidate_models_allocated")
        residual_sum = xp.zeros(len(models), dtype=xp.float64)
        residual_count = xp.zeros(len(models), dtype=xp.int64)
        hue_sum = xp.zeros(len(models), dtype=xp.float64)
        hue_count = xp.zeros(len(models), dtype=xp.int64)
        pass_started = time.perf_counter()
        batch_started = pass_started
        for feature in self._iter_stream_replay_batches(source, indices, "holdout_candidates"):
            for name, intensity_band, predicted_band, reference_band in (
                ("top", feature["top_intensity"], feature["top_chroma"], feature["top_fit_reference"]),
                ("bottom", feature["bottom_intensity"], feature["bottom_chroma"], feature["bottom_fit_reference"]),
            ):
                real, imag = self._conditioned_batch(
                    self.backend,
                    *field_parts[name],
                    intensity_band,
                    controls,
                    scales,
                    hues,
                )
                corrected = self._complex_batch(self.backend, predicted_band, real, imag)
                residual_sum += xp.sum(
                    xp.abs(reference_band[None, ...] - corrected),
                    axis=(1, 2, 3, 4),
                )
                residual_count += int(reference_band.size)
            top_real, top_imag = self._conditioned_batch(
                self.backend,
                *field_parts["top_edge"],
                feature["top_edge_intensity"],
                controls,
                scales,
                hues,
            )
            bottom_real, bottom_imag = self._conditioned_batch(
                self.backend,
                *field_parts["bottom_edge"],
                feature["bottom_edge_intensity"],
                controls,
                scales,
                hues,
            )
            top_corrected = self._complex_batch(
                self.backend, feature["top_edge_chroma"], top_real, top_imag
            )
            bottom_corrected = self._complex_batch(
                self.backend, feature["bottom_edge_chroma"], bottom_real, bottom_imag
            )
            top_hue, top_valid = self._hue_batch(
                self.backend, top_corrected, feature["top_fit_edge_reference"]
            )
            bottom_hue, bottom_valid = self._hue_batch(
                self.backend, bottom_corrected, feature["bottom_fit_edge_reference"]
            )
            hue_sum += top_hue + bottom_hue
            hue_count += top_valid + bottom_valid
            self.profile["replay_candidate_batches"] += 1
            self.profile["candidate_gpu"] += time.perf_counter() - batch_started
            batch_started = time.perf_counter()
        metrics = self._to_host(
            xp.stack(
                (
                    residual_sum,
                    residual_count.astype(xp.float64),
                    hue_sum,
                    hue_count.astype(xp.float64),
                ),
                axis=1,
            ),
            "final",
        )
        self.profile["pass_replay_candidates"] = time.perf_counter() - pass_started
        return {"metrics": metrics}

    def fit_shot(
        self,
        frame_stream: NativeFrameSource,
        config: Any,
        fit_scale: float = 1.0,
    ) -> dict[str, Any]:
        self.source_config = config
        self.fit_scale = v05._validate_fit_scale(fit_scale)
        fit_config, fit_geometry = v05._derive_fit_grid(config, self.fit_scale)
        self.fit_config = fit_config
        self.config = fit_config
        self.fit_seam_band = int(fit_config.seam_band)
        self.fit_chroma_sigma = float(v05.v1.CHROMA_SIGMA) * self.fit_scale
        self.fit_intensity_sigma = float(v05.v4.INTENSITY_SIGMA) * self.fit_scale
        self.same_reference_sigma = bool(
            np.isclose(self.fit_chroma_sigma, self.fit_intensity_sigma, rtol=0.0, atol=1e-12)
        )
        self.source_profiler = v05.Profiler()
        training, holdout = v05.sampled_split(frame_stream.count, self.fit_stride)
        started = time.perf_counter()
        self.profile = {
            "architecture": "gpu_shot_fitter_v4_streaming_constant_vram",
            "fit_scale_requested": self.fit_scale,
            "fit_scale_effective": self.fit_scale,
            "fit_geometry": fit_geometry,
            "source_geometry": fit_geometry["source"],
            "fit_grid": fit_geometry["fit"],
            "source_passes": 4,
            "source_pass_names": ["gain", "training_spatial", "training_intensity", "holdout_candidates"],
            "replay_passes": 3,
            "replay_pass_names": ["spatial_and_intensity_centre", "intensity_slopes", "candidates"],
            "ffmpeg_openings": 8,
            "decoded_pairs": int(frame_stream.count * 4),
            "decoded_individual_images": int(frame_stream.count * 8),
            "h2d": 0.0,
            "h2d_calls": 0,
            "h2d_batches": 0,
            "d2h": 0.0,
            "d2h_calls": 0,
            "d2h_compact": 0.0,
            "d2h_compact_calls": 0,
            "d2h_final": 0.0,
            "d2h_final_calls": 0,
            "d2h_gain_rows": 0.0,
            "d2h_gain_rows_calls": 0,
            "full_frame_d2h": 0,
            "per_frame_d2h": 0,
            "capture_gpu": 0.0,
            "spatial_gpu": 0.0,
            "intensity_gpu": 0.0,
            "candidate_gpu": 0.0,
            "solver_cpu": 0.0,
            "capture_batches": 0,
            "replay_spatial_batches": 0,
            "replay_intensity_batches": 0,
            "replay_candidate_batches": 0,
            "fit_resize_seconds": 0.0,
            "fit_resize_frames": 0,
            "fit_resize_images": 0,
            "fit_prepare_seconds": 0.0,
            "fit_prepare_frames": 0,
            "fit_producer_seconds": 0.0,
            "fit_queue_consumer_wait_seconds": 0.0,
            "fit_queue_consumer_wait_events": 0,
            "fit_queue_producer_wait_seconds": 0.0,
            "fit_queue_producer_wait_events": 0,
            "fit_queue_depth": 0,
            "fit_queue_high_water_mark": 0,
            "fit_direct_fill_batches": 0,
            "fit_stack_batches": 0,
            "fit_batch_path": "native_cuda_device_area_batches",
            "fit_hash_mode": "native_no_host_rgb_hash",
            "fit_source_traversals": 4,
            "fit_ffmpeg_openings": 8,
            "peak_vram_bytes": 0,
            "peak_vram_mib": 0.0,
            "vram_telemetry_available": False,
            "vram_samples": [],
            "pass_capture": 0.0,
            "pass_gain": 0.0,
            "pass_replay_spatial": 0.0,
            "pass_replay_intensity": 0.0,
            "pass_replay_candidates": 0.0,
        }
        self._sample_vram("fit_start")
        self.batch_size = self._choose_batch_size(fit_config)
        self._initialize_stream_geometry(fit_config)

        gain_started = time.perf_counter()
        gain_rows = self._gain_stream_pass(frame_stream, fit_config)["rows"]
        gain = self._finalize_stream_gain(gain_rows, fit_config)
        self.profile["pass_gain"] = time.perf_counter() - gain_started
        solver_started = time.perf_counter()
        gain_field_fit = v05.om.gain_field(fit_config, gain)
        self.gain_field_device = self.backend.asarray(gain_field_fit)
        self._sample_vram("gain_field_allocated")
        self.profile["h2d"] += time.perf_counter() - solver_started
        self.profile["h2d_calls"] += 1
        training_result = self._spatial_stream_pass(frame_stream, training)
        spatial = self._solve_spatial(training_result, training)
        spatial_field_fit = v05.v1.chroma_field(fit_config, spatial)
        host_weight = float(training_result["sum_weight"])
        v05.om.require(host_weight > 0.0, "No chroma-energy support for intensity-conditioned fit")
        centre = training_result["sum_intensity"] / host_weight
        variance = max(
            training_result["sum_intensity_squared"] / host_weight - centre * centre,
            0.0,
        )
        normalization = max(2.0 * math.sqrt(variance), v05.v4.MIN_INTENSITY_NORMALIZATION)
        self.profile["solver_cpu"] += time.perf_counter() - solver_started
        field_parts = self._make_field_parts(spatial_field_fit)
        self._sample_vram("field_parts_allocated")

        intensity_result = self._intensity_stream_pass(
            frame_stream,
            field_parts,
            centre,
            normalization,
            training,
        )
        denominator = intensity_result["denominator"]
        v05.om.require(denominator > 0.0, "Degenerate intensity-conditioned fit")
        solver_started = time.perf_counter()
        unconstrained_scale = intensity_result["scale_numerator"] / denominator
        unconstrained_hue = intensity_result["hue_numerator"] / denominator
        scale = float(
            np.clip(
                unconstrained_scale,
                -v05.v4.MAX_LOG_SATURATION_SLOPE,
                v05.v4.MAX_LOG_SATURATION_SLOPE,
            )
        )
        hue = float(
            np.clip(
                unconstrained_hue,
                -v05.v4.MAX_HUE_SLOPE_RADIANS,
                v05.v4.MAX_HUE_SLOPE_RADIANS,
            )
        )
        controls = {
            "intensity_centre": centre,
            "intensity_normalization": normalization,
            "log_saturation_slope": scale,
            "hue_slope_radians": hue,
            "hue_slope_degrees": math.degrees(hue),
            "unconstrained_log_saturation_slope": unconstrained_scale,
            "unconstrained_hue_slope_degrees": math.degrees(unconstrained_hue),
            "log_saturation_slope_limit": v05.v4.MAX_LOG_SATURATION_SLOPE,
            "hue_slope_limit_degrees": math.degrees(v05.v4.MAX_HUE_SLOPE_RADIANS),
            "training_frames": len(training),
            "valid_samples": intensity_result["valid_samples"],
            "seconds": self.profile["pass_replay_intensity"],
        }
        self.profile["solver_cpu"] += time.perf_counter() - solver_started

        models = v05.v4.candidate_models(controls)
        candidate_result = self._candidate_stream_pass(
            frame_stream,
            field_parts,
            controls,
            models,
            holdout,
        )
        selection = self._select_candidates(models, candidate_result["metrics"], len(holdout))
        selection["seconds"] = self.profile["pass_replay_candidates"]
        source_stats = self.source_profiler.result()
        self.profile["io"] = float(source_stats["stages"].get("io", {}).get("total_seconds", 0.0))
        self.profile["cpu_decode"] = float(
            source_stats["stages"].get("cpu_decode", {}).get("total_seconds", 0.0)
        )
        self.profile["decoded_frames"] = int(frame_stream.count * 4)
        self.profile["fit_total"] = time.perf_counter() - started
        self.profile["source_profiler"] = source_stats
        lifecycle = frame_stream.lifecycle
        lifecycle_violations = []
        acquired_equals_released = True
        in_flight_zero = True
        max_in_flight_within_ring = True
        for pass_record in lifecycle["passes"]:
            for source_name in ("hdr", "om"):
                counters = pass_record.get(source_name) or {}
                acquired = int(counters.get("acquired", 0))
                released = int(counters.get("released", 0))
                in_flight = int(counters.get("in_flight", 0))
                max_in_flight = int(counters.get("max_in_flight", 0))
                acquired_equals_released &= acquired == released
                in_flight_zero &= in_flight == 0
                max_in_flight_within_ring &= max_in_flight <= int(lifecycle["ring_size"])
                if acquired != released or in_flight != 0 or max_in_flight > int(lifecycle["ring_size"]):
                    lifecycle_violations.append(
                        {
                            "pass": pass_record["name"],
                            "source": source_name,
                            "counters": counters,
                        }
                    )
        self.profile["ring_lifecycle"] = lifecycle
        self.profile["ring_lifecycle_invariants"] = {
            "acquired_equals_released": bool(acquired_equals_released),
            "in_flight_zero": bool(in_flight_zero),
            "max_in_flight_within_ring": bool(max_in_flight_within_ring),
            "surface_capacity_exceeded": not bool(max_in_flight_within_ring),
            "violations": lifecycle_violations,
        }
        default_model = next(
            model for model in models if model["name"] == selection["default_model"]["name"]
        )
        review_model = next(
            model for model in models if model["name"] == selection["review_candidate"]["name"]
        )
        gain_field = self._upsample_scalar_field(gain_field_fit, config.om_size)
        spatial_field = self._upsample_complex_field(spatial_field_fit, config.om_size)
        self.profile["returned_gain_field_shape"] = [int(value) for value in gain_field.shape]
        self.profile["returned_spatial_field_shape"] = [int(value) for value in spatial_field.shape]
        self.profile["renderer_geometry"] = {
            "om_size": [int(config.om_size[0]), int(config.om_size[1])],
            "overlap": [int(value) for value in config.overlap],
            "fields_upsampled_before_renderer": self.fit_scale != 1.0,
        }
        result = {
            "training_indices": training,
            "holdout_indices": holdout,
            "gain": gain,
            "gain_field": gain_field,
            "spatial": spatial,
            "spatial_field": spatial_field,
            "controls": controls,
            "selection": selection,
            "default_model": default_model,
            "review_model": review_model,
            "rendered_model": default_model,
            "fit_seconds": self.profile["fit_total"],
            "fit_scale": self.fit_scale,
            "fit_source_passes": 4,
            "fit_storage_contract": "four sequential native decoder passes, exact host float64 gain rows, fixed-size GPU spatial/intensity/candidate accumulators, bounded CUDA batches only; no shot-sized GPU tape, full-frame D2H, or per-frame D2H",
            "fit_profile": self.profile,
        }
        self.tape.clear()
        self.gain_field_device = None
        return result


class NativeV4GpuShotFitter(NativeV4GpuStreamingShotFitter):
    """Default native V4 fitter using the constant-VRAM streaming implementation."""



def render_native(
    config: Any,
    source: NativeFrameSource,
    backend: Any,
    fit: dict[str, Any],
    output: Path,
    rendered_model: dict[str, Any],
    diagnostic_stride: int = 6,
) -> dict[str, Any]:
    """Run unchanged V4 per-frame equations on native device input.

    The phase-1 output boundary intentionally performs one final device-to-host
    PQ16 copy for the existing FFmpeg/NVENC stdin wrapper.  No decoded RGB frame
    is copied to host memory.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = v05.ManagedEncoder(config, output, config.om_size[0], config.om_size[1])
    profiler = v05.Profiler()
    quality = v05.QualityAccumulator()
    gain = backend.asarray(np.asarray(fit["gain_field"], dtype=np.float32))
    spatial = np.asarray(fit["spatial_field"])
    spatial_magnitude, spatial_angle = v05.v4.field_magnitude_angle(backend, spatial)
    frames = 0
    started = time.perf_counter()
    encoder.start()
    iterator = source.iter_frames(profiler, include_hashes=False)
    try:
        for frame in iterator:
            predicted = v05.v1.predict(
                backend,
                config,
                frame.sdr,
                backend.blur(frame.sdr, config.base_sigma),
                gain,
            )
            rgb, corrected, negative, above_peak = v05.v4.apply_intensity_conditioned(
                backend,
                predicted,
                spatial_magnitude,
                spatial_angle,
                rendered_model,
            )
            output_image = v05.v1.composite(backend, config, rgb, frame.hdr)
            quality_values: dict[str, float] = {
                "negative_rgb_fraction_before_clip": negative,
                "above_peak_rgb_fraction_before_clip": above_peak,
            }
            if frame.sequence % max(1, int(diagnostic_stride)) == 0:
                hdr_chroma = backend.blur(
                    backend.to_ictcp(frame.hdr * v05.fastcore.PEAK_NITS)[..., 1:],
                    v05.v4.INTENSITY_SIGMA,
                )
                quality_values["top_lowpass_chroma_residual"] = float(
                    backend.tohost(
                        backend.xp.mean(
                            backend.xp.abs(
                                hdr_chroma[: v05.v1.SEAM_BAND]
                                - corrected[
                                    config.overlap[1] : config.overlap[1] + v05.v1.SEAM_BAND
                                ]
                            )
                        )
                    )
                )
                quality_values["bottom_lowpass_chroma_residual"] = float(
                    backend.tohost(
                        backend.xp.mean(
                            backend.xp.abs(
                                hdr_chroma[-v05.v1.SEAM_BAND:]
                                - corrected[
                                    config.overlap[3] - v05.v1.SEAM_BAND : config.overlap[3]
                                ]
                            )
                        )
                    )
                )
                quality_values.update(v05.v2.seam_metrics(backend, output_image, config))
                quality.update(quality_values)
            pq16 = backend.to_pq16(output_image)
            encoder.write(np.asarray(pq16, dtype="<u2").tobytes())
            frames += 1
    except BaseException:
        encoder.abort()
        raise
    finally:
        iterator.close()
    encoder.close()
    elapsed = time.perf_counter() - started
    return {
        "frames": frames,
        "wall_seconds": elapsed,
        "fps": frames / max(elapsed, 1e-9),
        "quality": quality.result(),
        "profiler": profiler.result(),
        "output_boundary": {
            "mode": "temporary_host_pq16_to_ffmpeg_stdin",
            "zero_copy_nvenc": False,
            "full_frame_d2h": frames,
            "per_frame_d2h": frames,
            "decoded_rgb_host_transport": False,
        },
    }


class GpuOutputResampler:
    """Resize the final working-RGB frame on CUDA before GPU P010 conversion."""

    _kernel: Any = None

    def __init__(self, cupy: Any) -> None:
        self.cupy = cupy
        self._reported = False
        if GpuOutputResampler._kernel is None:
            GpuOutputResampler._kernel = cupy.RawKernel(
                r"""
                extern "C" __global__ void resize_linear_rgb(
                    const float *source,
                    float *target,
                    int source_width,
                    int source_height,
                    int target_width,
                    int target_height
                ) {
                    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
                    if (x >= target_width || y >= target_height) return;

                    const double scale_x = (double)source_width / (double)target_width;
                    const double scale_y = (double)source_height / (double)target_height;
                    const double source_x = ((double)x + 0.5) * scale_x - 0.5;
                    const double source_y = ((double)y + 0.5) * scale_y - 0.5;
                    int x0 = (int)floor(source_x);
                    int y0 = (int)floor(source_y);
                    double wx = source_x - (double)x0;
                    double wy = source_y - (double)y0;
                    if (x0 < 0) { x0 = 0; wx = 0.0; }
                    if (y0 < 0) { y0 = 0; wy = 0.0; }
                    if (x0 >= source_width - 1) { x0 = source_width - 2; wx = 1.0; }
                    if (y0 >= source_height - 1) { y0 = source_height - 2; wy = 1.0; }
                    const int x1 = x0 + 1;
                    const int y1 = y0 + 1;
                    const size_t p00 = ((size_t)y0 * source_width + x0) * 3;
                    const size_t p01 = ((size_t)y0 * source_width + x1) * 3;
                    const size_t p10 = ((size_t)y1 * source_width + x0) * 3;
                    const size_t p11 = ((size_t)y1 * source_width + x1) * 3;
                    const size_t destination = ((size_t)y * target_width + x) * 3;
                    #pragma unroll
                    for (int channel = 0; channel < 3; ++channel) {
                        const double top = (1.0 - wx) * (double)source[p00 + channel]
                            + wx * (double)source[p01 + channel];
                        const double bottom = (1.0 - wx) * (double)source[p10 + channel]
                            + wx * (double)source[p11 + channel];
                        target[destination + channel] = (float)((1.0 - wy) * top + wy * bottom);
                    }
                }
                """,
                "resize_linear_rgb",
            )

    def resize(self, rgb: Any, target_size: tuple[int, int], stream: Any) -> Any:
        target_width, target_height = (int(value) for value in target_size)
        source_height, source_width, channels = (int(value) for value in rgb.shape)
        if (source_width, source_height) == (target_width, target_height):
            return rgb
        if channels != 3 or min(source_width, source_height, target_width, target_height) < 2:
            raise NativeBridgeError(
                "GPU output resize requires an RGB frame with at least 2 pixels per axis: "
                f"source={rgb.shape}, target={target_size}"
            )
        if target_width % 2 or target_height % 2:
            raise NativeBridgeError(f"GPU output resize target is not even: {target_size}")
        if rgb.dtype != self.cupy.float32:
            rgb = rgb.astype(self.cupy.float32, copy=False)
        if not rgb.flags.c_contiguous:
            rgb = self.cupy.ascontiguousarray(rgb)
        target = self.cupy.empty((target_height, target_width, 3), dtype=self.cupy.float32)
        with stream:
            GpuOutputResampler._kernel(
                ((target_width + 15) // 16, (target_height + 15) // 16, 1),
                (16, 16, 1),
                (
                    rgb,
                    target,
                    source_width,
                    source_height,
                    target_width,
                    target_height,
                ),
                stream=stream,
            )
        if not self._reported:
            diagnostic_marker(
                "GPU_OUTPUT_RESIZE_OK",
                source_geometry=[source_width, source_height],
                target_geometry=[target_width, target_height],
                interpolation="INTER_LINEAR-equivalent bilinear",
                domain="normalized working RGB before PQ/P010",
                device_ptr=int(target.data.ptr),
            )
            self._reported = True
        return target


class NativeGpuEncoder:
    """FFmpeg/NVENC encoder fed by CUDA P010 planes without host pixel transfer."""

    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps_num: int,
        fps_den: int,
        *,
        bridge_path: Path = DEFAULT_BRIDGE,
        ffmpeg_bin: Path,
        device_index: int = 0,
        qp: int = 18,
    ) -> None:
        import cupy

        self.cupy = cupy
        self.output = Path(output)
        self.width = int(width)
        self.height = int(height)
        self._first_write_reported = False
        self._dll_handles: list[Any] = []
        self._error_capacity = 4096
        self._error = ctypes.create_string_buffer(self._error_capacity)
        bridge_path = Path(bridge_path).resolve()
        ffmpeg_bin = Path(ffmpeg_bin).resolve()
        if not bridge_path.is_file():
            raise NativeBridgeError(f"V5 native bridge is missing: {bridge_path}")
        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        for directory in (ffmpeg_bin, DEFAULT_CUDA_BIN):
            if directory.is_dir() and hasattr(os, "add_dll_directory"):
                self._dll_handles.append(os.add_dll_directory(str(directory)))
        self._dll = loader(str(bridge_path))
        self._configure_abi()
        self._handle = self._dll.v5_encoder_open(
            str(self.output).encode("utf-8"),
            self.width,
            self.height,
            int(fps_num),
            int(fps_den),
            int(device_index),
            int(qp),
            self._error,
            self._error_capacity,
        )
        if not self._handle:
            raise NativeBridgeError(
                f"v5_encoder_open({self.output}) failed: {_error_text(self._error)}"
            )
        diagnostic_marker(
            "NVENC_OK",
            output=str(self.output),
            bridge_path=str(bridge_path),
            width=self.width,
            height=self.height,
            device_index=int(device_index),
            qp=int(qp),
        )

    def _configure_abi(self) -> None:
        c_char_pointer = ctypes.POINTER(ctypes.c_char)
        self._dll.v5_encoder_open.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            c_char_pointer,
            ctypes.c_int,
        ]
        self._dll.v5_encoder_open.restype = ctypes.c_void_p
        self._dll.v5_encoder_write_p010.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_uint64,
            c_char_pointer,
            ctypes.c_int,
        ]
        self._dll.v5_encoder_write_p010.restype = ctypes.c_int
        self._dll.v5_encoder_close.argtypes = [
            ctypes.c_void_p,
            c_char_pointer,
            ctypes.c_int,
        ]
        self._dll.v5_encoder_close.restype = ctypes.c_int
        self._dll.v5_encoder_abort.argtypes = [ctypes.c_void_p]
        self._dll.v5_encoder_abort.restype = None

    def write(self, p010: Any, pts: int) -> None:
        if not self._handle:
            raise NativeBridgeError("V5 GPU encoder is closed")
        if p010.dtype != self.cupy.uint16 or tuple(p010.shape) != (
            self.height + self.height // 2,
            self.width,
        ):
            raise NativeBridgeError(
                f"unexpected GPU P010 shape/dtype: shape={p010.shape}, dtype={p010.dtype}"
            )
        if not p010.flags.c_contiguous:
            raise NativeBridgeError("GPU P010 buffer must be contiguous")
        pitch = int(p010.strides[0])
        pointer = int(p010.data.ptr)
        uv_pointer = pointer + self.height * pitch
        status = int(
            self._dll.v5_encoder_write_p010(
                self._handle,
                ctypes.c_uint64(pointer),
                pitch,
                ctypes.c_uint64(uv_pointer),
                pitch,
                int(pts),
                ctypes.c_uint64(0),
                self._error,
                self._error_capacity,
            )
        )
        if status != 0:
            raise NativeBridgeError(f"v5_encoder_write_p010 failed: {_error_text(self._error)}")
        if not self._first_write_reported:
            self._first_write_reported = True
            diagnostic_marker(
                "FIRST_FRAME_ENCODE_SUBMIT_OK",
                sequence=int(pts),
                y_device_ptr=pointer,
                uv_device_ptr=uv_pointer,
                pitch_bytes=pitch,
            )

    def close(self) -> None:
        if not self._handle:
            return
        status = int(self._dll.v5_encoder_close(self._handle, self._error, self._error_capacity))
        self._handle = None
        self._dll_handles.clear()
        if status != 0:
            raise NativeBridgeError(f"v5_encoder_close failed: {_error_text(self._error)}")

    def abort(self) -> None:
        if self._handle:
            self._dll.v5_encoder_abort(self._handle)
            self._handle = None
        self._dll_handles.clear()


class GpuP010Converter:
    """Convert normalized V5 BT.2020 RGB to limited-range 10-bit P010 on CUDA."""

    _kernel: Any = None

    def __init__(self, cupy: Any) -> None:
        self.cupy = cupy
        self._reported = False
        if GpuP010Converter._kernel is None:
            GpuP010Converter._kernel = cupy.RawKernel(
                r"""
                extern "C" __device__ float clamp01(float value) {
                    return fminf(fmaxf(value, 0.0f), 1.0f);
                }
                extern "C" __device__ float pq_oetf(float value) {
                    const float m1 = 2610.0f / 16384.0f;
                    const float m2 = (2523.0f / 4096.0f) * 128.0f;
                    const float c1 = 3424.0f / 4096.0f;
                    const float c2 = (2413.0f / 4096.0f) * 32.0f;
                    const float c3 = (2392.0f / 4096.0f) * 32.0f;
                    const float powered = powf(clamp01(value), m1);
                    return powf((c1 + c2 * powered) / (1.0f + c3 * powered), m2);
                }
                extern "C" __device__ unsigned short limited_y(float value) {
                    return (unsigned short)llrintf((64.0f + 876.0f * clamp01(value)) * 64.0f);
                }
                extern "C" __device__ unsigned short limited_c(float value) {
                    return (unsigned short)llrintf((64.0f + 896.0f * clamp01(value)) * 64.0f);
                }
                extern "C" __global__ void rgb_to_p010(
                    const float *rgb,
                    unsigned short *y_plane,
                    unsigned short *uv_plane,
                    int width,
                    int height
                ) {
                    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
                    if (x >= width || y >= height) return;
                    const int pixel = (y * width + x) * 3;
                    const float r = pq_oetf(rgb[pixel + 0]);
                    const float g = pq_oetf(rgb[pixel + 1]);
                    const float b = pq_oetf(rgb[pixel + 2]);
                    const float luma = 0.2627f * r + 0.6780f * g + 0.0593f * b;
                    y_plane[y * width + x] = limited_y(luma);
                    if ((x & 1) == 0 && (y & 1) == 0 && x + 1 < width && y + 1 < height) {
                        float u = 0.0f;
                        float v = 0.0f;
                        #pragma unroll
                        for (int dy = 0; dy < 2; ++dy) {
                            #pragma unroll
                            for (int dx = 0; dx < 2; ++dx) {
                                const int p = ((y + dy) * width + x + dx) * 3;
                                const float rr = pq_oetf(rgb[p + 0]);
                                const float gg = pq_oetf(rgb[p + 1]);
                                const float bb = pq_oetf(rgb[p + 2]);
                                const float yy = 0.2627f * rr + 0.6780f * gg + 0.0593f * bb;
                                u += (bb - yy) / (2.0f * (1.0f - 0.0593f)) + 0.5f;
                                v += (rr - yy) / (2.0f * (1.0f - 0.2627f)) + 0.5f;
                            }
                        }
                        uv_plane[(y / 2) * width + x] = limited_c(u * 0.25f);
                        uv_plane[(y / 2) * width + x + 1] = limited_c(v * 0.25f);
                    }
                }
                """,
                "rgb_to_p010",
            )

    def convert(self, rgb: Any, stream: Any) -> Any:
        if rgb.dtype != self.cupy.float32:
            rgb = rgb.astype(self.cupy.float32, copy=False)
        if not rgb.flags.c_contiguous:
            rgb = self.cupy.ascontiguousarray(rgb)
        height, width, channels = (int(value) for value in rgb.shape)
        if channels != 3 or width % 2 or height % 2:
            raise NativeBridgeError(f"RGB output is not even 4:2:0 geometry: {rgb.shape}")
        p010 = self.cupy.empty((height + height // 2, width), dtype=self.cupy.uint16)
        with stream:
            GpuP010Converter._kernel(
                ((width + 15) // 16, (height + 15) // 16, 1),
                (16, 16, 1),
                (rgb, p010[:height], p010[height:], width, height),
                stream=stream,
            )
        if not self._reported:
            diagnostic_marker(
                "P010_BUFFER_OK",
                shape=[int(value) for value in p010.shape],
                dtype=str(p010.dtype),
                device_ptr=int(p010.data.ptr),
                pitch_bytes=int(p010.strides[0]),
            )
            self._reported = True
        return p010


def _gpu_scalar(backend: Any, value: Any) -> float:
    return float(backend.tohost(value))


def _extension_contract(
    backend: Any,
    config: Any,
    predicted: Any,
    hdr: Any,
    output: Any,
) -> dict[str, Any]:
    _, y1, _, y2 = config.overlap
    feather = int(config.feather)
    top = output[:y1]
    bottom = output[y2:]
    center = output[y1 + feather : y2 - feather]
    center_hdr = hdr[feather : hdr.shape[0] - feather]
    top_predicted_error = _gpu_scalar(backend, backend.xp.max(backend.xp.abs(top - predicted[:y1])))
    bottom_predicted_error = _gpu_scalar(backend, backend.xp.max(backend.xp.abs(bottom - predicted[y2:])))
    center_hdr_error = _gpu_scalar(backend, backend.xp.max(backend.xp.abs(center - center_hdr)))
    top_signal = _gpu_scalar(backend, backend.xp.mean(backend.xp.abs(top)))
    bottom_signal = _gpu_scalar(backend, backend.xp.mean(backend.xp.abs(bottom)))
    finite = bool(backend.tohost(backend.xp.all(backend.xp.isfinite(output))))
    return {
        "output_geometry": [int(config.om_size[0]), int(config.om_size[1])],
        "overlap": list(config.overlap),
        "top_extension_rows": [0, y1],
        "bottom_extension_rows": [y2, int(config.om_size[1])],
        "top_predicted_max_abs_error": top_predicted_error,
        "bottom_predicted_max_abs_error": bottom_predicted_error,
        "center_hdr_max_abs_error_excluding_feather": center_hdr_error,
        "top_extension_mean_abs_signal": top_signal,
        "bottom_extension_mean_abs_signal": bottom_signal,
        "finite": finite,
        "extended_open_matte_pass": bool(
            finite
            and top_predicted_error <= 1e-5
            and bottom_predicted_error <= 1e-5
            and center_hdr_error <= 1e-5
            and top_signal > 0.0
            and bottom_signal > 0.0
        ),
        "validation_scope": "GPU compositing contract; no independent HDR ground truth exists for extension rows",
    }


def render_native_phase2(
    config: Any,
    source: NativeFrameSource,
    backend: Any,
    fit: dict[str, Any],
    pre_metadata_output: Path,
    rendered_model: dict[str, Any],
    *,
    bridge_path: Path = DEFAULT_BRIDGE,
    ring_size: int = 4,
    device_index: int = 0,
    hdr_decoder: str = "hevc_cuvid",
    om_decoder: str = "h264_cuvid",
    encoder_qp: int = 18,
    diagnostic_stride: int = 6,
) -> dict[str, Any]:
    """Run native NVDEC/V5/GPU-P010/NVENC with bounded stage overlap."""

    import cupy

    pre_metadata_output.parent.mkdir(parents=True, exist_ok=True)
    source.hdr_decoder = str(hdr_decoder)
    source.om_decoder = str(om_decoder)
    source.ring_size = int(ring_size)
    source.device_index = int(device_index)
    profiler = v05.Profiler()
    quality = v05.QualityAccumulator()
    gain = backend.asarray(np.asarray(fit["gain_field"], dtype=np.float32))
    spatial = np.asarray(fit["spatial_field"])
    spatial_magnitude, spatial_angle = v05.v4.field_magnitude_angle(backend, spatial)
    converter = GpuP010Converter(cupy)
    processing_width, processing_height = (int(value) for value in config.om_size)
    configured_output_size = getattr(config, "output_size", None)
    output_width, output_height = (
        (int(configured_output_size[0]), int(configured_output_size[1]))
        if configured_output_size is not None
        else (processing_width, processing_height)
    )
    if output_width % 2 or output_height % 2:
        raise NativeBridgeError(f"native output geometry must be even: {(output_width, output_height)}")
    output_resampler = (
        GpuOutputResampler(cupy)
        if (output_width, output_height) != (processing_width, processing_height)
        else None
    )
    compute_stream = cupy.cuda.Stream(non_blocking=True)
    decode_queue: queue.Queue[Any] = queue.Queue(maxsize=max(2, int(ring_size) - 1))
    encode_queue: queue.Queue[Any] = queue.Queue(maxsize=max(2, int(ring_size) - 1))
    surface_credits = threading.BoundedSemaphore(max(1, int(ring_size)))
    sentinel = object()
    errors: list[dict[str, Any]] = []
    error_lock = threading.Lock()
    extension: dict[str, Any] | None = None
    frames = 0

    def record_error(stage: str, error: BaseException, sequence: int | None = None) -> None:
        with error_lock:
            first = not errors
            if first:
                errors.append(
                    {
                        "stage": str(stage),
                        "sequence": None if sequence is None else int(sequence),
                        "error": error,
                    }
                )
        if first:
            diagnostic_exception(stage, error, sequence)
        else:
            diagnostic_marker(
                "PIPELINE_ERROR",
                stage=str(stage),
                sequence=None if sequence is None else int(sequence),
                exception_type=type(error).__name__,
                exception=str(error),
                traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
            )

    def first_error() -> dict[str, Any] | None:
        with error_lock:
            return dict(errors[0]) if errors else None

    def put_or_raise(target: queue.Queue[Any], item: Any, stage: str, sequence: int | None = None) -> None:
        while True:
            try:
                target.put(item, timeout=0.1)
                return
            except queue.Full:
                failure = first_error()
                if failure is not None:
                    raise NativeBridgeError(
                        f"{stage} stopped after {failure['stage']} failure: {failure['error']}"
                    ) from failure["error"]

    def acquire_surface_credit(stage: str, sequence: int | None = None) -> None:
        while not surface_credits.acquire(timeout=0.1):
            failure = first_error()
            if failure is not None:
                raise NativeBridgeError(
                    f"{stage} stopped after {failure['stage']} failure: {failure['error']}"
                ) from failure["error"]

    def release_pair_and_credit(pair: NativeGpuPair, stream: Any) -> None:
        try:
            # The decode conversion is recorded before ready_event. This wait
            # also protects exception/early-return cleanup before recording the
            # consumer_done event on the render stream.
            with stream:
                stream.wait_event(pair.ready_event)
            source.release_pair(pair, stream)
        finally:
            if pair.fully_released:
                pair.return_surface_credit()

    def drain_decode_queue() -> None:
        while True:
            try:
                item = decode_queue.get_nowait()
            except queue.Empty:
                return
            if item is sentinel:
                continue
            pair = item
            try:
                release_pair_and_credit(pair, compute_stream)
            except BaseException as cleanup_error:
                record_error("decode_queue_drain", cleanup_error, pair.sequence)

    def produce() -> None:
        expected: int | None = None
        try:
            source.open_pairs()
            for expected in range(source.count):
                pair: NativeGpuPair | None = None
                credit_held = False
                try:
                    acquire_surface_credit("surface_credit", expected)
                    credit_held = True
                    started = time.perf_counter()
                    pair = source.next_pair()
                    if pair is None:
                        raise NativeBridgeError(f"native source ended early at pair {expected}")
                    pair.surface_credit = surface_credits
                    credit_held = False
                    if pair.sequence != expected:
                        actual_sequence = pair.sequence
                        release_pair_and_credit(pair, compute_stream)
                        pair = None
                        raise NativeBridgeError(
                            f"native pair sequence mismatch: expected={expected}, got={actual_sequence}"
                        )
                    profiler.record("native_decode", time.perf_counter() - started)
                    put_or_raise(decode_queue, pair, "decode_queue", expected)
                    pair = None
                except BaseException:
                    if pair is not None:
                        try:
                            release_pair_and_credit(pair, compute_stream)
                        except BaseException as release_error:
                            record_error("nvdec_producer_pair_release", release_error, expected)
                    elif credit_held:
                        surface_credits.release()
                    raise
            put_or_raise(decode_queue, sentinel, "decode_queue_sentinel", expected)
        except BaseException as exc:
            record_error("nvdec_producer", exc, expected)
            try:
                put_or_raise(decode_queue, sentinel, "nvdec_producer_sentinel", expected)
            except BaseException as sentinel_error:
                record_error("nvdec_producer_sentinel", sentinel_error, expected)

    def encode() -> None:
        encoder: NativeGpuEncoder | None = None
        sequence: int | None = None
        try:
            fps_num, fps_den = _fps_fraction(config)
            encoder = NativeGpuEncoder(
                pre_metadata_output,
                output_width,
                output_height,
                fps_num,
                fps_den,
                bridge_path=bridge_path,
                ffmpeg_bin=config.ffmpeg.parent,
                device_index=device_index,
                qp=encoder_qp,
            )
            while True:
                item = encode_queue.get()
                if item is sentinel:
                    break
                sequence, p010, ready_event = item
                ready_event.synchronize()
                started = time.perf_counter()
                encoder.write(p010, sequence)
                profiler.record("nvenc", time.perf_counter() - started)
                del p010
            encoder.close()
            encoder = None
        except BaseException as exc:
            record_error("nvenc_consumer", exc, sequence)
            if encoder is not None:
                try:
                    encoder.abort()
                except BaseException as abort_error:
                    record_error("nvenc_abort", abort_error, sequence)

    producer = threading.Thread(target=produce, name="v05-native-nvdec-producer", daemon=True)
    encoder_thread = threading.Thread(target=encode, name="v05-native-nvenc-consumer", daemon=True)
    pipeline_started = time.perf_counter()
    producer.start()
    encoder_thread.start()
    render_started = time.perf_counter()
    try:
        while frames < source.count:
            try:
                item = decode_queue.get(timeout=0.1)
            except queue.Empty:
                if first_error() is not None:
                    break
                continue
            if item is sentinel:
                break
            pair = item
            try:
                with compute_stream:
                    compute_stream.wait_event(pair.ready_event)
                    predicted = v05.v1.predict(
                        backend,
                        config,
                        pair.sdr.array,
                        backend.blur(pair.sdr.array, config.base_sigma),
                        gain,
                    )
                    rgb, corrected, negative, above_peak = v05.v4.apply_intensity_conditioned(
                        backend,
                        predicted,
                        spatial_magnitude,
                        spatial_angle,
                        rendered_model,
                    )
                    output_image = v05.v1.composite(backend, config, rgb, pair.hdr.array)
                    if extension is None:
                        extension = _extension_contract(
                            backend,
                            config,
                            predicted,
                            pair.hdr.array,
                            output_image,
                        )
                        diagnostic_marker(
                            "GPU_RENDER_OK",
                            sequence=int(pair.sequence),
                            output_shape=[int(value) for value in output_image.shape],
                            output_dtype=str(output_image.dtype),
                            finite=bool(extension.get("finite")),
                            extended_open_matte_pass=bool(extension.get("extended_open_matte_pass")),
                        )
                    quality_values: dict[str, float] = {
                        "negative_rgb_fraction_before_clip": negative,
                        "above_peak_rgb_fraction_before_clip": above_peak,
                    }
                    if pair.sequence % max(1, int(diagnostic_stride)) == 0:
                        hdr_chroma = backend.blur(
                            backend.to_ictcp(pair.hdr.array * v05.fastcore.PEAK_NITS)[..., 1:],
                            v05.v4.INTENSITY_SIGMA,
                        )
                        quality_values["top_lowpass_chroma_residual"] = _gpu_scalar(
                            backend,
                            backend.xp.mean(
                                backend.xp.abs(
                                    hdr_chroma[: v05.v1.SEAM_BAND]
                                    - corrected[
                                        config.overlap[1] : config.overlap[1] + v05.v1.SEAM_BAND
                                    ]
                                )
                            ),
                        )
                        quality_values["bottom_lowpass_chroma_residual"] = _gpu_scalar(
                            backend,
                            backend.xp.mean(
                                backend.xp.abs(
                                    hdr_chroma[-v05.v1.SEAM_BAND:]
                                    - corrected[
                                        config.overlap[3] - v05.v1.SEAM_BAND : config.overlap[3]
                                    ]
                                )
                            ),
                        )
                        quality_values.update(v05.v2.seam_metrics(backend, output_image, config))
                        quality.update(quality_values)
                    encoded_rgb = (
                        output_image
                        if output_resampler is None
                        else output_resampler.resize(
                            output_image,
                            (output_width, output_height),
                            compute_stream,
                        )
                    )
                    p010 = converter.convert(encoded_rgb, compute_stream)
                    ready_event = cupy.cuda.Event()
                    ready_event.record(compute_stream)
                release_pair_and_credit(pair, compute_stream)
                put_or_raise(encode_queue, (pair.sequence, p010, ready_event), "encode_queue", pair.sequence)
                profiler.frame({})
                frames += 1
            except BaseException as exc:
                try:
                    release_pair_and_credit(pair, compute_stream)
                except BaseException as release_error:
                    record_error("gpu_render_release", release_error, pair.sequence)
                raise
        compute_stream.synchronize()
    except BaseException as exc:
        record_error("gpu_render", exc, frames)
    finally:
        producer.join(timeout=120)
        if producer.is_alive():
            record_error("nvdec_producer_join_timeout", TimeoutError("NVDEC producer did not stop within 120 seconds"), frames)
        drain_decode_queue()
        try:
            put_or_raise(encode_queue, sentinel, "encoder_shutdown", frames)
        except BaseException as sentinel_error:
            record_error("nvenc_sentinel", sentinel_error, frames)
        encoder_thread.join(timeout=120)
        if encoder_thread.is_alive():
            record_error("nvenc_consumer_join_timeout", TimeoutError("NVENC consumer did not stop within 120 seconds"), frames)
        try:
            source.close_pairs()
        except BaseException as close_error:
            record_error("decoder_cleanup", close_error, frames)
    render_elapsed = time.perf_counter() - render_started
    pipeline_elapsed = time.perf_counter() - pipeline_started
    if errors:
        first_failure = errors[0]
        first_error = first_failure["error"]
        raise NativeBridgeError(
            "native Phase 2 first failure "
            f"stage={first_failure['stage']} sequence={first_failure['sequence']}: {first_error}"
        ) from first_error
    if frames != source.count:
        raise NativeBridgeError(f"native Phase 2 rendered {frames} of {source.count} frames")
    if extension is None:
        raise NativeBridgeError("native Phase 2 produced no extension contract")
    return {
        "frames": frames,
        "wall_seconds": render_elapsed,
        "pipeline_seconds": pipeline_elapsed,
        "fps": frames / max(render_elapsed, 1e-9),
        "quality": quality.result(),
        "profiler": profiler.result(),
        "extension_contract": extension,
        "processing_geometry": [processing_width, processing_height],
        "encoded_geometry": [output_width, output_height],
        "output_boundary": {
            "mode": "CUDA P010 AVHWFramesContext -> hevc_nvenc -> Matroska",
            "zero_copy_nvenc": True,
            "gpu_output_resize_active": output_resampler is not None,
            "gpu_output_resize_contract": (
                "INTER_LINEAR-equivalent bilinear on normalized working RGB before PQ/P010"
                if output_resampler is not None
                else "no-op; configured output geometry equals processing geometry"
            ),
            "decoded_frame_d2h_bytes": 0,
            "render_output_d2h_bytes": 0,
            "d2h_bytes": 0,
            "h2d_bytes": 0,
            "p010_device_bytes_per_frame": int(output_width * (output_height + output_height // 2) * 2),
            "encoded_bitstream_host_retrieval": True,
        },
        "overlap_contract": {
            "enabled": True,
            "decode_stage": "NVDEC N+1",
            "render_stage": "CUDA V5 render N",
            "encode_stage": "NVENC N-1",
            "bounded_decode_queue": decode_queue.maxsize,
            "bounded_encode_queue": encode_queue.maxsize,
            "ordered_single_nvenc_consumer": True,
            "global_cuda_synchronization": False,
        },
    }
