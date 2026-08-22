"""CUDA implementations of the neutral backend contracts.

This module is an adapter boundary only.  The existing V5 CUDA bridge,
fastcore math, output resampler, and NVENC wrapper remain the implementations;
this module gives them the common interfaces without moving or rewriting their
math and kernels.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from auto_openmatte.backends import (
    BackendCapabilities,
    Frame,
    FrameBatch,
    FrameMemory,
    FrameTimestamps,
    MemoryDomain,
    Ownership,
)

CUDA_CAPABILITIES = BackendCapabilities(
    backend="cuda-v05",
    supported_codecs=("h264", "hevc"),
    supported_pixel_formats=("NV12", "P010LE", "RGB32F"),
    bit_depths=(8, 10, 32),
    hardware_decode=True,
    hardware_encode=True,
    resampling=True,
    zero_copy=True,
    device_memory_type="cuda-device",
)


class CudaComputeBackend:
    """Common compute facade over the existing ``fastcore.Backend``."""

    def __init__(self, implementation: Any) -> None:
        if not bool(getattr(implementation, "gpu", False)):
            raise ValueError("CudaComputeBackend requires the existing GPU fastcore backend")
        self._implementation = implementation
        self.capabilities = CUDA_CAPABILITIES

    @property
    def name(self) -> str:
        return "cuda-v05-compute"

    @property
    def gpu(self) -> bool:
        return True

    @property
    def xp(self) -> Any:
        return self._implementation.xp

    def asarray(self, value: Any) -> Any:
        return self._implementation.asarray(value)

    def tohost(self, value: Any) -> Any:
        return self._implementation.tohost(value)

    def blur(self, image: Any, sigma: float) -> Any:
        return self._implementation.blur(image, sigma)

    def luminance(self, rgb: Any) -> Any:
        return self._implementation.luminance(rgb)

    def matrix(self, transform: Any, rgb: Any) -> Any:
        return self._implementation.matrix(transform, rgb)

    def pq_oetf(self, nits: Any) -> Any:
        return self._implementation.pq_oetf(nits)

    def pq_eotf(self, signal: Any) -> Any:
        return self._implementation.pq_eotf(signal)

    def to_ictcp(self, rgb_nits: Any) -> Any:
        return self._implementation.to_ictcp(rgb_nits)

    def from_ictcp(self, ictcp: Any) -> Any:
        return self._implementation.from_ictcp(ictcp)

    def to_pq16(
        self,
        normalized: Any,
        out: Any = None,
        stream: object = None,
        blocking: bool = True,
    ) -> Any:
        return self._implementation.to_pq16(normalized, out=out, stream=stream, blocking=blocking)

    def percentile(self, values: Any, q: float) -> float:
        return self._implementation.percentile(values, q)

    def __getattr__(self, name: str) -> Any:
        """Preserve non-math diagnostics of fastcore without widening the contract."""
        return getattr(self._implementation, name)


class CudaDecoderBackend:
    """Neutral paired-decoder facade over the existing bounded CUDA source."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self.capabilities = CUDA_CAPABILITIES

    @property
    def count(self) -> int:
        return int(self._source.count)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._source.metadata

    @property
    def lifecycle(self) -> dict[str, Any]:
        return self._source.lifecycle

    @property
    def hdr_decoder(self) -> str:
        return str(self._source.hdr_decoder)

    @hdr_decoder.setter
    def hdr_decoder(self, value: str) -> None:
        self._source.hdr_decoder = str(value)

    @property
    def om_decoder(self) -> str:
        return str(self._source.om_decoder)

    @om_decoder.setter
    def om_decoder(self, value: str) -> None:
        self._source.om_decoder = str(value)

    @property
    def ring_size(self) -> int:
        return int(self._source.ring_size)

    @ring_size.setter
    def ring_size(self, value: int) -> None:
        self._source.ring_size = int(value)

    @property
    def device_index(self) -> int:
        return int(self._source.device_index)

    @device_index.setter
    def device_index(self, value: int) -> None:
        self._source.device_index = int(value)

    def configure(
        self,
        *,
        hdr_decoder: str | None = None,
        om_decoder: str | None = None,
        ring_size: int | None = None,
        device_index: int | None = None,
    ) -> None:
        if hdr_decoder is not None:
            self.hdr_decoder = hdr_decoder
        if om_decoder is not None:
            self.om_decoder = om_decoder
        if ring_size is not None:
            self.ring_size = ring_size
        if device_index is not None:
            self.device_index = device_index

    def open(self) -> None:
        self._source.open_pairs()

    def read(self) -> FrameBatch | None:
        native_pair = self._source.next_pair()
        if native_pair is None:
            return None
        try:
            hdr = _frame_from_native(native_pair.hdr, is_hdr=True, source=self._source)
            om = _frame_from_native(native_pair.sdr, is_hdr=False, source=self._source)
            batch = FrameBatch(
                frames=(hdr, om),
                sequence=int(native_pair.sequence),
                readiness=native_pair.ready_event,
                metadata={"native_pair": native_pair},
            )

            def wait_ready(context: object | None) -> None:
                if context is None:
                    return
                with context:
                    context.wait_event(native_pair.ready_event)

            def release(context: object | None) -> None:
                self._source.release_pair(native_pair, context)

            batch.metadata["wait_ready"] = wait_ready
            batch.release_callback = release
            return batch
        except BaseException as exc:
            try:
                # A failed wrapper must not return decode slots while the
                # producer stream may still be writing their device views.
                native_pair.ready_event.synchronize()
                self._source.release_pair(native_pair)
            except BaseException as cleanup_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "decoder pair rollback failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise

    def release(self, frame: Frame | FrameBatch, context: object | None = None) -> None:
        frame.release(context)

    def close(self) -> None:
        self._source.close_pairs()

    def iter_frames(self, *args: Any, **kwargs: Any) -> Any:
        """Retain the existing fitting iterator while exposing this source as a backend."""
        return self._source.iter_frames(*args, **kwargs)

    def __enter__(self) -> "CudaDecoderBackend":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _frame_from_native(native_frame: Any, *, is_hdr: bool, source: Any) -> Frame:
    """Describe the existing normalized CUDA working-RGB view neutrally."""

    software_format = int(native_frame.software_format)
    source_format = {23: "NV12", 158: "P010LE"}.get(
        software_format,
        f"AVPixelFormat({software_format})",
    )
    try:
        fps = Fraction(str(source.config.fps_text))
        time_base = (int(fps.denominator), int(fps.numerator))
    except (AttributeError, ValueError, ZeroDivisionError):
        time_base = None
    array = native_frame.array
    return Frame(
        width=native_frame.width,
        height=native_frame.height,
        pixel_format="RGB32F",
        bit_depth=32,
        color_range="full",
        primaries="BT.2020",
        transfer="linear",
        matrix="RGB",
        timestamps=FrameTimestamps(pts=int(native_frame.sequence), time_base=time_base),
        memory_domain=MemoryDomain.DEVICE,
        backend="cuda-v05",
        device=int(source.device_index),
        ownership=Ownership.BORROWED,
        memory=FrameMemory(
            payload=array,
            domain=MemoryDomain.DEVICE,
            nbytes=int(getattr(array, "nbytes", 0)),
            device=int(source.device_index),
            ownership=Ownership.BORROWED,
            release_callback=lambda context, owned_frame=native_frame: owned_frame.release(context),
        ),
        metadata={
            "source_role": "HDR" if is_hdr else "OM",
            "source_pixel_format": source_format,
            "source_pixel_format_code": software_format,
            "native_sequence": int(native_frame.sequence),
        },
    )


class CudaResampleBackend:
    """Neutral resample facade over the existing CUDA bilinear implementation."""

    def __init__(self, implementation: Any) -> None:
        self._implementation = implementation
        self.capabilities = CUDA_CAPABILITIES

    def process(self, frame: Any, request: Any, stream: object = None) -> Any:
        return self._implementation.process(frame, request, stream)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._implementation, name)


class CudaEncoderBackend:
    """Neutral ordered encoder facade over the existing CUDA P010/NVENC bridge."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs
        self._implementation: Any = None
        self.capabilities = CUDA_CAPABILITIES

    def open(self) -> None:
        if self._implementation is not None:
            return
        from v05_gpu_native import NativeGpuEncoder

        self._implementation = NativeGpuEncoder(*self._args, **self._kwargs)

    def write(self, frame: Frame, pts: int) -> None:
        if self._implementation is None:
            raise RuntimeError("CUDA encoder backend is not open")
        if frame.pixel_format != "P010LE" or frame.memory_domain != MemoryDomain.DEVICE:
            raise ValueError("CUDA encoder backend requires a device P010LE frame")
        self._implementation.write(frame.view(), int(pts))

    def close(self) -> None:
        if self._implementation is not None:
            self._implementation.close()
            self._implementation = None

    def abort(self) -> None:
        if self._implementation is not None:
            self._implementation.abort()
            self._implementation = None


def make_cuda_frame(
    payload: Any,
    *,
    width: int,
    height: int,
    pixel_format: str,
    bit_depth: int,
    pts: int,
    device: int,
) -> Frame:
    """Wrap an existing device allocation without copying it."""

    return Frame(
        width=int(width),
        height=int(height),
        pixel_format=str(pixel_format),
        bit_depth=int(bit_depth),
        color_range="limited" if pixel_format == "P010LE" else "full",
        primaries="BT.2020",
        transfer="SMPTE2084" if pixel_format == "P010LE" else "linear",
        matrix="BT.2020_NCL" if pixel_format == "P010LE" else "RGB",
        timestamps=FrameTimestamps(pts=int(pts)),
        memory_domain=MemoryDomain.DEVICE,
        backend="cuda-v05",
        device=int(device),
        ownership=Ownership.OWNED,
        memory=FrameMemory(
            payload=payload,
            domain=MemoryDomain.DEVICE,
            nbytes=int(getattr(payload, "nbytes", 0)),
            device=int(device),
            ownership=Ownership.OWNED,
        ),
    )


def make_cuda_frame_factory(device: int) -> Callable[..., Frame]:
    """Return a neutral frame factory backed by the selected CUDA device."""

    def factory(
        payload: Any,
        width: int,
        height: int,
        pixel_format: str,
        bit_depth: int,
        pts: int,
    ) -> Frame:
        return make_cuda_frame(
            payload,
            width=width,
            height=height,
            pixel_format=pixel_format,
            bit_depth=bit_depth,
            pts=pts,
            device=int(device),
        )

    return factory


def make_cuda_resample_backend() -> CudaResampleBackend:
    """Construct the existing CUDA output resampler behind its contract."""

    import cupy
    from v05_gpu_native import GpuOutputResampler

    return CudaResampleBackend(GpuOutputResampler(cupy))


def make_cuda_encoder_factory(
    *,
    bridge_path: Path,
    ffmpeg_bin: Path,
    device_index: int = 0,
    qp: int = 18,
) -> Callable[[Path, int, int, int, int], CudaEncoderBackend]:
    """Bind CUDA/NVENC options outside the neutral render loop."""

    def factory(
        output: Path,
        width: int,
        height: int,
        fps_num: int,
        fps_den: int,
    ) -> CudaEncoderBackend:
        return CudaEncoderBackend(
            output,
            width,
            height,
            fps_num,
            fps_den,
            bridge_path=bridge_path,
            ffmpeg_bin=ffmpeg_bin,
            device_index=int(device_index),
            qp=int(qp),
        )

    return factory


__all__ = [
    "CUDA_CAPABILITIES",
    "CudaComputeBackend",
    "CudaDecoderBackend",
    "CudaEncoderBackend",
    "CudaResampleBackend",
    "make_cuda_encoder_factory",
    "make_cuda_frame",
    "make_cuda_frame_factory",
    "make_cuda_resample_backend",
]
