"""Optional CPU/GPU backends for the existing SDR-to-HDR transform.

The CPU backend delegates to the production implementation.  The GPU backend
is an optional float64 adapter that mirrors the same numerical stages without
changing fitting, LUT construction, geometry, or compositing semantics.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import _get_pq_oetf_sqrt_lut
from auto_openmatte.processing.luminance import build_curve_lut
from auto_openmatte.processing.transform import (
    _LUM_B_2020,
    _LUM_G_2020,
    _LUM_R_2020,
    _M_709_TO_2020,
    apply_shot_transform,
)

logger = logging.getLogger(__name__)

_PEAK_NITS = 10000.0
_PQ_LUT_SIZE = 65536
_GPU_TRANSFER_PAIR = ("bt709", "smpte2084")


class BackendError(RuntimeError):
    """Base class for backend availability and runtime failures."""


class BackendUnavailableError(BackendError):
    """Raised when an optional backend cannot be initialized."""


class BackendRuntimeError(BackendError):
    """Raised when a backend fails while transforming an ROI."""


class TransformBackend(Protocol):
    """Common contract used by CPU and optional GPU transform backends."""

    @property
    def name(self) -> str:
        ...

    def prepare_shot(
        self,
        transform: ShotTransform,
        *,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> "TransformWorkspace":
        ...

    def transform_roi(
        self,
        om_frame: np.ndarray,
        transform: ShotTransform,
        *,
        workspace: "TransformWorkspace | None" = None,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> np.ndarray:
        ...


@dataclass
class TransformWorkspace:
    """Persistent per-shot state shared by repeated ROI transformations."""

    backend_name: str
    signature: str
    curve_lut: dict[str, Any] | None
    sdr_transfer: str
    hdr_transfer: str
    peak_nits: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _GPUBufferSet:
    """Preallocated float64 and index buffers for one ROI shape."""

    shape: tuple[int, int, int]
    input_flat: Any
    linear_709: Any
    matrix_flat: Any
    color_flat: Any
    chroma_flat: Any
    luminance: Any
    mapped_luminance: Any
    ratio: Any
    bridge_mask: Any
    fitted_mask: Any
    safe_mask: Any
    normalized: Any
    position: Any
    fraction: Any
    lower: Any
    upper: Any
    lower_values: Any
    upper_values: Any
    pq_normalized: Any
    pq_position: Any
    pq_fraction: Any
    pq_lower: Any
    pq_upper: Any
    pq_lower_values: Any
    pq_upper_values: Any
    output_flat: Any

    @property
    def bytes(self) -> int:
        arrays = (
            self.input_flat,
            self.linear_709,
            self.matrix_flat,
            self.color_flat,
            self.chroma_flat,
            self.luminance,
            self.mapped_luminance,
            self.ratio,
            self.bridge_mask,
            self.fitted_mask,
            self.safe_mask,
            self.normalized,
            self.position,
            self.fraction,
            self.lower,
            self.upper,
            self.lower_values,
            self.upper_values,
            self.pq_normalized,
            self.pq_position,
            self.pq_fraction,
            self.pq_lower,
            self.pq_upper,
            self.pq_lower_values,
            self.pq_upper_values,
            self.output_flat,
        )
        return int(sum(int(array.nbytes) for array in arrays))


@dataclass
class GPUTransformWorkspace(TransformWorkspace):
    """Per-shot GPU LUT state plus one reusable ROI workspace."""

    cupy: Any = None
    curve_lut_gpu: Any = None
    curve_start_gpu: Any = None
    curve_start_hdr_gpu: Any = None
    curve_inv_range_gpu: Any = None
    pq_lut_gpu: Any = None
    matrix_gpu: Any = None
    color_matrix_gpu: Any = None
    buffers: _GPUBufferSet | None = None
    device_id: int = 0


# The loader is injectable only to make unavailable-GPU fallback deterministic
# in tests.  Normal callers use importlib.import_module lazily.


def _canonical_transfer(value: str) -> str:
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _transform_signature(
    transform: ShotTransform,
    *,
    sdr_transfer: str,
    hdr_transfer: str,
    peak_nits: float,
) -> str:
    payload = {
        "shot_id": transform.shot_id,
        "curve": transform.luminance_curve,
        "color_matrix": transform.color_matrix,
        "saturation": transform.saturation,
        "sdr_transfer": sdr_transfer,
        "hdr_transfer": hdr_transfer,
        "peak_nits": peak_nits,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_roi(om_frame: np.ndarray) -> tuple[int, int, int]:
    if not isinstance(om_frame, np.ndarray):
        raise TypeError("om_frame must be a NumPy array at the backend boundary")
    if om_frame.ndim != 3 or om_frame.shape[2] != 3:
        raise ValueError("om_frame must have shape (height, width, 3)")
    if om_frame.shape[0] == 0 or om_frame.shape[1] == 0:
        raise ValueError("om_frame must contain at least one pixel")
    return tuple(int(value) for value in om_frame.shape)


class CPUTransformBackend:
    """Reference backend delegating to the unchanged production transform."""

    name = "cpu"

    def __init__(self) -> None:
        self._workspaces: dict[str, TransformWorkspace] = {}

    def prepare_shot(
        self,
        transform: ShotTransform,
        *,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> TransformWorkspace:
        signature = _transform_signature(
            transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
        )
        workspace = self._workspaces.get(signature)
        if workspace is not None:
            return workspace

        curve_lut = None
        if transform.luminance_curve and len(transform.luminance_curve) >= 2:
            curve_lut = build_curve_lut(transform.luminance_curve)
        workspace = TransformWorkspace(
            backend_name=self.name,
            signature=signature,
            curve_lut=curve_lut,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
            metadata={"curve_lut_built_once": curve_lut is not None},
        )
        self._workspaces[signature] = workspace
        return workspace

    def transform_roi(
        self,
        om_frame: np.ndarray,
        transform: ShotTransform,
        *,
        workspace: TransformWorkspace | None = None,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> np.ndarray:
        _validate_roi(om_frame)
        if workspace is None:
            workspace = self.prepare_shot(
                transform,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
        expected = _transform_signature(
            transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
        )
        if workspace.signature != expected or workspace.backend_name != self.name:
            raise ValueError("workspace does not match the requested CPU transform")
        return apply_shot_transform(
            om_frame,
            transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
            prebuilt_lut=workspace.curve_lut,
        )

    def clear(self) -> None:
        """Drop cached CPU shot workspaces."""
        self._workspaces.clear()


class GPUTransformBackend:
    """Optional float64 CuPy backend matching the CPU production stages."""

    name = "gpu"

    def __init__(
        self,
        *,
        device_id: int = 0,
        cupy_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.device_id = device_id
        self._cupy_loader = cupy_loader
        self._cupy: Any = None
        self._workspaces: dict[str, GPUTransformWorkspace] = {}
        self._pq_lut_gpu: Any = None
        self._matrix_gpu: Any = None
        self._initialized = False

    @property
    def cupy(self) -> Any:
        self._ensure_initialized()
        return self._cupy

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            loader = self._cupy_loader or (lambda: importlib.import_module("cupy"))
            cupy = loader()
            if cupy is None:
                raise ImportError("CuPy loader returned None")
            if int(cupy.cuda.runtime.getDeviceCount()) <= self.device_id:
                raise RuntimeError(f"CUDA device {self.device_id} is unavailable")
            with cupy.cuda.Device(self.device_id):
                # Force context/device validation before the first shot setup.
                cupy.cuda.runtime.getDevice()
            self._cupy = cupy
            self._initialized = True
        except Exception as exc:
            raise BackendUnavailableError(
                f"GPU backend initialization failed: {exc}"
            ) from exc

    def _require_supported_transfers(self, sdr_transfer: str, hdr_transfer: str) -> None:
        pair = (_canonical_transfer(sdr_transfer), _canonical_transfer(hdr_transfer))
        if pair != _GPU_TRANSFER_PAIR:
            raise BackendUnavailableError(
                "GPU backend currently supports only bt709 -> smpte2084; "
                f"received {sdr_transfer!r} -> {hdr_transfer!r}"
            )

    def prepare_shot(
        self,
        transform: ShotTransform,
        *,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> GPUTransformWorkspace:
        self._ensure_initialized()
        self._require_supported_transfers(sdr_transfer, hdr_transfer)
        signature = _transform_signature(
            transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
        )
        workspace = self._workspaces.get(signature)
        if workspace is not None:
            return workspace

        cupy = self._cupy
        try:
            with cupy.cuda.Device(self.device_id):
                curve_lut = None
                curve_lut_gpu = None
                curve_start_gpu = None
                curve_start_hdr_gpu = None
                curve_inv_range_gpu = None
                if transform.luminance_curve and len(transform.luminance_curve) >= 2:
                    curve_lut = build_curve_lut(transform.luminance_curve)
                    curve_lut_gpu = cupy.asarray(curve_lut["lut"], dtype=cupy.float64)
                    curve_start_gpu = cupy.asarray(
                        curve_lut["curve_start_linear"], dtype=cupy.float64
                    )
                    curve_start_hdr_gpu = cupy.asarray(
                        curve_lut["curve_start_hdr_linear"], dtype=cupy.float64
                    )
                    curve_inv_range_gpu = cupy.asarray(
                        curve_lut["inv_range"], dtype=cupy.float64
                    )

                if self._pq_lut_gpu is None:
                    self._pq_lut_gpu = cupy.asarray(
                        _get_pq_oetf_sqrt_lut(), dtype=cupy.float64
                    )
                if self._matrix_gpu is None:
                    self._matrix_gpu = cupy.asarray(
                        _M_709_TO_2020, dtype=cupy.float64
                    )

                color_matrix_gpu = None
                color_matrix = np.asarray(transform.color_matrix, dtype=np.float64)
                if transform.color_matrix and not np.allclose(
                    color_matrix, np.eye(3), atol=0.001
                ):
                    color_matrix_gpu = cupy.asarray(color_matrix, dtype=cupy.float64)
                cupy.cuda.Stream.null.synchronize()
        except BackendError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"GPU LUT/device setup failed: {exc}"
            ) from exc

        workspace = GPUTransformWorkspace(
            backend_name=self.name,
            signature=signature,
            curve_lut=curve_lut,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
            metadata={
                "dtype": "float64",
                "curve_lut_built_once": curve_lut is not None,
                "curve_lut_copied_once": curve_lut_gpu is not None,
                "pq_lut_copied_once_per_backend": True,
                "workspace_allocations": 0,
                "transform": transform,
            },
            cupy=cupy,
            curve_lut_gpu=curve_lut_gpu,
            curve_start_gpu=curve_start_gpu,
            curve_start_hdr_gpu=curve_start_hdr_gpu,
            curve_inv_range_gpu=curve_inv_range_gpu,
            pq_lut_gpu=self._pq_lut_gpu,
            matrix_gpu=self._matrix_gpu,
            color_matrix_gpu=color_matrix_gpu,
            device_id=self.device_id,
        )
        self._workspaces[signature] = workspace
        return workspace

    def _get_buffers(
        self,
        workspace: GPUTransformWorkspace,
        shape: tuple[int, int, int],
    ) -> _GPUBufferSet:
        if workspace.buffers is not None and workspace.buffers.shape == shape:
            return workspace.buffers
        cupy = workspace.cupy
        elements = shape[0] * shape[1]

        def float_buffer() -> Any:
            return cupy.empty((elements, 3), dtype=cupy.float64)

        def scalar_buffer() -> Any:
            return cupy.empty(elements, dtype=cupy.float64)

        def bool_buffer() -> Any:
            return cupy.empty(elements, dtype=cupy.bool_)

        def index_buffer() -> Any:
            return cupy.empty(elements, dtype=cupy.int64)

        def pixel_scalar_buffer() -> Any:
            return cupy.empty((elements, 3), dtype=cupy.float64)

        def pixel_index_buffer() -> Any:
            return cupy.empty((elements, 3), dtype=cupy.int64)
        workspace.buffers = _GPUBufferSet(
            shape=shape,
            input_flat=float_buffer(),
            linear_709=float_buffer(),
            matrix_flat=float_buffer(),
            color_flat=float_buffer(),
            chroma_flat=float_buffer(),
            luminance=scalar_buffer(),
            mapped_luminance=scalar_buffer(),
            ratio=scalar_buffer(),
            bridge_mask=bool_buffer(),
            fitted_mask=bool_buffer(),
            safe_mask=bool_buffer(),
            normalized=scalar_buffer(),
            position=scalar_buffer(),
            fraction=scalar_buffer(),
            lower=index_buffer(),
            upper=index_buffer(),
            lower_values=scalar_buffer(),
            upper_values=scalar_buffer(),
            pq_normalized=pixel_scalar_buffer(),
            pq_position=pixel_scalar_buffer(),
            pq_fraction=pixel_scalar_buffer(),
            pq_lower=pixel_index_buffer(),
            pq_upper=pixel_index_buffer(),
            pq_lower_values=pixel_scalar_buffer(),
            pq_upper_values=pixel_scalar_buffer(),
            output_flat=float_buffer(),
        )
        workspace.metadata["workspace_allocations"] = int(
            workspace.metadata.get("workspace_allocations", 0) + 1
        )
        return workspace.buffers

    def _upload_to_device(
        self, om_frame: np.ndarray, workspace: GPUTransformWorkspace
    ) -> _GPUBufferSet:
        shape = _validate_roi(om_frame)
        buffers = self._get_buffers(workspace, shape)
        host_frame = np.asarray(om_frame, dtype=np.float64, order="C")
        buffers.input_flat.set(host_frame.reshape(-1, 3))
        return buffers

    def _transform_device(
        self,
        workspace: GPUTransformWorkspace,
        buffers: _GPUBufferSet,
    ) -> None:
        cupy = workspace.cupy
        linear_709 = buffers.linear_709
        cupy.clip(buffers.input_flat, 0.0, 1.0, out=linear_709)
        cupy.power(linear_709, 2.4, out=linear_709)

        cupy.matmul(linear_709, workspace.matrix_gpu.T, out=buffers.matrix_flat)
        cupy.maximum(buffers.matrix_flat, 0.0, out=buffers.matrix_flat)
        linear = buffers.matrix_flat

        if workspace.curve_lut_gpu is not None:
            luminance = buffers.luminance
            cupy.multiply(linear[:, 0], _LUM_R_2020, out=luminance)
            cupy.multiply(linear[:, 1], _LUM_G_2020, out=buffers.ratio)
            cupy.add(luminance, buffers.ratio, out=luminance)
            cupy.multiply(linear[:, 2], _LUM_B_2020, out=buffers.ratio)
            cupy.add(luminance, buffers.ratio, out=luminance)

            cupy.greater(luminance, 0.0, out=buffers.bridge_mask)
            cupy.less(luminance, workspace.curve_start_gpu, out=buffers.fitted_mask)
            cupy.logical_and(
                buffers.bridge_mask, buffers.fitted_mask, out=buffers.bridge_mask
            )
            cupy.divide(
                luminance, workspace.curve_start_gpu, out=buffers.ratio
            )
            cupy.multiply(
                buffers.ratio,
                workspace.curve_start_hdr_gpu,
                out=buffers.ratio,
            )
            buffers.mapped_luminance.fill(0.0)
            cupy.copyto(
                buffers.mapped_luminance,
                buffers.ratio,
                where=buffers.bridge_mask,
            )

            cupy.greater_equal(luminance, workspace.curve_start_gpu, out=buffers.fitted_mask)
            cupy.subtract(
                luminance, workspace.curve_start_gpu, out=buffers.position
            )
            cupy.multiply(
                buffers.position,
                workspace.curve_inv_range_gpu,
                out=buffers.position,
            )
            cupy.clip(buffers.position, 0.0, 1.0, out=buffers.position)
            buffers.lower[...] = buffers.position * (workspace.curve_lut_gpu.size - 1)
            cupy.clip(
                buffers.lower,
                0,
                workspace.curve_lut_gpu.size - 1,
                out=buffers.lower,
            )
            cupy.take(
                workspace.curve_lut_gpu,
                buffers.lower,
                out=buffers.ratio,
            )
            cupy.copyto(
                buffers.mapped_luminance,
                buffers.ratio,
                where=buffers.fitted_mask,
            )

            cupy.greater(luminance, 1e-6, out=buffers.safe_mask)
            buffers.ratio.fill(1.0)
            cupy.copyto(buffers.ratio, luminance, where=buffers.safe_mask)
            cupy.divide(
                buffers.mapped_luminance,
                buffers.ratio,
                out=buffers.mapped_luminance,
            )
            cupy.copyto(
                buffers.ratio,
                buffers.mapped_luminance,
                where=buffers.safe_mask,
            )
            cupy.multiply(
                linear,
                buffers.ratio[:, None],
                out=linear,
            )
            cupy.maximum(linear, 0.0, out=linear)

        if workspace.color_matrix_gpu is not None:
            cupy.matmul(
                linear,
                workspace.color_matrix_gpu.T,
                out=buffers.color_flat,
            )
            cupy.maximum(buffers.color_flat, 0.0, out=buffers.color_flat)
            linear = buffers.color_flat

        transform = self._transform_for_signature(workspace.signature)
        if abs(transform.saturation - 1.0) > 0.01:
            luminance = buffers.luminance
            cupy.multiply(linear[:, 0], _LUM_R_2020, out=luminance)
            cupy.multiply(linear[:, 1], _LUM_G_2020, out=buffers.ratio)
            cupy.add(luminance, buffers.ratio, out=luminance)
            cupy.multiply(linear[:, 2], _LUM_B_2020, out=buffers.ratio)
            cupy.add(luminance, buffers.ratio, out=luminance)
            cupy.subtract(linear, luminance[:, None], out=buffers.chroma_flat)
            cupy.multiply(
                buffers.chroma_flat,
                transform.saturation,
                out=buffers.chroma_flat,
            )
            cupy.add(
                buffers.chroma_flat,
                luminance[:, None],
                out=linear,
            )
            cupy.maximum(linear, 0.0, out=linear)

        cupy.multiply(
            linear,
            workspace.peak_nits / _PEAK_NITS,
            out=buffers.pq_normalized,
        )
        cupy.clip(buffers.pq_normalized, 0.0, 1.0, out=buffers.pq_normalized)
        cupy.sqrt(buffers.pq_normalized, out=buffers.pq_position)
        cupy.multiply(
            buffers.pq_position,
            workspace.pq_lut_gpu.size - 1,
            out=buffers.pq_position,
        )
        buffers.pq_lower[...] = buffers.pq_position
        cupy.clip(
            buffers.pq_lower,
            0,
            workspace.pq_lut_gpu.size - 2,
            out=buffers.pq_lower,
        )
        cupy.subtract(
            buffers.pq_position,
            buffers.pq_lower,
            out=buffers.pq_fraction,
        )
        cupy.add(buffers.pq_lower, 1, out=buffers.pq_upper)
        cupy.take(
            workspace.pq_lut_gpu,
            buffers.pq_lower,
            out=buffers.pq_lower_values,
        )
        cupy.take(
            workspace.pq_lut_gpu,
            buffers.pq_upper,
            out=buffers.pq_upper_values,
        )
        cupy.multiply(
            buffers.pq_lower_values,
            1.0 - buffers.pq_fraction,
            out=buffers.output_flat,
        )
        cupy.multiply(
            buffers.pq_upper_values,
            buffers.pq_fraction,
            out=buffers.pq_upper_values,
        )
        cupy.add(
            buffers.output_flat,
            buffers.pq_upper_values,
            out=buffers.output_flat,
        )
        cupy.clip(buffers.output_flat, 0.0, 1.0, out=buffers.output_flat)

    def _transform_for_signature(self, signature: str) -> ShotTransform:
        for workspace in self._workspaces.values():
            if workspace.signature == signature:
                return workspace.metadata["transform"]
        raise BackendRuntimeError("GPU workspace transform metadata is unavailable")

    def _download_to_host(
        self, workspace: GPUTransformWorkspace, buffers: _GPUBufferSet
    ) -> np.ndarray:
        output = workspace.cupy.asnumpy(buffers.output_flat)
        return output.reshape(buffers.shape)

    def transform_roi(
        self,
        om_frame: np.ndarray,
        transform: ShotTransform,
        *,
        workspace: TransformWorkspace | None = None,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> np.ndarray:
        shape = _validate_roi(om_frame)
        if workspace is None:
            workspace = self.prepare_shot(
                transform,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
        if not isinstance(workspace, GPUTransformWorkspace):
            raise ValueError("workspace does not belong to the GPU backend")
        expected = _transform_signature(
            transform,
            sdr_transfer=sdr_transfer,
            hdr_transfer=hdr_transfer,
            peak_nits=peak_nits,
        )
        if workspace.signature != expected or workspace.backend_name != self.name:
            raise ValueError("workspace does not match the requested GPU transform")
        workspace.metadata["transform"] = transform
        try:
            with workspace.cupy.cuda.Device(self.device_id):
                buffers = self._upload_to_device(om_frame, workspace)
                self._transform_device(workspace, buffers)
                workspace.cupy.cuda.Stream.null.synchronize()
                return self._download_to_host(workspace, buffers)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendRuntimeError(f"GPU transform failed for {shape}: {exc}") from exc

    def memory_report(self, workspace: GPUTransformWorkspace) -> dict[str, Any]:
        """Return current persistent-buffer and CuPy pool memory counters."""
        cupy = workspace.cupy
        pool = cupy.get_default_memory_pool()
        try:
            free_bytes, total_bytes = cupy.cuda.Device(self.device_id).mem_info
        except (AttributeError, RuntimeError):
            free_bytes, total_bytes = None, None
        persistent_buffers = workspace.buffers.bytes if workspace.buffers else 0
        curve_bytes = (
            int(workspace.curve_lut_gpu.nbytes)
            if workspace.curve_lut_gpu is not None
            else 0
        )
        pq_bytes = int(workspace.pq_lut_gpu.nbytes) if workspace.pq_lut_gpu is not None else 0
        matrix_bytes = int(workspace.matrix_gpu.nbytes)
        color_bytes = (
            int(workspace.color_matrix_gpu.nbytes)
            if workspace.color_matrix_gpu is not None
            else 0
        )
        return {
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_reserved_bytes": int(pool.total_bytes()),
            "device_free_bytes": int(free_bytes) if free_bytes is not None else None,
            "device_total_bytes": int(total_bytes) if total_bytes is not None else None,
            "persistent_roi_buffers_bytes": persistent_buffers,
            "curve_lut_device_bytes": curve_bytes,
            "pq_lut_device_bytes": pq_bytes,
            "matrix_device_bytes": matrix_bytes,
            "color_matrix_device_bytes": color_bytes,
            "persistent_bytes_total": persistent_buffers
            + curve_bytes
            + pq_bytes
            + matrix_bytes
            + color_bytes,
            "workspace_allocations": workspace.metadata.get("workspace_allocations", 0),
        }

    def clear(self) -> None:
        """Release backend references; caller may then trim CuPy's pool."""
        self._workspaces.clear()
        self._pq_lut_gpu = None
        self._matrix_gpu = None


class FallbackTransformBackend:
    """GPU-primary backend that permanently falls back to CPU on GPU failure."""

    def __init__(
        self,
        primary: TransformBackend,
        fallback: CPUTransformBackend | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or CPUTransformBackend()
        self._active: TransformBackend = primary
        self.fallback_reason: str | None = None

    @property
    def name(self) -> str:
        return self._active.name if self._active is self.primary else "cpu-fallback"

    @property
    def using_gpu(self) -> bool:
        return self._active is self.primary

    def _switch_to_cpu(self, exc: BackendError) -> None:
        self.fallback_reason = str(exc)
        self._active = self.fallback
        logger.warning("GPU transform backend unavailable; using CPU fallback: %s", exc)

    def prepare_shot(
        self,
        transform: ShotTransform,
        *,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> TransformWorkspace:
        if self._active is self.fallback:
            return self.fallback.prepare_shot(
                transform,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
        try:
            return self.primary.prepare_shot(
                transform,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
        except BackendError as exc:
            self._switch_to_cpu(exc)
            return self.fallback.prepare_shot(
                transform,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )

    def transform_roi(
        self,
        om_frame: np.ndarray,
        transform: ShotTransform,
        *,
        workspace: TransformWorkspace | None = None,
        sdr_transfer: str = "bt709",
        hdr_transfer: str = "smpte2084",
        peak_nits: float = _PEAK_NITS,
    ) -> np.ndarray:
        if self._active is self.fallback:
            return self.fallback.transform_roi(
                om_frame,
                transform,
                workspace=None,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
        try:
            return self.primary.transform_roi(
                om_frame,
                transform,
                workspace=workspace,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )
        except BackendError as exc:
            self._switch_to_cpu(exc)
            return self.fallback.transform_roi(
                om_frame,
                transform,
                workspace=None,
                sdr_transfer=sdr_transfer,
                hdr_transfer=hdr_transfer,
                peak_nits=peak_nits,
            )

    def clear(self) -> None:
        for backend in (self.primary, self.fallback):
            clear = getattr(backend, "clear", None)
            if clear is not None:
                clear()


def create_transform_backend(
    experimental_gpu: bool = False,
    *,
    device_id: int = 0,
    cupy_loader: Callable[[], Any] | None = None,
) -> TransformBackend:
    """Create CPU by default, or GPU-primary with sticky CPU fallback."""
    cpu = CPUTransformBackend()
    if not experimental_gpu:
        return cpu
    gpu = GPUTransformBackend(device_id=device_id, cupy_loader=cupy_loader)
    return FallbackTransformBackend(gpu, fallback=cpu)


def transform_roi(
    om_frame: np.ndarray,
    transform: ShotTransform,
    *,
    backend: TransformBackend | None = None,
    workspace: TransformWorkspace | None = None,
    sdr_transfer: str = "bt709",
    hdr_transfer: str = "smpte2084",
    peak_nits: float = _PEAK_NITS,
) -> np.ndarray:
    """Transform one ROI through the selected backend.

    With no explicit backend this function uses a CPU backend.  Callers that
    process multiple frames should retain a backend and prepared workspace.
    """
    selected = backend or CPUTransformBackend()
    return selected.transform_roi(
        om_frame,
        transform,
        workspace=workspace,
        sdr_transfer=sdr_transfer,
        hdr_transfer=hdr_transfer,
        peak_nits=peak_nits,
    )


__all__ = [
    "BackendError",
    "BackendRuntimeError",
    "BackendUnavailableError",
    "CPUTransformBackend",
    "FallbackTransformBackend",
    "GPUTransformBackend",
    "GPUTransformWorkspace",
    "TransformBackend",
    "TransformWorkspace",
    "create_transform_backend",
    "transform_roi",
]
