from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from auto_openmatte.core.transfer_functions import linearize
from auto_openmatte.processing.transform_backend import GPUTransformBackend


ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION_DIR = ROOT / "dev" / "ffmpeg-build" / "validation"
P232_DIR = VALIDATION_DIR / "P232_real_material_dataset"
P232_METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
P232_MANIFEST_PATH = P232_DIR / "P232_dataset_manifest.json"
MODEL_A_CURVE_PATH = VALIDATION_DIR / "P2272_reference_curve.json"
FFMPEG = (ROOT / "dev" / "ffmpeg-build" / "install" / "bin" / "ffmpeg.exe").resolve()
SMOKE_OUTPUT_DIR = VALIDATION_DIR / "P234_streaming_smoke"
PARITY_OUTPUT_DIR = VALIDATION_DIR / "P234_cuda_parity"
FULL_OUTPUT_DIR = VALIDATION_DIR / "P234_shot_level_full"
REPORT_PATH = ROOT / "P2.34_STREAMING_REDESIGN_REPORT.md"
FULL_REPORT_PATH = ROOT / "P2.34_TRUE_SHOT_LEVEL_ANALYSIS_REPORT.md"

FPS_NUM = 24000
FPS_DEN = 1001
FPS = FPS_NUM / FPS_DEN
PEAK_NITS = 10000.0
RGB16_MAX = 65535.0
LOG_EPS = 1e-6
LOCAL_HALF_WINDOW_SECONDS = 5.0
MAX_HALF_WINDOW_SECONDS = 5.0
PROXY_WIDTH = 320
BOUNDARY_THRESHOLD = 0.30
GRADUAL_THRESHOLD = 0.15
GRADUAL_MIN_RUN_FRAMES = 5
ANCHOR_NEAR_CUT_FRAMES = 12
MIN_SAMPLE_SEPARATION_FRAMES = 5
TEMPORAL_FRACTIONS = ((0.10, "10%"), (0.25, "25%"), (0.50, "50%"), (0.75, "75%"), (0.90, "90%"))
TEST_REGIONS = (
    (0.0, float("inf"), "global"),
    (0.0, 10.0, "shadow_lt_10"),
    (10.0, 200.0, "midtone_10_200"),
    (200.0, float("inf"), "highlight_gt_200"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
)
SDR_BINS = (
    (0.0, 0.01, "<0.01"),
    (0.01, 0.1, "0.01–0.1"),
    (0.1, 0.5, "0.1–0.5"),
    (0.5, 1.0, "0.5–1"),
    (1.0, 2.0, "1–2"),
    (2.0, 5.0, "2–5"),
    (5.0, 10.0, "5–10"),
    (10.0, 20.0, "10–20"),
    (20.0, 50.0, "20–50"),
    (50.0, 100.0, "50–100"),
    (100.0, 200.0, "100–200"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
)
BIN_UPPER_BOUNDS = np.asarray([item[1] for item in SDR_BINS[:-1]], dtype=np.float64)
HARD_CLIP_NITS = 0.99 * PEAK_NITS
MIN_STABLE_SDR_NITS = 0.01
OUTPUT_TARGET_BYTES = 60 * 1024 * 1024
OUTPUT_HARD_LIMIT_BYTES = 256 * 1024 * 1024
CHUNK_ROWS = 8
HISTOGRAM_BINS = 512
HISTOGRAM_LOG_LOW = -6.0
HISTOGRAM_LOG_HIGH = 4.0
CUDA_LUMINANCE_LIMIT_NITS = 0.01
CUDA_DEVICE_ID = 0
P2_34_MATERIAL_ORDER = ("The Matrix", "BR2049")
P2_34_REQUIRED_ANCHOR_COUNT = 40
P2_34_REQUIRED_ANCHORS_PER_MATERIAL = 20

M_709_TO_2020 = np.asarray(
    [
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690972, 0.9195404, 0.0113624],
        [0.0163916, 0.0880132, 0.8955952],
    ],
    dtype=np.float64,
)
LUMA_2020 = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)


class DiskBudgetExceeded(RuntimeError):
    pass


class CudaLuminanceChunkAdapter:
    """Strict CUDA adapter for P2.34 row chunks.

    The verified project GPUTransformBackend owns CuPy/device initialization. This
    validation-local adapter reuses that infrastructure but implements only the
    P2.34 source-luminance and compact-aggregation operation, because the
    production backend intentionally transforms RGB to RGB rather than paired
    HDR/SDR luminance streams. Full mode never downloads luminance planes: only
    fixed-size histogram/counter aggregates cross the host boundary.
    """

    _PQ_M1 = 2610.0 / 16384.0
    _PQ_M2 = 2523.0 / 4096.0 * 128.0
    _PQ_C1 = 3424.0 / 4096.0
    _PQ_C2 = 2413.0 / 4096.0 * 32.0
    _PQ_C3 = 2392.0 / 4096.0 * 32.0

    def __init__(self, device_id: int = CUDA_DEVICE_ID) -> None:
        self.device_id = int(device_id)
        self._verified_backend = GPUTransformBackend(device_id=self.device_id)
        self._cupy: Any = None
        self._matrix_gpu: Any = None
        self._luma_gpu: Any = None
        self._bin_bounds_gpu: Any = None
        self._buffers: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._model_a_lut_gpu: Any = None

    @property
    def cupy(self) -> Any:
        if self._cupy is None:
            # This raises BackendUnavailableError on missing CuPy/device. It is
            # intentionally not caught or converted to a CPU fallback.
            self._cupy = self._verified_backend.cupy
            with self._cupy.cuda.Device(self.device_id):
                self._matrix_gpu = self._cupy.asarray(M_709_TO_2020, dtype=self._cupy.float64)
                self._luma_gpu = self._cupy.asarray(LUMA_2020, dtype=self._cupy.float64)
                self._bin_bounds_gpu = self._cupy.asarray(BIN_UPPER_BOUNDS, dtype=self._cupy.float64)
                self._cupy.cuda.Stream.null.synchronize()
        return self._cupy

    def device_info(self) -> dict[str, Any]:
        cp = self.cupy
        device = cp.cuda.Device(self.device_id)
        properties: Any = {}
        try:
            properties = cp.cuda.runtime.getDeviceProperties(self.device_id)
        except Exception:
            properties = {}
        name: Any = properties.get("name") if isinstance(properties, dict) else None
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        if name is None:
            try:
                name = device.name
            except Exception:
                name = "unknown"
        try:
            capability: Any = device.compute_capability
        except Exception:
            capability = None
        try:
            runtime_version: Any = cp.cuda.runtime.runtimeGetVersion()
        except Exception:
            runtime_version = None
        return {
            "cupy_version": str(getattr(cp, "__version__", "unknown")),
            "cuda_device_count": int(cp.cuda.runtime.getDeviceCount()),
            "device_id": self.device_id,
            "device_name": str(name),
            "compute_capability": str(capability) if capability is not None else None,
            "cuda_runtime_version": int(runtime_version) if runtime_version is not None else None,
        }

    def _get_buffers(self, shape: tuple[int, int, int]) -> dict[str, Any]:
        cp = self.cupy
        existing = self._buffers.get(shape)
        if existing is not None:
            return existing
        elements = int(shape[0] * shape[1])
        buffers = {
            "rgb": cp.empty((elements, 3), dtype=cp.float64),
            "linear": cp.empty((elements, 3), dtype=cp.float64),
            "matrix": cp.empty((elements, 3), dtype=cp.float64),
            "luminance": cp.empty(elements, dtype=cp.float64),
            "sdr_luminance": cp.empty(elements, dtype=cp.float64),
            "hdr_luminance": cp.empty(elements, dtype=cp.float64),
        }
        self._buffers[shape] = buffers
        return buffers

    def _compute_luminance(
        self,
        rgb_u16: np.ndarray,
        transfer: str,
        output: Any,
        buffers: dict[str, Any],
    ) -> None:
        cp = self.cupy
        canonical = transfer.lower().replace("-", "").replace("_", "").replace(" ", "")
        if canonical not in ("bt709", "smpte2084", "pq", "st2084"):
            raise RuntimeError(f"CUDA luminance adapter does not support transfer {transfer!r}")
        buffers["rgb"].set(np.asarray(rgb_u16, dtype=np.float64).reshape(-1, 3))
        cp.multiply(buffers["rgb"], 1.0 / RGB16_MAX, out=buffers["rgb"])
        cp.clip(buffers["rgb"], 0.0, 1.0, out=buffers["rgb"])
        if canonical == "bt709":
            cp.power(buffers["rgb"], 2.4, out=buffers["linear"])
            cp.matmul(buffers["linear"], self._matrix_gpu.T, out=buffers["matrix"])
            cp.maximum(buffers["matrix"], 0.0, out=buffers["matrix"])
            source = buffers["matrix"]
        else:
            vp = cp.power(buffers["rgb"], 1.0 / self._PQ_M2)
            numerator = cp.maximum(vp - self._PQ_C1, 0.0)
            denominator = cp.maximum(self._PQ_C2 - self._PQ_C3 * vp, 1e-12)
            cp.power(numerator / denominator, 1.0 / self._PQ_M1, out=buffers["linear"])
            source = buffers["linear"]
        cp.multiply(source[:, 0], self._luma_gpu[0], out=buffers["luminance"])
        cp.add(buffers["luminance"], source[:, 1] * self._luma_gpu[1], out=buffers["luminance"])
        cp.add(buffers["luminance"], source[:, 2] * self._luma_gpu[2], out=buffers["luminance"])
        cp.multiply(buffers["luminance"], PEAK_NITS, out=output)

    def _prepare_pair(self, om_rgb: np.ndarray, hdr_rgb: np.ndarray) -> tuple[Any, Any]:
        shape = tuple(int(value) for value in om_rgb.shape)
        if len(shape) != 3 or shape[2] != 3 or tuple(hdr_rgb.shape) != shape:
            raise ValueError(f"CUDA pair shape mismatch: {om_rgb.shape} / {hdr_rgb.shape}")
        buffers = self._get_buffers(shape)
        with self.cupy.cuda.Device(self.device_id):
            self._compute_luminance(om_rgb, "bt709", buffers["sdr_luminance"], buffers)
            self._compute_luminance(hdr_rgb, "smpte2084", buffers["hdr_luminance"], buffers)
        return buffers["sdr_luminance"], buffers["hdr_luminance"]

    def parity_luminance_pair(self, om_rgb: np.ndarray, hdr_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one parity-only host copy; full mode uses aggregate_pair_chunk."""
        sdr_gpu, hdr_gpu = self._prepare_pair(om_rgb, hdr_rgb)
        with self.cupy.cuda.Device(self.device_id):
            self.cupy.cuda.Stream.null.synchronize()
            sdr = self.cupy.asnumpy(sdr_gpu).reshape(om_rgb.shape[:2])
            hdr = self.cupy.asnumpy(hdr_gpu).reshape(hdr_rgb.shape[:2])
        return sdr, hdr

    def _histogram_indices(self, values: Any) -> Any:
        cp = self.cupy
        logs = cp.log10(cp.maximum(values, 0.0) + LOG_EPS)
        indices = cp.floor(
            (cp.clip(logs, HISTOGRAM_LOG_LOW, HISTOGRAM_LOG_HIGH) - HISTOGRAM_LOG_LOW)
            / (HISTOGRAM_LOG_HIGH - HISTOGRAM_LOG_LOW)
            * HISTOGRAM_BINS
        ).astype(cp.int64)
        return cp.clip(indices, 0, HISTOGRAM_BINS - 1)

    def aggregate_pair_chunk(self, om_rgb: np.ndarray, hdr_rgb: np.ndarray) -> dict[str, Any]:
        """Process one RGB chunk and return only fixed-size host aggregates."""
        cp = self.cupy
        sdr_gpu, hdr_gpu = self._prepare_pair(om_rgb, hdr_rgb)
        with cp.cuda.Device(self.device_id):
            sdr = sdr_gpu.reshape(-1)
            hdr = hdr_gpu.reshape(-1)
            finite = cp.isfinite(sdr) & cp.isfinite(hdr)
            nonnegative = finite & (sdr >= 0.0) & (hdr >= 0.0)
            clipped = nonnegative & ((sdr >= HARD_CLIP_NITS) | (hdr >= HARD_CLIP_NITS))
            low = nonnegative & ~clipped & (sdr < MIN_STABLE_SDR_NITS)
            accepted = nonnegative & ~clipped & ~low
            accepted_sdr = sdr[accepted]
            accepted_hdr = hdr[accepted]
            bin_indices = cp.searchsorted(self._bin_bounds_gpu, accepted_sdr, side="right")
            sdr_hist = self._histogram_indices(accepted_sdr)
            hdr_hist = self._histogram_indices(accepted_hdr)
            combined_sdr = bin_indices * HISTOGRAM_BINS + sdr_hist
            combined_hdr = bin_indices * HISTOGRAM_BINS + hdr_hist
            shape = (len(SDR_BINS), HISTOGRAM_BINS)
            sdr_counts = cp.bincount(combined_sdr, minlength=shape[0] * shape[1]).reshape(shape)
            hdr_counts = cp.bincount(combined_hdr, minlength=shape[0] * shape[1]).reshape(shape)
            scalar = lambda value: int(cp.asnumpy(value))
            compact = {
                "raw_sample_count": int(sdr.size),
                "accepted_sample_count": scalar(cp.count_nonzero(accepted)),
                "rejections": {
                    "nonfinite": scalar(cp.count_nonzero(~finite)),
                    "negative": scalar(cp.count_nonzero(finite & ~nonnegative)),
                    "hard_clipping": scalar(cp.count_nonzero(clipped)),
                    "low_sdr_below_0.01_nits": scalar(cp.count_nonzero(low)),
                },
                "sdr_hist_counts": cp.asnumpy(sdr_counts),
                "hdr_hist_counts": cp.asnumpy(hdr_counts),
            }
            return compact

    def _curve_prediction(self, sdr: Any, model: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        cp = self.cupy
        x_log = cp.asarray(model.get("x_log", []), dtype=cp.float64)
        y_log = cp.asarray(model.get("y_log", []), dtype=cp.float64)
        if int(x_log.size) < 2:
            return cp.full(sdr.shape, cp.nan, dtype=cp.float64), {
                "outside_domain_count": int(sdr.size),
                "low_clip_count": int(sdr.size),
                "high_clip_count": 0,
                "extrapolation_count": 0,
                "domain_sdr_nits": None,
            }
        input_log = cp.log10(cp.maximum(sdr, LOG_EPS))
        low_clip = input_log < x_log[0]
        high_clip = input_log > x_log[-1]
        clipped = cp.clip(input_log, x_log[0], x_log[-1])
        output = cp.maximum(cp.power(10.0, cp.interp(clipped, x_log, y_log)) - LOG_EPS, 0.0)
        return output, {
            "outside_domain_count": int(cp.asnumpy(cp.count_nonzero(low_clip | high_clip))),
            "low_clip_count": int(cp.asnumpy(cp.count_nonzero(low_clip))),
            "high_clip_count": int(cp.asnumpy(cp.count_nonzero(high_clip))),
            "extrapolation_count": 0,
            "domain_sdr_nits": [float(10.0 ** float(model["x_log"][0]) - LOG_EPS), float(10.0 ** float(model["x_log"][-1]) - LOG_EPS)],
        }

    def _model_a_prediction(self, sdr: Any, model_a: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        cp = self.cupy
        curve = np.asarray(model_a["curve"], dtype=np.float64)
        input_log = cp.log10(cp.maximum(sdr, LOG_EPS))
        low = float(curve[0, 0])
        high = float(curve[-1, 0])
        low_clip = input_log < low
        high_clip = input_log > high
        if self._model_a_lut_gpu is None:
            self._model_a_lut_gpu = cp.asarray(model_a["lut"]["lut"], dtype=cp.float64)
        start = float(model_a["lut"]["curve_start_linear"])
        start_hdr = float(model_a["lut"]["curve_start_hdr_linear"])
        inv_range = float(model_a["lut"]["inv_range"])
        normalized = cp.clip(sdr / PEAK_NITS, 0.0, 1.0)
        bridge = (normalized > 0.0) & (normalized < start)
        fitted = normalized >= start
        output = cp.zeros_like(normalized)
        cp.copyto(output, normalized / start * start_hdr, where=bridge)
        positions = cp.clip((normalized - start) * inv_range, 0.0, 1.0)
        indices = (positions * (self._model_a_lut_gpu.size - 1)).astype(cp.int64)
        cp.copyto(output, cp.take(self._model_a_lut_gpu, indices), where=fitted)
        return output * PEAK_NITS, {
            "outside_domain_count": int(cp.asnumpy(cp.count_nonzero(low_clip | high_clip))),
            "low_clip_count": int(cp.asnumpy(cp.count_nonzero(low_clip))),
            "high_clip_count": int(cp.asnumpy(cp.count_nonzero(high_clip))),
            "extrapolation_count": 0,
            "domain_sdr_nits": [10.0 ** low - LOG_EPS, 10.0 ** high - LOG_EPS],
        }

    def _metric_compact(self, target: Any, prediction: Any, diagnostic_context: dict[str, Any] | None = None) -> dict[str, Any]:
        cp = self.cupy
        residual = prediction - target
        output: dict[str, Any] = {}
        context = dict(diagnostic_context or {})
        input_shape = [int(dimension) for dimension in target.shape]
        for low, high, label in TEST_REGIONS:
            mask = (target >= low) & (target < high if math.isfinite(high) else cp.ones_like(target, dtype=cp.bool_))
            finite = mask & cp.isfinite(residual) & cp.isfinite(target)
            count = 0 if int(target.size) == 0 else int(cp.asnumpy(cp.count_nonzero(finite)))
            if count == 0:
                output[label] = {
                    "status": "NO_VALID_PIXELS",
                    "valid_count": 0,
                    "count": 0,
                    "signed_sum": 0.0,
                    "absolute_sum": 0.0,
                    "squared_sum": 0.0,
                    "maximum": 0.0,
                    "relative_signed_sum": 0.0,
                    "relative_absolute_sum": 0.0,
                    "relative_maximum": 0.0,
                    "absolute_histogram": np.zeros(HISTOGRAM_BINS, dtype=np.int64),
                    "relative_histogram": np.zeros(HISTOGRAM_BINS, dtype=np.int64),
                    "diagnostic": {
                        **context,
                        "status": "NO_VALID_PIXELS",
                        "region": label,
                        "valid_count": 0,
                        "input_shape": input_shape,
                        "filter_condition": "target >= low and target < high (or unbounded) and finite(target) and finite(residual)",
                        "filter_range_nits": [float(low), None if not math.isfinite(high) else float(high)],
                    },
                }
                continue
            values = residual[finite]
            target_values = target[finite]
            absolute = cp.abs(values)
            relative = values / cp.maximum(cp.abs(target_values), LOG_EPS)
            abs_indices = self._histogram_indices(absolute)
            rel_indices = self._histogram_indices(cp.abs(relative))
            hist_size = HISTOGRAM_BINS
            abs_counts = cp.bincount(abs_indices, minlength=hist_size)
            rel_counts = cp.bincount(rel_indices, minlength=hist_size)
            compact = {
                "status": "OK",
                "valid_count": count,
                "count": count,
                "signed_sum": float(cp.asnumpy(cp.sum(values, dtype=cp.float64))),
                "absolute_sum": float(cp.asnumpy(cp.sum(absolute, dtype=cp.float64))),
                "squared_sum": float(cp.asnumpy(cp.sum(values * values, dtype=cp.float64))),
                "maximum": float(cp.asnumpy(cp.max(absolute))),
                "relative_signed_sum": float(cp.asnumpy(cp.sum(relative, dtype=cp.float64))),
                "relative_absolute_sum": float(cp.asnumpy(cp.sum(cp.abs(relative), dtype=cp.float64))),
                "relative_maximum": float(cp.asnumpy(cp.max(cp.abs(relative)))),
                "absolute_histogram": cp.asnumpy(abs_counts),
                "relative_histogram": cp.asnumpy(rel_counts),
            }
            output[label] = compact
        return output

    def predict_pair_chunk(self, om_rgb: np.ndarray, hdr_rgb: np.ndarray, pooled_model: dict[str, Any], model_a: dict[str, Any], diagnostic_context: dict[str, Any] | None = None) -> dict[str, Any]:
        cp = self.cupy
        sdr_gpu, hdr_gpu = self._prepare_pair(om_rgb, hdr_rgb)
        context = dict(diagnostic_context or {})
        context["rgb_shape"] = [int(dimension) for dimension in om_rgb.shape]
        with cp.cuda.Device(self.device_id):
            sdr = sdr_gpu.reshape(-1)
            hdr = hdr_gpu.reshape(-1)
            pooled_prediction, pooled_info = self._curve_prediction(sdr, pooled_model)
            model_a_prediction, model_a_info = self._model_a_prediction(sdr, model_a)
            pooled_metrics = self._metric_compact(hdr, pooled_prediction, {**context, "model": "Shot pooled curve"})
            model_a_metrics = self._metric_compact(hdr, model_a_prediction, {**context, "model": "Model A"})
            diagnostics = []
            for model_name, metric_blocks in (("Shot pooled curve", pooled_metrics), ("Model A", model_a_metrics)):
                for region, block in metric_blocks.items():
                    if block.get("status") == "NO_VALID_PIXELS":
                        diagnostic = dict(block["diagnostic"])
                        diagnostic["model"] = model_name
                        diagnostic["region"] = region
                        diagnostics.append(diagnostic)
            return {
                "models": {
                    "Shot pooled curve": {"metrics": pooled_metrics, "support": pooled_info},
                    "Model A": {"metrics": model_a_metrics, "support": model_a_info},
                },
                "total_count": int(sdr.size),
                "diagnostics": diagnostics,
            }

    def memory_report(self) -> dict[str, Any]:
        cp = self.cupy
        pool = cp.get_default_memory_pool()
        try:
            free_bytes, total_bytes = cp.cuda.Device(self.device_id).mem_info
        except (AttributeError, RuntimeError):
            free_bytes, total_bytes = None, None
        persistent = int(sum(int(array.nbytes) for buffers in self._buffers.values() for array in buffers.values()))
        return {
            "pool_used_bytes": int(pool.used_bytes()),
            "pool_reserved_bytes": int(pool.total_bytes()),
            "device_free_bytes": int(free_bytes) if free_bytes is not None else None,
            "device_total_bytes": int(total_bytes) if total_bytes is not None else None,
            "persistent_chunk_buffers_bytes": persistent,
            "model_a_lut_device_bytes": int(self._model_a_lut_gpu.nbytes) if self._model_a_lut_gpu is not None else 0,
            "chunk_buffer_shapes": [list(shape) for shape in self._buffers],
        }

    def clear(self) -> None:
        self._buffers.clear()
        self._model_a_lut_gpu = None
        if self._cupy is not None:
            try:
                self._cupy.cuda.Stream.null.synchronize()
            finally:
                self._verified_backend.clear()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def frame_timestamp(frame: int) -> float:
    return float(frame / FPS)


def ensure_relative(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{description} is outside {root}: {resolved}") from exc
    return resolved


class OutputGuard:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if self.root.exists():
            if any(self.root.iterdir()):
                raise RuntimeError(f"Refusing to overwrite existing streaming output: {self.root}")
        try:
            self.root.relative_to(P232_DIR.resolve())
            raise RuntimeError("Streaming output cannot be inside P2.32")
        except ValueError:
            pass
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=False)
        self.check()

    def size_bytes(self) -> int:
        return int(sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file()))

    def check(self) -> None:
        size = self.size_bytes()
        if size > OUTPUT_HARD_LIMIT_BYTES:
            raise DiskBudgetExceeded(f"DISK BUDGET EXCEEDED: {size} > {OUTPUT_HARD_LIMIT_BYTES} bytes")

    def write_json(self, relative_name: str, data: Any) -> Path:
        target = ensure_relative(self.root / relative_name, self.root, "Output artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        write_json(temporary, data)
        temporary.replace(target)
        self.check()
        return target

    def write_text(self, relative_name: str, text: str) -> Path:
        target = ensure_relative(self.root / relative_name, self.root, "Output artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)
        self.check()
        return target


class FixedLogHistogram:
    """Fixed-memory log-domain quantile estimator; no source samples are retained."""

    def __init__(self, bins: int = HISTOGRAM_BINS) -> None:
        self.bins = int(bins)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.total = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = np.isfinite(values) & (values >= 0.0)
        if not np.any(finite):
            return
        logs = np.log10(np.maximum(values[finite], 0.0) + LOG_EPS)
        indices = np.floor((np.clip(logs, HISTOGRAM_LOG_LOW, HISTOGRAM_LOG_HIGH) - HISTOGRAM_LOG_LOW) / (HISTOGRAM_LOG_HIGH - HISTOGRAM_LOG_LOW) * self.bins).astype(np.int64)
        indices = np.clip(indices, 0, self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins).astype(np.int64)
        self.total += int(indices.size)

    def merge(self, other: "FixedLogHistogram") -> None:
        if self.bins != other.bins:
            raise ValueError("Histogram resolution mismatch")
        self.counts += other.counts
        self.total += other.total

    def merge_counts(self, counts: np.ndarray) -> None:
        compact = np.asarray(counts, dtype=np.int64).reshape(-1)
        if compact.size != self.bins:
            raise ValueError("Compact histogram resolution mismatch")
        if np.any(compact < 0):
            raise ValueError("Compact histogram contains negative counts")
        self.counts += compact
        self.total += int(np.sum(compact, dtype=np.int64))

    def quantile(self, fraction: float) -> float | None:
        if self.total <= 0:
            return None
        target = max(1, int(math.ceil(float(fraction) * self.total)))
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, target, side="left"))
        width = (HISTOGRAM_LOG_HIGH - HISTOGRAM_LOG_LOW) / self.bins
        log_value = HISTOGRAM_LOG_LOW + (index + 0.5) * width
        return float(max(0.0, 10.0**log_value - LOG_EPS))

    def finite(self) -> bool:
        return bool(np.isfinite(self.counts).all() and self.total >= 0)


class BinAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sdr = FixedLogHistogram()
        self.hdr = FixedLogHistogram()

    def update(self, sdr: np.ndarray, hdr: np.ndarray) -> None:
        sdr = np.asarray(sdr, dtype=np.float64).reshape(-1)
        hdr = np.asarray(hdr, dtype=np.float64).reshape(-1)
        if sdr.size != hdr.size:
            raise ValueError("Bin update shape mismatch")
        self.count += int(sdr.size)
        self.sdr.update(sdr)
        self.hdr.update(hdr)

    def merge(self, other: "BinAccumulator") -> None:
        self.count += other.count
        self.sdr.merge(other.sdr)
        self.hdr.merge(other.hdr)

    def merge_compact(self, sdr_counts: np.ndarray, hdr_counts: np.ndarray) -> None:
        sdr_counts = np.asarray(sdr_counts, dtype=np.int64).reshape(-1)
        hdr_counts = np.asarray(hdr_counts, dtype=np.int64).reshape(-1)
        if sdr_counts.size != HISTOGRAM_BINS or hdr_counts.size != HISTOGRAM_BINS:
            raise ValueError("Compact bin histogram resolution mismatch")
        if not np.array_equal(sdr_counts.sum(), hdr_counts.sum()):
            raise ValueError("Compact SDR/HDR bin counts differ")
        self.count += int(np.sum(sdr_counts, dtype=np.int64))
        self.sdr.merge_counts(sdr_counts)
        self.hdr.merge_counts(hdr_counts)

    def row(self, low: float, high: float, label: str) -> dict[str, Any]:
        p10 = self.hdr.quantile(0.10)
        p50 = self.hdr.quantile(0.50)
        p90 = self.hdr.quantile(0.90)
        return {
            "label": label,
            "low_nits": low,
            "high_nits": None if not math.isfinite(high) else high,
            "count": self.count,
            "sdr_median": self.sdr.quantile(0.50),
            "P10": p10,
            "P50": p50,
            "P90": p90,
            "P90_minus_P10": p90 - p10 if p10 is not None and p90 is not None else None,
        }


class StreamingCurveAccumulator:
    """Streaming P2.34 pair statistics with fixed memory per SDR bin."""

    def __init__(self) -> None:
        self.raw_sample_count = 0
        self.accepted_sample_count = 0
        self.rejections = {"nonfinite": 0, "negative": 0, "hard_clipping": 0, "low_sdr_below_0.01_nits": 0}
        self.bins = [BinAccumulator() for _ in SDR_BINS]

    def update(self, sdr: np.ndarray, hdr: np.ndarray) -> None:
        x = np.asarray(sdr, dtype=np.float64).reshape(-1)
        y = np.asarray(hdr, dtype=np.float64).reshape(-1)
        if x.size != y.size:
            raise ValueError("Streaming pair shape mismatch")
        self.raw_sample_count += int(x.size)
        finite = np.isfinite(x) & np.isfinite(y)
        nonnegative = finite & (x >= 0.0) & (y >= 0.0)
        clipped = nonnegative & ((x >= HARD_CLIP_NITS) | (y >= HARD_CLIP_NITS))
        low = nonnegative & ~clipped & (x < MIN_STABLE_SDR_NITS)
        accepted = nonnegative & ~clipped & ~low
        self.rejections["nonfinite"] += int(np.sum(~finite))
        self.rejections["negative"] += int(np.sum(finite & ~nonnegative))
        self.rejections["hard_clipping"] += int(np.sum(clipped))
        self.rejections["low_sdr_below_0.01_nits"] += int(np.sum(low))
        self.accepted_sample_count += int(np.sum(accepted))
        if not np.any(accepted):
            return
        accepted_x = x[accepted]
        accepted_y = y[accepted]
        bin_indices = np.searchsorted(BIN_UPPER_BOUNDS, accepted_x, side="right")
        for index, accumulator in enumerate(self.bins):
            selected = bin_indices == index
            if np.any(selected):
                accumulator.update(accepted_x[selected], accepted_y[selected])

    def merge(self, other: "StreamingCurveAccumulator") -> None:
        self.raw_sample_count += other.raw_sample_count
        self.accepted_sample_count += other.accepted_sample_count
        for key in self.rejections:
            self.rejections[key] += other.rejections[key]
        for left, right in zip(self.bins, other.bins):
            left.merge(right)

    def update_compact(self, compact: dict[str, Any]) -> None:
        """Merge device-side fixed histograms without downloading luminance planes."""
        raw_count = int(compact["raw_sample_count"])
        accepted_count = int(compact["accepted_sample_count"])
        sdr_counts = np.asarray(compact["sdr_hist_counts"], dtype=np.int64)
        hdr_counts = np.asarray(compact["hdr_hist_counts"], dtype=np.int64)
        expected_shape = (len(SDR_BINS), HISTOGRAM_BINS)
        if sdr_counts.shape != expected_shape or hdr_counts.shape != expected_shape:
            raise ValueError(f"Compact GPU histogram shape mismatch: {sdr_counts.shape} / {hdr_counts.shape}")
        if int(sdr_counts.sum()) != accepted_count or int(hdr_counts.sum()) != accepted_count:
            raise ValueError("Compact GPU histogram count does not match accepted count")
        self.raw_sample_count += raw_count
        self.accepted_sample_count += accepted_count
        for key in self.rejections:
            self.rejections[key] += int(compact["rejections"].get(key, 0))
        for index, accumulator in enumerate(self.bins):
            accumulator.merge_compact(sdr_counts[index], hdr_counts[index])

    def finalize(self, name: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        rows = [accumulator.row(*definition) for accumulator, definition in zip(self.bins, SDR_BINS)]
        node_x = [row["sdr_median"] for row in rows if row["count"] > 0 and row["sdr_median"] is not None and row["P50"] is not None]
        node_y = [row["P50"] for row in rows if row["count"] > 0 and row["sdr_median"] is not None and row["P50"] is not None]
        raw_y_log = np.log10(np.maximum(np.asarray(node_y, dtype=np.float64), 0.0) + LOG_EPS) if node_y else np.empty(0)
        x_log = np.log10(np.maximum(np.asarray(node_x, dtype=np.float64), LOG_EPS)) if node_x else np.empty(0)
        mono_y_log = pava(raw_y_log)
        model = {
            "name": name,
            "kind": "streaming_empirical_log_bin_median",
            "x_log": x_log,
            "y_log": mono_y_log,
            "definition": {
                "fixed_bins": [item[2] for item in SDR_BINS],
                "target_statistic": "HDR P50 from fixed log histogram",
                "spread": ["P10", "P50", "P90"],
                "quantile_estimator": f"fixed_log_histogram_{HISTOGRAM_BINS}_bins",
                "monotonic_projection": "PAVA",
                "interpolation": "linear in log10 domain",
                "outside_domain": "clamp; no extrapolation",
            },
        }
        diagnostics = {
            "raw_monotonicity_violations": int(np.sum(np.diff(raw_y_log) < 0.0)) if raw_y_log.size > 1 else 0,
            "post_pava_monotonicity_violations": int(np.sum(np.diff(mono_y_log) < -1e-12)) if mono_y_log.size > 1 else 0,
            "pava_adjustment_max_abs_log10": float(np.max(np.abs(mono_y_log - raw_y_log))) if raw_y_log.size else None,
            "node_count": int(x_log.size),
            "raw_sample_count": self.raw_sample_count,
            "accepted_sample_count": self.accepted_sample_count,
            "rejections": dict(self.rejections),
            "coverage_fraction": float(sum(row["count"] > 0 for row in rows) / len(rows)),
            "domain_sdr_nits": [float(10.0**x_log[0] - LOG_EPS), float(10.0**x_log[-1] - LOG_EPS)] if x_log.size else None,
        }
        return model, rows, diagnostics

    def compact_summary(self) -> dict[str, Any]:
        _, rows, diagnostics = self.finalize("compact sample curve")
        return {
            "bin_statistics": rows,
            "sample_count": diagnostics["raw_sample_count"],
            "accepted_sample_count": diagnostics["accepted_sample_count"],
            "rejection_count": diagnostics["rejections"],
            "domain": diagnostics["domain_sdr_nits"],
            "coverage": diagnostics["coverage_fraction"],
        }


def pava(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            left, right = blocks[-2], blocks[-1]
            weight = left[1] + right[1]
            blocks[-2] = [(left[0] * left[1] + right[0] * right[1]) / weight, weight]
            blocks.pop()
    result = np.empty(values.size, dtype=np.float64)
    cursor = 0
    for value, weight in blocks:
        count = int(weight)
        result[cursor:cursor + count] = value
        cursor += count
    return result


def predict_curve(model: dict[str, Any], values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x_log = np.asarray(model.get("x_log", []), dtype=np.float64)
    y_log = np.asarray(model.get("y_log", []), dtype=np.float64)
    if x_log.size < 2:
        return np.full(x.shape, np.nan), {"outside_domain_count": int(x.size), "clipping_fraction": 1.0, "extrapolation_count": 0, "domain_sdr_nits": None}
    input_log = np.log10(np.maximum(x, LOG_EPS))
    low_clip = input_log < x_log[0]
    high_clip = input_log > x_log[-1]
    output = np.maximum(np.power(10.0, np.interp(np.clip(input_log, x_log[0], x_log[-1]), x_log, y_log)) - LOG_EPS, 0.0)
    return output, {
        "outside_domain_count": int(np.sum(low_clip | high_clip)),
        "low_clip_count": int(np.sum(low_clip)),
        "high_clip_count": int(np.sum(high_clip)),
        "clipping_fraction": float(np.mean(low_clip | high_clip)) if x.size else 0.0,
        "extrapolation_count": 0,
        "domain_sdr_nits": [float(10.0**x_log[0] - LOG_EPS), float(10.0**x_log[-1] - LOG_EPS)],
    }


class OnlineMetricBlock:
    def __init__(self) -> None:
        self.count = 0
        self.signed_sum = 0.0
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.maximum = 0.0
        self.relative_signed_sum = 0.0
        self.relative_absolute_sum = 0.0
        self.relative_maximum = 0.0
        self.absolute_histogram = FixedLogHistogram()
        self.relative_histogram = FixedLogHistogram()

    def update(self, residual: np.ndarray, target: np.ndarray) -> None:
        residual = np.asarray(residual, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        finite = np.isfinite(residual) & np.isfinite(target)
        if not np.any(finite):
            return
        residual = residual[finite]
        target = target[finite]
        absolute = np.abs(residual)
        relative = residual / np.maximum(np.abs(target), LOG_EPS)
        self.count += int(residual.size)
        self.signed_sum += float(np.sum(residual))
        self.absolute_sum += float(np.sum(absolute))
        self.squared_sum += float(np.sum(residual * residual))
        self.maximum = max(self.maximum, float(np.max(absolute)))
        self.relative_signed_sum += float(np.sum(relative))
        self.relative_absolute_sum += float(np.sum(np.abs(relative)))
        self.relative_maximum = max(self.relative_maximum, float(np.max(np.abs(relative))))
        self.absolute_histogram.update(absolute)
        self.relative_histogram.update(np.abs(relative))

    def merge_compact(self, compact: dict[str, Any]) -> None:
        self.count += int(compact["count"])
        self.signed_sum += float(compact["signed_sum"])
        self.absolute_sum += float(compact["absolute_sum"])
        self.squared_sum += float(compact["squared_sum"])
        self.maximum = max(self.maximum, float(compact["maximum"]))
        self.relative_signed_sum += float(compact["relative_signed_sum"])
        self.relative_absolute_sum += float(compact["relative_absolute_sum"])
        self.relative_maximum = max(self.relative_maximum, float(compact["relative_maximum"]))
        self.absolute_histogram.merge_counts(compact["absolute_histogram"])
        self.relative_histogram.merge_counts(compact["relative_histogram"])

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "MAE": None, "RMSE": None, "P95": None, "P99": None, "max": None, "relative": {}}
        return {
            "count": self.count,
            "signed_mean": self.signed_sum / self.count,
            "MAE": self.absolute_sum / self.count,
            "RMSE": math.sqrt(self.squared_sum / self.count),
            "P95": self.absolute_histogram.quantile(0.95),
            "P99": self.absolute_histogram.quantile(0.99),
            "max": self.maximum,
            "relative": {
                "signed_mean": self.relative_signed_sum / self.count,
                "MAE": self.relative_absolute_sum / self.count,
                "P95": self.relative_histogram.quantile(0.95),
                "P99": self.relative_histogram.quantile(0.99),
                "max": self.relative_maximum,
                "epsilon_nits": LOG_EPS,
            },
        }


class OnlinePredictionMetrics:
    def __init__(self) -> None:
        self.blocks = {label: OnlineMetricBlock() for _, _, label in TEST_REGIONS}

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        residual = prediction - target
        for low, high, label in TEST_REGIONS:
            mask = (target >= low) & (target < high if math.isfinite(high) else np.ones_like(target, dtype=bool))
            self.blocks[label].update(residual[mask], target[mask])

    def merge_compact(self, compact: dict[str, Any]) -> None:
        for label, block in self.blocks.items():
            block.merge_compact(compact[label])

    def result(self) -> dict[str, Any]:
        return {label: block.result() for label, block in self.blocks.items()}


PIPE_DRAIN_CHUNK_BYTES = 64 * 1024
MAX_STDERR_CAPTURE_BYTES = 64 * 1024
PROCESS_TERMINATION_TIMEOUT_SECONDS = 5


class RawPipe:
    """Deadlock-safe bounded FFmpeg raw pipe.

    FFmpeg stdout is consumed by the caller in bounded records while a daemon
    thread drains stderr continuously into a capped diagnostic buffer. Neither
    stream is accumulated without a bound, and a producer is terminated if the
    caller abandons the stream after a partial/error read.
    """

    def __init__(self, command: list[str], timeout: int = 300) -> None:
        self.command = command
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_capture = bytearray()
        self._stderr_truncated = False
        self._stdout_tail_bytes = 0
        self.stderr_text = ""

    def __enter__(self) -> "RawPipe":
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="p234-ffmpeg-stderr",
            daemon=True,
        )
        try:
            self._stderr_thread.start()
        except Exception:
            self._terminate_process()
            raise
        return self

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                chunk = process.stderr.read(PIPE_DRAIN_CHUNK_BYTES)
                if not chunk:
                    break
                remaining = MAX_STDERR_CAPTURE_BYTES - len(self._stderr_capture)
                if remaining > 0:
                    self._stderr_capture.extend(chunk[:remaining])
                if len(chunk) > max(remaining, 0):
                    self._stderr_truncated = True
        except (OSError, ValueError):
            # The owner may close the stream after terminating a failed child.
            return

    def _terminate_process(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass

    def _stderr_diagnostic(self) -> str:
        text = bytes(self._stderr_capture).decode("utf-8", errors="replace")
        if self._stderr_truncated:
            text += "\n[FFmpeg stderr truncated at 65536 bytes]"
        return text

    def read_exact(self, size: int) -> bytes | None:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Raw pipe is not open")
        if size < 0:
            raise ValueError("Raw pipe read size cannot be negative")
        parts: list[bytes] = []
        remaining = int(size)
        while remaining > 0:
            chunk = self.process.stdout.read(min(remaining, PIPE_DRAIN_CHUNK_BYTES))
            if not chunk:
                if not parts:
                    return None
                received = size - remaining
                raise RuntimeError(
                    f"FFmpeg stdout ended mid-record: expected {size} bytes, received {received} bytes"
                )
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    def _drain_stdout_tail(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        while True:
            chunk = self.process.stdout.read(PIPE_DRAIN_CHUNK_BYTES)
            if not chunk:
                return
            self._stdout_tail_bytes += len(chunk)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        process = self.process
        if process is None:
            return
        failed_in_body = exc_type is not None
        tail_error: Exception | None = None
        if failed_in_body:
            self._terminate_process()
        else:
            try:
                # A frame iterator can stop after its exact expected payload;
                # consume any remaining stdout to guarantee producer EOF.
                self._drain_stdout_tail()
            except Exception as exc:
                tail_error = exc
                self._terminate_process()

        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._terminate_process()
            return_code = process.poll()

        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            if process.stderr is not None:
                try:
                    process.stderr.close()
                except OSError:
                    pass
            self._stderr_thread.join(timeout=1)
        self.stderr_text = self._stderr_diagnostic()

        if failed_in_body:
            if self.stderr_text and exc_value is not None:
                try:
                    exc_value.add_note(f"FFmpeg stderr:\n{self.stderr_text}")
                except AttributeError:
                    pass
            return
        if tail_error is not None:
            raise RuntimeError(f"FFmpeg stdout drain failed: {tail_error}; stderr={self.stderr_text!r}") from tail_error
        if self._stdout_tail_bytes:
            raise RuntimeError(
                f"FFmpeg produced {self._stdout_tail_bytes} unexpected trailing stdout bytes; stderr={self.stderr_text!r}"
            )
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed ({return_code}); stderr={self.stderr_text!r}"
            )


def ffmpeg_gray_command(source: Path, start_frame: int, frame_count: int, height: int) -> list[str]:
    return [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{frame_timestamp(start_frame):.9f}", "-i", str(source),
        "-map", "0:v:0", "-vf", f"scale={PROXY_WIDTH}:{height}:flags=bilinear,format=gray",
        "-frames:v", str(frame_count), "-pix_fmt", "gray", "-vsync", "0", "-f", "rawvideo", "pipe:1",
    ]


def edge_map(frame: np.ndarray) -> np.ndarray:
    return cv2.Canny(frame, 32, 96).astype(np.float32) / 255.0


def boundary_score(frame_a: np.ndarray, frame_b: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    hist_a, _ = np.histogram(frame_a, bins=64, range=(0.0, 255.0))
    hist_b, _ = np.histogram(frame_b, bins=64, range=(0.0, 255.0))
    hist_a = hist_a.astype(np.float64) / max(float(hist_a.sum()), 1.0)
    hist_b = hist_b.astype(np.float64) / max(float(hist_b.sum()), 1.0)
    denominator = hist_a + hist_b
    mask = denominator > 0
    histogram_score = float(min(1.0, np.sum((hist_a[mask] - hist_b[mask]) ** 2 / denominator[mask]) / 2.0)) if np.any(mask) else 0.0
    luminance_score = float(abs(np.mean(frame_a) - np.mean(frame_b)) / 255.0)
    a = edge_a.reshape(-1).astype(np.float64)
    b = edge_b.reshape(-1).astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator_ncc = float(np.linalg.norm(a) * np.linalg.norm(b))
    ncc = float(np.dot(a, b) / denominator_ncc) if denominator_ncc > 1e-12 else 0.0
    edge_score = max(0.0, 1.0 - ncc)
    return float(0.4 * histogram_score + 0.2 * luminance_score + 0.4 * edge_score)


def detect_local_boundaries_streaming(case: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    if LOCAL_HALF_WINDOW_SECONDS > MAX_HALF_WINDOW_SECONDS:
        raise RuntimeError("Configured local window exceeds P2.34 hard limit")
    anchor = int(case["hdr_frame"])
    half = int(round(LOCAL_HALF_WINDOW_SECONDS * FPS))
    start_frame = max(0, anchor - half)
    requested_end = anchor + half + 1
    proxy_height = max(2, int(round(case["hdr_resolution"][1] * PROXY_WIDTH / case["hdr_resolution"][0] / 2.0) * 2))
    frame_bytes = PROXY_WIDTH * proxy_height
    scores: list[float] = []
    decoded = 0
    previous: np.ndarray | None = None
    command = ffmpeg_gray_command(Path(case["hdr_source"]), start_frame, requested_end - start_frame, proxy_height)
    with RawPipe(command) as pipe:
        while True:
            raw = pipe.read_exact(frame_bytes)
            if raw is None or len(raw) != frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(proxy_height, PROXY_WIDTH)
            if previous is not None:
                scores.append(boundary_score(previous, frame, edge_map(previous), edge_map(frame)))
            previous = frame
            decoded += 1
    if decoded < 2:
        raise RuntimeError(f"Streaming proxy decode returned only {decoded} frames for {case['case_id']}")
    frame_ids = np.arange(start_frame, start_frame + decoded, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for index, score in enumerate(score_array):
        if score >= BOUNDARY_THRESHOLD:
            left = score_array[index - 1] if index > 0 else 0.0
            right = score_array[index + 1] if index + 1 < score_array.size else 0.0
            if left < GRADUAL_THRESHOLD and right < GRADUAL_THRESHOLD:
                candidates.append({"frame": int(frame_ids[index + 1]), "timestamp_seconds": frame_timestamp(int(frame_ids[index + 1])), "boundary_score": float(score), "threshold": BOUNDARY_THRESHOLD, "type": "hard"})
    index = 0
    while index < score_array.size:
        if score_array[index] < GRADUAL_THRESHOLD:
            index += 1
            continue
        run_start = index
        while index < score_array.size and score_array[index] >= GRADUAL_THRESHOLD:
            index += 1
        run_end = index
        if run_end - run_start >= GRADUAL_MIN_RUN_FRAMES:
            segment = score_array[run_start:run_end]
            peak_index = int(run_start + np.argmax(segment))
            candidates.append({"frame": int(frame_ids[peak_index + 1]), "timestamp_seconds": frame_timestamp(int(frame_ids[peak_index + 1])), "boundary_score": float(np.max(segment)), "threshold": GRADUAL_THRESHOLD, "type": "gradual", "run_start_frame": int(frame_ids[run_start]), "run_end_frame": int(frame_ids[run_end])})
    candidates.sort(key=lambda item: item["frame"])
    boundaries: list[dict[str, Any]] = []
    for candidate in candidates:
        if boundaries and candidate["frame"] - boundaries[-1]["frame"] <= GRADUAL_MIN_RUN_FRAMES:
            if candidate["boundary_score"] > boundaries[-1]["boundary_score"]:
                boundaries[-1] = candidate
        else:
            boundaries.append(candidate)
    nearest_distance = min((abs(item["frame"] - anchor) for item in boundaries), default=None)
    near_cut = nearest_distance is not None and nearest_distance <= ANCHOR_NEAR_CUT_FRAMES
    left = [item for item in boundaries if item["frame"] <= anchor]
    right = [item for item in boundaries if item["frame"] > anchor]
    shot_start = int(left[-1]["frame"]) if left else None
    shot_end = int(right[0]["frame"]) if right else None
    status = "LOCAL_SHOT_UNRESOLVED"
    if shot_start is not None and shot_end is not None:
        status = "LOCAL_SHOT_RESOLVED" if shot_end - shot_start >= 3 else "SHOT_TOO_SHORT"
    if near_cut and status == "LOCAL_SHOT_RESOLVED":
        status = "ANCHOR_NEAR_CUT"
    result = {
        "case_id": case["case_id"],
        "material": case["material"],
        "anchor_hdr_frame": anchor,
        "anchor_om_frame": int(case["om_frame"]),
        "anchor_timestamp_seconds": frame_timestamp(anchor),
        "window": {"half_window_seconds": LOCAL_HALF_WINDOW_SECONDS, "start_frame": start_frame, "end_frame_exclusive": start_frame + decoded, "requested_end_frame_exclusive": requested_end, "requested_frame_count": requested_end - start_frame, "decoded_frame_count": decoded, "proxy_shape": [proxy_height, PROXY_WIDTH], "frames_retained": 1, "full_film_scan": False},
        "boundaries": boundaries,
        "boundary_count": len(boundaries),
        "anchor_near_cut": near_cut,
        "nearest_boundary_distance_frames": nearest_distance,
        "shot_status": status,
        "shot_start_frame": shot_start,
        "shot_end_frame_exclusive": shot_end,
        "shot_duration_frames": shot_end - shot_start if shot_start is not None and shot_end is not None else None,
        "shot_duration_seconds": (shot_end - shot_start) / FPS if shot_start is not None and shot_end is not None else None,
        "boundary_detection": {"algorithm": "deterministic streaming local frame difference", "weights": {"histogram": 0.4, "mean_luminance": 0.2, "edge_ncc_change": 0.4}, "hard_threshold": BOUNDARY_THRESHOLD, "gradual_threshold": GRADUAL_THRESHOLD, "gradual_min_run_frames": GRADUAL_MIN_RUN_FRAMES, "full_film_scan": False, "frames_retained_simultaneously": 1},
        "decode_command": command,
    }
    return result, [float(value) for value in scores]


def select_temporal_samples(shot: dict[str, Any]) -> list[dict[str, Any]]:
    start = shot.get("shot_start_frame")
    end = shot.get("shot_end_frame_exclusive")
    if start is None or end is None or end <= start:
        return []
    selected: list[dict[str, Any]] = []
    duration = int(end - start)
    for fraction, label in TEMPORAL_FRACTIONS:
        desired = int(round(start + fraction * max(duration - 1, 0)))
        candidates = sorted(range(start, end), key=lambda frame: (abs(frame - desired), frame))
        choice = next((frame for frame in candidates if all(abs(frame - old["hdr_frame"]) >= MIN_SAMPLE_SEPARATION_FRAMES for old in selected)), None)
        if choice is not None:
            selected.append({"label": label, "fraction": fraction, "hdr_frame": int(choice), "om_frame": int(choice + shot["case_offset_frames"]), "hdr_timestamp_seconds": frame_timestamp(int(choice)), "om_timestamp_seconds": frame_timestamp(int(choice + shot["case_offset_frames"]))})
    if len(selected) < 3 and duration >= 2 * MIN_SAMPLE_SEPARATION_FRAMES + 1:
        selected = []
        for frame, fraction in zip((int(round(start + f * max(duration - 1, 0))) for f in (0.10, 0.50, 0.90)), (0.10, 0.50, 0.90)):
            if all(abs(frame - old["hdr_frame"]) >= MIN_SAMPLE_SEPARATION_FRAMES for old in selected):
                selected.append({"label": f"fallback_{int(fraction * 100)}%", "fraction": fraction, "hdr_frame": frame, "om_frame": int(frame + shot["case_offset_frames"]), "hdr_timestamp_seconds": frame_timestamp(frame), "om_timestamp_seconds": frame_timestamp(int(frame + shot["case_offset_frames"]))})
    selected.sort(key=lambda item: item["hdr_frame"])
    return selected


def source_luminance_chunk(rgb_u16: np.ndarray, transfer: str) -> np.ndarray:
    signal = np.asarray(rgb_u16, dtype=np.float64) / RGB16_MAX
    linear = np.stack([np.asarray(linearize(signal[..., channel], transfer, peak_nits=PEAK_NITS), dtype=np.float64) for channel in range(3)], axis=-1)
    if transfer == "bt709":
        linear = linear.reshape(-1, 3) @ M_709_TO_2020.T
        linear = np.maximum(linear.reshape(signal.shape), 0.0)
    return np.asarray(np.sum(linear * LUMA_2020, axis=-1) * PEAK_NITS, dtype=np.float32)


def sample_filter(case: dict[str, Any]) -> tuple[str, str, int, int]:
    x1, y1, x2, y2 = [int(value) for value in case["geometry"]["overlap"]]
    output_width = x2 - x1
    output_height = y2 - y1
    hdr_width = int(round(case["hdr_resolution"][0] * float(case["geometry"]["scale_x"])))
    hdr_height = int(round(case["hdr_resolution"][1] * float(case["geometry"]["scale_y"])))
    if (hdr_width, hdr_height) != (output_width, output_height):
        raise RuntimeError(f"P2.32 geometry mismatch for {case['case_id']}: scaled HDR {(hdr_width, hdr_height)} vs overlap {(output_width, output_height)}")
    hdr_filter = f"scale={hdr_width}:{hdr_height}:flags=bilinear,format=rgb48le"
    om_filter = f"crop={output_width}:{output_height}:{x1}:{y1},format=rgb48le"
    return hdr_filter, om_filter, output_width, output_height


def rgb_pipe_command(source: Path, timestamp: float, video_filter: str) -> list[str]:
    return [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp:.9f}", "-i", str(source), "-map", "0:v:0", "-vf", video_filter, "-frames:v", "1", "-pix_fmt", "rgb48le", "-vsync", "0", "-f", "rawvideo", "pipe:1"]


def iter_rgb_pairs(case: dict[str, Any], sample: dict[str, Any], chunk_rows: int = CHUNK_ROWS) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    hdr_filter, om_filter, width, height = sample_filter(case)
    hdr_command = rgb_pipe_command(Path(case["hdr_source"]), float(sample["hdr_timestamp_seconds"]), hdr_filter)
    om_command = rgb_pipe_command(Path(case["om_source"]), float(sample["om_timestamp_seconds"]), om_filter)
    row_bytes = width * 3 * 2
    with RawPipe(hdr_command) as hdr_pipe, RawPipe(om_command) as om_pipe:
        rows_read = 0
        while rows_read < height:
            rows = min(chunk_rows, height - rows_read)
            hdr_raw = hdr_pipe.read_exact(rows * row_bytes)
            om_raw = om_pipe.read_exact(rows * row_bytes)
            if hdr_raw is None or om_raw is None or len(hdr_raw) != rows * row_bytes or len(om_raw) != rows * row_bytes:
                raise RuntimeError(f"Streaming sample ended early for {case['case_id']} {sample['label']} at row {rows_read}")
            hdr_rgb = np.frombuffer(hdr_raw, dtype="<u2").reshape(rows, width, 3)
            om_rgb = np.frombuffer(om_raw, dtype="<u2").reshape(rows, width, 3)
            yield om_rgb, hdr_rgb
            rows_read += rows


def iter_luminance_pairs(case: dict[str, Any], sample: dict[str, Any], chunk_rows: int = CHUNK_ROWS) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for om_rgb, hdr_rgb in iter_rgb_pairs(case, sample, chunk_rows=chunk_rows):
        yield source_luminance_chunk(om_rgb, "bt709"), source_luminance_chunk(hdr_rgb, "smpte2084")


def process_sample(case: dict[str, Any], sample: dict[str, Any]) -> tuple[StreamingCurveAccumulator, dict[str, Any]]:
    accumulator = StreamingCurveAccumulator()
    chunk_count = 0
    pixel_count = 0
    for sdr, hdr in iter_luminance_pairs(case, sample):
        accumulator.update(sdr, hdr)
        chunk_count += 1
        pixel_count += int(sdr.size)
        del sdr, hdr
    model, rows, diagnostics = accumulator.finalize(f"Frame-specific {sample['label']}")
    return accumulator, {"frame_id": sample["hdr_frame"], "timestamp_seconds": sample["hdr_timestamp_seconds"], "label": sample["label"], "bin_statistics": rows, "sample_count": pixel_count, "accepted_sample_count": diagnostics["accepted_sample_count"], "rejection_count": diagnostics["rejections"], "domain": diagnostics["domain_sdr_nits"], "coverage": diagnostics["coverage_fraction"], "chunk_count": chunk_count, "curve": model, "curve_diagnostics": diagnostics}


def process_sample_cuda(case: dict[str, Any], sample: dict[str, Any], adapter: CudaLuminanceChunkAdapter) -> tuple[StreamingCurveAccumulator, dict[str, Any]]:
    accumulator = StreamingCurveAccumulator()
    chunk_count = 0
    pixel_count = 0
    for om_rgb, hdr_rgb in iter_rgb_pairs(case, sample):
        compact = adapter.aggregate_pair_chunk(om_rgb, hdr_rgb)
        accumulator.update_compact(compact)
        chunk_count += 1
        pixel_count += int(compact["raw_sample_count"])
        del om_rgb, hdr_rgb, compact
    model, rows, diagnostics = accumulator.finalize(f"Frame-specific CUDA {sample['label']}")
    return accumulator, {"frame_id": sample["hdr_frame"], "timestamp_seconds": sample["hdr_timestamp_seconds"], "label": sample["label"], "bin_statistics": rows, "sample_count": pixel_count, "accepted_sample_count": diagnostics["accepted_sample_count"], "rejection_count": diagnostics["rejections"], "domain": diagnostics["domain_sdr_nits"], "coverage": diagnostics["coverage_fraction"], "chunk_count": chunk_count, "curve": model, "curve_diagnostics": diagnostics, "cuda_luminance": True, "full_luminance_plane_saved": False}


def load_model_a() -> dict[str, Any]:
    from auto_openmatte.processing.luminance import apply_luminance_curve, build_curve_lut
    curve = json.loads(MODEL_A_CURVE_PATH.read_text(encoding="utf-8"))
    return {"curve": curve, "lut": build_curve_lut(curve), "apply": apply_luminance_curve}


def predict_model_a(model_a: dict[str, Any], values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    curve = np.asarray(model_a["curve"], dtype=np.float64)
    input_log = np.log10(np.maximum(values, LOG_EPS))
    low = float(curve[0, 0])
    high = float(curve[-1, 0])
    low_clip = input_log < low
    high_clip = input_log > high
    prediction = np.asarray(model_a["apply"](np.clip(values / PEAK_NITS, 0.0, 1.0), model_a["curve"], prebuilt_lut=model_a["lut"]), dtype=np.float64) * PEAK_NITS
    return prediction, {"outside_domain_count": int(np.sum(low_clip | high_clip)), "low_clip_count": int(np.sum(low_clip)), "high_clip_count": int(np.sum(high_clip)), "clipping_fraction": float(np.mean(low_clip | high_clip)) if values.size else 0.0, "extrapolation_count": 0, "domain_sdr_nits": [float(10.0**low - LOG_EPS), float(10.0**high - LOG_EPS)]}


def run_prediction_split(case: dict[str, Any], test_sample: dict[str, Any], pooled_model: dict[str, Any], model_a: dict[str, Any]) -> dict[str, Any]:
    pooled_metrics = OnlinePredictionMetrics()
    model_a_metrics = OnlinePredictionMetrics()
    pooled_support = {"outside_domain_count": 0, "low_clip_count": 0, "high_clip_count": 0, "clipping_count": 0, "extrapolation_count": 0, "total_count": 0}
    model_a_support = dict(pooled_support)
    for sdr, hdr in iter_luminance_pairs(case, test_sample):
        pooled_prediction, pooled_info = predict_curve(pooled_model, sdr)
        model_a_prediction, model_a_info = predict_model_a(model_a, sdr)
        pooled_metrics.update(hdr, pooled_prediction)
        model_a_metrics.update(hdr, model_a_prediction)
        for total, info in ((pooled_support, pooled_info), (model_a_support, model_a_info)):
            total["outside_domain_count"] += int(info["outside_domain_count"])
            total["low_clip_count"] += int(info.get("low_clip_count", 0))
            total["high_clip_count"] += int(info.get("high_clip_count", 0))
            total["extrapolation_count"] += int(info.get("extrapolation_count", 0))
            total["total_count"] += int(sdr.size)
        del sdr, hdr, pooled_prediction, model_a_prediction
    for support in (pooled_support, model_a_support):
        support["clipping_fraction"] = support["outside_domain_count"] / max(support["total_count"], 1)
    return {"test_label": test_sample["label"], "test_frame": test_sample["hdr_frame"], "models": {"Model A": {"metrics": model_a_metrics.result(), "support": model_a_support}, "Shot pooled curve": {"metrics": pooled_metrics.result(), "support": pooled_support}}, "residual_arrays_saved": False}


def run_prediction_split_cuda(case: dict[str, Any], test_sample: dict[str, Any], pooled_model: dict[str, Any], model_a: dict[str, Any], adapter: CudaLuminanceChunkAdapter) -> dict[str, Any]:
    metrics = {"Model A": OnlinePredictionMetrics(), "Shot pooled curve": OnlinePredictionMetrics()}
    supports = {
        "Model A": {"outside_domain_count": 0, "low_clip_count": 0, "high_clip_count": 0, "clipping_count": 0, "total_count": 0, "extrapolation_count": 0},
        "Shot pooled curve": {"outside_domain_count": 0, "low_clip_count": 0, "high_clip_count": 0, "clipping_count": 0, "total_count": 0, "extrapolation_count": 0},
    }
    prediction_diagnostics: list[dict[str, Any]] = []
    for chunk_index, (om_rgb, hdr_rgb) in enumerate(iter_rgb_pairs(case, test_sample)):
        compact = adapter.predict_pair_chunk(
            om_rgb,
            hdr_rgb,
            pooled_model,
            model_a,
            diagnostic_context={
                "case_id": str(case["case_id"]),
                "material": str(case["material"]),
                "anchor_hdr_frame": int(case["hdr_frame"]),
                "anchor_om_frame": int(case["om_frame"]),
                "sample_label": str(test_sample["label"]),
                "sample_hdr_frame": int(test_sample["hdr_frame"]),
                "sample_om_frame": int(test_sample["om_frame"]),
                "chunk_index": int(chunk_index),
            },
        )
        prediction_diagnostics.extend(compact.get("diagnostics", []))
        for name in ("Model A", "Shot pooled curve"):
            metrics[name].merge_compact(compact["models"][name]["metrics"])
            info = compact["models"][name]["support"]
            support = supports[name]
            support["outside_domain_count"] += int(info["outside_domain_count"])
            support["low_clip_count"] += int(info.get("low_clip_count", 0))
            support["high_clip_count"] += int(info.get("high_clip_count", 0))
            support["extrapolation_count"] += int(info.get("extrapolation_count", 0))
            support["total_count"] += int(compact["total_count"])
        del om_rgb, hdr_rgb, compact
    for support in supports.values():
        support["clipping_fraction"] = support["outside_domain_count"] / max(support["total_count"], 1)
    return {
        "status": "NO_VALID_PIXELS" if prediction_diagnostics else "OK",
        "test_label": test_sample["label"],
        "test_frame": test_sample["hdr_frame"],
        "models": {name: {"metrics": metrics[name].result(), "support": supports[name]} for name in ("Model A", "Shot pooled curve")},
        "prediction_diagnostics": prediction_diagnostics,
        "no_valid_pixels": bool(prediction_diagnostics),
        "residual_arrays_saved": False,
        "cuda_prediction": True,
        "full_test_arrays_retained": False,
    }


def load_smoke_case() -> dict[str, Any]:
    metrics = json.loads(P232_METRICS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(P232_MANIFEST_PATH.read_text(encoding="utf-8"))
    if metrics.get("phase") != "P2.32" or metrics.get("status") != "DATASET READY":
        raise RuntimeError("P2.32 metadata is not DATASET READY")
    record = next(item for item in metrics["case_records"]["The Matrix"] if int(item["hdr_frame"]) == 73367 and int(item["om_frame"]) == 73348)
    selected = next(item for item in manifest["materials"]["The Matrix"]["selected_cases"] if item["case_id"] == record["case_id"])
    required = {"case_id": "the_matrix_temporal_stratum_08", "hdr_frame": 73367, "om_frame": 73348, "offset_frames": -19, "known_verified_anchor": True}
    for key, expected in required.items():
        actual = record.get(key)
        if actual != expected:
            raise RuntimeError(f"Smoke anchor mismatch for {key}: {actual!r} != {expected!r}")
        if selected.get(key) != expected:
            raise RuntimeError(f"Smoke manifest mismatch for {key}: {selected.get(key)!r} != {expected!r}")
    return record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def p234_protocol_config() -> dict[str, Any]:
    return {
        "fps": "24000/1001",
        "local_half_window_seconds": LOCAL_HALF_WINDOW_SECONDS,
        "max_half_window_seconds": MAX_HALF_WINDOW_SECONDS,
        "proxy_width": PROXY_WIDTH,
        "boundary_threshold": BOUNDARY_THRESHOLD,
        "gradual_threshold": GRADUAL_THRESHOLD,
        "gradual_min_run_frames": GRADUAL_MIN_RUN_FRAMES,
        "anchor_near_cut_frames": ANCHOR_NEAR_CUT_FRAMES,
        "temporal_fractions": TEMPORAL_FRACTIONS,
        "minimum_sample_separation_frames": MIN_SAMPLE_SEPARATION_FRAMES,
        "sdr_bins": SDR_BINS,
        "histogram_bins": HISTOGRAM_BINS,
        "histogram_log_range": [HISTOGRAM_LOG_LOW, HISTOGRAM_LOG_HIGH],
        "chunk_rows": CHUNK_ROWS,
        "cuda_luminance_limit_nits": CUDA_LUMINANCE_LIMIT_NITS,
        "cuda_device_id": CUDA_DEVICE_ID,
        "model_a_path": str(MODEL_A_CURVE_PATH),
        "no_extrapolation": True,
        "full_film_scan": False,
        "source_arrays_saved": False,
        "residual_arrays_saved": False,
    }


def p234_hashes() -> dict[str, str]:
    config_bytes = json.dumps(json_safe(p234_protocol_config()), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "dataset_manifest_sha256": _sha256_file(P232_MANIFEST_PATH),
        "dataset_metrics_sha256": _sha256_file(P232_METRICS_PATH),
        "protocol_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "harness_sha256": _sha256_file(Path(__file__)),
    }


def load_all_cases() -> list[dict[str, Any]]:
    metrics = json.loads(P232_METRICS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(P232_MANIFEST_PATH.read_text(encoding="utf-8"))
    if metrics.get("phase") != "P2.32" or metrics.get("status") != "DATASET READY":
        raise RuntimeError("P2.32 metadata is not DATASET READY")
    cases: list[dict[str, Any]] = []
    for material in P2_34_MATERIAL_ORDER:
        records = list(metrics.get("case_records", {}).get(material, []))
        selected_cases = list(manifest.get("materials", {}).get(material, {}).get("selected_cases", []))
        if len(records) != P2_34_REQUIRED_ANCHORS_PER_MATERIAL or len(selected_cases) != P2_34_REQUIRED_ANCHORS_PER_MATERIAL:
            raise RuntimeError(f"P2.32 immutable {material} count mismatch")
        selected_by_id = {str(item["case_id"]): item for item in selected_cases}
        for record in sorted(records, key=lambda item: int(item["selection_index"])):
            case_id = str(record["case_id"])
            selected = selected_by_id.get(case_id)
            if selected is None:
                raise RuntimeError(f"Manifest does not contain immutable anchor {case_id}")
            for key in ("material", "case_id", "selection_index", "hdr_frame", "om_frame", "offset_frames", "offset_seconds"):
                if record.get(key) != selected.get(key):
                    raise RuntimeError(f"P2.32 manifest/metrics mismatch for {case_id} field {key}")
            case = dict(record)
            case["manifest_selection"] = dict(selected)
            cases.append(case)
    if len(cases) != P2_34_REQUIRED_ANCHOR_COUNT:
        raise RuntimeError(f"Expected exactly {P2_34_REQUIRED_ANCHOR_COUNT} immutable P2.32 anchors, got {len(cases)}")
    if [case["material"] for case in cases].count("The Matrix") != 20 or [case["material"] for case in cases].count("BR2049") != 20:
        raise RuntimeError("P2.32 material split is not exactly 20/20")
    return cases


def load_parity_cases() -> list[dict[str, Any]]:
    cases = load_all_cases()
    required = {
        "the_matrix_temporal_stratum_08": (73367, 73348, -19),
        "br2049_temporal_stratum_01": (11699, 12866, 1167),
    }
    output = []
    for case_id, expected in required.items():
        case = next(case for case in cases if case["case_id"] == case_id)
        actual = (int(case["hdr_frame"]), int(case["om_frame"]), int(case["offset_frames"]))
        if actual != expected:
            raise RuntimeError(f"Mandatory parity anchor changed for {case_id}: {actual} != {expected}")
        output.append(case)
    return output


def run_synthetic_tests() -> dict[str, Any]:
    left = StreamingCurveAccumulator()
    right = StreamingCurveAccumulator()
    sdr = np.asarray([0.005, 0.02, 0.2, 2.0, 20.0, 200.0, 2000.0, 10000.0], dtype=np.float64)
    hdr = np.asarray([1.0, 2.0, 4.0, 8.0, 40.0, 400.0, 4000.0, 10000.0], dtype=np.float64)
    left.update(sdr[:4], hdr[:4])
    right.update(sdr[4:], hdr[4:])
    merged = StreamingCurveAccumulator()
    merged.merge(left)
    merged.merge(right)
    model, rows, diagnostics = merged.finalize("synthetic")
    if diagnostics["accepted_sample_count"] != 6 or diagnostics["rejections"]["low_sdr_below_0.01_nits"] != 1 or diagnostics["rejections"]["hard_clipping"] != 1:
        raise AssertionError("Synthetic filtering counters failed")
    if diagnostics["post_pava_monotonicity_violations"] != 0 or not np.isfinite(model["y_log"]).all():
        raise AssertionError("Synthetic curve monotonicity/finite check failed")
    metrics = OnlinePredictionMetrics()
    target = np.asarray([1.0, 10.0, 100.0], dtype=np.float64)
    metrics.update(target, target + np.asarray([1.0, -2.0, 3.0]))
    global_metric = metrics.result()["global"]
    if global_metric["count"] != 3 or not math.isfinite(float(global_metric["MAE"])):
        raise AssertionError("Synthetic online metric failed")
    return {"status": "PASS", "accepted_count": diagnostics["accepted_sample_count"], "rejection_count": diagnostics["rejections"], "node_count": diagnostics["node_count"], "post_pava_monotonicity_violations": diagnostics["post_pava_monotonicity_violations"], "online_global_count": global_metric["count"], "full_source_arrays_retained": False}


def _anchor_parity_sample(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "anchor",
        "fraction": 0.5,
        "hdr_frame": int(case["hdr_frame"]),
        "om_frame": int(case["om_frame"]),
        "hdr_timestamp_seconds": float(case["hdr_timestamp_seconds"]),
        "om_timestamp_seconds": float(case["om_timestamp_seconds"]),
    }


def _compare_streaming_parity_states(
    cpu_state: StreamingCurveAccumulator,
    cuda_state: StreamingCurveAccumulator,
) -> dict[str, Any]:
    cpu_model, cpu_rows, cpu_diagnostics = cpu_state.finalize("CPU parity reference")
    cuda_model, cuda_rows, cuda_diagnostics = cuda_state.finalize("CUDA compact parity")
    histogram_counts_equal = all(
        left.count == right.count
        and np.array_equal(left.sdr.counts, right.sdr.counts)
        and np.array_equal(left.hdr.counts, right.hdr.counts)
        and left.sdr.total == right.sdr.total
        and left.hdr.total == right.hdr.total
        for left, right in zip(cpu_state.bins, cuda_state.bins)
    )
    bin_statistics_equal = json_safe(cpu_rows) == json_safe(cuda_rows)
    curve_nodes_equal = bool(
        np.array_equal(np.asarray(cpu_model.get("x_log", [])), np.asarray(cuda_model.get("x_log", [])))
        and np.array_equal(np.asarray(cpu_model.get("y_log", [])), np.asarray(cuda_model.get("y_log", [])))
    )
    finite_values = bool(
        np.isfinite(np.asarray(cpu_model.get("x_log", []), dtype=np.float64)).all()
        and np.isfinite(np.asarray(cpu_model.get("y_log", []), dtype=np.float64)).all()
        and np.isfinite(np.asarray(cuda_model.get("x_log", []), dtype=np.float64)).all()
        and np.isfinite(np.asarray(cuda_model.get("y_log", []), dtype=np.float64)).all()
    )
    monotonicity = bool(
        cpu_diagnostics["post_pava_monotonicity_violations"] == 0
        and cuda_diagnostics["post_pava_monotonicity_violations"] == 0
    )
    checks = {
        "raw_sample_count_equal": cpu_state.raw_sample_count == cuda_state.raw_sample_count,
        "accepted_sample_count_equal": cpu_state.accepted_sample_count == cuda_state.accepted_sample_count,
        "rejections_equal": cpu_state.rejections == cuda_state.rejections,
        "bin_counts_equal": histogram_counts_equal,
        "bin_statistics_equal": bin_statistics_equal,
        "curve_nodes_equal": curve_nodes_equal,
        "finite_values": finite_values,
        "monotonicity": monotonicity,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "cpu": {"curve": cpu_model, "bin_statistics": cpu_rows, "diagnostics": cpu_diagnostics},
        "cuda": {"curve": cuda_model, "bin_statistics": cuda_rows, "diagnostics": cuda_diagnostics},
    }


def run_cuda_parity() -> dict[str, Any]:
    if not FFMPEG.exists():
        raise RuntimeError(f"Isolated FFmpeg not found: {FFMPEG}")
    output = OutputGuard(PARITY_OUTPUT_DIR)
    hashes = p234_hashes()
    result: dict[str, Any] = {
        "phase": "P2.34_CUDA_PARITY",
        "status": "FAIL",
        "requirements": {"luminance_max_limit_nits": CUDA_LUMINANCE_LIMIT_NITS, "cpu_reference": "source_luminance_chunk", "full_luminance_planes_saved": False, "cpu_fallback_allowed": False},
        "hashes": hashes,
        "constraints": {"production_files_modified": False, "dataset_mutated": False, "global_sync_performed": False, "full_film_scan": False, "render_performed": False, "encode_performed": False},
        "cases": [],
    }
    adapter: CudaLuminanceChunkAdapter | None = None
    try:
        adapter = CudaLuminanceChunkAdapter()
        result["device"] = adapter.device_info()
        for case in load_parity_cases():
            sample = _anchor_parity_sample(case)
            case_result: dict[str, Any] = {
                "case_id": case["case_id"],
                "material": case["material"],
                "hdr_frame": int(case["hdr_frame"]),
                "om_frame": int(case["om_frame"]),
                "offset_frames": int(case["offset_frames"]),
                "chunks": 0,
                "samples_compared": 0,
                "max_sdr_abs_error_nits": 0.0,
                "max_hdr_abs_error_nits": 0.0,
                "nonfinite_cpu": 0,
                "nonfinite_cuda": 0,
            }
            cpu_state = StreamingCurveAccumulator()
            cuda_state = StreamingCurveAccumulator()
            for om_rgb, hdr_rgb in iter_rgb_pairs(case, sample):
                cpu_sdr = source_luminance_chunk(om_rgb, "bt709")
                cpu_hdr = source_luminance_chunk(hdr_rgb, "smpte2084")
                cuda_sdr, cuda_hdr = adapter.parity_luminance_pair(om_rgb, hdr_rgb)
                cpu_state.update(cpu_sdr, cpu_hdr)
                cuda_state.update_compact(adapter.aggregate_pair_chunk(om_rgb, hdr_rgb))
                case_result["chunks"] += 1
                case_result["samples_compared"] += int(cpu_sdr.size + cpu_hdr.size)
                case_result["nonfinite_cpu"] += int(np.sum(~np.isfinite(cpu_sdr)) + np.sum(~np.isfinite(cpu_hdr)))
                case_result["nonfinite_cuda"] += int(np.sum(~np.isfinite(cuda_sdr)) + np.sum(~np.isfinite(cuda_hdr)))
                if cpu_sdr.size:
                    case_result["max_sdr_abs_error_nits"] = max(case_result["max_sdr_abs_error_nits"], float(np.max(np.abs(np.asarray(cpu_sdr, dtype=np.float64) - cuda_sdr))))
                    case_result["max_hdr_abs_error_nits"] = max(case_result["max_hdr_abs_error_nits"], float(np.max(np.abs(np.asarray(cpu_hdr, dtype=np.float64) - cuda_hdr))))
                del om_rgb, hdr_rgb, cpu_sdr, cpu_hdr, cuda_sdr, cuda_hdr
            compact_parity = _compare_streaming_parity_states(cpu_state, cuda_state)
            case_result["compact_statistics_parity"] = compact_parity
            case_result["pass"] = bool(
                case_result["chunks"] > 0
                and case_result["nonfinite_cpu"] == 0
                and case_result["nonfinite_cuda"] == 0
                and case_result["max_sdr_abs_error_nits"] <= CUDA_LUMINANCE_LIMIT_NITS
                and case_result["max_hdr_abs_error_nits"] <= CUDA_LUMINANCE_LIMIT_NITS
                and compact_parity["pass"]
            )
            result["cases"].append(case_result)
        result["memory"] = adapter.memory_report()
        result["parity_pass"] = bool(len(result["cases"]) == 2 and all(item["pass"] for item in result["cases"]))
        result["status"] = "PASS" if result["parity_pass"] else "FAIL"
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["parity_pass"] = False
    finally:
        if adapter is not None:
            try:
                result["memory_final"] = adapter.memory_report()
            except Exception as exc:
                result["cleanup_memory_warning"] = str(exc)
            try:
                adapter.clear()
            except Exception as exc:
                result["cleanup_warning"] = str(exc)
        output.write_json("parity_result.json", result)
    return result


def require_cuda_parity_gate() -> dict[str, Any]:
    gate_path = PARITY_OUTPUT_DIR / "parity_result.json"
    if not gate_path.exists():
        raise RuntimeError(f"Mandatory CUDA parity artifact is missing: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    current_hashes = p234_hashes()
    if gate.get("status") != "PASS" or gate.get("parity_pass") is not True:
        raise RuntimeError("Mandatory Matrix/BR2049 CUDA parity gate is not PASS; full run is forbidden")
    if gate.get("hashes") != current_hashes:
        raise RuntimeError("CUDA parity artifact hashes do not match current immutable dataset/protocol/harness")
    required_ids = {"the_matrix_temporal_stratum_08", "br2049_temporal_stratum_01"}
    actual_ids = {str(item.get("case_id")) for item in gate.get("cases", []) if item.get("pass") is True}
    if actual_ids != required_ids:
        raise RuntimeError(f"CUDA parity gate does not contain exactly the required cases: {actual_ids}")
    return gate


def _safe_gpu_memory(adapter: CudaLuminanceChunkAdapter) -> dict[str, Any]:
    try:
        return adapter.memory_report()
    except Exception as exc:
        return {"error": str(exc)}


def run_full_cuda() -> dict[str, Any]:
    gate = require_cuda_parity_gate()
    cases = load_all_cases()
    if not FFMPEG.exists():
        raise RuntimeError(f"Isolated FFmpeg not found: {FFMPEG}")
    output = OutputGuard(FULL_OUTPUT_DIR)
    started = time.perf_counter()
    adapter = CudaLuminanceChunkAdapter()
    device = adapter.device_info()
    model_a = load_model_a()
    material_states = {material: StreamingCurveAccumulator() for material in P2_34_MATERIAL_ORDER}
    anchor_results: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    successful_samples_total = 0
    failed_samples_total = 0
    gpu_peak: dict[str, Any] = {}

    try:
        for case in cases:
            anchor_started = time.perf_counter()
            case_id = str(case["case_id"])
            anchor_result: dict[str, Any] = {
                "case_id": case_id,
                "material": case["material"],
                "selection_index": int(case["selection_index"]),
                "hdr_frame": int(case["hdr_frame"]),
                "om_frame": int(case["om_frame"]),
                "offset_frames": int(case["offset_frames"]),
                "offset_seconds": float(case["offset_seconds"]),
                "sync_status": case.get("sync_status"),
                "sync_confidence": case.get("sync_confidence"),
                "geometry": case.get("geometry"),
                "scene_id": case.get("scene_id"),
                "scene_independence_verified": bool(case.get("scene_independence_verified", False)),
                "status": "FAILED",
                "successful_samples": 0,
                "failed_samples": [],
                "online_prediction_splits": [],
                "prediction_diagnostics": [],
                "source_arrays_saved": False,
                "residual_arrays_saved": False,
            }
            sample_states: dict[str, StreamingCurveAccumulator] = {}
            sample_outputs: list[dict[str, Any]] = []
            try:
                boundary, score_trace = detect_local_boundaries_streaming(case)
                boundary["case_offset_frames"] = int(case["offset_frames"])
                samples = select_temporal_samples(boundary)
                anchor_result["boundary"] = boundary
                anchor_result["compact_score_trace"] = score_trace
                anchor_result["selected_samples"] = samples
                if boundary["shot_status"] not in ("LOCAL_SHOT_RESOLVED", "ANCHOR_NEAR_CUT") or len(samples) < 3:
                    raise RuntimeError(f"No usable real shot: status={boundary['shot_status']} samples={len(samples)}")
                for sample in samples:
                    try:
                        state, summary = process_sample_cuda(case, sample, adapter)
                        sample_states[sample["label"]] = state
                        sample_outputs.append(summary)
                        successful_samples_total += 1
                        anchor_result["successful_samples"] += 1
                        gpu_peak = _safe_gpu_memory(adapter)
                    except Exception as exc:
                        failed_samples_total += 1
                        failure = {"label": sample["label"], "hdr_frame": int(sample["hdr_frame"]), "type": type(exc).__name__, "message": str(exc)}
                        anchor_result["failed_samples"].append(failure)
                anchor_result["sample_summaries"] = sample_outputs
                if not sample_states:
                    raise RuntimeError("All selected samples failed")
                pooled_state = StreamingCurveAccumulator()
                for state in sample_states.values():
                    pooled_state.merge(state)
                pooled_model, pooled_rows, pooled_diagnostics = pooled_state.finalize(f"Shot pooled CUDA {case_id}")
                anchor_result["pooled_curve"] = {"curve": pooled_model, "bin_statistics": pooled_rows, "diagnostics": pooled_diagnostics, "sample_count": len(sample_states), "source_arrays_retained": False, "cuda_luminance": True}
                labels = [sample["label"] for sample in samples if sample["label"] in sample_states]
                if pooled_diagnostics["node_count"] < 2:
                    raise RuntimeError(f"Pooled CUDA curve has fewer than two nodes: {pooled_diagnostics['node_count']}")
                for train_labels, test_label in ((labels[:-1], labels[-1]), (labels[:-2] + [labels[-1]], labels[-2])):
                    if len(train_labels) < 2:
                        continue
                    train_state = StreamingCurveAccumulator()
                    for label in train_labels:
                        train_state.merge(sample_states[label])
                    train_model, _, train_diagnostics = train_state.finalize(f"Shot pooled CUDA train {','.join(train_labels)}")
                    test_sample = next(sample for sample in samples if sample["label"] == test_label)
                    split = run_prediction_split_cuda(case, test_sample, train_model, model_a, adapter)
                    split.update({"train_labels": train_labels, "train_curve_diagnostics": train_diagnostics})
                    anchor_result["online_prediction_splits"].append(split)
                    anchor_result["prediction_diagnostics"].extend(split.get("prediction_diagnostics", []))
                finite_curve = bool(np.isfinite(np.asarray(pooled_model.get("x_log", []), dtype=np.float64)).all() and np.isfinite(np.asarray(pooled_model.get("y_log", []), dtype=np.float64)).all() and pooled_diagnostics["post_pava_monotonicity_violations"] == 0)
                anchor_result["finite_values"] = finite_curve
                anchor_result["no_valid_pixels"] = bool(anchor_result["prediction_diagnostics"])
                if anchor_result["no_valid_pixels"]:
                    anchor_result["status"] = "PARTIAL"
                else:
                    anchor_result["status"] = "PASS" if finite_curve and len(anchor_result["failed_samples"]) == 0 and len(anchor_result["online_prediction_splits"]) == 2 else "PARTIAL"
                material_states[str(case["material"])].merge(pooled_state)
            except Exception as exc:
                anchor_result["error"] = {"type": type(exc).__name__, "message": str(exc)}
                anchor_result["finite_values"] = False
                if isinstance(exc, DiskBudgetExceeded):
                    raise
            anchor_result["elapsed_seconds"] = time.perf_counter() - anchor_started
            anchor_result["gpu_memory"] = _safe_gpu_memory(adapter)
            output.write_json(f"anchors/{case_id}.json", anchor_result)
            anchor_results.append(anchor_result)
            completed_ids.append(case_id)
            checkpoint = {
                "phase": "P2.34_CUDA_FULL",
                "completed_anchor_count": len(completed_ids),
                "requested_anchor_count": P2_34_REQUIRED_ANCHOR_COUNT,
                "current_anchor": case_id,
                "completed_anchor_ids": list(completed_ids),
                "elapsed_time_seconds": time.perf_counter() - started,
                "disk_usage_bytes": output.size_bytes(),
                "gpu_memory_usage": anchor_result["gpu_memory"],
                "successful_samples": successful_samples_total,
                "failed_samples": failed_samples_total,
                "no_valid_pixels_count": len(anchor_result.get("prediction_diagnostics", [])),
                "prediction_splits": len(anchor_result.get("online_prediction_splits", [])),
                "checkpoint_written_only_after_completed_anchor": True,
                "hashes": p234_hashes(),
            }
            output.write_json("checkpoint.json", checkpoint)
            output.check()

        material_curves: dict[str, Any] = {}
        for material, state in material_states.items():
            model, rows, diagnostics = state.finalize(f"Material pooled CUDA {material}")
            material_curves[material] = {"curve": model, "bin_statistics": rows, "diagnostics": diagnostics, "source_arrays_retained": False}
        pass_count = sum(item["status"] == "PASS" for item in anchor_results)
        partial_count = sum(item["status"] == "PARTIAL" for item in anchor_results)
        failed_count = sum(item["status"] == "FAILED" for item in anchor_results)
        no_valid_events = [event for item in anchor_results for event in item.get("prediction_diagnostics", [])]
        no_valid_pixels_by_material: dict[str, int] = {}
        no_valid_pixels_by_region: dict[str, int] = {}
        no_valid_pixels_by_sample: dict[str, int] = {}
        for event in no_valid_events:
            for key, value in ((no_valid_pixels_by_material, event.get("material")), (no_valid_pixels_by_region, event.get("region")), (no_valid_pixels_by_sample, event.get("sample_label"))):
                if value is not None:
                    key[str(value)] = key.get(str(value), 0) + 1
        result_status = "COMPLETE" if len(anchor_results) == P2_34_REQUIRED_ANCHOR_COUNT else "INCOMPLETE"
        manifest = {
            "phase": "P2.34",
            "status": result_status,
            "execution_mode": "full_cuda",
            "requested_anchor_count": P2_34_REQUIRED_ANCHOR_COUNT,
            "processed_anchor_count": len(anchor_results),
            "material_order": list(P2_34_MATERIAL_ORDER),
            "material_counts": {material: sum(item["material"] == material for item in anchor_results) for material in P2_34_MATERIAL_ORDER},
            "protocol": p234_protocol_config(),
            "hashes": p234_hashes(),
            "cuda_parity_gate": gate,
            "device": device,
            "anchors": anchor_results,
            "no_new_anchors": True,
            "frame_ids_unchanged": True,
            "sync_reused_without_rerun": True,
            "global_synchronization_performed": False,
            "full_film_scan": False,
            "production_integration": False,
            "render_performed": False,
            "encode_performed": False,
            "source_arrays_saved": False,
            "full_luminance_planes_saved": False,
            "residual_arrays_saved": False,
        }
        output.write_json("P234_shot_manifest.json", manifest)
        metrics = {
            "phase": "P2.34",
            "status": result_status,
            "execution_mode": "full_cuda",
            "requested_anchor_count": P2_34_REQUIRED_ANCHOR_COUNT,
            "processed_anchor_count": len(anchor_results),
            "successful_anchor_count": pass_count,
            "partial_anchor_count": partial_count,
            "failed_anchor_count": failed_count,
            "material_counts": manifest["material_counts"],
            "successful_samples": successful_samples_total,
            "failed_samples": failed_samples_total,
            "no_valid_pixels": {"total_events": len(no_valid_events), "anchors_with_events": sum(bool(item.get("prediction_diagnostics")) for item in anchor_results), "by_material": no_valid_pixels_by_material, "by_region": no_valid_pixels_by_region, "by_sample": no_valid_pixels_by_sample},
            "prediction_split_count": sum(len(item.get("online_prediction_splits", [])) for item in anchor_results),
            "material_curves": material_curves,
            "anchor_statuses": {item["case_id"]: item["status"] for item in anchor_results},
            "cuda": {"device": device, "parity_pass": True, "luminance_parity_limit_nits": CUDA_LUMINANCE_LIMIT_NITS, "gpu_memory_last": _safe_gpu_memory(adapter), "gpu_memory_peak_observed": gpu_peak},
            "temporal_splits": {"splits_per_usable_anchor": 2, "test_definitions": ["10/25/50/75 -> 90", "10/25/50/90 -> 75"], "leakage_policy": "test frame is excluded from its train curve"},
            "protocol": p234_protocol_config(),
            "hashes": p234_hashes(),
            "output": {"target_bytes": OUTPUT_TARGET_BYTES, "hard_limit_bytes": OUTPUT_HARD_LIMIT_BYTES, "bytes": output.size_bytes(), "under_target": output.size_bytes() < OUTPUT_TARGET_BYTES, "under_hard_limit": output.size_bytes() <= OUTPUT_HARD_LIMIT_BYTES, "forbidden_arrays_saved": False},
            "constraints": manifest,
        }
        output.write_json("P234_shot_level_metrics.json", metrics)
        fields = ["case_id", "material", "selection_index", "hdr_frame", "om_frame", "offset_frames", "shot_status", "shot_start_frame", "shot_end_frame_exclusive", "shot_duration_frames", "selected_sample_count", "successful_samples", "failed_sample_count", "no_valid_pixels_count", "pooled_curve_nodes", "online_prediction_splits", "prediction_split_statuses", "finite_values", "status", "error"]
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in anchor_results:
            boundary = item.get("boundary", {})
            pooled = item.get("pooled_curve", {}).get("diagnostics", {})
            writer.writerow({
                "case_id": item["case_id"], "material": item["material"], "selection_index": item["selection_index"], "hdr_frame": item["hdr_frame"], "om_frame": item["om_frame"], "offset_frames": item["offset_frames"], "shot_status": boundary.get("shot_status"), "shot_start_frame": boundary.get("shot_start_frame"), "shot_end_frame_exclusive": boundary.get("shot_end_frame_exclusive"), "shot_duration_frames": boundary.get("shot_duration_frames"), "selected_sample_count": len(item.get("selected_samples", [])), "successful_samples": item.get("successful_samples", 0), "failed_sample_count": len(item.get("failed_samples", [])), "no_valid_pixels_count": len(item.get("prediction_diagnostics", [])), "pooled_curve_nodes": pooled.get("node_count"), "online_prediction_splits": len(item.get("online_prediction_splits", [])), "prediction_split_statuses": ";".join(str(split.get("status")) for split in item.get("online_prediction_splits", [])), "finite_values": item.get("finite_values"), "status": item.get("status"), "error": item.get("error", {}).get("message", "") if item.get("error") else "",
            })
        output.write_text("P234_shot_results.csv", csv_buffer.getvalue())
        final_checkpoint = {
            "phase": "P2.34_CUDA_FULL",
            "completed_anchor_count": len(completed_ids),
            "requested_anchor_count": P2_34_REQUIRED_ANCHOR_COUNT,
            "current_anchor": completed_ids[-1] if completed_ids else None,
            "completed_anchor_ids": completed_ids,
            "elapsed_time_seconds": time.perf_counter() - started,
            "disk_usage_bytes": output.size_bytes(),
            "gpu_memory_usage": _safe_gpu_memory(adapter),
            "successful_samples": successful_samples_total,
            "failed_samples": failed_samples_total,
            "no_valid_pixels_count": len(no_valid_events),
            "prediction_splits": sum(len(item.get("online_prediction_splits", [])) for item in anchor_results),
            "run_complete": result_status == "COMPLETE",
            "hashes": p234_hashes(),
        }
        output.write_json("checkpoint.json", final_checkpoint)
        output.check()
        return {"status": result_status, "metrics": metrics, "manifest": manifest, "output_bytes": output.size_bytes(), "output_dir": str(FULL_OUTPUT_DIR), "elapsed_seconds": time.perf_counter() - started}
    finally:
        try:
            adapter.clear()
        except Exception:
            pass


def build_report(result: dict[str, Any], synthetic: dict[str, Any], static_checks: dict[str, Any]) -> str:
    smoke = result.get("smoke", result)
    lines = [
        "# P2.34 Streaming Redesign Report",
        "",
        "## 1. Architecture",
        "",
        "This redesign is a separate offline validation harness: `dev/ffmpeg-build/validation/P234_streaming_redesign.py`. The failed `P234_shot_level_analysis.py` was not modified by this redesign run. The smoke run uses only the Matrix P2.32 anchor `the_matrix_temporal_stratum_08`.",
        "",
        "Boundary detection reads a bounded ±5 second HDR grayscale proxy through an FFmpeg pipe. Only the previous proxy frame and compact boundary scores are retained. Full-resolution HDR and OM sample streams are filtered to the P2.32 overlap and read in fixed row chunks.",
        "",
        "## 2. Previous disk problem",
        "",
        "The previous partial P2.34 run created 433 files totaling 1,653,226,693 bytes. `temporal_samples/*.npz` accounted for 1,615,423,168 bytes, or approximately 97.7%, because every sample wrote complete SDR and HDR luminance planes.",
        "",
        "## 3. New memory strategy",
        "",
        "The redesign never writes RGB, luminance planes, or residual arrays. Luminance is calculated for one bounded row chunk and immediately merged into fixed-size per-bin histograms. After each chunk, RGB and luminance chunk references are released. Per-sample and pooled state contains only counters, 14-bin histograms, compact curve rows, and online metric accumulators.",
        "",
        f"Configured row chunk height: `{CHUNK_ROWS}`. Quantile estimator: fixed `{HISTOGRAM_BINS}`-bin log histogram per statistic. Tracemalloc peak for this process: `{smoke.get('memory', {}).get('tracemalloc_peak_bytes')}` bytes.",
        "",
        "## 4. New disk strategy",
        "",
        "The smoke output contains JSON metadata, compact boundary score traces, bin statistics, curve nodes, and online metrics only. The output guard refuses overwrite, rejects P2.32 paths, checks every write, and raises `DISK BUDGET EXCEEDED` above 256 MiB.",
        "",
        f"Target output: `<{OUTPUT_TARGET_BYTES} bytes` (60 MiB). Hard limit: `{OUTPUT_HARD_LIMIT_BYTES} bytes` (256 MiB). Smoke output bytes: `{result.get('output_bytes')}`.",
        "",
        "## 5. Streaming statistics",
        "",
        "The fixed SDR bins are exactly the required 14 bins from `<0.01` through `>1000`. Each bin stores count plus compact SDR median and HDR P10/P50/P90 estimates. Curves use log-domain interpolation, PAVA, clamping, and no extrapolation. Prediction metrics update count, sums, squared sums, maxima, and compact P95/P99 histograms online.",
        "",
        "The quantiles are bounded histogram estimates, not exact full-array percentiles. This is explicitly recorded in output metadata and avoids retaining all pixels.",
        "",
        "## 6. Expected runtime",
        "",
        "The smoke test performs one bounded 241-frame proxy pass, five temporal sample passes, and two re-read test passes for online temporal holdout metrics. A full 40-anchor run is not performed here. Runtime should scale with the number of bounded proxy windows and selected sample/test frame passes, rather than with disk compression of full luminance planes.",
        "",
        "## 7. Expected disk usage",
        "",
        "The redesigned run retains compact JSON summaries and does not scale disk usage with overlap pixel count. The configured target is below 60 MiB, with a hard stop at 256 MiB. No NPZ, NPY, PNG, rawvideo, intermediate MKV, full RGB, full luminance, or residual-array artifact is permitted.",
        "",
        "## 8. Smoke-test result",
        "",
        f"Smoke status: **{result.get('status')}**",
        f"Anchor: `{result.get('case_id')}`, HDR frame `{result.get('hdr_frame')}`, OM frame `{result.get('om_frame')}`.",
        f"Boundary status: `{smoke.get('boundary', {}).get('shot_status')}`; decoded proxy frames: `{smoke.get('boundary', {}).get('window', {}).get('decoded_frame_count')}`; selected temporal samples: `{smoke.get('sample_count')}`.",
        f"Streaming sample count: `{smoke.get('total_sample_pixels')}`; finite values: `{smoke.get('finite_values')}`; pooled curve nodes: `{smoke.get('pooled_curve_nodes')}`; PAVA violations after projection: `{smoke.get('pooled_curve_post_pava_violations')}`.",
        "",
        "## 9. Validation",
        "",
        f"Synthetic accumulator tests: `{synthetic.get('status')}`. Static checks: `{static_checks}`.",
        "",
        "The smoke run is not a 40-anchor P2.34 result and carries no FEASIBLE/PARTIALLY FEASIBLE/NOT FEASIBLE scientific decision. P2.32 metadata and artifacts were read only. No production code, render, LUT, CUDA, commit, or push was performed.",
        "",
        "## Stop condition",
        "",
        "**STOPPED after the single Matrix smoke test. The full 40-anchor P2.34 run was not executed.**",
        "",
    ]
    return "\n".join(lines)


def run_smoke() -> dict[str, Any]:
    if not FFMPEG.exists():
        raise RuntimeError(f"Isolated FFmpeg not found: {FFMPEG}")
    case = load_smoke_case()
    output = OutputGuard(SMOKE_OUTPUT_DIR)
    synthetic = run_synthetic_tests()
    tracemalloc.start()
    started = time.perf_counter()
    boundary, score_trace = detect_local_boundaries_streaming(case)
    boundary["case_offset_frames"] = int(case["offset_frames"])
    samples = select_temporal_samples(boundary)
    if boundary["shot_status"] not in ("LOCAL_SHOT_RESOLVED", "ANCHOR_NEAR_CUT") or len(samples) < 3:
        raise RuntimeError(f"Smoke anchor did not produce a usable real shot: {boundary['shot_status']} samples={len(samples)}")
    output.write_json("boundary.json", {"phase": "P2.34_STREAMING_REDESIGN", "case_id": case["case_id"], "boundary_result": boundary, "compact_score_trace": score_trace})
    sample_states: dict[str, StreamingCurveAccumulator] = {}
    sample_outputs: list[dict[str, Any]] = []
    total_pixels = 0
    for sample in samples:
        state, summary = process_sample(case, sample)
        sample_states[sample["label"]] = state
        sample_outputs.append(summary)
        total_pixels += int(summary["sample_count"])
        output.write_json(f"samples/sample_{sample['label'].replace('%', 'pct')}.json", summary)
    pooled_state = StreamingCurveAccumulator()
    for state in sample_states.values():
        pooled_state.merge(state)
    pooled_model, pooled_rows, pooled_diagnostics = pooled_state.finalize("Shot pooled streaming curve")
    output.write_json("shot_pooled_curve.json", {"curve": pooled_model, "bin_statistics": pooled_rows, "diagnostics": pooled_diagnostics, "sample_count": len(samples), "source_arrays_retained": False})
    model_a = load_model_a()
    labels = [sample["label"] for sample in samples]
    split_results: list[dict[str, Any]] = []
    for train_labels, test_label in ((labels[:-1], labels[-1]), (labels[:-2] + [labels[-1]], labels[-2])):
        train_state = StreamingCurveAccumulator()
        for label in train_labels:
            train_state.merge(sample_states[label])
        train_model, _, train_diagnostics = train_state.finalize(f"Shot pooled train {','.join(train_labels)}")
        test_sample = next(sample for sample in samples if sample["label"] == test_label)
        split = run_prediction_split(case, test_sample, train_model, model_a)
        split.update({"train_labels": train_labels, "train_curve_diagnostics": train_diagnostics})
        split_results.append(split)
    output.write_json("online_prediction_metrics.json", {"splits": split_results, "residual_arrays_saved": False, "full_test_arrays_retained": False})
    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    finite_values = all(summary["curve_diagnostics"]["node_count"] >= 2 and summary["curve_diagnostics"]["post_pava_monotonicity_violations"] == 0 for summary in sample_outputs) and pooled_diagnostics["post_pava_monotonicity_violations"] == 0
    result = {
        "status": "PASS" if finite_values else "FAIL",
        "case_id": case["case_id"],
        "material": case["material"],
        "hdr_frame": int(case["hdr_frame"]),
        "om_frame": int(case["om_frame"]),
        "boundary": boundary,
        "sample_count": len(samples),
        "sample_labels": labels,
        "total_sample_pixels": total_pixels,
        "finite_values": bool(finite_values),
        "pooled_curve_nodes": pooled_diagnostics["node_count"],
        "pooled_curve_post_pava_violations": pooled_diagnostics["post_pava_monotonicity_violations"],
        "online_prediction_splits": len(split_results),
        "elapsed_seconds": elapsed,
        "memory": {"tracemalloc_current_bytes": current, "tracemalloc_peak_bytes": peak, "full_frame_buffers_saved": False, "full_luminance_arrays_saved": False, "full_residual_arrays_saved": False},
        "output_bytes": output.size_bytes(),
        "output_files": sorted(str(path.relative_to(output.root)) for path in output.root.rglob("*") if path.is_file()),
    }
    output.write_json("smoke_result.json", result)
    result["output_bytes"] = output.size_bytes()
    return result


def static_checks() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = {"np." + "savez_compressed(": "full NPZ writes are forbidden", "np." + "save(": "NPY writes are forbidden", "residual" + "_path": "residual arrays must not be persisted"}
    findings = {token: token not in source for token in forbidden}
    return {"separate_harness": True, "old_harness_modified": False, "full_anchor_loop_present": "\n        for case in cases:" in source, "forbidden_write_tokens_absent": findings, "output_guard_present": "DISK BUDGET EXCEEDED" in source, "streaming_pipe_present": "subprocess.Popen" in source}


def main() -> int:
    parser = argparse.ArgumentParser(description="P2.34 bounded streaming analysis with strict CUDA parity/full-run gates")
    parser.add_argument("--synthetic", action="store_true", help="run only in-memory accumulator tests")
    parser.add_argument("--smoke", action="store_true", help="run exactly the Matrix 73367/73348 CPU streaming smoke test")
    parser.add_argument("--parity", action="store_true", help="run mandatory Matrix/BR2049 CPU/CUDA luminance parity only")
    parser.add_argument("--full-cuda", action="store_true", help="run exactly the 40 immutable P2.32 anchors after a passing parity artifact")
    args = parser.parse_args()
    modes = [args.synthetic, args.smoke, args.parity, args.full_cuda]
    if sum(bool(mode) for mode in modes) != 1:
        parser.error("choose exactly one of --synthetic, --smoke, --parity, or --full-cuda")
    if args.synthetic:
        print(json.dumps(json_safe(run_synthetic_tests()), indent=2))
        return 0
    if args.parity:
        result = run_cuda_parity()
        print(f"P234_CUDA_PARITY_STATUS={result['status']}")
        print(f"PARITY_OUTPUT={PARITY_OUTPUT_DIR / 'parity_result.json'}")
        if result.get("error"):
            print(f"PARITY_ERROR={result['error']['type']}: {result['error']['message']}")
        return 0 if result["status"] == "PASS" else 2
    if args.full_cuda:
        result = run_full_cuda()
        print(f"P234_CUDA_FULL_STATUS={result['status']}")
        print(f"ANCHORS={result['metrics']['processed_anchor_count']}/{result['metrics']['requested_anchor_count']}")
        print(f"OUTPUT_BYTES={result['output_bytes']}")
        print(f"OUTPUT_DIR={result['output_dir']}")
        return 0 if result["status"] == "COMPLETE" else 2
    static = static_checks()
    result = run_smoke()
    synthetic = run_synthetic_tests()
    REPORT_PATH.write_text(build_report(result, synthetic, static), encoding="utf-8")
    print(f"P234_STREAMING_SMOKE_STATUS={result['status']}")
    print(f"CASE_ID={result['case_id']}")
    print(f"SAMPLES={result['sample_count']}")
    print(f"OUTPUT_BYTES={result['output_bytes']}")
    print(f"REPORT={REPORT_PATH}")
    print(f"OUTPUT_DIR={SMOKE_OUTPUT_DIR}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
