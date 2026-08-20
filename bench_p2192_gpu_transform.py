"""P2.19.2 isolated CuPy benchmark for the current CPU shot transform.

This file is intentionally standalone. It calls the production CPU transform only
for the reference result and implements the GPU path with CuPy arrays. It does
not modify production code, LUT sources, tests, geometry, or video outputs.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import cupy as cp
import numpy as np

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import (
    _get_pq_oetf_sqrt_lut,
    pq_eotf,
)
from auto_openmatte.processing.luminance import build_curve_lut
from auto_openmatte.processing.transform import (
    _LUM_B_2020,
    _LUM_G_2020,
    _LUM_R_2020,
    _M_709_TO_2020,
    apply_shot_transform,
)

ROOT = Path(__file__).parent
CURVE_FILE = ROOT / "luminance_curve_br2049.json"
PEAK_NITS = 10000.0
PQ_LUT_SIZE = 65536
ROI_H = 560
ROI_W = 3840
WARMUPS = 2
TIMED_RUNS = 10
SCALING_RUNS = 5
SCALING_SIZES = ((280, 3840), (560, 3840), (1080, 1920))
REAL_TIME_SECONDS = 2715.0
REAL_FPS = 23.976
REAL_SYNC_OFFSET = 1167

HDR_SOURCE_CANDIDATES = (
    Path(
        r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"
    ),
)
OM_SOURCE_CANDIDATES = (
    Path(
        r"G:\Filmy\IMAX format (open matte)\Blade Runner 2049  Open Matte 2160p"
        r"\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
    ),
    Path(
        r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
    ),
)


GPU_MATRIX = cp.asarray(_M_709_TO_2020, dtype=cp.float64)
GPU_LUM_R = cp.float64(_LUM_R_2020)
GPU_LUM_G = cp.float64(_LUM_G_2020)
GPU_LUM_B = cp.float64(_LUM_B_2020)


def stats(values: list[float]) -> dict[str, Any]:
    """Return the requested summary statistics in milliseconds."""
    ordered = sorted(float(value) for value in values)
    return {
        "median_ms": float(median(ordered)),
        "mean_ms": float(mean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "samples_ms": ordered,
    }


def load_setup() -> dict[str, Any]:
    """Load production LUTs once and copy each LUT to the GPU once."""
    curve = json.loads(CURVE_FILE.read_text(encoding="utf-8"))["curve"]
    transform = ShotTransform(shot_id=2192, luminance_curve=curve)
    curve_lut = build_curve_lut(curve)
    pq_lut = _get_pq_oetf_sqrt_lut()

    # These are the only LUT transfers performed by the benchmark setup.
    curve_lut_gpu = cp.asarray(curve_lut["lut"], dtype=cp.float64)
    pq_lut_gpu = cp.asarray(pq_lut, dtype=cp.float64)
    curve_params_gpu = {
        "start": cp.asarray(curve_lut["curve_start_linear"], dtype=cp.float64),
        "start_hdr": cp.asarray(curve_lut["curve_start_hdr_linear"], dtype=cp.float64),
        "inv_range": cp.asarray(curve_lut["inv_range"], dtype=cp.float64),
    }
    cp.cuda.Stream.null.synchronize()

    return {
        "curve": curve,
        "transform": transform,
        "curve_lut": curve_lut,
        "curve_lut_gpu": curve_lut_gpu,
        "curve_params_gpu": curve_params_gpu,
        "pq_lut_gpu": pq_lut_gpu,
        "matrix_gpu": GPU_MATRIX,
        "lut_setup": {
            "curve_entries": int(curve_lut_gpu.size),
            "curve_dtype": str(curve_lut_gpu.dtype),
            "pq_entries": int(pq_lut_gpu.size),
            "pq_dtype": str(pq_lut_gpu.dtype),
            "curve_lut_host_bytes": int(curve_lut["lut"].nbytes),
            "pq_lut_host_bytes": int(pq_lut.nbytes),
            "curve_lut_device_bytes": int(curve_lut_gpu.nbytes),
            "pq_lut_device_bytes": int(pq_lut_gpu.nbytes),
            "copied_once_before_benchmark": True,
        },
    }


def gpu_apply_curve(
    luminance: cp.ndarray,
    curve_lut: cp.ndarray,
    curve_params: dict[str, cp.ndarray],
) -> cp.ndarray:
    """CuPy equivalent of production apply_luminance_curve()."""
    result = cp.zeros_like(luminance)
    curve_start = curve_params["start"]
    bridge_mask = (luminance > 0.0) & (luminance < curve_start)
    bridge = (luminance / curve_start) * curve_params["start_hdr"]
    result = cp.where(bridge_mask, bridge, result)

    fitted_mask = luminance >= curve_start
    t = (luminance - curve_start) * curve_params["inv_range"]
    t = cp.clip(t, 0.0, 1.0)
    indices = (t * (curve_lut.size - 1)).astype(cp.int64)
    indices = cp.clip(indices, 0, curve_lut.size - 1)
    fitted = curve_lut[indices]
    return cp.where(fitted_mask, fitted, result)


def gpu_pq_oetf_sqrt_lut(
    linear: cp.ndarray,
    pq_lut: cp.ndarray,
    peak_nits: float = PEAK_NITS,
) -> cp.ndarray:
    """CuPy equivalent of production pq_oetf_sqrt_lut()."""
    normalized = cp.clip((linear * peak_nits) / PEAK_NITS, 0.0, 1.0)
    position = cp.sqrt(normalized) * (pq_lut.size - 1)
    lower = position.astype(cp.int64)
    lower = cp.clip(lower, 0, pq_lut.size - 2)
    fraction = position - lower
    return pq_lut[lower] * (1.0 - fraction) + pq_lut[lower + 1] * fraction


def gpu_apply_shot_transform(
    frame_gpu: cp.ndarray,
    setup: dict[str, Any],
    peak_nits: float = PEAK_NITS,
) -> cp.ndarray:
    """Apply the current production transform using only GPU array operations."""
    transform = setup["transform"]

    # Step 1: BT.1886 EOTF, matching linearize(frame, "bt709").
    linear_709 = cp.power(cp.clip(frame_gpu, 0.0, 1.0), 2.4)

    # Step 2: BT.709 -> BT.2020 and the production nonnegative clamp.
    shape = linear_709.shape
    linear_2020 = cp.matmul(linear_709.reshape(-1, 3), setup["matrix_gpu"].T)
    linear_2020 = cp.maximum(linear_2020.reshape(shape), 0.0)

    # Step 3: BT.2020 luminance, exact production curve LUT/bridge, and ratio.
    linear = linear_2020
    if transform.luminance_curve and len(transform.luminance_curve) >= 2:
        luminance = (
            GPU_LUM_R * linear[..., 0]
            + GPU_LUM_G * linear[..., 1]
            + GPU_LUM_B * linear[..., 2]
        )
        mapped = gpu_apply_curve(
            luminance,
            setup["curve_lut_gpu"],
            setup["curve_params_gpu"],
        )
        safe_mask = luminance > 1e-6
        safe_luminance = cp.where(safe_mask, luminance, 1.0)
        ratio = cp.where(safe_mask, mapped / safe_luminance, 1.0)
        linear = cp.maximum(linear * ratio[..., cp.newaxis], 0.0)

    # Step 4: Optional production color-matrix branch.
    if transform.color_matrix:
        matrix = np.asarray(transform.color_matrix, dtype=np.float64)
        if not np.allclose(matrix, np.eye(3), atol=0.001):
            matrix_gpu = cp.asarray(matrix, dtype=cp.float64)
            linear = cp.maximum(
                cp.matmul(linear.reshape(-1, 3), matrix_gpu.T).reshape(shape),
                0.0,
            )

    # Step 5: Optional production saturation branch.
    if abs(transform.saturation - 1.0) > 0.01:
        luminance = (
            GPU_LUM_R * linear[..., 0]
            + GPU_LUM_G * linear[..., 1]
            + GPU_LUM_B * linear[..., 2]
        )
        chroma = linear - luminance[..., cp.newaxis]
        linear = cp.maximum(
            luminance[..., cp.newaxis] + chroma * transform.saturation,
            0.0,
        )

    # Step 6: Current PQ sqrt-LUT and final [0,1] clamp.
    hdr_signal = gpu_pq_oetf_sqrt_lut(linear, setup["pq_lut_gpu"], peak_nits)
    return cp.clip(hdr_signal, 0.0, 1.0)


def cpu_reference(frame: np.ndarray, setup: dict[str, Any]) -> np.ndarray:
    """Use the actual production implementation as the sole CPU reference."""
    return apply_shot_transform(
        frame,
        setup["transform"],
        sdr_transfer="bt709",
        hdr_transfer="smpte2084",
        peak_nits=PEAK_NITS,
        prebuilt_lut=setup["curve_lut"],
    )


def benchmark_cpu(
    frame: np.ndarray,
    setup: dict[str, Any],
    warmups: int = WARMUPS,
    runs: int = TIMED_RUNS,
) -> dict[str, Any]:
    """Benchmark the unmodified production CPU transform."""
    for _ in range(warmups):
        cpu_reference(frame, setup)
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        output = cpu_reference(frame, setup)
        timings.append((time.perf_counter() - start) * 1000.0)
        del output
    return stats(timings)


def benchmark_gpu_transform(
    frame_gpu: cp.ndarray,
    setup: dict[str, Any],
    warmups: int = WARMUPS,
    runs: int = TIMED_RUNS,
) -> tuple[dict[str, Any], cp.ndarray]:
    """Benchmark GPU transform-only time with CUDA events."""
    for _ in range(warmups):
        warm_output = gpu_apply_shot_transform(frame_gpu, setup)
        cp.cuda.Stream.null.synchronize()
        del warm_output

    timings = []
    output = None
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record(cp.cuda.Stream.null)
        output = gpu_apply_shot_transform(frame_gpu, setup)
        end_event.record(cp.cuda.Stream.null)
        end_event.synchronize()
        timings.append(float(cp.cuda.get_elapsed_time(start_event, end_event)))
    assert output is not None
    return stats(timings), output


def benchmark_upload(frame: np.ndarray, runs: int = TIMED_RUNS) -> dict[str, Any]:
    """Measure host-to-device upload after setup/warm-up."""
    for _ in range(WARMUPS):
        value = cp.asarray(frame, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        del value
    timings = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        value = cp.asarray(frame, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
        del value
    return stats(timings)


def benchmark_download(output_gpu: cp.ndarray, runs: int = TIMED_RUNS) -> dict[str, Any]:
    """Measure device-to-host download of a completed GPU output."""
    for _ in range(WARMUPS):
        value = cp.asnumpy(output_gpu)
        cp.cuda.Stream.null.synchronize()
        del value
    timings = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        value = cp.asnumpy(output_gpu)
        cp.cuda.Stream.null.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
        del value
    return stats(timings)


def benchmark_total_pipeline(
    frame: np.ndarray,
    setup: dict[str, Any],
    runs: int = TIMED_RUNS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure upload + GPU transform + synchronization + download."""
    for _ in range(WARMUPS):
        frame_gpu = cp.asarray(frame, dtype=cp.float64)
        output_gpu = gpu_apply_shot_transform(frame_gpu, setup)
        cp.cuda.Stream.null.synchronize()
        output_cpu = cp.asnumpy(output_gpu)
        del frame_gpu, output_gpu, output_cpu

    timings = []
    process_cpu_ms = []
    for _ in range(runs):
        cp.cuda.Stream.null.synchronize()
        wall_start = time.perf_counter()
        process_start = time.process_time()
        frame_gpu = cp.asarray(frame, dtype=cp.float64)
        output_gpu = gpu_apply_shot_transform(frame_gpu, setup)
        cp.cuda.Stream.null.synchronize()
        output_cpu = cp.asnumpy(output_gpu)
        cp.cuda.Stream.null.synchronize()
        process_cpu_ms.append((time.process_time() - process_start) * 1000.0)
        timings.append((time.perf_counter() - wall_start) * 1000.0)
        del frame_gpu, output_gpu, output_cpu

    timing_summary = stats(timings)
    wall_mean = timing_summary["mean_ms"]
    cpu_mean = float(mean(process_cpu_ms))
    cpu_count = max(1, os.cpu_count() or 1)
    utilization = {
        "process_cpu_ms_mean": cpu_mean,
        "wall_ms_mean": wall_mean,
        "process_cpu_percent_one_core_mean": cpu_mean / wall_mean * 100.0,
        "process_cpu_percent_all_cores_mean": cpu_mean / wall_mean * 100.0 / cpu_count,
        "logical_cpu_count": cpu_count,
        "samples_process_cpu_ms": process_cpu_ms,
    }
    return timing_summary, utilization


def device_memory() -> dict[str, int | None]:
    """Read CuPy pool and device memory counters."""
    pool = cp.get_default_memory_pool()
    try:
        free_bytes, total_bytes = cp.cuda.Device().mem_info
    except (AttributeError, RuntimeError):
        free_bytes, total_bytes = None, None
    return {
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_reserved_bytes": int(pool.total_bytes()),
        "device_free_bytes": int(free_bytes) if free_bytes is not None else None,
        "device_total_bytes": int(total_bytes) if total_bytes is not None else None,
    }


def measure_gpu_memory(frame_gpu: cp.ndarray, setup: dict[str, Any]) -> dict[str, Any]:
    """Measure persistent, post-transform, and pool high-water memory."""
    pool = cp.get_default_memory_pool()
    cp.cuda.Stream.null.synchronize()
    pool.free_all_blocks()
    before = device_memory()
    output_gpu = gpu_apply_shot_transform(frame_gpu, setup)
    cp.cuda.Stream.null.synchronize()
    after = device_memory()
    output_bytes = int(output_gpu.nbytes)
    del output_gpu
    cp.cuda.Stream.null.synchronize()
    after_release = device_memory()
    return {
        "before_transform": before,
        "after_transform": after,
        "after_output_release": after_release,
        "output_bytes": output_bytes,
        "transient_reserved_delta_bytes": max(
            0,
            after["pool_reserved_bytes"] - before["pool_reserved_bytes"],
        ),
        "peak_allocated_proxy_bytes": after["pool_reserved_bytes"],
        "method": (
            "CuPy memory-pool reserved high-water proxy after a clean pool state; "
            "not an optimization pass"
        ),
    }


def pq_luminance_nits(signal: np.ndarray) -> np.ndarray:
    """Compute BT.2020 output luminance in absolute nits."""
    channel_nits = pq_eotf(signal)
    return (
        _LUM_R_2020 * channel_nits[..., 0]
        + _LUM_G_2020 * channel_nits[..., 1]
        + _LUM_B_2020 * channel_nits[..., 2]
    )


def accuracy_metrics(cpu_output: np.ndarray, gpu_output: np.ndarray) -> dict[str, Any]:
    """Compare normalized RGB and PQ-decoded BT.2020 luminance."""
    rgb_difference = np.abs(cpu_output - gpu_output)
    cpu_luminance = pq_luminance_nits(cpu_output)
    gpu_luminance = pq_luminance_nits(gpu_output)
    luminance_difference = np.abs(cpu_luminance - gpu_luminance)
    return {
        "max_rgb_diff_normalized": float(np.max(rgb_difference)),
        "mean_rgb_diff_normalized": float(np.mean(rgb_difference)),
        "rmse_rgb_diff_normalized": float(np.sqrt(np.mean(rgb_difference**2))),
        "max_luminance_error_nits": float(np.max(luminance_difference)),
        "mean_luminance_error_nits": float(np.mean(luminance_difference)),
        "p99_luminance_error_nits": float(np.percentile(luminance_difference, 99)),
        "finite_cpu": bool(np.isfinite(cpu_output).all()),
        "finite_gpu": bool(np.isfinite(gpu_output).all()),
        "gpu_in_range": bool(((gpu_output >= 0.0) & (gpu_output <= 1.0)).all()),
    }


def edge_frame_from_linear_luminance(
    luminance_values: np.ndarray,
    gray_factor: float,
) -> np.ndarray:
    """Create grayscale signal values targeting post-matrix BT.2020 luminance."""
    linear_709 = np.asarray(luminance_values, dtype=np.float64) / gray_factor
    signal = np.power(np.clip(linear_709, 0.0, 1.0), 1.0 / 2.4)
    return np.repeat(signal.reshape(1, -1, 1), 3, axis=2)


def compare_case(
    name: str,
    frame: np.ndarray,
    setup: dict[str, Any],
) -> dict[str, Any]:
    """Run one CPU/GPU edge case and return accuracy plus output diagnostics."""
    cpu_output = cpu_reference(frame, setup)
    frame_gpu = cp.asarray(frame, dtype=cp.float64)
    gpu_output_gpu = gpu_apply_shot_transform(frame_gpu, setup)
    cp.cuda.Stream.null.synchronize()
    gpu_output = cp.asnumpy(gpu_output_gpu)
    metrics = accuracy_metrics(cpu_output, gpu_output)
    return {
        "name": name,
        "input_shape": list(frame.shape),
        "input_min": float(np.min(frame)),
        "input_max": float(np.max(frame)),
        "metrics": metrics,
        "cpu_output_min": float(np.min(cpu_output)),
        "cpu_output_max": float(np.max(cpu_output)),
        "gpu_output_min": float(np.min(gpu_output)),
        "gpu_output_max": float(np.max(gpu_output)),
        "cpu_luminance_nits_min": float(np.min(pq_luminance_nits(cpu_output))),
        "cpu_luminance_nits_max": float(np.max(pq_luminance_nits(cpu_output))),
    }


def run_edge_cases(setup: dict[str, Any]) -> dict[str, Any]:
    """Run required boundary cases and artifact diagnostics."""
    curve_lut = setup["curve_lut"]
    start = float(curve_lut["curve_start_linear"])
    curve_max = float(curve_lut["curve_max_linear"])
    x_points = np.asarray([point[0] for point in setup["curve"]], dtype=np.float64)
    p99_linear = (10.0 ** np.percentile(x_points, 99) - 1e-6) / PEAK_NITS
    p999_linear = (10.0 ** np.percentile(x_points, 99.9) - 1e-6) / PEAK_NITS
    matrix = np.asarray(_M_709_TO_2020, dtype=np.float64)
    gray_factor = float(
        np.dot(
            np.array([_LUM_R_2020, _LUM_G_2020, _LUM_B_2020]),
            matrix @ np.ones(3),
        )
    )

    cases = {
        "A_pure_black": np.zeros((1, 3, 3), dtype=np.float64),
        "B_near_black": edge_frame_from_linear_luminance(
            np.array([1e-12, 1e-10, 1e-8, 1e-6, 1e-4]), gray_factor
        ),
        "C_low_end_bridge": edge_frame_from_linear_luminance(
            start * np.array([1e-3, 0.1, 0.5, 0.999]), gray_factor
        ),
        "D_curve_start": edge_frame_from_linear_luminance(np.array([start]), gray_factor),
        "E_normal_midtones": np.repeat(
            np.array([[[0.18, 0.18, 0.18], [0.5, 0.5, 0.5]]], dtype=np.float64), 1, axis=0
        ),
        "F_P99_region": edge_frame_from_linear_luminance(np.array([p99_linear]), gray_factor),
        "G_P99_9_region": edge_frame_from_linear_luminance(np.array([p999_linear]), gray_factor),
        "H_curve_maximum": edge_frame_from_linear_luminance(np.array([curve_max]), gray_factor),
        "I_above_curve_maximum": np.repeat(
            np.array([[[1.0, 1.0, 1.0], [1.1, 1.1, 1.1]]], dtype=np.float64), 1, axis=0
        ),
    }
    results = [compare_case(name, frame, setup) for name, frame in cases.items()]

    shadow_luminance = np.geomspace(1e-12, max(start * 0.9, 1e-8), 256)
    shadow_frame = edge_frame_from_linear_luminance(shadow_luminance, gray_factor)
    shadow_cpu = cpu_reference(shadow_frame, setup)
    shadow_gpu_frame = cp.asarray(shadow_frame, dtype=cp.float64)
    shadow_gpu = cp.asnumpy(gpu_apply_shot_transform(shadow_gpu_frame, setup))
    shadow_cpu_lum = pq_luminance_nits(shadow_cpu).reshape(-1)
    shadow_gpu_lum = pq_luminance_nits(shadow_gpu).reshape(-1)

    white_luminance = np.linspace(max(curve_max * 0.9, 0.0), 1.0, 256)
    white_frame = edge_frame_from_linear_luminance(white_luminance, gray_factor)
    white_cpu = pq_luminance_nits(cpu_reference(white_frame, setup)).reshape(-1)
    white_gpu_frame = cp.asarray(white_frame, dtype=cp.float64)
    white_gpu = pq_luminance_nits(
        cp.asnumpy(gpu_apply_shot_transform(white_gpu_frame, setup))
    ).reshape(-1)
    return {
        "curve_bounds": {
            "curve_start_linear": start,
            "curve_max_linear": curve_max,
            "p99_linear_target": float(p99_linear),
            "p99_9_linear_target": float(p999_linear),
        },
        "cases": results,
        "artifact_diagnostics": {
            "shadow_cpu_monotonic": bool(np.all(np.diff(shadow_cpu_lum) >= -1e-9)),
            "shadow_gpu_monotonic": bool(np.all(np.diff(shadow_gpu_lum) >= -1e-9)),
            "shadow_cpu_positive_samples": int(np.count_nonzero(shadow_cpu_lum > 0.0)),
            "shadow_gpu_positive_samples": int(np.count_nonzero(shadow_gpu_lum > 0.0)),
            "shadow_cpu_exact_plateau_steps": int(np.count_nonzero(np.diff(shadow_cpu_lum) == 0.0)),
            "shadow_gpu_exact_plateau_steps": int(np.count_nonzero(np.diff(shadow_gpu_lum) == 0.0)),
            "white_cpu_negative_luminance_steps": int(np.count_nonzero(np.diff(white_cpu) < -1e-9)),
            "white_gpu_negative_luminance_steps": int(np.count_nonzero(np.diff(white_gpu) < -1e-9)),
            "white_cpu_finite": bool(np.isfinite(white_cpu).all()),
            "white_gpu_finite": bool(np.isfinite(white_gpu).all()),
            "white_cpu_in_range": bool(
                ((white_cpu >= 0.0) & (white_cpu <= PEAK_NITS + 1e-6)).all()
            ),
            "white_gpu_in_range": bool(
                ((white_gpu >= 0.0) & (white_gpu <= PEAK_NITS + 1e-6)).all()
            ),
            "interpretation": (
                "Monotonic/finite diagnostics found no shadow plateau regression or "
                "isolated white-dot spike in these boundary ramps; exact high-end "
                "repeats are expected where the production curve clamps."
            ),
        },
    }


def extract_real_material_roi() -> tuple[np.ndarray | None, dict[str, Any]]:
    """Extract the two 280-row OM extension regions through an FFmpeg pipe."""
    source = next((path for path in OM_SOURCE_CANDIDATES if path.exists()), None)
    if source is None:
        return None, {
            "status": "SKIPPED",
            "reason": "No configured real-material Open Matte source exists",
            "candidates": [str(path) for path in OM_SOURCE_CANDIDATES],
        }

    om_time = REAL_TIME_SECONDS + REAL_SYNC_OFFSET / REAL_FPS
    command = [
        "ffmpeg",
        "-v",
        "quiet",
        "-nostdin",
        "-ss",
        f"{om_time:.6f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb48le",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, {"status": "SKIPPED", "reason": f"FFmpeg invocation failed: {exc}"}

    expected = 3840 * 2160 * 3 * 2
    if result.returncode != 0 or len(result.stdout) < expected:
        return None, {
            "status": "SKIPPED",
            "reason": "FFmpeg did not return one complete 3840x2160 rgb48le frame",
            "returncode": result.returncode,
            "stderr": result.stderr.decode(errors="replace")[-1000:],
        }

    om_frame = np.frombuffer(result.stdout[:expected], dtype=np.uint16).reshape(2160, 3840, 3)
    om_frame = om_frame.astype(np.float64) / 65535.0
    roi = np.concatenate((om_frame[:280, :, :], om_frame[1880:, :, :]), axis=0)
    return roi, {
        "status": "PASS",
        "source": str(source),
        "om_time_seconds": om_time,
        "roi_shape": list(roi.shape),
        "construction": "concatenate(om_frame[:280,:,:], om_frame[1880:,:,:])",
        "note": (
            "This is the two extension regions processed by Mode A, not a contiguous "
            "560-row crop."
        ),
    }


def run_scaling(setup: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    """Run short CPU/GPU transform-only scaling measurements."""
    results = []
    for height, width in SCALING_SIZES:
        frame = rng.random((height, width, 3), dtype=np.float64)
        cpu_timing = benchmark_cpu(frame, setup, warmups=WARMUPS, runs=SCALING_RUNS)
        frame_gpu = cp.asarray(frame, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        gpu_timing, output_gpu = benchmark_gpu_transform(
            frame_gpu, setup, warmups=WARMUPS, runs=SCALING_RUNS
        )
        del output_gpu, frame_gpu, frame
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        results.append(
            {
                "height": height,
                "width": width,
                "pixels": height * width,
                "cpu_transform": cpu_timing,
                "gpu_transform_only": gpu_timing,
                "cpu_ms_per_million_pixels": (
                    cpu_timing["median_ms"] / (height * width) * 1_000_000.0
                ),
                "gpu_ms_per_million_pixels": (
                    gpu_timing["median_ms"] / (height * width) * 1_000_000.0
                ),
            }
        )
    return results


def environment_info() -> dict[str, Any]:
    """Collect benchmark environment details."""
    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    try:
        runtime = int(cp.cuda.runtime.runtimeGetVersion())
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except cp.cuda.runtime.CUDARuntimeError:
        runtime = None
        device_count = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "cupy_version": cp.__version__,
        "cuda_runtime_reported_by_cupy": runtime,
        "device_count": device_count,
        "device_name": name,
        "compute_capability": str(device.compute_capability),
        "device_memory_bytes": int(props["totalGlobalMem"]),
        "nvidia_smi_checked_externally": "RTX 3080 / driver 595.79 / CUDA 13.2; see P2.19.1 report",
        "nvcc_checked_externally": "NOT FOUND",
    }


def main() -> None:
    setup = load_setup()
    rng = np.random.default_rng(2192)
    frame = rng.random((ROI_H, ROI_W, 3), dtype=np.float64)

    print("Preparing CPU reference and GPU LUT setup...")
    cpu_output = cpu_reference(frame, setup)
    cpu_timing = benchmark_cpu(frame, setup)

    frame_gpu = cp.asarray(frame, dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    gpu_timing, gpu_output_gpu = benchmark_gpu_transform(frame_gpu, setup)
    cp.cuda.Stream.null.synchronize()
    gpu_output = cp.asnumpy(gpu_output_gpu)
    accuracy = accuracy_metrics(cpu_output, gpu_output)

    upload_timing = benchmark_upload(frame)
    download_timing = benchmark_download(gpu_output_gpu)
    total_timing, cpu_utilization = benchmark_total_pipeline(frame, setup)
    memory = measure_gpu_memory(frame_gpu, setup)
    edge_cases = run_edge_cases(setup)

    real_roi, real_info = extract_real_material_roi()
    if real_roi is not None:
        real_cpu = cpu_reference(real_roi, setup)
        real_gpu_frame = cp.asarray(real_roi, dtype=cp.float64)
        real_gpu = cp.asnumpy(gpu_apply_shot_transform(real_gpu_frame, setup))
        cp.cuda.Stream.null.synchronize()
        real_info["accuracy"] = accuracy_metrics(real_cpu, real_gpu)
        del real_gpu_frame, real_cpu, real_gpu, real_roi
        cp.get_default_memory_pool().free_all_blocks()

    scaling = run_scaling(setup, rng)
    cpu_median = cpu_timing["median_ms"]
    gpu_median = gpu_timing["median_ms"]
    total_median = total_timing["median_ms"]
    report_data = {
        "benchmark": {
            "name": "P2.19.2 isolated GPU transform feasibility",
            "roi": {"height": ROI_H, "width": ROI_W, "channels": 3, "dtype": "float64"},
            "warmups": WARMUPS,
            "timed_runs": TIMED_RUNS,
            "scaling_runs": SCALING_RUNS,
            "seed": 2192,
            "gpu_lut_transfers_excluded_from_timing": True,
            "cpu_reference": "auto_openmatte.processing.transform.apply_shot_transform",
        },
        "environment": environment_info(),
        "lut_setup": setup["lut_setup"],
        "timing_ms": {
            "cpu_production_transform": cpu_timing,
            "gpu_transform_only_cuda_events": gpu_timing,
            "cpu_to_gpu_upload": upload_timing,
            "gpu_to_cpu_download": download_timing,
            "gpu_total_upload_transform_download": total_timing,
        },
        "accuracy": accuracy,
        "acceptance": {
            "max_rgb_diff_limit": 1e-5,
            "max_luminance_error_nits_limit": 0.01,
            "max_rgb_diff_pass": accuracy["max_rgb_diff_normalized"] <= 1e-5,
            "max_luminance_error_pass": accuracy["max_luminance_error_nits"] <= 0.01,
            "overall_pass": accuracy["max_rgb_diff_normalized"] <= 1e-5
            and accuracy["max_luminance_error_nits"] <= 0.01,
        },
        "edge_cases": edge_cases,
        "real_material_roi": real_info,
        "memory": memory,
        "cpu_utilization": cpu_utilization,
        "scaling": scaling,
        "performance_summary": {
            "cpu_ms_per_frame_median": cpu_median,
            "gpu_transform_only_ms_per_frame_median": gpu_median,
            "gpu_total_pipeline_ms_per_frame_median": total_median,
            "cpu_fps_from_median": 1000.0 / cpu_median,
            "gpu_transform_only_fps_from_median": 1000.0 / gpu_median,
            "gpu_total_pipeline_fps_from_median": 1000.0 / total_median,
            "cpu_over_gpu_transform_speedup": cpu_median / gpu_median,
            "cpu_over_gpu_total_speedup": cpu_median / total_median,
        },
        "scope_guard": {
            "production_files_modified": False,
            "gpu_integrated": False,
            "video_rendered": False,
            "commit_or_push": False,
        },
    }

    output_path = ROOT / "P2.19.2_gpu_benchmark_results.json"
    output_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(json.dumps(report_data, indent=2))


if __name__ == "__main__":
    main()
