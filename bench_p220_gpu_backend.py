"""P2.20 benchmark for the optional persistent float64 GPU backend.

The benchmark processes one ROI at a time.  It uses the production CPU backend
as reference, instruments the GPU backend's upload/transform/download stages,
and records workspace/LUT reuse and CuPy/device memory.  It does not render
video or modify production transform math.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import pq_eotf
from auto_openmatte.processing.transform import (
    _LUM_B_2020,
    _LUM_G_2020,
    _LUM_R_2020,
)
from auto_openmatte.processing.transform_backend import (
    CPUTransformBackend,
    GPUTransformBackend,
)

try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only on CPU-only machines
    cp = None

ROOT = Path(__file__).parent
CURVE_FILE = ROOT / "luminance_curve_br2049.json"
PEAK_NITS = 10000.0
ROI_H = 560
ROI_W = 3840
WARMUPS = 2
TIMED_ROIS = 10
SCALING_RUNS = 3
SCALING_SIZES = ((280, 3840), (560, 3840), (1080, 1920))
SEED = 22020


def stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    return {
        "median_ms": float(median(ordered)),
        "mean_ms": float(mean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
        "fps_from_median": float(1000.0 / median(ordered)),
        "samples_ms": ordered,
    }


def luminance_nits(signal: np.ndarray) -> np.ndarray:
    """Decode PQ output and compute BT.2020 luminance in nits."""
    nits = pq_eotf(signal)
    return (
        _LUM_R_2020 * nits[..., 0]
        + _LUM_G_2020 * nits[..., 1]
        + _LUM_B_2020 * nits[..., 2]
    )


def cuda_event_ms(action: Any) -> tuple[float, Any]:
    """Run one action and measure its default-stream CUDA duration."""
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    result = action()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)), result


def make_material_like_roi(
    rng: np.random.Generator,
    index: int,
    height: int = ROI_H,
    width: int = ROI_W,
) -> np.ndarray:
    """Generate one varied ROI without retaining a sequence in memory."""
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)
    phase = index * 0.17
    base = 0.015 + 0.58 * (0.25 * x + 0.75 * y) ** 1.18
    texture = (
        0.025 * np.sin(xx * 0.031 + phase)
        + 0.018 * np.cos(yy * 0.047 - phase)
        + 0.008 * rng.standard_normal((height, width))
    )
    frame = np.stack(
        [
            base + texture,
            base * 0.86 - texture * 0.35,
            base * 0.68 + texture * 0.55,
        ],
        axis=-1,
    )
    return np.clip(frame, 0.0, 1.0).astype(np.float64, copy=False)


def run_scaling(
    cpu_backend: CPUTransformBackend,
    gpu_backend: GPUTransformBackend,
    cpu_workspace: Any,
    gpu_workspace: Any,
    transform: ShotTransform,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure representative ROI sizes one frame at a time."""
    results: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []
    for size_index, (height, width) in enumerate(SCALING_SIZES):
        warmup_frame = make_material_like_roi(rng, 100 + size_index, height, width)
        cpu_backend.transform_roi(warmup_frame, transform, workspace=cpu_workspace)
        measure_gpu_roi(gpu_backend, gpu_workspace, warmup_frame, transform)
        del warmup_frame

        cpu_samples: list[float] = []
        gpu_total_samples: list[float] = []
        gpu_transform_samples: list[float] = []
        max_diff = 0.0
        for run_index in range(SCALING_RUNS):
            frame = make_material_like_roi(
                rng,
                200 + size_index * SCALING_RUNS + run_index,
                height,
                width,
            )
            cpu_start = time.perf_counter()
            cpu_output = cpu_backend.transform_roi(
                frame,
                transform,
                workspace=cpu_workspace,
            )
            cpu_samples.append((time.perf_counter() - cpu_start) * 1000.0)
            gpu_timing, gpu_output = measure_gpu_roi(
                gpu_backend,
                gpu_workspace,
                frame,
                transform,
            )
            gpu_total_samples.append(gpu_timing["total_ms"])
            gpu_transform_samples.append(gpu_timing["gpu_transform_ms"])
            max_diff = max(max_diff, float(np.max(np.abs(cpu_output - gpu_output))))
            del frame, cpu_output, gpu_output

        memory = memory_snapshot(gpu_backend, gpu_workspace)
        memories.append(memory)
        results.append(
            {
                "height": height,
                "width": width,
                "pixels": height * width,
                "runs": SCALING_RUNS,
                "cpu_transform": stats(cpu_samples),
                "gpu_total": stats(gpu_total_samples),
                "gpu_transform_cuda_events": stats(gpu_transform_samples),
                "max_rgb_diff_normalized": max_diff,
                "workspace_allocations_after_shape": gpu_workspace.metadata[
                    "workspace_allocations"
                ],
                "persistent_roi_buffers_bytes": memory[
                    "persistent_roi_buffers_bytes"
                ],
            }
        )
    return results, memories


def environment_info() -> dict[str, Any]:
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    device = cp.cuda.Device(0)
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "cupy_version": cp.__version__,
        "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
        "device_name": name,
        "compute_capability": str(device.compute_capability),
        "device_total_memory_bytes": int(props["totalGlobalMem"]),
    }


def measure_gpu_roi(
    backend: GPUTransformBackend,
    workspace: Any,
    frame: np.ndarray,
    transform: ShotTransform,
) -> tuple[dict[str, float], np.ndarray]:
    """Measure one ROI through the backend's exact staged implementation."""
    workspace.metadata["transform"] = transform
    cp.cuda.Stream.null.synchronize()
    total_start = time.perf_counter()
    upload_ms, buffers = cuda_event_ms(
        lambda: backend._upload_to_device(frame, workspace)
    )
    transform_ms, _ = cuda_event_ms(
        lambda: backend._transform_device(workspace, buffers)
    )
    download_ms, output = cuda_event_ms(
        lambda: backend._download_to_host(workspace, buffers)
    )
    cp.cuda.Stream.null.synchronize()
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return {
        "upload_ms": upload_ms,
        "gpu_transform_ms": transform_ms,
        "download_ms": download_ms,
        "total_ms": total_ms,
    }, output


def memory_snapshot(backend: GPUTransformBackend, workspace: Any) -> dict[str, Any]:
    return backend.memory_report(workspace)


def main() -> int:
    if cp is None:
        print("CuPy is not installed; P2.20 GPU benchmark was not run.")
        return 2

    curve_data = json.loads(CURVE_FILE.read_text(encoding="utf-8"))
    transform = ShotTransform(shot_id=22020, luminance_curve=curve_data["curve"])
    cpu_backend = CPUTransformBackend()
    gpu_backend = GPUTransformBackend()
    cpu_workspace = cpu_backend.prepare_shot(transform, peak_nits=PEAK_NITS)
    gpu_workspace = gpu_backend.prepare_shot(transform, peak_nits=PEAK_NITS)

    rng = np.random.default_rng(SEED)
    print(
        f"P2.20 benchmark: {WARMUPS} warmups + {TIMED_ROIS} sequential "
        f"float64 ROIs of {ROI_H}x{ROI_W}"
    )

    # Warmups establish CUDA context, kernel caches, and the persistent ROI buffers.
    for index in range(WARMUPS):
        frame = make_material_like_roi(rng, index)
        cpu_backend.transform_roi(frame, transform, workspace=cpu_workspace)
        measure_gpu_roi(gpu_backend, gpu_workspace, frame, transform)
        del frame

    initial_buffers = gpu_workspace.buffers
    warmup_memory = memory_snapshot(gpu_backend, gpu_workspace)
    cpu_samples: list[float] = []
    upload_samples: list[float] = []
    transform_samples: list[float] = []
    download_samples: list[float] = []
    total_samples: list[float] = []
    memory_samples: list[dict[str, Any]] = []
    max_rgb_diff = 0.0
    max_luminance_error_nits = 0.0

    for index in range(TIMED_ROIS):
        frame = make_material_like_roi(rng, WARMUPS + index)
        cpu_start = time.perf_counter()
        cpu_output = cpu_backend.transform_roi(
            frame,
            transform,
            workspace=cpu_workspace,
            peak_nits=PEAK_NITS,
        )
        cpu_samples.append((time.perf_counter() - cpu_start) * 1000.0)

        gpu_timing, gpu_output = measure_gpu_roi(
            gpu_backend,
            gpu_workspace,
            frame,
            transform,
        )
        upload_samples.append(gpu_timing["upload_ms"])
        transform_samples.append(gpu_timing["gpu_transform_ms"])
        download_samples.append(gpu_timing["download_ms"])
        total_samples.append(gpu_timing["total_ms"])
        max_rgb_diff = max(max_rgb_diff, float(np.max(np.abs(cpu_output - gpu_output))))
        max_luminance_error_nits = max(
            max_luminance_error_nits,
            float(
                np.max(
                    np.abs(luminance_nits(cpu_output) - luminance_nits(gpu_output))
                )
            ),
        )
        memory_samples.append(memory_snapshot(gpu_backend, gpu_workspace))
        del frame, cpu_output, gpu_output

    main_final_memory = memory_snapshot(gpu_backend, gpu_workspace)
    main_buffer_reused = gpu_workspace.buffers is initial_buffers
    main_workspace_allocations = gpu_workspace.metadata["workspace_allocations"]
    scaling, scaling_memories = run_scaling(
        cpu_backend,
        gpu_backend,
        cpu_workspace,
        gpu_workspace,
        transform,
        rng,
    )
    final_memory = memory_snapshot(gpu_backend, gpu_workspace)
    all_memory_samples = memory_samples + scaling_memories
    pool_reserved_high_water = max(
        [warmup_memory["pool_reserved_bytes"]]
        + [int(sample["pool_reserved_bytes"]) for sample in all_memory_samples]
    )
    persistent_bytes = int(main_final_memory["persistent_bytes_total"])
    results = {
        "benchmark": {
            "name": "P2.20 optional persistent float64 GPU backend",
            "roi": {"height": ROI_H, "width": ROI_W, "channels": 3},
            "dtype": "float64",
            "warmups": WARMUPS,
            "timed_sequential_rois": TIMED_ROIS,
            "scaling_runs_per_size": SCALING_RUNS,
            "seed": SEED,
            "one_roi_at_a_time": True,
            "full_film_rendered": False,
        },
        "environment": environment_info(),
        "timing_ms": {
            "cpu_backend_transform": stats(cpu_samples),
            "gpu_upload": stats(upload_samples),
            "gpu_transform_cuda_events": stats(transform_samples),
            "gpu_download": stats(download_samples),
            "gpu_total_upload_transform_download": stats(total_samples),
        },
        "accuracy": {
            "max_rgb_diff_normalized": max_rgb_diff,
            "max_luminance_error_nits": max_luminance_error_nits,
            "rgb_acceptance_limit": 1e-5,
            "luminance_acceptance_limit_nits": 0.01,
            "pass": max_rgb_diff <= 1e-5 and max_luminance_error_nits <= 0.01,
            "comparison": "CPUTransformBackend vs GPUTransformBackend",
        },
        "workspace_reuse": {
            "same_main_roi_buffer_object": main_buffer_reused,
            "main_roi_workspace_allocations": main_workspace_allocations,
            "curve_lut_built_once": gpu_workspace.metadata["curve_lut_built_once"],
            "curve_lut_copied_once": gpu_workspace.metadata["curve_lut_copied_once"],
            "pq_lut_copied_once_per_backend": gpu_workspace.metadata[
                "pq_lut_copied_once_per_backend"
            ],
            "curve_lut_entries": int(gpu_workspace.curve_lut_gpu.size),
            "pq_lut_entries": int(gpu_workspace.pq_lut_gpu.size),
            "curve_lut_dtype": str(gpu_workspace.curve_lut_gpu.dtype),
            "pq_lut_dtype": str(gpu_workspace.pq_lut_gpu.dtype),
        },
        "memory": {
            "warmup": warmup_memory,
            "main_roi_final": main_final_memory,
            "post_scaling_final": final_memory,
            "pool_reserved_high_water_bytes": pool_reserved_high_water,
            "persistent_bytes_total": persistent_bytes,
            "temporary_pool_reserved_over_persistent_bytes": max(
                0, pool_reserved_high_water - persistent_bytes
            ),
            "sample_count": len(all_memory_samples),
            "sample_pool_reserved_bytes": [
                int(sample["pool_reserved_bytes"]) for sample in all_memory_samples
            ],
        },
        "scaling": scaling,
        "scope_guard": {
            "cpu_default_preserved": True,
            "gpu_requires_explicit_experimental_flag": True,
            "float32_production": False,
            "video_rendered": False,
            "commit_or_push": False,
        },
    }
    output_path = ROOT / "P2.20_gpu_backend_benchmark_results.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
