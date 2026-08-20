"""P2.21 read-only audit of the P2.19.2 -> P2.20 GPU timing regression.

This script deliberately does not alter production math, backend buffers, or
configuration.  It invokes the existing P2.19.2 implementation and the current
P2.20 backend, then records comparable 5-warmup/20-iteration statistics,
CUDA-event stage traces, synchronization/copy inventories, memory counters, and
optional nvidia-smi snapshots.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import cupy as cp
import numpy as np

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.processing.transform_backend import (
    CPUTransformBackend,
    GPUTransformBackend,
)

ROOT = Path(__file__).parent
CURVE_FILE = ROOT / "luminance_curve_br2049.json"
PEAK_NITS = 10000.0
ROI_H = 560
ROI_W = 3840
WARMUPS = 5
TIMED_RUNS = 20
SCALING_SIZES = ((280, 3840), (560, 3840), (1080, 1920))
SEED = 2192


# The trace covers the default production transform: identity color matrix and
# saturation=1.0.  Array-method assignments/fill are listed separately because
# they cannot be wrapped by a module-level CuPy function proxy.
STAGE_GROUPS = (
    ("B_bt1886_linearization", 2),
    ("C_bt709_to_bt2020_matrix", 2),
    ("D_bt2020_luminance", 5),
    ("E_luminance_curve_lut", 13),
    ("F_ratio", 4),
    ("G_rgb_times_ratio", 1),
    ("H_rgb_clamp", 1),
    ("I_pq_sqrt_lut", 13),
)
TRACED_FUNCTIONS = {
    "clip",
    "power",
    "matmul",
    "maximum",
    "multiply",
    "add",
    "greater",
    "less",
    "logical_and",
    "divide",
    "copyto",
    "greater_equal",
    "subtract",
    "take",
    "sqrt",
}


def stats(values: list[float]) -> dict[str, Any]:
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "count": int(ordered.size),
        "median_ms": float(np.median(ordered)),
        "mean_ms": float(np.mean(ordered)),
        "min_ms": float(np.min(ordered)),
        "max_ms": float(np.max(ordered)),
        "p10_ms": float(np.percentile(ordered, 10)),
        "p90_ms": float(np.percentile(ordered, 90)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "samples_ms": [float(value) for value in ordered],
    }


def event_elapsed(action: Callable[[], Any]) -> tuple[float, Any]:
    """Measure one default-stream action with CUDA events and wait for completion."""
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    result = action()
    end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(start, end)), result


def device_memory() -> dict[str, int | None]:
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


def nvidia_smi_snapshot() -> dict[str, Any]:
    query = (
        "name,utilization.gpu,utilization.memory,power.draw,clocks.sm,"
        "clocks.mem,memory.used,memory.total"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}
    if result.returncode != 0:
        return {
            "available": False,
            "reason": result.stderr.strip() or f"returncode={result.returncode}",
        }
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 8:
            continue
        rows.append(
            {
                "name": values[0],
                "gpu_utilization_percent": float(values[1]),
                "memory_utilization_percent": float(values[2]),
                "power_watts": float(values[3]),
                "sm_clock_mhz": float(values[4]),
                "memory_clock_mhz": float(values[5]),
                "memory_used_mib": float(values[6]),
                "memory_total_mib": float(values[7]),
            }
        )
    return {"available": True, "gpus": rows}


def static_sync_audit() -> dict[str, Any]:
    paths = (
        ROOT / "bench_p2192_gpu_transform.py",
        ROOT / "bench_p220_gpu_backend.py",
        ROOT / "src" / "auto_openmatte" / "processing" / "transform_backend.py",
    )
    patterns = (
        "Stream.null.synchronize",
        "Device().synchronize",
        "Device(self.device_id)",
        "Event(",
        ".synchronize()",
        "asnumpy(",
        ".set(",
        "cp.asarray(",
        "cupy.asarray(",
    )
    inventory: dict[str, Any] = {}
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        entries: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            for pattern in patterns:
                if pattern in line:
                    entries.append(
                        {
                            "line": line_number,
                            "pattern": pattern,
                            "text": line.strip(),
                        }
                    )
        inventory[str(path.relative_to(ROOT))] = entries
    return inventory


def load_curve() -> list[list[float]]:
    return json.loads(CURVE_FILE.read_text(encoding="utf-8"))["curve"]


def p2192_setup() -> tuple[Any, np.ndarray]:
    import bench_p2192_gpu_transform as legacy

    setup = legacy.load_setup()
    frame = np.random.default_rng(SEED).random((ROI_H, ROI_W, 3), dtype=np.float64)
    return setup, frame


def legacy_upload(frame: np.ndarray) -> dict[str, Any]:
    for _ in range(WARMUPS):
        value = cp.asarray(frame, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        del value
    samples: list[float] = []
    for _ in range(TIMED_RUNS):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        value = cp.asarray(frame, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        del value
    return stats(samples)


def legacy_download(output_gpu: Any) -> dict[str, Any]:
    for _ in range(WARMUPS):
        value = cp.asnumpy(output_gpu)
        cp.cuda.Stream.null.synchronize()
        del value
    samples: list[float] = []
    for _ in range(TIMED_RUNS):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        value = cp.asnumpy(output_gpu)
        cp.cuda.Stream.null.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        del value
    return stats(samples)


def legacy_total(frame: np.ndarray, setup: dict[str, Any]) -> dict[str, Any]:
    import bench_p2192_gpu_transform as legacy

    for _ in range(WARMUPS):
        frame_gpu = cp.asarray(frame, dtype=cp.float64)
        output_gpu = legacy.gpu_apply_shot_transform(frame_gpu, setup)
        cp.cuda.Stream.null.synchronize()
        output_cpu = cp.asnumpy(output_gpu)
        del frame_gpu, output_gpu, output_cpu
    samples: list[float] = []
    for _ in range(TIMED_RUNS):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        frame_gpu = cp.asarray(frame, dtype=cp.float64)
        output_gpu = legacy.gpu_apply_shot_transform(frame_gpu, setup)
        cp.cuda.Stream.null.synchronize()
        output_cpu = cp.asnumpy(output_gpu)
        cp.cuda.Stream.null.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
        del frame_gpu, output_gpu, output_cpu
    return stats(samples)


def legacy_mode() -> dict[str, Any]:
    import bench_p2192_gpu_transform as legacy

    setup, frame = p2192_setup()
    frame_gpu = cp.asarray(frame, dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    before = nvidia_smi_snapshot()
    gpu_timing, output_gpu = legacy.benchmark_gpu_transform(
        frame_gpu,
        setup,
        warmups=WARMUPS,
        runs=TIMED_RUNS,
    )
    cp.cuda.Stream.null.synchronize()
    after = nvidia_smi_snapshot()
    cpu_timing = legacy.benchmark_cpu(
        frame,
        setup,
        warmups=WARMUPS,
        runs=TIMED_RUNS,
    )
    upload_timing = legacy_upload(frame)
    download_timing = legacy_download(output_gpu)
    total_timing = legacy_total(frame, setup)
    scaling: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for height, width in SCALING_SIZES:
        scaling_frame = rng.random((height, width, 3), dtype=np.float64)
        cpu = legacy.benchmark_cpu(
            scaling_frame,
            setup,
            warmups=WARMUPS,
            runs=TIMED_RUNS,
        )
        scaling_gpu_frame = cp.asarray(scaling_frame, dtype=cp.float64)
        cp.cuda.Stream.null.synchronize()
        gpu, scaling_output = legacy.benchmark_gpu_transform(
            scaling_gpu_frame,
            setup,
            warmups=WARMUPS,
            runs=TIMED_RUNS,
        )
        del scaling_output, scaling_gpu_frame, scaling_frame
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        scaling.append(
            {
                "height": height,
                "width": width,
                "pixels": height * width,
                "cpu_transform": cpu,
                "gpu_transform_only": gpu,
            }
        )
    result = {
        "implementation": "P2.19.2 existing temporary-array benchmark",
        "warmups": WARMUPS,
        "timed_runs": TIMED_RUNS,
        "roi": [ROI_H, ROI_W, 3],
        "timing_ms": {
            "cpu_transform": cpu_timing,
            "gpu_transform_cuda_events": gpu_timing,
            "upload_wall_clock": upload_timing,
            "download_wall_clock": download_timing,
            "total_wall_clock": total_timing,
        },
        "memory_after_primary": device_memory(),
        "scaling": scaling,
        "telemetry": {"before": before, "after": after},
        "static_sync_audit": static_sync_audit(),
    }
    del output_gpu, frame_gpu, frame, setup
    cp.cuda.Stream.null.synchronize()
    return result


def make_material_like_roi(
    rng: np.random.Generator,
    index: int,
    height: int = ROI_H,
    width: int = ROI_W,
) -> np.ndarray:
    """Match the P2.20 benchmark's one-ROI-at-a-time input generator."""
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


def stage_backend_frame(
    backend: GPUTransformBackend,
    workspace: Any,
    frame: np.ndarray,
    transform: ShotTransform,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run the exact P2.20 staged private path and return stage timings."""
    workspace.metadata["transform"] = transform
    cp.cuda.Stream.null.synchronize()
    wall_start = time.perf_counter()
    upload_event_ms, buffers = event_elapsed(
        lambda: backend._upload_to_device(frame, workspace)
    )
    transform_event_ms, _ = event_elapsed(
        lambda: backend._transform_device(workspace, buffers)
    )
    download_event_ms, output = event_elapsed(
        lambda: backend._download_to_host(workspace, buffers)
    )
    cp.cuda.Stream.null.synchronize()
    return (
        {
            "upload_event_ms": upload_event_ms,
            "transform_event_ms": transform_event_ms,
            "download_event_ms": download_event_ms,
            "total_wall_ms": (time.perf_counter() - wall_start) * 1000.0,
            "input_bytes": int(frame.nbytes),
            "output_bytes": int(output.nbytes),
        },
        output,
    )


def backend_public_loop(
    backend: GPUTransformBackend,
    workspace: Any,
    frame: np.ndarray,
    transform: ShotTransform,
) -> dict[str, Any]:
    for _ in range(WARMUPS):
        output = backend.transform_roi(frame, transform, workspace=workspace)
        del output
    cp.cuda.Stream.null.synchronize()
    samples: list[float] = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        output = backend.transform_roi(frame, transform, workspace=workspace)
        samples.append((time.perf_counter() - start) * 1000.0)
        del output
    cp.cuda.Stream.null.synchronize()
    return stats(samples)


class TraceCupy:
    """Proxy selected CuPy functions with non-blocking CUDA event pairs."""

    def __init__(self, actual: Any, trace: list[dict[str, Any]]) -> None:
        self._actual = actual
        self._trace = trace

    def __getattr__(self, name: str) -> Any:
        function = getattr(self._actual, name)
        if name not in TRACED_FUNCTIONS:
            return function

        def traced(*args: Any, **kwargs: Any) -> Any:
            start = self._actual.cuda.Event()
            end = self._actual.cuda.Event()
            start.record()
            result = function(*args, **kwargs)
            end.record()
            self._trace.append({"name": name, "start": start, "end": end})
            return result

        return traced


def group_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    cp.cuda.Stream.null.synchronize()
    calls = []
    for entry in trace:
        calls.append(
            {
                "name": entry["name"],
                "elapsed_ms": float(
                    cp.cuda.get_elapsed_time(entry["start"], entry["end"])
                ),
            }
        )

    groups: dict[str, list[float]] = {}
    cursor = 0
    for name, count in STAGE_GROUPS:
        values = [entry["elapsed_ms"] for entry in calls[cursor : cursor + count]]
        groups[name] = values
        cursor += count
    groups["unassigned_calls"] = [
        entry["elapsed_ms"] for entry in calls[cursor:]
    ]
    return {
        "call_count": len(calls),
        "expected_traced_call_count": sum(count for _, count in STAGE_GROUPS),
        "calls": calls,
        "stage_event_samples_ms": groups,
        "stage_event_sum_ms": {
            name: float(sum(values)) for name, values in groups.items()
        },
        "untraced_device_operations": [
            "buffers.mapped_luminance.fill(0.0)",
            "buffers.lower[...] = buffers.position * (lut_size - 1)",
            "buffers.pq_lower[...] = buffers.pq_position",
        ],
    }


def trace_stages(
    backend: GPUTransformBackend,
    workspace: Any,
    frame: np.ndarray,
    transform: ShotTransform,
    runs: int = 3,
) -> dict[str, Any]:
    """Profile operation groups without synchronizing after each operation."""
    workspace.metadata["transform"] = transform
    samples: list[dict[str, Any]] = []
    for _ in range(runs):
        buffers = backend._upload_to_device(frame, workspace)
        cp.cuda.Stream.null.synchronize()
        trace: list[dict[str, Any]] = []
        actual = workspace.cupy
        outer_start = cp.cuda.Event()
        outer_end = cp.cuda.Event()
        proxy = TraceCupy(actual, trace)
        workspace.cupy = proxy
        outer_start.record()
        backend._transform_device(workspace, buffers)
        outer_end.record()
        workspace.cupy = actual
        outer_end.synchronize()
        grouped = group_trace(trace)
        grouped["outer_transform_event_ms"] = float(
            cp.cuda.get_elapsed_time(outer_start, outer_end)
        )
        samples.append(grouped)
    return {
        "runs": runs,
        "instrumentation": (
            "selected module-level CuPy functions wrapped with non-blocking CUDA "
            "events; array-method assignments/fill are reported separately."
        ),
        "samples": samples,
        "stage_median_ms": {
            name: float(
                np.median([sample["stage_event_sum_ms"][name] for sample in samples])
            )
            for name, _ in STAGE_GROUPS
        },
        "outer_transform_median_ms": float(
            np.median([sample["outer_transform_event_ms"] for sample in samples])
        ),
    }


def p220_scaling(
    backend: GPUTransformBackend,
    workspace: Any,
    transform: ShotTransform,
    cpu_backend: CPUTransformBackend,
    cpu_workspace: Any,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for height, width in SCALING_SIZES:
        frame = rng.random((height, width, 3), dtype=np.float64)
        for _ in range(WARMUPS):
            timing, output = stage_backend_frame(backend, workspace, frame, transform)
            del output, timing
        cpu_samples: list[float] = []
        gpu_transform_samples: list[float] = []
        gpu_total_samples: list[float] = []
        for _ in range(TIMED_RUNS):
            cpu_start = time.perf_counter()
            output = cpu_backend.transform_roi(
                frame,
                transform,
                workspace=cpu_workspace,
            )
            cpu_samples.append((time.perf_counter() - cpu_start) * 1000.0)
            del output
            timing, output = stage_backend_frame(backend, workspace, frame, transform)
            gpu_transform_samples.append(timing["transform_event_ms"])
            gpu_total_samples.append(timing["total_wall_ms"])
            del output, timing
        cp.cuda.Stream.null.synchronize()
        memory = device_memory()
        rows.append(
            {
                "height": height,
                "width": width,
                "pixels": height * width,
                "cpu_transform": stats(cpu_samples),
                "gpu_transform_cuda_events": stats(gpu_transform_samples),
                "gpu_total_wall": stats(gpu_total_samples),
                "workspace_allocations": workspace.metadata[
                    "workspace_allocations"
                ],
                "memory": memory,
            }
        )
        del frame
        cp.cuda.Stream.null.synchronize()
    return rows


def p220_mode() -> dict[str, Any]:
    transform = ShotTransform(shot_id=2192, luminance_curve=load_curve())
    cpu_backend = CPUTransformBackend()
    cpu_workspace = cpu_backend.prepare_shot(transform)
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before_prepare_memory = device_memory()
    prepare_start = time.perf_counter()
    backend = GPUTransformBackend()
    workspace = backend.prepare_shot(transform)
    cp.cuda.Stream.null.synchronize()
    prepare_wall_ms = (time.perf_counter() - prepare_start) * 1000.0
    after_prepare_memory = device_memory()

    frame = np.random.default_rng(SEED).random((ROI_H, ROI_W, 3), dtype=np.float64)
    allocation_start = time.perf_counter()
    buffers = backend._upload_to_device(frame, workspace)
    cp.cuda.Stream.null.synchronize()
    allocation_wall_ms = (time.perf_counter() - allocation_start) * 1000.0
    allocation_memory = device_memory()
    del buffers

    # Warm up the same staged path used by bench_p220_gpu_backend.py.
    for _ in range(WARMUPS):
        timing, output = stage_backend_frame(backend, workspace, frame, transform)
        del timing, output

    staged_samples: list[dict[str, Any]] = []
    telemetry_samples: list[dict[str, Any]] = []
    for _ in range(TIMED_RUNS):
        timing, output = stage_backend_frame(backend, workspace, frame, transform)
        staged_samples.append(timing)
        del output
        telemetry_samples.append(nvidia_smi_snapshot())

    public_timing = backend_public_loop(backend, workspace, frame, transform)
    trace = trace_stages(backend, workspace, frame, transform, runs=3)
    primary_workspace_allocations = workspace.metadata["workspace_allocations"]
    primary_memory = device_memory()
    primary_workspace_report = backend.memory_report(workspace)
    scaling = p220_scaling(
        backend,
        workspace,
        transform,
        cpu_backend,
        cpu_workspace,
    )
    cp.cuda.Stream.null.synchronize()
    post_scaling_memory = device_memory()
    post_scaling_workspace_report = backend.memory_report(workspace)
    result = {
        "implementation": "P2.20 current persistent-buffer backend",
        "warmups": WARMUPS,
        "timed_runs": TIMED_RUNS,
        "roi": [ROI_H, ROI_W, 3],
        "dtype": "float64",
        "primary_input": {
            "generator": "np.random.default_rng(2192).random",
            "c_contiguous": bool(frame.flags.c_contiguous),
            "dtype": str(frame.dtype),
            "bytes": int(frame.nbytes),
        },
        "timing_ms": {
            "staged_upload_event": stats(
                [sample["upload_event_ms"] for sample in staged_samples]
            ),
            "staged_transform_cuda_event": stats(
                [sample["transform_event_ms"] for sample in staged_samples]
            ),
            "staged_download_event": stats(
                [sample["download_event_ms"] for sample in staged_samples]
            ),
            "staged_total_wall": stats(
                [sample["total_wall_ms"] for sample in staged_samples]
            ),
            "public_backend_transform_roi_wall": public_timing,
        },
        "stage_profile": trace,
        "initialization": {
            "prepare_shot_wall_ms": prepare_wall_ms,
            "memory_before_prepare": before_prepare_memory,
            "memory_after_prepare": after_prepare_memory,
            "first_shape_upload_and_allocation_wall_ms": allocation_wall_ms,
            "memory_after_first_shape_allocation": allocation_memory,
            "workspace_allocations_after_primary": primary_workspace_allocations,
            "curve_lut_entries": int(workspace.curve_lut_gpu.size),
            "pq_lut_entries": int(workspace.pq_lut_gpu.size),
            "curve_lut_bytes": int(workspace.curve_lut_gpu.nbytes),
            "pq_lut_bytes": int(workspace.pq_lut_gpu.nbytes),
        },
        "copy_audit": {
            "host_to_gpu_input_bytes_per_timed_roi": int(frame.nbytes),
            "gpu_to_host_output_bytes_per_timed_roi": int(frame.nbytes),
            "host_normalization_copy_bytes": 0,
            "host_normalization_reason": (
                "C-contiguous float64 input; np.asarray(order='C') returns a "
                "same-layout view/reference."
            ),
            "reshape_copy_bytes": 0,
            "reshape_reason": "reshape(-1, 3) is a view for the contiguous input.",
            "gpu_lut_setup_bytes": {
                "curve_lut": int(workspace.curve_lut_gpu.nbytes),
                "pq_lut": int(workspace.pq_lut_gpu.nbytes),
                "matrix": int(workspace.matrix_gpu.nbytes),
                "curve_scalars": 24,
            },
            "gpu_to_gpu_operations": [
                "buffers.lower[...] assignment: N int64 indices",
                "buffers.pq_lower[...] assignment: N*3 int64 indices",
                "three curve copyto masked writes: N float64 values",
                "two PQ take writes: N*3 float64 values",
                "two PQ weighted multiply writes and one output add: N*3 float64 values",
            ],
            "additional_full_roi_host_copies_in_steady_state": 0,
        },
        "synchronization_audit": {
            "per_timed_staged_roi": [
                "one null-stream synchronize before staged timing",
                "one end-event synchronize after upload",
                "one end-event synchronize after transform",
                "one end-event synchronize after download",
                "one null-stream synchronize after staged timing",
            ],
            "public_backend_transform_roi": [
                (
                    "one null-stream synchronize inside "
                    "GPUTransformBackend.transform_roi after _transform_device "
                    "and before asnumpy"
                ),
                "cp.asnumpy implicitly waits/materializes the output",
            ],
            "initialization_only": [
                (
                    "prepare_shot uses Device context and one "
                    "Stream.null.synchronize after LUT/matrix setup"
                ),
                "first shape allocation is asynchronous until the explicit audit synchronize",
            ],
            "stage_trace_policy": (
                "per-function events are recorded without synchronizing after each "
                "function; one final event wait is used."
            ),
            "static_inventory": static_sync_audit(),
        },
        "telemetry": {
            "samples_during_timed_loop": telemetry_samples,
            "before_scaling": nvidia_smi_snapshot(),
        },
        "memory": {
            "before_scaling": primary_memory,
            "after_scaling": post_scaling_memory,
            "workspace_persistent_bytes_before_scaling": int(
                primary_workspace_report["persistent_bytes_total"]
            ),
            "workspace_persistent_bytes_after_scaling": int(
                post_scaling_workspace_report["persistent_bytes_total"]
            ),
            "workspace_report_before_scaling": primary_workspace_report,
            "workspace_report_after_scaling": post_scaling_workspace_report,
        },
        "scaling": scaling,
    }
    del frame, workspace, backend, cpu_workspace, cpu_backend
    cp.cuda.Stream.null.synchronize()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("p2192", "p220"),
        required=True,
        help="Run one implementation in a clean Python process.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = legacy_mode() if args.mode == "p2192" else p220_mode()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
