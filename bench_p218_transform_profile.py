"""P2.18: post-optimization transform profiler.

This is an observational harness only. It intentionally reproduces the current
production operations and does not modify production code.
"""

from __future__ import annotations

import gc
import io
import json
import os
import time
import tracemalloc
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from auto_openmatte.core.models import ShotTransform
from auto_openmatte.core.transfer_functions import (
    _get_pq_oetf_sqrt_lut,
    delinearize,
    linearize,
)
from auto_openmatte.processing.luminance import (
    apply_luminance_curve,
    build_curve_lut,
)
from auto_openmatte.processing.transform import (
    _LUM_B_2020,
    _LUM_G_2020,
    _LUM_R_2020,
    _M_709_TO_2020,
    apply_shot_transform,
)

_PEAK_NITS = 10000.0
_ROI_H = 560
_ROI_W = 3840
_MEASURED_RUNS = 10
_WARMUPS = 2


def load_configuration():
    curve = json.loads(
        Path("luminance_curve_br2049.json").read_text(encoding="utf-8")
    )["curve"]
    transform = ShotTransform(shot_id=218, luminance_curve=curve)
    lut = build_curve_lut(curve)
    _get_pq_oetf_sqrt_lut()
    return curve, transform, lut


def mark_stage(stage_data, name, operation):
    start = time.perf_counter()
    value = operation()
    elapsed = (time.perf_counter() - start) * 1000.0
    stage_data[name] = elapsed
    return value


def profile_pipeline(frame, transform, prebuilt_lut, capture_memory=False):
    """Reproduce apply_shot_transform with per-stage timers.

    Operation order matches the current production implementation. The
    identity color-matrix and saturation branches are intentionally retained.
    """
    timings = {}
    memory = {}
    if capture_memory:
        tracemalloc.start()

    def mark(name, operation):
        if capture_memory:
            _, before_peak = tracemalloc.get_traced_memory()
        value = mark_stage(timings, name, operation)
        if capture_memory:
            current, after_peak = tracemalloc.get_traced_memory()
            memory[name] = {
                "current_bytes": int(current),
                "peak_delta_bytes": int(after_peak - before_peak),
                "peak_bytes": int(after_peak),
            }
        return value

    total_start = time.perf_counter()
    prepared = mark("A_input_preparation", lambda: np.asarray(frame))
    linear_709 = mark("B_bt1886_linearization", lambda: linearize(prepared, "bt709"))

    def matrix_stage():
        shape = linear_709.shape
        converted = linear_709.reshape(-1, 3) @ _M_709_TO_2020.T
        converted = converted.reshape(shape)
        return np.maximum(converted, 0.0)

    linear_2020 = mark("C_709_to_2020_matrix", matrix_stage)

    def luminance_stage():
        return (
            _LUM_R_2020 * linear_2020[..., 0]
            + _LUM_G_2020 * linear_2020[..., 1]
            + _LUM_B_2020 * linear_2020[..., 2]
        )

    lum = mark("D_bt2020_luminance", luminance_stage)
    lum_mapped = mark(
        "E_luminance_curve_lut",
        lambda: apply_luminance_curve(
            lum, transform.luminance_curve, prebuilt_lut=prebuilt_lut
        ),
    )

    def safe_ratio_stage():
        safe_mask = lum > 1e-6
        ratio = np.ones_like(lum)
        np.divide(lum_mapped, lum, out=ratio, where=safe_mask)
        return safe_mask, ratio

    safe_mask, ratio = mark("F_safe_ratio", safe_ratio_stage)
    linear_scaled_holder = [
        mark("G_rgb_times_ratio", lambda: linear_2020 * ratio[..., np.newaxis])
    ]
    linear = mark(
        "H_rgb_maximum_clamp",
        lambda: np.maximum(linear_scaled_holder[0], 0.0),
    )
    # Production assigns the clamped result back to `linear`; the temporary
    # product from stage G is released after that assignment.
    linear_scaled_holder.clear()

    def other_stage():
        # This is the current identity-matrix and saturation branch logic.
        if transform.color_matrix:
            mat = np.array(transform.color_matrix)
            if not np.allclose(mat, np.eye(3), atol=0.001):
                raise AssertionError("Profiler expected identity color matrix")
        if abs(transform.saturation - 1.0) > 0.01:
            raise AssertionError("Profiler expected identity saturation")
        return linear

    linear = mark("K_other_branch_checks", other_stage)
    hdr_signal = mark(
        "I_pq_oetf_sqrt_lut",
        lambda: delinearize(linear, "smpte2084", peak_nits=_PEAK_NITS),
    )
    output = mark("J_final_output_clip", lambda: np.clip(hdr_signal, 0.0, 1.0))
    total_ms = (time.perf_counter() - total_start) * 1000.0

    if capture_memory:
        _, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        final_peak = None

    # Keep references alive while returning inventory; this reflects the
    # arrays retained at the end of each production stage in this harness.
    arrays = {
        "input": prepared,
        "linear_709": linear_709,
        "linear_2020": linear_2020,
        "luminance": lum,
        "mapped_luminance": lum_mapped,
        "safe_mask": safe_mask,
        "ratio": ratio,
        "linear_clamped": linear,
        "hdr_signal": hdr_signal,
        "output": output,
    }
    inventory = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": int(value.nbytes),
        }
        for name, value in arrays.items()
    }
    timings["TOTAL"] = total_ms
    timings["K_residual_not_separately_isolated"] = max(
        0.0,
        total_ms - sum(
            value
            for name, value in timings.items()
            if name not in {"TOTAL", "K_residual_not_separately_isolated"}
        ),
    )
    return {
        "timings": timings,
        "memory": memory,
        "trace_peak_bytes": final_peak,
        "inventory": inventory,
        "output": output,
    }


def stage_runs(frame, transform, lut):
    for _ in range(_WARMUPS):
        profile_pipeline(frame, transform, lut)
    samples = []
    for _ in range(_MEASURED_RUNS):
        samples.append(profile_pipeline(frame, transform, lut)["timings"])
    names = list(samples[0])
    summary = {}
    for name in names:
        values = np.array([sample[name] for sample in samples])
        summary[name] = {
            "median_ms": float(np.median(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "samples_ms": values.tolist(),
        }
    total = summary["TOTAL"]["median_ms"]
    for name, value in summary.items():
        value["percent_total"] = (
            float(value["median_ms"] / total * 100.0) if name != "TOTAL" else 100.0
        )
    memory_run = profile_pipeline(frame, transform, lut, capture_memory=True)
    return summary, memory_run


def benchmark_transform(frame, transform, lut):
    for _ in range(_WARMUPS):
        apply_shot_transform(frame, transform, prebuilt_lut=lut)
    values = []
    for _ in range(_MEASURED_RUNS):
        start = time.perf_counter()
        apply_shot_transform(frame, transform, prebuilt_lut=lut)
        values.append((time.perf_counter() - start) * 1000.0)
    values = np.array(values)
    return {
        "median_ms": float(np.median(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "samples_ms": values.tolist(),
    }


def benchmark_cpu(frame, transform, lut):
    wall = []
    cpu = []
    for _ in range(5):
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        apply_shot_transform(frame, transform, prebuilt_lut=lut)
        wall.append((time.perf_counter() - start_wall) * 1000.0)
        cpu.append((time.process_time() - start_cpu) * 1000.0)
    return {
        "wall_median_ms": float(np.median(wall)),
        "cpu_median_ms": float(np.median(cpu)),
        "cpu_wall_ratio": float(np.median(cpu) / np.median(wall)),
        "wall_samples_ms": wall,
        "cpu_samples_ms": cpu,
    }


def scaling_check(transform, lut, rng):
    sizes = [
        (540, 960),
        (1080, 1920),
        (2160, 3840),
        (280, 3840),
        (560, 3840),
    ]
    results = []
    for height, width in sizes:
        frame = rng.random((height, width, 3), dtype=np.float64)
        timing = benchmark_transform(frame, transform, lut)
        pixels = height * width
        results.append(
            {
                "height": height,
                "width": width,
                "pixels": pixels,
                "timing": timing,
                "ms_per_million_pixels": timing["median_ms"] / pixels * 1_000_000,
            }
        )
        del frame
        gc.collect()
    return results


def backend_info():
    output = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(output):
            np.show_config()
    return {
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numpy_config": output.getvalue(),
    }


def main():
    rng = np.random.default_rng(217)
    curve, transform, lut = load_configuration()
    frame = rng.random((_ROI_H, _ROI_W, 3), dtype=np.float64)

    stage_summary, memory_run = stage_runs(frame, transform, lut)
    transform_timing = benchmark_transform(frame, transform, lut)
    cpu = benchmark_cpu(frame, transform, lut)
    scaling = scaling_check(transform, lut, rng)

    results = {
        "roi": {"height": _ROI_H, "width": _ROI_W, "pixels": _ROI_H * _ROI_W},
        "measured_runs": _MEASURED_RUNS,
        "warmups": _WARMUPS,
        "stage_summary": stage_summary,
        "memory_run": {
            "trace_peak_bytes": memory_run["trace_peak_bytes"],
            "memory": memory_run["memory"],
            "inventory": memory_run["inventory"],
        },
        "actual_transform_timing": transform_timing,
        "cpu": cpu,
        "scaling": scaling,
        "backend": backend_info(),
        "allocation_audit": {
            "production_files": [
                "src/auto_openmatte/processing/transform.py",
                "src/auto_openmatte/core/transfer_functions.py",
                "src/auto_openmatte/processing/luminance.py",
            ],
            "note": "Static audit is documented in the P2.18 report; no source was changed.",
        },
    }
    Path("P2.18_profile_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
