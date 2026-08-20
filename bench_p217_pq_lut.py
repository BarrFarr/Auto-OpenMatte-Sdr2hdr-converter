"""P2.17: PQ OETF sqrt-LUT production regression and benchmark.

This script does not modify production code and does not render video.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np

import auto_openmatte.processing.transform as transform_module
from auto_openmatte.core.models import GeometryModel, ShotTransform
from auto_openmatte.core.transfer_functions import (
    _get_pq_oetf_sqrt_lut,
    pq_eotf,
    pq_oetf,
    pq_oetf_sqrt_lut,
)
from auto_openmatte.pipeline.compose import composite_extend
from auto_openmatte.processing.transform import apply_shot_transform

_PEAK_NITS = 10000.0
_LUM_WEIGHTS = np.array([0.2627, 0.6780, 0.0593])
ROI_H = 560
ROI_W = 3840
RUNS = 10

HDR_SOURCE = (
    Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U")
    / "Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"
)
OM_SOURCE = (
    Path(r"G:\Filmy\IMAX format (open matte)\Blade Runner 2049  Open Matte 2160p")
    / "Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
)
CURVE_FILE = Path("luminance_curve_br2049.json")


def reference_delinearize(linear, transfer, peak_nits=10000.0):
    """Old production behavior, using the preserved mathematical pq_oetf."""
    transfer_lower = transfer.lower().replace("-", "").replace("_", "").replace(" ", "")
    if transfer_lower in ("smpte2084", "pq", "st2084"):
        return pq_oetf(linear * peak_nits)
    return transform_module._ORIGINAL_DELINEARIZE(linear, transfer, peak_nits)


def run_old_transform(frame, transform, peak_nits=10000.0):
    """Run apply_shot_transform with the old mathematical PQ OETF."""
    original = transform_module.delinearize
    transform_module.delinearize = reference_delinearize
    try:
        return apply_shot_transform(frame, transform, peak_nits=peak_nits)
    finally:
        transform_module.delinearize = original


def run_new_transform(frame, transform, peak_nits=10000.0):
    """Run current production apply_shot_transform with the LUT path."""
    return apply_shot_transform(frame, transform, peak_nits=peak_nits)


def benchmark(function, runs=RUNS):
    function()
    function()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1000.0)
    timings.sort()
    return {
        "median": float(timings[len(timings) // 2]),
        "min": float(timings[0]),
        "max": float(timings[-1]),
        "all": timings,
    }


def errors(reference, candidate):
    pq_error = np.abs(reference - candidate)
    reference_nits = pq_eotf(reference)
    candidate_nits = pq_eotf(candidate)
    nits_error = np.abs(reference_nits - candidate_nits)
    return {
        "max_pq": float(np.max(pq_error)),
        "mean_pq": float(np.mean(pq_error)),
        "rmse_pq": float(np.sqrt(np.mean(pq_error**2))),
        "max_nits": float(np.max(nits_error)),
        "mean_nits": float(np.mean(nits_error)),
        "rmse_nits": float(np.sqrt(np.mean(nits_error**2))),
        "p99_nits": float(np.percentile(nits_error, 99)),
        "p999_nits": float(np.percentile(nits_error, 99.9)),
        "p9999_nits": float(np.percentile(nits_error, 99.99)),
    }


def luminance_nits(signal):
    channel_nits = pq_eotf(signal)
    return np.tensordot(channel_nits, _LUM_WEIGHTS, axes=([-1], [0]))


def extract_crop(path, timestamp, width, height, x, y):
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(timestamp), "-i", str(path), "-frames:v", "1",
        "-vf", f"crop={width}:{height}:{x}:{y}",
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))
    expected = width * height * 3 * 2
    if len(result.stdout) < expected:
        raise RuntimeError(f"Short ffmpeg output: {len(result.stdout)} < {expected}")
    values = np.frombuffer(result.stdout[:expected], dtype=np.uint16)
    return values.reshape(height, width, 3).astype(np.float64) / 65535.0


def old_new_composite(hdr_frame, om_frame, transform, geometry, mask):
    original = transform_module.delinearize
    transform_module.delinearize = reference_delinearize
    try:
        old = composite_extend(hdr_frame, om_frame, transform, geometry, mask)
    finally:
        transform_module.delinearize = original
    new = composite_extend(hdr_frame, om_frame, transform, geometry, mask)
    return old, new


def windows_memory():
    """Return current/peak working set when available on Windows."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if ok:
            return {
                "working_set_mb": counters.WorkingSetSize / 1024**2,
                "peak_working_set_mb": counters.PeakWorkingSetSize / 1024**2,
            }
    except (AttributeError, OSError, TypeError):
        pass

    # The psapi call is restricted in some IDE sandboxes. Query the same
    # process through PowerShell as a Windows-native fallback.
    try:
        command = (
            f"$p=Get-Process -Id {os.getpid()}; "
            "Write-Output ($p.WorkingSet64); "
            "Write-Output ($p.PeakWorkingSet64)"
        )
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            timeout=10,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        if len(values) >= 2:
            return {
                "working_set_mb": values[0] / 1024**2,
                "peak_working_set_mb": values[1] / 1024**2,
            }
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {"working_set_mb": None, "peak_working_set_mb": None}


def allocation_measurement(function):
    """Measure Python-visible peak allocation for one transform invocation."""
    gc.collect()
    tracemalloc.start()
    function()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"current_bytes": int(current), "peak_bytes": int(peak)}


def stats(values):
    return {
        "p1": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "gt100": int(np.sum(values > 100.0)),
        "gt200": int(np.sum(values > 200.0)),
        "gt400": int(np.sum(values > 400.0)),
    }


def main():
    rng = np.random.default_rng(217)
    curve = json.loads(CURVE_FILE.read_text(encoding="utf-8"))["curve"]
    transform = ShotTransform(shot_id=217, luminance_curve=curve)

    # Build once before timing and verify singleton lifecycle.
    _get_pq_oetf_sqrt_lut.cache_clear()
    lut = _get_pq_oetf_sqrt_lut()
    cache_after_build = _get_pq_oetf_sqrt_lut.cache_info()

    roi_frame = rng.random((ROI_H, ROI_W, 3), dtype=np.float64)
    roi_luminance = rng.random((ROI_H, ROI_W, 3), dtype=np.float64) * _PEAK_NITS

    # Direct PQ accuracy datasets.
    dense = np.linspace(0.0, _PEAK_NITS, 1_000_001)
    random_values = rng.random(1_000_000) * _PEAK_NITS
    low_ranges = {
        "0-0.001": np.linspace(0.0, 0.001, 250_001) * _PEAK_NITS,
        "0.001-0.01": np.linspace(0.001, 0.01, 250_001) * _PEAK_NITS,
        "0.01-0.1": np.linspace(0.01, 0.1, 250_001) * _PEAK_NITS,
        "0.1-1.0": np.linspace(0.1, 1.0, 250_001) * _PEAK_NITS,
    }
    boundaries = np.array(
        [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4,
         1e-3, 0.01, 0.1, 0.5, 1.0]
    ) * _PEAK_NITS

    dense_errors = errors(pq_oetf(dense), pq_oetf_sqrt_lut(dense))
    random_errors = errors(pq_oetf(random_values), pq_oetf_sqrt_lut(random_values))
    boundary_errors = errors(pq_oetf(boundaries), pq_oetf_sqrt_lut(boundaries))
    range_errors = {
        name: errors(pq_oetf(values), pq_oetf_sqrt_lut(values))
        for name, values in low_ranges.items()
    }

    # Direct PQ 3-channel benchmark.
    old_pq_timing = benchmark(lambda: pq_oetf(roi_luminance))
    new_pq_timing = benchmark(lambda: pq_oetf_sqrt_lut(roi_luminance))

    # Full production transform benchmark, actual 3-channel ROI.
    old_output = run_old_transform(roi_frame, transform)
    new_output = run_new_transform(roi_frame, transform)
    transform_rgb_error = np.abs(old_output - new_output)
    transform_luminance_error = np.abs(
        luminance_nits(old_output) - luminance_nits(new_output)
    )
    transform_errors = {
        "max_rgb": float(np.max(transform_rgb_error)),
        "mean_rgb": float(np.mean(transform_rgb_error)),
        "rmse_rgb": float(np.sqrt(np.mean(transform_rgb_error**2))),
        "max_luminance_nits": float(np.max(transform_luminance_error)),
        "mean_luminance_nits": float(np.mean(transform_luminance_error)),
        "p99_luminance_nits": float(np.percentile(transform_luminance_error, 99)),
        "p999_luminance_nits": float(np.percentile(transform_luminance_error, 99.9)),
    }
    old_transform_timing = benchmark(lambda: run_old_transform(roi_frame, transform))
    new_transform_timing = benchmark(lambda: run_new_transform(roi_frame, transform))

    # HDR overlap invariance: composite's HDR region is copied directly.
    hdr_frame = rng.random((16, 48, 3), dtype=np.float64)
    om_small = rng.random((32, 48, 3), dtype=np.float64)
    geometry = GeometryModel(overlap_bbox=[0, 8, 48, 24], offset_y=8)
    mask = np.ones((32, 48), dtype=np.float64)
    mask[8:24, :] = 0.0
    old_composite, new_composite = old_new_composite(
        hdr_frame, om_small, transform, geometry, mask
    )
    hdr_diff = np.abs(old_composite[8:24] - new_composite[8:24])
    extension_diff = np.abs(
        np.concatenate((old_composite[:8], old_composite[24:]))
        - np.concatenate((new_composite[:8], new_composite[24:]))
    )

    # Jacket ROI from the same BR2049 material as P2.15.
    jacket = None
    jacket_errors = None
    if OM_SOURCE.exists():
        jacket = extract_crop(OM_SOURCE, 2714.0, 1000, 300, 1400, 1800)
        jacket_old = run_old_transform(jacket, transform)
        jacket_new = run_new_transform(jacket, transform)
        jacket_old_nits = luminance_nits(jacket_old)
        jacket_new_nits = luminance_nits(jacket_new)
        jacket_errors = {
            "old": stats(jacket_old_nits),
            "new": stats(jacket_new_nits),
            "diff": errors(jacket_old, jacket_new),
            "max_rgb": float(np.max(np.abs(jacket_old - jacket_new))),
            "mean_rgb": float(np.mean(np.abs(jacket_old - jacket_new))),
        }

    # Peak memory around one old and one new production transform.
    # tracemalloc tracks Python-visible allocations; Windows counters include
    # native NumPy working-set memory.
    memory_before = windows_memory()
    old_allocations = allocation_measurement(
        lambda: run_old_transform(roi_frame, transform)
    )
    new_allocations = allocation_measurement(
        lambda: run_new_transform(roi_frame, transform)
    )
    memory_after = windows_memory()

    results = {
        "lut_entries": int(len(lut)),
        "lut_bytes": int(lut.nbytes),
        "cache": {
            "hits": cache_after_build.hits,
            "misses": cache_after_build.misses,
            "maxsize": cache_after_build.maxsize,
            "currsize": cache_after_build.currsize,
        },
        "dense_errors": dense_errors,
        "random_errors": random_errors,
        "boundary_errors": boundary_errors,
        "range_errors": range_errors,
        "pq_old_timing": old_pq_timing,
        "pq_new_timing": new_pq_timing,
        "transform_errors": transform_errors,
        "transform_old_timing": old_transform_timing,
        "transform_new_timing": new_transform_timing,
        "hdr_overlap_max_diff": float(np.max(hdr_diff)),
        "hdr_overlap_mean_diff": float(np.mean(hdr_diff)),
        "extension_max_diff": float(np.max(extension_diff)),
        "extension_mean_diff": float(np.mean(extension_diff)),
        "jacket": jacket_errors,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "old_transform_allocations": old_allocations,
        "new_transform_allocations": new_allocations,
        "source_exists": {
            "hdr": HDR_SOURCE.exists(),
            "om": OM_SOURCE.exists(),
            "curve": CURVE_FILE.exists(),
        },
    }

    Path("P2.17_benchmark_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
