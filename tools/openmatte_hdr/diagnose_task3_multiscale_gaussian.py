#!/usr/bin/env python3
"""Short numerical diagnostic for the isolated TASK 3 multi-scale Gaussian experiment.

This file is intentionally unimported by production code. It compares the existing
CPU sigma-16 Gaussian on the production-domain ICtCp fields with:

    INTER_AREA downsample by 16 -> sigma/16 Gaussian -> INTER_LINEAR upsample.

The stored P232 uint16 samples are decoded with the same BT.1886/PQ and geometry
contracts as the production decoder. No fitting, rendering, encoding, or Task 2
pipeline code is invoked.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH_SRC = ROOT / "research" / "src"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

import fastcore  # noqa: E402
from reshaping_research.utils.color_spaces import bt709_to_bt2020  # noqa: E402
from reshaping_research.utils.transfer_functions import (  # noqa: E402
    bt1886_eotf,
    pq_eotf,
)

SCALE = 16
SIGMA = 16.0
SIGMA_SMALL = SIGMA / SCALE
RGB16_MAX = 65535.0
PEAK_NITS = 10000.0
SEAM_BAND = 48
OVERLAP = (0, 140, 1920, 940)
OM_SIZE = (1920, 1080)
HDR_SIZE = (3840, 1600)
DEFAULT_FRAMES = (1, 8, 20)
DEFAULT_OUTPUT = ROOT / "task3_out" / "task3_multiscale_gaussian.json"
RANGE_TOLERANCE = 1e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare full-resolution and isolated TASK 3 multi-scale Gaussian paths."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset",
        help="P232_real_material_dataset directory containing raw_rgb_u16.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON report path; defaults to task3_out/task3_multiscale_gaussian.json.",
    )
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=list(DEFAULT_FRAMES),
        help="The Matrix temporal-stratum indices to inspect (default: 1 8 20).",
    )
    parser.add_argument(
        "--timing-repetitions",
        type=int,
        default=8,
        help="Timed repetitions per frame after warmup (default: 8).",
    )
    parser.add_argument(
        "--timing-warmups",
        type=int,
        default=2,
        help="Untimed warmup repetitions per frame (default: 2).",
    )
    return parser


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _frame_paths(dataset_root: Path, index: int) -> tuple[Path, Path]:
    _require(1 <= index <= 20, f"P232 Matrix stratum must be in 1..20, got {index}")
    stem = f"the_matrix_temporal_stratum_{index:02d}"
    directory = dataset_root / "raw_rgb_u16" / "the_matrix"
    return directory / f"{stem}_om_rgb_u16.npy", directory / f"{stem}_hdr_rgb_u16.npy"


def _load_raw(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    _require(path.is_file(), f"Missing P232 input: {path}")
    raw = np.load(path)
    _require(raw.shape == expected_shape, f"Unexpected shape for {path}: {raw.shape}")
    _require(raw.ndim == 3 and raw.shape[-1] == 3, f"Expected RGB array: {path}")
    return np.asarray(raw)


def _raw_to_code(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32, copy=False) / RGB16_MAX


def _decode_hdr_overlap(raw: np.ndarray) -> np.ndarray:
    """Match openmatte_hdr.decode_hdr for the fixed P232 geometry."""
    x1, y1, x2, y2 = OVERLAP
    code = cv2.resize(
        raw.astype(np.float64) / RGB16_MAX,
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )
    return (pq_eotf(code) / PEAK_NITS).astype(np.float32)


def _decode_sdr(raw: np.ndarray) -> np.ndarray:
    """Match openmatte_hdr.decode_sdr, including BT.709 -> BT.2020."""
    return np.maximum(
        bt709_to_bt2020(bt1886_eotf(raw.astype(np.float64) / RGB16_MAX)),
        0.0,
    ).astype(np.float32)


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Canonical CPU blur: independent channels and BORDER_REFLECT_101."""
    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _small_size(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return max(1, int(round(width / SCALE))), max(1, int(round(height / SCALE)))


def _multiscale_parts(
    image: np.ndarray, include_attribution: bool = True
) -> dict[str, Any]:
    """Return stages of the path, optionally omitting attribution-only variants."""
    height, width = image.shape[:2]
    small_width, small_height = _small_size(image)
    area = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    small_blur = _gaussian(area, SIGMA_SMALL)
    upsampled = cv2.resize(small_blur, (width, height), interpolation=cv2.INTER_LINEAR)
    result: dict[str, Any] = {
        "area": area,
        "small_blur": small_blur,
        "upsampled": upsampled,
        "small_size": [small_width, small_height],
    }
    if include_attribution:
        result["area_linear"] = cv2.resize(
            area, (width, height), interpolation=cv2.INTER_LINEAR
        )
        result["nearest"] = cv2.resize(
            small_blur, (width, height), interpolation=cv2.INTER_NEAREST
        )
    return result


def _range_stats(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    result: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(np.all(finite)),
        "nonfinite_count": int(np.size(values) - np.count_nonzero(finite)),
    }
    if np.any(finite):
        result["min"] = float(np.min(values[finite]))
        result["max"] = float(np.max(values[finite]))
    else:
        result["min"] = None
        result["max"] = None
    if values.ndim >= 3 and values.shape[-1] > 1:
        result["channels"] = [
            {
                "min": float(np.min(values[..., channel])),
                "max": float(np.max(values[..., channel])),
            }
            for channel in range(values.shape[-1])
        ]
    return result


def _source_range_ok(array: np.ndarray, lower: float | None = None) -> bool:
    values = np.asarray(array)
    if not bool(np.all(np.isfinite(values))):
        return False
    return lower is None or bool(np.min(values) >= lower - RANGE_TOLERANCE)


def _within_range(candidate: np.ndarray, source: np.ndarray) -> bool:
    candidate_values = np.asarray(candidate)
    source_values = np.asarray(source)
    if not np.all(np.isfinite(candidate_values)):
        return False
    return bool(
        np.min(candidate_values) >= np.min(source_values) - RANGE_TOLERANCE
        and np.max(candidate_values) <= np.max(source_values) + RANGE_TOLERANCE
    )


def _error_summary(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_values = np.asarray(reference)
    candidate_values = np.asarray(candidate)
    delta = candidate_values.astype(np.float64) - reference_values.astype(np.float64)
    absolute = np.abs(delta)
    result: dict[str, Any] = {
        "max_abs": float(np.max(absolute)),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "mae": float(np.mean(absolute)),
        "element_count": int(delta.size),
    }
    if delta.ndim >= 3 and delta.shape[-1] > 1:
        result["channels"] = [
            {
                "max_abs": float(np.max(absolute[..., channel])),
                "rms": float(np.sqrt(np.mean(delta[..., channel] * delta[..., channel]))),
                "mae": float(np.mean(absolute[..., channel])),
            }
            for channel in range(delta.shape[-1])
        ]
    return result


def _masked_error(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    reference_values = np.asarray(reference)[mask]
    candidate_values = np.asarray(candidate)[mask]
    return _error_summary(reference_values, candidate_values)


def _edge_mask(shape: tuple[int, int], band: int) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    mask[:band] = True
    mask[-band:] = True
    mask[:, :band] = True
    mask[:, -band:] = True
    return mask


def _field_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    height, width = reference.shape[:2]
    edge = _edge_mask((height, width), min(SEAM_BAND, height // 2, width // 2))
    interior = ~edge
    return {
        "all": _error_summary(reference, candidate),
        "edge_band": _masked_error(reference, candidate, edge),
        "interior": _masked_error(reference, candidate, interior),
        "relative_rms_to_reference_range_percent": float(
            100.0
            * _error_summary(reference, candidate)["rms"]
            / max(float(np.ptp(reference)), RANGE_TOLERANCE)
        ),
    }


def _ictcp(backend: fastcore.Backend, rgb_linear: np.ndarray) -> np.ndarray:
    ictcp = backend.to_ictcp(backend.asarray(rgb_linear * PEAK_NITS))
    return np.asarray(backend.tohost(ictcp), dtype=np.float32)


def _blur_pair(ictcp: np.ndarray, multiscale: bool) -> tuple[np.ndarray, np.ndarray]:
    intensity = ictcp[..., 0]
    chroma = ictcp[..., 1:]
    if multiscale:
        intensity_result = _multiscale_parts(intensity, include_attribution=False)["upsampled"]
        chroma_result = _multiscale_parts(chroma, include_attribution=False)["upsampled"]
    else:
        intensity_result = _gaussian(intensity, SIGMA)
        chroma_result = _gaussian(chroma, SIGMA)
    return intensity_result, chroma_result


def _blur_field(ictcp: np.ndarray, multiscale: bool) -> np.ndarray:
    intensity, chroma = _blur_pair(ictcp, multiscale)
    return np.concatenate((intensity[..., None], chroma), axis=-1)


def _relative_change_percent(baseline: float, candidate: float) -> float:
    return float(100.0 * (candidate - baseline) / max(abs(baseline), RANGE_TOLERANCE))


def _band_residuals(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    _, y1, _, y2 = OVERLAP
    top = float(np.mean(np.abs(predicted[y1 : y1 + SEAM_BAND] - reference[:SEAM_BAND])))
    bottom = float(np.mean(np.abs(predicted[y2 - SEAM_BAND : y2] - reference[-SEAM_BAND:])))
    return {"top": top, "bottom": bottom, "mean": (top + bottom) / 2.0}


def _reconstructed_chroma(om_chroma: np.ndarray, hdr_chroma: np.ndarray) -> np.ndarray:
    """Build the field-space equivalent of the seam-metric reconstructed output."""
    _, y1, _, y2 = OVERLAP
    reconstructed = om_chroma.copy()
    reconstructed[y1:y2] = hdr_chroma
    return reconstructed


def _seam_steps(reconstructed_chroma: np.ndarray) -> dict[str, float]:
    _, y1, _, y2 = OVERLAP
    magnitudes = np.hypot(reconstructed_chroma[..., 0], reconstructed_chroma[..., 1])
    top = float(np.mean(np.abs(magnitudes[y1 - 1] - magnitudes[y1])))
    bottom = float(np.mean(np.abs(magnitudes[y2 - 1] - magnitudes[y2])))
    return {"top": top, "bottom": bottom, "mean": (top + bottom) / 2.0}


def _stage_attribution(
    source: np.ndarray,
    full: np.ndarray,
    parts: dict[str, Any],
) -> dict[str, Any]:
    candidate = parts["upsampled"]
    return {
        "candidate_vs_full": _field_error(full, candidate),
        "area_downsample_plus_linear_reconstruction_without_small_gaussian_vs_full": _field_error(
            full, parts["area_linear"]
        ),
        "small_grid_gaussian_delta_vs_area_linear": _error_summary(
            parts["area_linear"], candidate
        ),
        "bilinear_reconstruction_delta_vs_nearest": _error_summary(
            parts["nearest"], candidate
        ),
        "source_range": _range_stats(source),
        "small_area_range": _range_stats(parts["area"]),
        "small_blur_range": _range_stats(parts["small_blur"]),
    }


def _timing_summary(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values) * 1000.0),
        "median_ms": float(np.median(values) * 1000.0),
        "p95_ms": float(np.percentile(values, 95.0) * 1000.0),
        "max_ms": float(np.max(values) * 1000.0),
    }


def _time_frame(
    om_ictcp: np.ndarray,
    hdr_ictcp: np.ndarray,
    repetitions: int,
    warmups: int,
) -> dict[str, Any]:
    def full_operations() -> tuple[float, float]:
        started = time.perf_counter()
        _blur_pair(om_ictcp, multiscale=False)
        om_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        _gaussian(hdr_ictcp[..., 1:], SIGMA)
        hdr_elapsed = time.perf_counter() - started
        return om_elapsed, hdr_elapsed

    def multiscale_operations() -> tuple[float, float]:
        started = time.perf_counter()
        _blur_pair(om_ictcp, multiscale=True)
        om_elapsed = time.perf_counter() - started
        started = time.perf_counter()
        _multiscale_parts(hdr_ictcp[..., 1:], include_attribution=False)
        hdr_elapsed = time.perf_counter() - started
        return om_elapsed, hdr_elapsed

    for _ in range(warmups):
        full_operations()
        multiscale_operations()
    full_samples: list[float] = []
    full_om_samples: list[float] = []
    full_hdr_samples: list[float] = []
    multiscale_samples: list[float] = []
    multiscale_om_samples: list[float] = []
    multiscale_hdr_samples: list[float] = []
    for _ in range(repetitions):
        full_om, full_hdr = full_operations()
        full_om_samples.append(full_om)
        full_hdr_samples.append(full_hdr)
        full_samples.append(full_om + full_hdr)
        multiscale_om, multiscale_hdr = multiscale_operations()
        multiscale_om_samples.append(multiscale_om)
        multiscale_hdr_samples.append(multiscale_hdr)
        multiscale_samples.append(multiscale_om + multiscale_hdr)
    return {
        "full_resolution_total_blur": _timing_summary(full_samples),
        "multi_scale_total_blur": _timing_summary(multiscale_samples),
        "stage_breakdown": {
            "full_resolution_om_intensity_plus_chroma": _timing_summary(full_om_samples),
            "full_resolution_hdr_overlap_chroma": _timing_summary(full_hdr_samples),
            "multi_scale_om_intensity_plus_chroma": _timing_summary(multiscale_om_samples),
            "multi_scale_hdr_overlap_chroma": _timing_summary(multiscale_hdr_samples),
        },
        "timed_operations": "OM intensity + OM chroma + fixed HDR overlap chroma",
        "repetitions": repetitions,
        "warmups": warmups,
    }


def _frame_record(
    index: int,
    dataset_root: Path,
    backend: fastcore.Backend,
    timing_repetitions: int,
    timing_warmups: int,
) -> dict[str, Any]:
    om_path, hdr_path = _frame_paths(dataset_root, index)
    om_raw = _load_raw(om_path, (OM_SIZE[1], OM_SIZE[0], 3))
    hdr_raw = _load_raw(hdr_path, (HDR_SIZE[1], HDR_SIZE[0], 3))
    om_code = _raw_to_code(om_raw)
    hdr_code = _raw_to_code(hdr_raw)
    om_linear = _decode_sdr(om_raw)
    hdr_overlap_linear = _decode_hdr_overlap(hdr_raw)
    om_ictcp = _ictcp(backend, om_linear)
    hdr_ictcp = _ictcp(backend, hdr_overlap_linear)

    om_full = _blur_field(om_ictcp, multiscale=False)
    om_parts_intensity = _multiscale_parts(om_ictcp[..., 0])
    om_parts_chroma = _multiscale_parts(om_ictcp[..., 1:])
    om_multi = np.concatenate(
        (om_parts_intensity["upsampled"][..., None], om_parts_chroma["upsampled"]), axis=-1
    )
    hdr_full = _blur_field(hdr_ictcp, multiscale=False)
    hdr_parts_chroma = _multiscale_parts(hdr_ictcp[..., 1:])
    hdr_multi = np.concatenate(
        (hdr_full[..., :1], hdr_parts_chroma["upsampled"]), axis=-1
    )

    om_chroma_full = om_full[..., 1:]
    om_chroma_multi = om_multi[..., 1:]
    hdr_chroma_full = hdr_full[..., 1:]
    residual_full = _band_residuals(om_chroma_full, hdr_chroma_full)
    residual_multi = _band_residuals(om_chroma_multi, hdr_chroma_full)
    residual_change = {
        name: _relative_change_percent(residual_full[name], residual_multi[name])
        for name in ("top", "bottom", "mean")
    }
    seam_full = _seam_steps(_reconstructed_chroma(om_chroma_full, hdr_chroma_full))
    seam_multi = _seam_steps(_reconstructed_chroma(om_chroma_multi, hdr_chroma_full))
    seam_change = {
        name: _relative_change_percent(seam_full[name], seam_multi[name])
        for name in ("top", "bottom", "mean")
    }

    om_rgb_full = _gaussian(om_linear, SIGMA)
    om_rgb_parts = _multiscale_parts(om_linear)
    timing = _time_frame(
        om_ictcp,
        hdr_ictcp,
        repetitions=timing_repetitions,
        warmups=timing_warmups,
    )
    source_checks = {
        "om_raw_uint16": _range_stats(om_raw),
        "hdr_raw_uint16": _range_stats(hdr_raw),
        "om_code_0_1": _range_stats(om_code),
        "hdr_code_0_1": _range_stats(hdr_code),
        "om_linear_bt2020_nonnegative": _range_stats(om_linear),
        "hdr_overlap_linear_bt2020_nonnegative": _range_stats(hdr_overlap_linear),
    }
    field_checks = {
        "om_ictcp_input": _range_stats(om_ictcp),
        "hdr_overlap_ictcp_input": _range_stats(hdr_ictcp),
        "om_ictcp_full": _range_stats(om_full),
        "om_ictcp_multiscale": _range_stats(om_multi),
        "hdr_ictcp_full": _range_stats(hdr_full),
        "hdr_ictcp_multiscale_chroma": _range_stats(hdr_multi),
        "om_multiscale_within_input_field_range": _within_range(om_multi, om_ictcp),
        "hdr_multiscale_within_input_field_range": _within_range(hdr_multi, hdr_ictcp),
        "all_field_arrays_finite": all(
            bool(np.all(np.isfinite(array)))
            for array in (om_ictcp, hdr_ictcp, om_full, om_multi, hdr_full, hdr_multi)
        ),
    }
    return {
        "stratum": index,
        "inputs": {
            "om_raw_rgb_u16": str(om_path),
            "hdr_raw_rgb_u16": str(hdr_path),
            "om_shape": list(om_raw.shape),
            "hdr_shape": list(hdr_raw.shape),
        },
        "geometry": {
            "overlap_open_matte_coordinates": list(OVERLAP),
            "hdr_resize": "INTER_LINEAR to 1920x800, matching production decode_hdr",
            "om_source_size": list(OM_SIZE),
            "hdr_source_size": list(HDR_SIZE),
            "om_small_size": list(_small_size(om_ictcp)),
            "hdr_overlap_small_size": list(_small_size(hdr_ictcp)),
        },
        "blur_error": {
            "decoded_bt2020_rgb": _field_error(om_rgb_full, om_rgb_parts["upsampled"]),
            "om_ictcp_all_channels": _field_error(om_full, om_multi),
            "om_ictcp_chroma_field_units": _field_error(om_chroma_full, om_chroma_multi),
            "hdr_overlap_ictcp_all_channels": _field_error(hdr_full, hdr_multi),
            "hdr_overlap_ictcp_chroma_field_units": _field_error(
                hdr_chroma_full, hdr_multi[..., 1:]
            ),
        },
        "stage_attribution": {
            "om_ictcp_chroma": _stage_attribution(
                om_ictcp[..., 1:], om_chroma_full, om_parts_chroma
            ),
            "hdr_overlap_ictcp_chroma": _stage_attribution(
                hdr_ictcp[..., 1:], hdr_chroma_full, hdr_parts_chroma
            ),
        },
        "lowpass_chroma_residual_against_fixed_full_hdr_reference": {
            "full": residual_full,
            "multi_scale": residual_multi,
            "change_percent": residual_change,
            "absolute_change_percent": {
                name: abs(value) for name, value in residual_change.items()
            },
            "reference": "full-resolution HDR overlap chroma; not replaced by the candidate path",
        },
        "reconstructed_field_space_seam_chroma_step": {
            "full": seam_full,
            "multi_scale": seam_multi,
            "change_percent": seam_change,
            "absolute_change_percent": {name: abs(value) for name, value in seam_change.items()},
            "row_pairs": {
                "top": [OVERLAP[1] - 1, OVERLAP[1]],
                "bottom": [OVERLAP[3] - 1, OVERLAP[3]],
            },
            "formula": "mean(abs(hypot(Ct,Cp)[row0] - hypot(Ct,Cp)[row1]))",
        },
        "source_checks": source_checks,
        "field_checks": field_checks,
        "timing": timing,
    }


def _aggregate_error(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    summaries: list[dict[str, Any]] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        summaries.append(value)
    counts = np.asarray([summary["element_count"] for summary in summaries], dtype=np.float64)
    total = max(float(np.sum(counts)), 1.0)
    return {
        "max_abs": float(max(summary["max_abs"] for summary in summaries)),
        "rms": float(
            np.sqrt(
                np.sum(
                    np.asarray([summary["rms"] ** 2 for summary in summaries]) * counts
                )
                / total
            )
        ),
        "mae": float(
            np.sum(np.asarray([summary["mae"] for summary in summaries]) * counts) / total
        ),
        "frames": len(summaries),
    }


def _aggregate_scalar(records: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    values: list[float] = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
        "frames": len(values),
    }


def _aggregate_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    residual_changes = {
        name: _aggregate_scalar(
            records,
            (
                "lowpass_chroma_residual_against_fixed_full_hdr_reference",
                "absolute_change_percent",
                name,
            ),
        )
        for name in ("top", "bottom", "mean")
    }
    seam_changes = {
        name: _aggregate_scalar(
            records,
            ("reconstructed_field_space_seam_chroma_step", "absolute_change_percent", name),
        )
        for name in ("top", "bottom", "mean")
    }
    multi_timing = _aggregate_scalar(
        records, ("timing", "multi_scale_total_blur", "median_ms")
    )
    full_timing = _aggregate_scalar(
        records, ("timing", "full_resolution_total_blur", "median_ms")
    )
    return {
        "blur_error": {
            "om_ictcp_all_channels": _aggregate_error(
                records, ("blur_error", "om_ictcp_all_channels", "all")
            ),
            "om_ictcp_chroma_field_units": _aggregate_error(
                records, ("blur_error", "om_ictcp_chroma_field_units", "all")
            ),
            "hdr_overlap_ictcp_chroma_field_units": _aggregate_error(
                records, ("blur_error", "hdr_overlap_ictcp_chroma_field_units", "all")
            ),
        },
        "lowpass_residual_absolute_change_percent": residual_changes,
        "seam_step_absolute_change_percent": seam_changes,
        "timing_median_ms_per_frame": {
            "full_resolution_total_blur": full_timing,
            "multi_scale_total_blur": multi_timing,
            "threshold_ms_per_frame": 2.0,
        },
    }


def _pass_fail(records: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    residual = aggregate["lowpass_residual_absolute_change_percent"]
    seam = aggregate["seam_step_absolute_change_percent"]
    residual_max = max(value["max"] for value in residual.values())
    seam_max = max(value["max"] for value in seam.values())
    timing_max = aggregate["timing_median_ms_per_frame"]["multi_scale_total_blur"]["max"]
    # The range booleans are recorded directly in each frame; keep the source-code
    # check explicit here without reconstructing arrays from JSON.
    source_code_ok = all(
        record["source_checks"]["om_code_0_1"]["finite"]
        and record["source_checks"]["hdr_code_0_1"]["finite"]
        and record["source_checks"]["om_code_0_1"]["min"] >= -RANGE_TOLERANCE
        and record["source_checks"]["om_code_0_1"]["max"] <= 1.0 + RANGE_TOLERANCE
        and record["source_checks"]["hdr_code_0_1"]["min"] >= -RANGE_TOLERANCE
        and record["source_checks"]["hdr_code_0_1"]["max"] <= 1.0 + RANGE_TOLERANCE
        for record in records
    )
    finite_and_range = all(
        record["field_checks"]["all_field_arrays_finite"]
        and record["field_checks"]["om_multiscale_within_input_field_range"]
        and record["field_checks"]["hdr_multiscale_within_input_field_range"]
        for record in records
    ) and source_code_ok
    criteria = {
        "total_blur_cost_le_2_ms_per_frame": timing_max <= 2.0,
        "lowpass_chroma_residual_change_lt_1_percent": residual_max < 1.0,
        "seam_chroma_step_change_lt_1_percent": seam_max < 1.0,
        "no_new_nonfinite_or_out_of_range": finite_and_range,
    }
    return {
        "criteria": criteria,
        "max_observed_absolute_residual_change_percent": residual_max,
        "max_observed_absolute_seam_step_change_percent": seam_max,
        "max_observed_multi_scale_median_blur_ms_per_frame": timing_max,
        "pass": bool(all(criteria.values())),
        "note": (
            "A failed one-percent criterion is reported as measured; this diagnostic does not tune "
            "sigma or geometry. Stage attribution identifies AREA/sampling, small-grid Gaussian, "
            "bilinear reconstruction, and boundary contributions."
        ),
    }


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.timing_repetitions > 0, "--timing-repetitions must be positive")
    _require(args.timing_warmups >= 0, "--timing-warmups must be nonnegative")
    _require(args.frames, "At least one frame index is required")
    dataset_root = args.dataset_root.resolve()
    backend = fastcore.Backend(False)
    records = [
        _frame_record(
            index,
            dataset_root,
            backend,
            timing_repetitions=args.timing_repetitions,
            timing_warmups=args.timing_warmups,
        )
        for index in args.frames
    ]
    aggregate = _aggregate_report(records)
    return {
        "task": "TASK 3 — MULTI-SCALE GAUSSIAN",
        "mode": "isolated numerical diagnostic; no production caller changed",
        "protocol": {
            "source_material": "The Matrix P232 real-material dataset",
            "strata": list(args.frames),
            "full_resolution_sigma_px": SIGMA,
            "scale": SCALE,
            "small_resolution_sigma_px": SIGMA_SMALL,
            "downsample": "cv2.INTER_AREA",
            "upsample": "cv2.INTER_LINEAR",
            "gaussian_border": "cv2.BORDER_REFLECT_101",
            "border_contract": "CPU equivalent of the production GPU mirror mode",
            "blur_domain": (
                "linear BT.2020 RGB converted to ICtCp; intensity and Ct/Cp "
                "blurred independently"
            ),
            "fixed_hdr_reference": "full-resolution sigma-16 HDR overlap chroma",
            "seam_band_rows": SEAM_BAND,
            "overlap": list(OVERLAP),
            "pass_thresholds": {
                "total_blur_ms_per_frame": 2.0,
                "lowpass_residual_change_percent": 1.0,
                "seam_step_change_percent": 1.0,
            },
        },
        "dataset_root": str(dataset_root),
        "frames": records,
        "aggregate": aggregate,
        "pass_fail": _pass_fail(records, aggregate),
        "scope_guard": {
            "fitting_run": False,
            "model_selection_run": False,
            "renderer_run": False,
            "nvdec_run": False,
            "nvenc_run": False,
            "task_2_changed": False,
            "tasks_4_plus_run": False,
            "full_film_run": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _build_report(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["pass_fail"], indent=2, sort_keys=True))
    print(f"JSON report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
