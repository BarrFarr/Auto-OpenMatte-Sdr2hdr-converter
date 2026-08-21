#!/usr/bin/env python3
"""Bounded v3: intensity-conditioned residual for the low-frequency chroma field.

The frozen spatial model remains unchanged: 12 real ICtCp chroma controls,
sigma-16 smoothing, two quadratic seam profiles, vertical blending, untouched
intensity/detail, and a copied HDR centre. This iteration adds only two global,
shot-static residual slopes driven by *smoothed predicted ICtCp intensity*:

    saturation_residual(t) = exp(a * t)
    hue_residual(t)        = exp(i * b * t)

where t is a robustly normalized low-frequency intensity in [-1, 1]. Both
a and b are fitted on training frames, hard-bounded, and must beat the frozen
spatial model on held-out overlap frames before being accepted. No new spatial
field, temporal state, AI, LUT, MMR, or HDR-centre generation is introduced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The v2 helper provides the fixed train/held-out split and v1 reusable contracts.
v2 = _load("fast_chroma_field_hue_damped_v2", HERE / "fast_chroma_field_hue_damped.py")
v1 = v2.v1
fastcore = v2.fastcore
om = v2.om

INTENSITY_SIGMA = v1.CHROMA_SIGMA
MIN_INTENSITY_NORMALIZATION = 0.02
MAX_LOG_SATURATION_SLOPE = math.log(1.25)  # t=+/-1 limits residual saturation to 1.25x / 0.80x
MAX_HUE_SLOPE_RADIANS = math.radians(5.0)  # t=+/-1 permits at most +/-5 degrees
MIN_HOLDOUT_RESIDUAL_IMPROVEMENT = 0.002
MAX_HOLDOUT_HUE_STEP_REGRESSION_DEGREES = 0.02
DEFAULT_OUTPUT = HERE / "out" / "chroma_field_intensity"


def predicted_low_frequency(be: fastcore.Backend, config: om.Config, sdr: Any, gain: Any) -> tuple[Any, Any, Any, Any]:
    """Return prediction plus low-frequency I and Ct/Cp from the frozen path."""
    predicted = v1.predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
    ictcp = be.to_ictcp(predicted * fastcore.PEAK_NITS)
    return predicted, ictcp, be.blur(ictcp[..., 0], INTENSITY_SIGMA), be.blur(ictcp[..., 1:], INTENSITY_SIGMA)


def normalized_intensity(be: fastcore.Backend, intensity: Any, centre: float, normalization: float) -> Any:
    return be.xp.clip((intensity - centre) / normalization, -1.0, 1.0)


def field_magnitude_angle(be: fastcore.Backend, field: np.ndarray) -> tuple[Any, Any]:
    return be.asarray(np.abs(field).astype(np.float32)), be.asarray(np.angle(field).astype(np.float32))


def conditioned_components(
    be: fastcore.Backend,
    spatial_magnitude: Any,
    spatial_angle: Any,
    intensity: Any,
    controls: dict[str, float],
) -> tuple[Any, Any]:
    """Bound the final complex factor after applying the two intensity slopes."""
    t = normalized_intensity(be, intensity, controls["intensity_centre"], controls["intensity_normalization"])
    magnitude = be.xp.clip(
        spatial_magnitude * be.xp.exp(controls["log_saturation_slope"] * t),
        v1.SCALE_CLAMP[0],
        v1.SCALE_CLAMP[1],
    )
    rotation_limit = math.radians(v1.ROTATION_CLAMP_DEG)
    angle = be.xp.clip(spatial_angle + controls["hue_slope_radians"] * t, -rotation_limit, rotation_limit)
    return magnitude * be.xp.cos(angle), magnitude * be.xp.sin(angle)


def fit_intensity_controls(
    be: fastcore.Backend,
    config: om.Config,
    sdr_cache: np.memmap,
    hdr_cache: np.memmap,
    gain: Any,
    spatial_field: np.ndarray,
    training_indices: list[int],
) -> dict[str, Any]:
    """Fit one global log-saturation slope and one global hue slope on training data."""
    _, y1, _, y2 = config.overlap
    field_parts = {
        "top": field_magnitude_angle(be, spatial_field[y1:y1 + v1.SEAM_BAND]),
        "bottom": field_magnitude_angle(be, spatial_field[y2 - v1.SEAM_BAND:y2]),
    }
    sum_weight = be.xp.float64(0.0)
    sum_intensity = be.xp.float64(0.0)
    sum_intensity_squared = be.xp.float64(0.0)
    started = time.perf_counter()
    # Pass 1: chroma-energy-weighted centre/spread for a stable, portable t domain.
    for index in training_indices:
        sdr = be.asarray(sdr_cache[index])
        _, _, intensity, chroma = predicted_low_frequency(be, config, sdr, gain)
        for intensity_band, chroma_band in (
            (intensity[y1:y1 + v1.SEAM_BAND], chroma[y1:y1 + v1.SEAM_BAND]),
            (intensity[y2 - v1.SEAM_BAND:y2], chroma[y2 - v1.SEAM_BAND:y2]),
        ):
            weight = be.xp.sum(chroma_band * chroma_band, axis=-1)
            valid_weight = be.xp.where(weight > v2.HUE_CHROMA_FLOOR**2, weight, 0.0)
            sum_weight += be.xp.sum(valid_weight)
            sum_intensity += be.xp.sum(valid_weight * intensity_band)
            sum_intensity_squared += be.xp.sum(valid_weight * intensity_band * intensity_band)
    host_weight = float(be.tohost(sum_weight))
    om.require(host_weight > 0.0, "No chroma-energy support for intensity-conditioned fit")
    centre = float(be.tohost(sum_intensity)) / host_weight
    variance = max(float(be.tohost(sum_intensity_squared)) / host_weight - centre * centre, 0.0)
    normalization = max(2.0 * math.sqrt(variance), MIN_INTENSITY_NORMALIZATION)

    scale_numerator = be.xp.float64(0.0)
    hue_numerator = be.xp.float64(0.0)
    denominator = be.xp.float64(0.0)
    valid_samples = be.xp.int64(0)
    # Pass 2: linear residual fit in log magnitude and circular hue difference.
    for index in training_indices:
        sdr = be.asarray(sdr_cache[index])
        hdr = be.asarray(hdr_cache[index])
        _, _, intensity, predicted_chroma = predicted_low_frequency(be, config, sdr, gain)
        hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], INTENSITY_SIGMA)
        for name, intensity_band, predicted_band, reference_band in (
            ("top", intensity[y1:y1 + v1.SEAM_BAND], predicted_chroma[y1:y1 + v1.SEAM_BAND], hdr_chroma[:v1.SEAM_BAND]),
            ("bottom", intensity[y2 - v1.SEAM_BAND:y2], predicted_chroma[y2 - v1.SEAM_BAND:y2], hdr_chroma[-v1.SEAM_BAND:]),
        ):
            spatial = v2.complex_transform(be, predicted_band, *field_parts[name])
            predicted_magnitude = be.xp.hypot(spatial[..., 0], spatial[..., 1])
            reference_magnitude = be.xp.hypot(reference_band[..., 0], reference_band[..., 1])
            valid = (predicted_magnitude > v2.HUE_CHROMA_FLOOR) & (reference_magnitude > v2.HUE_CHROMA_FLOOR)
            weight = be.xp.where(valid, predicted_magnitude * predicted_magnitude, 0.0)
            t = normalized_intensity(be, intensity_band, centre, normalization)
            log_ratio = be.xp.log(be.xp.maximum(reference_magnitude, fastcore.EPS) / be.xp.maximum(predicted_magnitude, fastcore.EPS))
            phase = be.xp.arctan2(
                spatial[..., 0] * reference_band[..., 1] - spatial[..., 1] * reference_band[..., 0],
                spatial[..., 0] * reference_band[..., 0] + spatial[..., 1] * reference_band[..., 1],
            )
            scale_numerator += be.xp.sum(weight * t * log_ratio)
            hue_numerator += be.xp.sum(weight * t * phase)
            denominator += be.xp.sum(weight * t * t)
            valid_samples += be.xp.sum(valid)
    host_denominator = float(be.tohost(denominator))
    om.require(host_denominator > 0.0, "Degenerate intensity-conditioned fit")
    unconstrained_scale = float(be.tohost(scale_numerator)) / host_denominator
    unconstrained_hue = float(be.tohost(hue_numerator)) / host_denominator
    scale = float(np.clip(unconstrained_scale, -MAX_LOG_SATURATION_SLOPE, MAX_LOG_SATURATION_SLOPE))
    hue = float(np.clip(unconstrained_hue, -MAX_HUE_SLOPE_RADIANS, MAX_HUE_SLOPE_RADIANS))
    return {
        "intensity_centre": centre,
        "intensity_normalization": normalization,
        "log_saturation_slope": scale,
        "hue_slope_radians": hue,
        "hue_slope_degrees": math.degrees(hue),
        "unconstrained_log_saturation_slope": unconstrained_scale,
        "unconstrained_hue_slope_degrees": math.degrees(unconstrained_hue),
        "log_saturation_slope_limit": MAX_LOG_SATURATION_SLOPE,
        "hue_slope_limit_degrees": math.degrees(MAX_HUE_SLOPE_RADIANS),
        "training_frames": len(training_indices),
        "valid_samples": int(be.tohost(valid_samples)),
        "seconds": time.perf_counter() - started,
    }


def candidate_models(controls: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "intensity_centre": float(controls["intensity_centre"]),
        "intensity_normalization": float(controls["intensity_normalization"]),
    }
    return [
        {"name": "spatial_only", **base, "log_saturation_slope": 0.0, "hue_slope_radians": 0.0},
        {"name": "intensity_saturation", **base, "log_saturation_slope": float(controls["log_saturation_slope"]), "hue_slope_radians": 0.0},
        {"name": "intensity_hue", **base, "log_saturation_slope": 0.0, "hue_slope_radians": float(controls["hue_slope_radians"])},
        {"name": "intensity_joint", **base, "log_saturation_slope": float(controls["log_saturation_slope"]), "hue_slope_radians": float(controls["hue_slope_radians"])},
    ]


def evaluate_candidates(
    be: fastcore.Backend,
    config: om.Config,
    sdr_cache: np.memmap,
    hdr_cache: np.memmap,
    gain: Any,
    spatial_field: np.ndarray,
    controls: dict[str, Any],
    holdout_indices: list[int],
) -> dict[str, Any]:
    """Compare the no-slope baseline and three bounded candidate ablations on held-out frames."""
    _, y1, _, y2 = config.overlap
    parts = {
        "top_band": field_magnitude_angle(be, spatial_field[y1:y1 + v1.SEAM_BAND]),
        "bottom_band": field_magnitude_angle(be, spatial_field[y2 - v1.SEAM_BAND:y2]),
        "top_edge": field_magnitude_angle(be, spatial_field[y1 - 1]),
        "bottom_edge": field_magnitude_angle(be, spatial_field[y2]),
    }
    models = candidate_models(controls)
    accumulators = {model["name"]: {"residual_sum": 0.0, "residual_count": 0, "hue_sum": 0.0, "hue_count": 0} for model in models}
    started = time.perf_counter()
    for index in holdout_indices:
        sdr = be.asarray(sdr_cache[index])
        hdr = be.asarray(hdr_cache[index])
        _, _, intensity, predicted_chroma = predicted_low_frequency(be, config, sdr, gain)
        hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], INTENSITY_SIGMA)
        entries = (
            ("top_band", intensity[y1:y1 + v1.SEAM_BAND], predicted_chroma[y1:y1 + v1.SEAM_BAND], hdr_chroma[:v1.SEAM_BAND]),
            ("bottom_band", intensity[y2 - v1.SEAM_BAND:y2], predicted_chroma[y2 - v1.SEAM_BAND:y2], hdr_chroma[-v1.SEAM_BAND:]),
        )
        for model in models:
            accumulator = accumulators[model["name"]]
            for name, intensity_band, predicted_band, reference_band in entries:
                real, imag = conditioned_components(be, *parts[name], intensity_band, model)
                corrected = v2.complex_transform(be, predicted_band, real, imag)
                accumulator["residual_sum"] += float(be.tohost(be.xp.sum(be.xp.abs(reference_band - corrected))))
                accumulator["residual_count"] += int(reference_band.size)
            top_real, top_imag = conditioned_components(be, *parts["top_edge"], intensity[y1 - 1], model)
            bottom_real, bottom_imag = conditioned_components(be, *parts["bottom_edge"], intensity[y2], model)
            for corrected, reference in (
                (v2.complex_transform(be, predicted_chroma[y1 - 1], top_real, top_imag), hdr_chroma[0]),
                (v2.complex_transform(be, predicted_chroma[y2], bottom_real, bottom_imag), hdr_chroma[-1]),
            ):
                hue_sum, hue_count = v2.circular_hue_sum(be, corrected, reference)
                accumulator["hue_sum"] += hue_sum
                accumulator["hue_count"] += hue_count
    values = []
    for model in models:
        accumulator = accumulators[model["name"]]
        values.append(
            {
                "name": model["name"],
                "log_saturation_slope": model["log_saturation_slope"],
                "hue_slope_degrees": math.degrees(model["hue_slope_radians"]),
                "lowpass_band_chroma_error": accumulator["residual_sum"] / max(accumulator["residual_count"], 1),
                "lowpass_boundary_hue_step_degrees": accumulator["hue_sum"] / max(accumulator["hue_count"], 1),
                "hue_valid_samples": accumulator["hue_count"],
            }
        )
    baseline = next(value for value in values if value["name"] == "spatial_only")
    nonbaseline = [value for value in values if value["name"] != "spatial_only"]
    review_candidate = min(
        nonbaseline,
        key=lambda value: (value["lowpass_band_chroma_error"], value["lowpass_boundary_hue_step_degrees"]),
    )
    eligible = [
        value for value in nonbaseline
        if value["lowpass_band_chroma_error"] <= baseline["lowpass_band_chroma_error"] * (1.0 - MIN_HOLDOUT_RESIDUAL_IMPROVEMENT)
        and value["lowpass_boundary_hue_step_degrees"] <= baseline["lowpass_boundary_hue_step_degrees"] + MAX_HOLDOUT_HUE_STEP_REGRESSION_DEGREES
    ]
    selected = min(eligible, key=lambda value: value["lowpass_band_chroma_error"]) if eligible else baseline
    return {
        "candidates": values,
        "baseline": baseline,
        "default_model": selected,
        "review_candidate": review_candidate,
        "accepted_as_default": selected["name"] != "spatial_only",
        "minimum_residual_improvement_fraction": MIN_HOLDOUT_RESIDUAL_IMPROVEMENT,
        "maximum_hue_step_regression_degrees": MAX_HOLDOUT_HUE_STEP_REGRESSION_DEGREES,
        "holdout_frames": len(holdout_indices),
        "seconds": time.perf_counter() - started,
    }


def apply_intensity_conditioned(
    be: fastcore.Backend,
    predicted: Any,
    spatial_magnitude: Any,
    spatial_angle: Any,
    controls: dict[str, float],
) -> tuple[Any, Any, float, float]:
    """Apply a low-frequency factor that depends only on low-frequency predicted I."""
    ictcp = be.to_ictcp(predicted * fastcore.PEAK_NITS)
    chroma = ictcp[..., 1:]
    base = be.blur(chroma, INTENSITY_SIGMA)
    detail = chroma - base
    intensity = be.blur(ictcp[..., 0], INTENSITY_SIGMA)
    real, imag = conditioned_components(be, spatial_magnitude, spatial_angle, intensity, controls)
    corrected = v2.complex_transform(be, base, real, imag)
    ictcp[..., 1:] = corrected + detail
    raw = be.from_ictcp(ictcp) / fastcore.PEAK_NITS
    negative = float(be.tohost(be.xp.mean(raw < -1e-4)))
    above_peak = float(be.tohost(be.xp.mean(raw > 1.0001)))
    return be.xp.clip(raw, 0.0, 1.0), corrected, negative, above_peak


def region_sheet(path: Path, seam: int, samples: dict[str, np.ndarray], height: int) -> None:
    top = max(0, min(height - 270, seam - 135))
    rows = []
    for name, x in v1.REGIONS:
        tiles = []
        for variant in ("spatial_field", "intensity_conditioned"):
            patch = np.round(om.preview(samples[variant][top:top + 270, x:x + 480]) * 255.0).astype(np.uint8)
            cv2.rectangle(patch, (0, 0), (480, 22), (0, 0, 0), -1)
            cv2.putText(patch, f"{name} / {variant}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(patch, (0, seam - top), (480, seam - top), (0, 0, 0), 1)
            tiles.append(patch)
        rows.append(np.concatenate(tiles, axis=1))
    om.write_png(path, np.concatenate(rows, axis=0))


def render(
    be: fastcore.Backend,
    config: om.Config,
    sdr_cache: np.memmap,
    hdr_cache: np.memmap,
    gain: Any,
    spatial_field: np.ndarray,
    controls: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Render frozen spatial-field versus the held-out-selected intensity variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = config.om_size
    _, y1, _, y2 = config.overlap
    spatial_real, spatial_imag = be.asarray(spatial_field.real.astype(np.float32)), be.asarray(spatial_field.imag.astype(np.float32))
    spatial_magnitude, spatial_angle = field_magnitude_angle(be, spatial_field)
    outputs = {
        "spatial_field": output_dir / "v02_spatial_chroma_hdr10.mkv",
        "intensity_conditioned": output_dir / "v04_intensity_chroma_hdr10.mkv",
        "side_by_side": output_dir / "side_by_side_hdr10.mkv",
    }
    pipes = [
        v1.nvenc(config, outputs["spatial_field"], width, height),
        v1.nvenc(config, outputs["intensity_conditioned"], width, height),
        v1.nvenc(config, outputs["side_by_side"], width * 2, height),
    ]
    writer = fastcore.ParallelWriter(pipes)
    sample_index = int(round((len(sdr_cache) - 1) * 0.5))
    samples: dict[str, np.ndarray] = {}
    metrics_rows: dict[str, list[dict[str, float]]] = {"spatial_field": [], "intensity_conditioned": []}
    band_residuals: dict[str, list[float]] = {"spatial_field": [], "intensity_conditioned": []}
    negative_totals = {"spatial_field": 0.0, "intensity_conditioned": 0.0}
    above_peak_totals = {"spatial_field": 0.0, "intensity_conditioned": 0.0}
    started = time.perf_counter()
    try:
        for index in range(len(sdr_cache)):
            sdr = be.asarray(sdr_cache[index])
            hdr = be.asarray(hdr_cache[index])
            predicted = v1.predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
            spatial_rgb, spatial_chroma, spatial_negative, spatial_above_peak = v2.apply_chroma_with_range(be, predicted, spatial_real, spatial_imag)
            intensity_rgb, intensity_chroma, intensity_negative, intensity_above_peak = apply_intensity_conditioned(be, predicted, spatial_magnitude, spatial_angle, controls)
            rendered = {
                "spatial_field": v1.composite(be, config, spatial_rgb, hdr),
                "intensity_conditioned": v1.composite(be, config, intensity_rgb, hdr),
            }
            spatial_pq, intensity_pq = be.to_pq16(rendered["spatial_field"]), be.to_pq16(rendered["intensity_conditioned"])
            writer.submit([spatial_pq.tobytes(), intensity_pq.tobytes(), np.concatenate((spatial_pq, intensity_pq), axis=1).tobytes()])
            negative_totals["spatial_field"] += spatial_negative
            negative_totals["intensity_conditioned"] += intensity_negative
            above_peak_totals["spatial_field"] += spatial_above_peak
            above_peak_totals["intensity_conditioned"] += intensity_above_peak
            if index % v2.DIAGNOSTIC_STRIDE == 0:
                hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], INTENSITY_SIGMA)
                for name, image, corrected_chroma in (
                    ("spatial_field", rendered["spatial_field"], spatial_chroma),
                    ("intensity_conditioned", rendered["intensity_conditioned"], intensity_chroma),
                ):
                    band_residuals[name].append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_chroma[:v1.SEAM_BAND] - corrected_chroma[y1:y1 + v1.SEAM_BAND])))))
                    band_residuals[name].append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_chroma[-v1.SEAM_BAND:] - corrected_chroma[y2 - v1.SEAM_BAND:y2])))))
                    metrics_rows[name].append(v2.seam_metrics(be, image, config))
            if index == sample_index:
                samples = {name: be.tohost(image) for name, image in rendered.items()}
    finally:
        writer.close()
        for pipe in pipes:
            if pipe.stdin:
                pipe.stdin.close()
        codes = [pipe.wait() for pipe in pipes]
        errors = [pipe.stderr.read().decode("utf-8", errors="replace") for pipe in pipes]
        om.require(all(code == 0 for code in codes), f"NVENC failed: {codes}; {errors}")
    om.require(samples, "Representative frame was not rendered")
    region_sheet(output_dir / "top_seam_regions.png", y1, samples, height)
    region_sheet(output_dir / "bottom_seam_regions.png", y2, samples, height)
    full_pair = np.concatenate(
        [np.round(om.preview(cv2.resize(samples[name], (960, 540), interpolation=cv2.INTER_AREA)) * 255.0).astype(np.uint8) for name in ("spatial_field", "intensity_conditioned")],
        axis=1,
    )
    om.write_png(output_dir / "full_frame_pair.png", full_pair)
    summary: dict[str, Any] = {}
    for name, rows in metrics_rows.items():
        summary[name] = {
            "lowpass_band_chroma_error": float(np.mean(band_residuals[name])),
            "negative_rgb_fraction_before_clip": negative_totals[name] / len(sdr_cache),
            "above_peak_rgb_fraction_before_clip": above_peak_totals[name] / len(sdr_cache),
            "seam": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
        }
    tags = {name: om.verify_hdr_tags(config, path) for name, path in outputs.items()}
    return {"seconds": time.perf_counter() - started, "frames": len(sdr_cache), "outputs": {name: str(path) for name, path in outputs.items()}, "tags": tags, "metrics": summary}


def markdown_report(payload: dict[str, Any]) -> str:
    controls = payload["intensity_controls"]
    selection = payload["selection"]
    baseline = payload["result"]["metrics"]["spatial_field"]
    candidate = payload["result"]["metrics"]["intensity_conditioned"]
    default = selection["default_model"]
    review = selection["review_candidate"]
    rendered = payload["rendered_model"]
    status = "accepted as default" if selection["accepted_as_default"] else "rejected as default; a non-default review candidate was still rendered for visual comparison"
    lines = [
        "# PoC v3 — bounded intensity-conditioned chroma residual", "",
        "The 12-control low-frequency ICtCp spatial field is unchanged. This iteration adds at most two shot-global residual slopes driven by sigma-16 predicted ICtCp intensity: log saturation versus intensity and hue rotation versus intensity.", "",
        "## Fitted controls", "",
        f"Training frames: `{controls['training_frames']}`; valid chroma samples: `{controls['valid_samples']}`.",
        f"Intensity centre `{controls['intensity_centre']:.6f}`, normalization `{controls['intensity_normalization']:.6f}`, log-saturation slope `{controls['log_saturation_slope']:+.6f}`, hue slope `{controls['hue_slope_degrees']:+.4f}°` at `t=+1`.",
        f"Hard limits: residual saturation `[0.80, 1.25]` at |t|=1; residual hue `±{controls['hue_slope_limit_degrees']:.1f}°` at |t|=1.", "",
        "## Held-out model gate", "",
        f"Default-model status: **{status}**. Selection used `{selection['holdout_frames']}` distinct held-out frames. A candidate must reduce low-pass chroma disagreement by at least `{selection['minimum_residual_improvement_fraction'] * 100:.1f}%` while increasing low-pass hue boundary step by no more than `{selection['maximum_hue_step_regression_degrees']:.2f}°`.",
        f"Default model: `{default['name']}`; held-out chroma error `{default['lowpass_band_chroma_error']:.8f}`, held-out hue step `{default['lowpass_boundary_hue_step_degrees']:.4f}°`.",
        f"Visual-review candidate: `{review['name']}`; held-out chroma error `{review['lowpass_band_chroma_error']:.8f}`, held-out hue step `{review['lowpass_boundary_hue_step_degrees']:.4f}°`.", "",
        "## Full render diagnostics", "",
        f"| Quantity | Frozen spatial field | Rendered `{rendered['name']}` |",
        "|---|---:|---:|",
        f"| Low-pass seam-band chroma disagreement | {baseline['lowpass_band_chroma_error']:.8f} | {candidate['lowpass_band_chroma_error']:.8f} |",
        f"| Top instantaneous hue step | {baseline['seam']['top_hue_step_degrees']:.4f}° | {candidate['seam']['top_hue_step_degrees']:.4f}° |",
        f"| Bottom instantaneous hue step | {baseline['seam']['bottom_hue_step_degrees']:.4f}° | {candidate['seam']['bottom_hue_step_degrees']:.4f}° |",
        f"| Negative RGB fraction before clip | {baseline['negative_rgb_fraction_before_clip']:.8f} | {candidate['negative_rgb_fraction_before_clip']:.8f} |",
        f"| Above-peak RGB fraction before clip | {baseline['above_peak_rgb_fraction_before_clip']:.8f} | {candidate['above_peak_rgb_fraction_before_clip']:.8f} |",
        "",
        "## Review artifacts", "",
        "- `v02_spatial_chroma_hdr10.mkv` — frozen spatial-field baseline",
        f"- `v04_intensity_chroma_hdr10.mkv` — rendered `{rendered['name']}`; it is the default only when the held-out gate accepts it",
        "- `side_by_side_hdr10.mkv` — baseline left, candidate right",
        "- `top_seam_regions.png`, `bottom_seam_regions.png`, `full_frame_pair.png`", "",
        "The extensions have no HDR ground truth. Visual HDR review remains decisive: inspect skin, rice paper, walls, dark clothes, and saturated elements for a more natural merge without intensity-dependent colour artifacts.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="fast-chroma-field-intensity")
    parser.add_argument("--max-frames", type=int, default=None, help="Render the first N frames for a smoke test")
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride for the chroma train/held-out split")
    parser.add_argument("--cpu", action="store_true", help="Force the multicore NumPy/OpenCV fallback")
    parser.add_argument("--render-model", choices=("default", "review"), default="default", help="Render the held-out-accepted default or the best residual review candidate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for this iteration's review artifacts")
    args = parser.parse_args()
    om.require(args.stride > 0, "--stride must be positive")
    argv = [
        "--hdr", v1.MATRIX["hdr"], "--om", v1.MATRIX["om"], "--output", str(args.output_dir / "unused.mkv"),
        "--shot-start", str(v1.MATRIX["shot_start"]), "--shot-end", str(v1.MATRIX["shot_end"]),
        "--offset-frames", str(v1.MATRIX["offset"]), "--seam-band", str(v1.SEAM_BAND),
    ]
    if args.max_frames is not None:
        argv += ["--max-frames", str(args.max_frames)]
    config = om.build_config(argv)
    count = config.frame_count
    training_indices, holdout_indices = v2.sampled_split(count, args.stride)
    be = fastcore.Backend(use_gpu=not args.cpu and fastcore.gpu_available())
    cache = v1.cache_for(config, count)
    print(f"intensity-conditioned chroma field: {count} frames, backend {be.name}, training {len(training_indices)}, hold-out {len(holdout_indices)}")
    total = time.perf_counter()
    if cache.valid():
        decode_seconds = 0.0
        print("cache: reusing decoded frames")
    else:
        started = time.perf_counter()
        cache.build(lambda: om.pair_stream(config, count))
        decode_seconds = time.perf_counter() - started
        print(f"cache: decoded once in {decode_seconds:.1f} s")
    sdr_cache, hdr_cache, manifest = cache.open()
    gain = v1.measure_gain(be, config, sdr_cache, hdr_cache, stride=1)
    gain_field = be.asarray(om.gain_field(config, gain))
    spatial = v2.measure_spatial_field(be, config, sdr_cache, hdr_cache, gain_field, training_indices)
    spatial_field = v1.chroma_field(config, spatial)
    controls = fit_intensity_controls(be, config, sdr_cache, hdr_cache, gain_field, spatial_field, training_indices)
    selection = evaluate_candidates(be, config, sdr_cache, hdr_cache, gain_field, spatial_field, controls, holdout_indices)
    default_model = next(model for model in candidate_models(controls) if model["name"] == selection["default_model"]["name"])
    review_model = next(model for model in candidate_models(controls) if model["name"] == selection["review_candidate"]["name"])
    rendered_model = default_model if args.render_model == "default" else review_model
    print(f"gain {gain['seconds']:.1f} s; spatial {spatial['seconds']:.1f} s; intensity fit {controls['seconds']:.1f} s; held-out gate {selection['seconds']:.1f} s")
    print(f"controls: log-sat {controls['log_saturation_slope']:+.5f}, hue {controls['hue_slope_degrees']:+.3f} deg; default {default_model['name']}; review {review_model['name']}; rendered {rendered_model['name']}")
    result = render(be, config, sdr_cache, hdr_cache, gain_field, spatial_field, rendered_model, args.output_dir)
    elapsed = time.perf_counter() - total
    print(f"render {result['seconds']:.1f} s ({result['seconds'] / count:.3f} s/frame); total {elapsed:.1f} s")
    payload = {
        "schema": "openmatte-hdr-intensity-conditioned-chroma-field/v1",
        "backend": be.name,
        "frames": count,
        "stride": args.stride,
        "split": {"fold_modulo": v2.FOLD_MODULO, "holdout_remainder": v2.HOLDOUT_REMAINDER, "training_indices": training_indices, "holdout_indices": holdout_indices},
        "cache": {"gigabytes": cache.gigabytes, "reused": decode_seconds == 0.0, "source_frame_hashes": manifest["source_frame_hashes"]},
        "timing_seconds": {"decode_cache": decode_seconds, "gain_pass": gain["seconds"], "spatial_fit": spatial["seconds"], "intensity_fit": controls["seconds"], "heldout_gate": selection["seconds"], "render": result["seconds"], "total": elapsed},
        "gain": {key: value for key, value in gain.items() if not key.endswith("_profile_stops")},
        "spatial_field": {
            "parameter_count": spatial["parameter_count"],
            "frames_used": spatial["frames_used"],
            "z_shot": [spatial["z_shot"].real, spatial["z_shot"].imag],
            "clamped_fraction": spatial["clamped_fraction"],
            "band_chroma_error_before": spatial["band_chroma_error_before"],
            "coefficients_highest_order_first": {name: [[value.real, value.imag] for value in values] for name, values in spatial["coefficients"].items()},
        },
        "intensity_controls": controls,
        "selection": selection,
        "default_model": default_model,
        "review_model": review_model,
        "rendered_model": rendered_model,
        "result": result,
        "guarantees": {"temporal_state": False, "ai": False, "lut": False, "mmr": False, "hdr_centre_regenerated": False, "intensity_modified_by_chroma_step": False, "chroma_detail_modified": False, "new_spatial_parameters": 0},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "intensity_chroma_field.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "INTENSITY_CHROMA_FIELD_POC.md").write_text(markdown_report(payload), encoding="utf-8")
    print(args.output_dir / "INTENSITY_CHROMA_FIELD_POC.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
