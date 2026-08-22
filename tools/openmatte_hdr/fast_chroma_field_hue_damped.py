#!/usr/bin/env python3
"""Bounded v2 iteration of the accelerated OpenMatte chroma-field PoC.

It retains the frozen v1 spatial field exactly in form: sigma-16 ICtCp chroma,
a quadratic x profile at each seam, vertical blending, and no correction to
intensity or chroma detail. The only extra flexibility is one shot-global hue
damping scalar lambda in [0, 1]:

    z_damped(x, y) = abs(z(x, y)) * exp(i * lambda * arg(z(x, y)))

The 12 spatial controls are fitted on fixed training frames. Lambda is selected
on disjoint held-out frames for the smallest low-pass boundary hue step while
allowing no more than a 0.5% low-pass chroma-residual increase over lambda=1.
It adds no spatial parameters, temporal state, LUT, MMR, AI, or HDR-centre
regeneration.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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


# Reuse the frozen v1 decode, base/detail, GPU backend and encoder contracts.
v1 = _load("fast_chroma_field_v1", HERE / "fast_chroma_field.py")
fastcore = v1.fastcore
om = v1.om

FOLD_MODULO = 5
HOLDOUT_REMAINDER = 0
LAMBDA_CANDIDATES = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
MAX_RESIDUAL_REGRESSION_FRACTION = 0.005
HUE_CHROMA_FLOOR = 1e-4
DIAGNOSTIC_STRIDE = 6
DEFAULT_OUTPUT = HERE / "out" / "chroma_field_hue_damped"


def sampled_split(count: int, stride: int) -> tuple[list[int], list[int]]:
    sampled = list(range(0, count, stride))
    training = [index for index in sampled if index % FOLD_MODULO != HOLDOUT_REMAINDER]
    holdout = [index for index in sampled if index % FOLD_MODULO == HOLDOUT_REMAINDER]
    om.require(training and holdout, "Need both training and held-out samples; increase --max-frames or lower --stride")
    return training, holdout


def measure_spatial_field(
    be: fastcore.Backend,
    config: om.Config,
    sdr_cache: np.memmap,
    hdr_cache: np.memmap,
    gain: Any,
    indices: list[int],
) -> dict[str, Any]:
    """Fit the frozen 12-control spatial model from training frames only."""
    _, y1, _, y2 = config.overlap
    width = config.om_size[0]
    totals = {name: {"num_re": np.zeros(width), "num_im": np.zeros(width), "den": np.zeros(width)} for name in ("top", "bottom")}
    before: list[float] = []
    started = time.perf_counter()
    for index in indices:
        sdr = be.asarray(sdr_cache[index])
        hdr = be.asarray(hdr_cache[index])
        predicted = v1.predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
        pred_chroma = be.blur(be.to_ictcp(predicted * fastcore.PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
        hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
        bands = {
            "top": (pred_chroma[y1:y1 + v1.SEAM_BAND], hdr_chroma[:v1.SEAM_BAND]),
            "bottom": (pred_chroma[y2 - v1.SEAM_BAND:y2], hdr_chroma[-v1.SEAM_BAND:]),
        }
        for name, (predicted_band, hdr_band) in bands.items():
            p_re, p_im = predicted_band[..., 0], predicted_band[..., 1]
            h_re, h_im = hdr_band[..., 0], hdr_band[..., 1]
            totals[name]["num_re"] += be.tohost(be.xp.sum(p_re * h_re + p_im * h_im, axis=0)).astype(np.float64)
            totals[name]["num_im"] += be.tohost(be.xp.sum(p_re * h_im - p_im * h_re, axis=0)).astype(np.float64)
            totals[name]["den"] += be.tohost(be.xp.sum(p_re * p_re + p_im * p_im, axis=0)).astype(np.float64)
            before.append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_band - predicted_band)))))
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    coefficients: dict[str, list[complex]] = {}
    per_column: dict[str, np.ndarray] = {}
    clamped: dict[str, float] = {}
    total_num = sum(entry["num_re"].sum() + 1j * entry["num_im"].sum() for entry in totals.values())
    total_den = sum(entry["den"].sum() for entry in totals.values())
    z_shot, _ = v1.clamp_z(np.asarray([total_num / max(total_den, 1e-30)]))
    for name, entry in totals.items():
        den = np.maximum(entry["den"], 1e-30)
        z_column = (entry["num_re"] + 1j * entry["num_im"]) / den
        weights = np.sqrt(den)
        real = np.polyfit(x, z_column.real, v1.POLYNOMIAL_DEGREE, w=weights)
        imag = np.polyfit(x, z_column.imag, v1.POLYNOMIAL_DEGREE, w=weights)
        limited, fraction = v1.clamp_z(np.polyval(real, x) + 1j * np.polyval(imag, x))
        coefficients[name] = [complex(value, imaginary) for value, imaginary in zip(real, imag)]
        per_column[name] = limited
        clamped[name] = fraction
    return {
        "coefficients": coefficients,
        "per_column": per_column,
        "z_shot": complex(z_shot[0]),
        "clamped_fraction": clamped,
        "band_chroma_error_before": float(np.mean(before)),
        "frames_used": len(indices),
        "seconds": time.perf_counter() - started,
        "parameter_count": 2 * 2 * (v1.POLYNOMIAL_DEGREE + 1),
    }


def damp_field(field: np.ndarray, hue_damping: float) -> np.ndarray:
    """Preserve fitted saturation while damping only the fitted hue rotation."""
    return np.abs(field) * np.exp(1j * hue_damping * np.angle(field))


def complex_transform(be: fastcore.Backend, chroma: Any, real: Any, imag: Any) -> Any:
    ct, cp = chroma[..., 0], chroma[..., 1]
    return be.xp.stack((ct * real - cp * imag, ct * imag + cp * real), axis=-1)


def circular_hue_sum(be: fastcore.Backend, predicted: Any, reference: Any) -> tuple[float, int]:
    """Return hue-distance sum/count after excluding chroma-neutral pixels."""
    predicted_magnitude = be.xp.hypot(predicted[..., 0], predicted[..., 1])
    reference_magnitude = be.xp.hypot(reference[..., 0], reference[..., 1])
    valid = (predicted_magnitude > HUE_CHROMA_FLOOR) & (reference_magnitude > HUE_CHROMA_FLOOR)
    difference = be.xp.abs(
        ((be.xp.degrees(be.xp.arctan2(predicted[..., 1], predicted[..., 0])) - be.xp.degrees(be.xp.arctan2(reference[..., 1], reference[..., 0])) + 180.0) % 360.0) - 180.0
    )
    return float(be.tohost(be.xp.sum(be.xp.where(valid, difference, 0.0)))), int(be.tohost(be.xp.sum(valid)))


def candidate_selection(
    be: fastcore.Backend,
    config: om.Config,
    sdr_cache: np.memmap,
    hdr_cache: np.memmap,
    gain: Any,
    spatial_field: np.ndarray,
    holdout_indices: list[int],
) -> dict[str, Any]:
    """Choose lambda on held-out frames, not on spatial-field training data."""
    _, y1, _, y2 = config.overlap
    parts = {
        "top_band": spatial_field[y1:y1 + v1.SEAM_BAND],
        "bottom_band": spatial_field[y2 - v1.SEAM_BAND:y2],
        "top_edge": spatial_field[y1 - 1],
        "bottom_edge": spatial_field[y2],
    }
    values: list[dict[str, Any]] = []
    started = time.perf_counter()
    for hue_damping in LAMBDA_CANDIDATES:
        transforms = {}
        for name, value in parts.items():
            damped = damp_field(value, hue_damping)
            transforms[name] = (
                be.asarray(damped.real.astype(np.float32)),
                be.asarray(damped.imag.astype(np.float32)),
            )
        residual_sum = 0.0
        residual_count = 0
        hue_sum = 0.0
        hue_count = 0
        for index in holdout_indices:
            sdr = be.asarray(sdr_cache[index])
            hdr = be.asarray(hdr_cache[index])
            predicted = v1.predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
            predicted_chroma = be.blur(be.to_ictcp(predicted * fastcore.PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
            hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
            for part, predicted_band, reference_band in (
                ("top_band", predicted_chroma[y1:y1 + v1.SEAM_BAND], hdr_chroma[:v1.SEAM_BAND]),
                ("bottom_band", predicted_chroma[y2 - v1.SEAM_BAND:y2], hdr_chroma[-v1.SEAM_BAND:]),
            ):
                corrected = complex_transform(be, predicted_band, *transforms[part])
                residual_sum += float(be.tohost(be.xp.sum(be.xp.abs(reference_band - corrected))))
                residual_count += int(reference_band.size)
            top_corrected = complex_transform(be, predicted_chroma[y1 - 1], *transforms["top_edge"])
            bottom_corrected = complex_transform(be, predicted_chroma[y2], *transforms["bottom_edge"])
            for corrected, reference in ((top_corrected, hdr_chroma[0]), (bottom_corrected, hdr_chroma[-1])):
                current_sum, current_count = circular_hue_sum(be, corrected, reference)
                hue_sum += current_sum
                hue_count += current_count
        values.append(
            {
                "lambda": hue_damping,
                "lowpass_band_chroma_error": residual_sum / max(residual_count, 1),
                "lowpass_boundary_hue_step_degrees": hue_sum / max(hue_count, 1),
                "hue_valid_samples": hue_count,
            }
        )
    baseline = next(value for value in values if value["lambda"] == 1.0)
    residual_limit = baseline["lowpass_band_chroma_error"] * (1.0 + MAX_RESIDUAL_REGRESSION_FRACTION)
    eligible = [value for value in values if value["lowpass_band_chroma_error"] <= residual_limit]
    selected = min(eligible, key=lambda value: (value["lowpass_boundary_hue_step_degrees"], -value["lambda"]))
    return {
        "candidates": values,
        "selected": selected,
        "baseline_lambda_one": baseline,
        "residual_limit": residual_limit,
        "residual_regression_budget_fraction": MAX_RESIDUAL_REGRESSION_FRACTION,
        "holdout_frames": len(holdout_indices),
        "seconds": time.perf_counter() - started,
    }


def apply_chroma_with_range(be: fastcore.Backend, predicted: Any, real: Any, imag: Any) -> tuple[Any, Any, float, float]:
    """Same frozen chroma decomposition as v1, with unclipped range diagnostics."""
    ictcp = be.to_ictcp(predicted * fastcore.PEAK_NITS)
    chroma = ictcp[..., 1:]
    base = be.blur(chroma, v1.CHROMA_SIGMA)
    detail = chroma - base
    corrected = complex_transform(be, base, real, imag)
    ictcp[..., 1:] = corrected + detail
    raw = be.from_ictcp(ictcp) / fastcore.PEAK_NITS
    negative = float(be.tohost(be.xp.mean(raw < -1e-4)))
    above_peak = float(be.tohost(be.xp.mean(raw > 1.0001)))
    return be.xp.clip(raw, 0.0, 1.0), corrected, negative, above_peak


def seam_metrics(be: fastcore.Backend, output: Any, config: om.Config) -> dict[str, float]:
    """Masked instantaneous hue diagnostics at the two HDR/extension boundaries."""
    _, y1, _, y2 = config.overlap
    metrics: dict[str, float] = {}
    for name, outside, inside in (("top", y1 - 1, y1), ("bottom", y2 - 1, y2)):
        pair = be.to_ictcp(output[[outside, inside]] * fastcore.PEAK_NITS)[..., 1:]
        magnitudes = be.xp.hypot(pair[..., 0], pair[..., 1])
        metrics[f"{name}_chroma_step"] = float(be.tohost(be.xp.mean(be.xp.abs(magnitudes[0] - magnitudes[1]))))
        hue_sum, hue_count = circular_hue_sum(be, pair[0], pair[1])
        metrics[f"{name}_hue_step_degrees"] = hue_sum / max(hue_count, 1)
        metrics[f"{name}_hue_valid_fraction"] = hue_count / float(pair.shape[1])
    return metrics


def region_sheet(path: Path, seam: int, samples: dict[str, np.ndarray], height: int) -> None:
    top = max(0, min(height - 270, seam - 135))
    rows = []
    for name, x in v1.REGIONS:
        tiles = []
        for variant in ("spatial_field", "hue_damped"):
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
    damped_field: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Render the frozen spatial field versus the one-scalar damped variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = config.om_size
    _, y1, _, y2 = config.overlap
    spatial_real, spatial_imag = be.asarray(spatial_field.real.astype(np.float32)), be.asarray(spatial_field.imag.astype(np.float32))
    damped_real, damped_imag = be.asarray(damped_field.real.astype(np.float32)), be.asarray(damped_field.imag.astype(np.float32))
    outputs = {
        "spatial_field": output_dir / "v02_spatial_chroma_hdr10.mkv",
        "hue_damped": output_dir / "v03_hue_damped_chroma_hdr10.mkv",
        "side_by_side": output_dir / "side_by_side_hdr10.mkv",
    }
    pipes = [
        v1.nvenc(config, outputs["spatial_field"], width, height),
        v1.nvenc(config, outputs["hue_damped"], width, height),
        v1.nvenc(config, outputs["side_by_side"], width * 2, height),
    ]
    writer = fastcore.ParallelWriter(pipes)
    sample_index = int(round((len(sdr_cache) - 1) * 0.5))
    samples: dict[str, np.ndarray] = {}
    metrics_rows: dict[str, list[dict[str, float]]] = {"spatial_field": [], "hue_damped": []}
    band_residuals: dict[str, list[float]] = {"spatial_field": [], "hue_damped": []}
    negative_totals = {"spatial_field": 0.0, "hue_damped": 0.0}
    above_peak_totals = {"spatial_field": 0.0, "hue_damped": 0.0}
    started = time.perf_counter()
    try:
        for index in range(len(sdr_cache)):
            sdr = be.asarray(sdr_cache[index])
            hdr = be.asarray(hdr_cache[index])
            predicted = v1.predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
            spatial_rgb, spatial_chroma, spatial_negative, spatial_above_peak = apply_chroma_with_range(be, predicted, spatial_real, spatial_imag)
            damped_rgb, damped_chroma, damped_negative, damped_above_peak = apply_chroma_with_range(be, predicted, damped_real, damped_imag)
            rendered = {
                "spatial_field": v1.composite(be, config, spatial_rgb, hdr),
                "hue_damped": v1.composite(be, config, damped_rgb, hdr),
            }
            pq_spatial, pq_damped = be.to_pq16(rendered["spatial_field"]), be.to_pq16(rendered["hue_damped"])
            writer.submit([pq_spatial.tobytes(), pq_damped.tobytes(), np.concatenate((pq_spatial, pq_damped), axis=1).tobytes()])
            negative_totals["spatial_field"] += spatial_negative
            negative_totals["hue_damped"] += damped_negative
            above_peak_totals["spatial_field"] += spatial_above_peak
            above_peak_totals["hue_damped"] += damped_above_peak
            if index % DIAGNOSTIC_STRIDE == 0:
                hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], v1.CHROMA_SIGMA)
                for name, image, corrected_chroma in (
                    ("spatial_field", rendered["spatial_field"], spatial_chroma),
                    ("hue_damped", rendered["hue_damped"], damped_chroma),
                ):
                    band_residuals[name].append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_chroma[:v1.SEAM_BAND] - corrected_chroma[y1:y1 + v1.SEAM_BAND])))))
                    band_residuals[name].append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_chroma[-v1.SEAM_BAND:] - corrected_chroma[y2 - v1.SEAM_BAND:y2])))))
                    metrics_rows[name].append(seam_metrics(be, image, config))
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
    pair = np.concatenate(
        [np.round(om.preview(cv2.resize(samples[name], (960, 540), interpolation=cv2.INTER_AREA)) * 255.0).astype(np.uint8) for name in ("spatial_field", "hue_damped")],
        axis=1,
    )
    om.write_png(output_dir / "full_frame_pair.png", pair)
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
    selection = payload["selection"]
    selected = selection["selected"]
    result = payload["result"]["metrics"]
    baseline = result["spatial_field"]
    damped = result["hue_damped"]
    lines = [
        "# PoC v2 — bounded hue damping", "",
        "This is the next iteration of the frozen low-frequency ICtCp chroma field. The 12 fitted spatial controls, sigma 16 px base/detail split, HDR-centre copy, and static-per-shot behavior are unchanged.", "",
        "## One additional control", "",
        "The original spatial complex field is transformed as `abs(z) * exp(i * lambda * arg(z))`. Saturation correction is preserved; `lambda` damps only hue rotation. It adds **one** scalar and no spatial degrees of freedom.", "",
        f"Training / held-out split: frame index modulo {FOLD_MODULO}; held-out remainder {HOLDOUT_REMAINDER}. The spatial field used `{payload['spatial_field']['frames_used']}` training frames; lambda selection used `{selection['holdout_frames']}` unseen frames.",
        f"Selected `lambda = {selected['lambda']:.2f}`. Its held-out low-pass boundary hue step was `{selected['lowpass_boundary_hue_step_degrees']:.4f}°`; permitted chroma-residual ceiling was `{selection['residual_limit']:.8f}` (lambda=1 plus {selection['residual_regression_budget_fraction'] * 100:.1f}%).", "",
        "## Full render diagnostics", "",
        "| Quantity | Frozen spatial field | Hue-damped field |",
        "|---|---:|---:|",
        f"| Low-pass seam-band chroma disagreement | {baseline['lowpass_band_chroma_error']:.8f} | {damped['lowpass_band_chroma_error']:.8f} |",
        f"| Top instantaneous hue step | {baseline['seam']['top_hue_step_degrees']:.4f}° | {damped['seam']['top_hue_step_degrees']:.4f}° |",
        f"| Bottom instantaneous hue step | {baseline['seam']['bottom_hue_step_degrees']:.4f}° | {damped['seam']['bottom_hue_step_degrees']:.4f}° |",
        f"| Negative RGB fraction before clip | {baseline['negative_rgb_fraction_before_clip']:.8f} | {damped['negative_rgb_fraction_before_clip']:.8f} |",
        f"| Above-peak RGB fraction before clip | {baseline['above_peak_rgb_fraction_before_clip']:.8f} | {damped['above_peak_rgb_fraction_before_clip']:.8f} |",
        "",
        "## Review artifacts", "",
        "- `v02_spatial_chroma_hdr10.mkv` — frozen spatial-field baseline",
        "- `v03_hue_damped_chroma_hdr10.mkv` — one-scalar hue-damped candidate",
        "- `side_by_side_hdr10.mkv` — baseline left, candidate right",
        "- `top_seam_regions.png`, `bottom_seam_regions.png`, `full_frame_pair.png`", "",
        "The extensions still have no HDR ground truth. Use the HDR10 media to judge the actual criterion: whether paper, skin, walls, clothing, and saturated elements merge more naturally without creating a hue cutoff. The metrics constrain the experiment; they do not establish a recovered creative grade.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(prog="fast-chroma-field-hue-damped")
    parser.add_argument("--max-frames", type=int, default=None, help="Render the first N frames for a smoke test")
    parser.add_argument("--stride", type=int, default=1, help="Sampling stride for the chroma train/held-out split")
    parser.add_argument("--cpu", action="store_true", help="Force the multicore NumPy/OpenCV fallback")
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
    training_indices, holdout_indices = sampled_split(count, args.stride)
    be = fastcore.Backend(use_gpu=not args.cpu and fastcore.gpu_available())
    cache = v1.cache_for(config, count)
    print(f"hue-damped chroma field: {count} frames, backend {be.name}, training {len(training_indices)}, hold-out {len(holdout_indices)}")
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
    print(f"gain pass {gain['seconds']:.1f} s: {gain['shot_gain_stops']:+.4f} stops")
    chroma = measure_spatial_field(be, config, sdr_cache, hdr_cache, gain_field, training_indices)
    spatial_field = v1.chroma_field(config, chroma)
    selection = candidate_selection(be, config, sdr_cache, hdr_cache, gain_field, spatial_field, holdout_indices)
    selected_lambda = float(selection["selected"]["lambda"])
    damped = damp_field(spatial_field, selected_lambda)
    print(f"spatial fit {chroma['seconds']:.1f} s; held-out lambda search {selection['seconds']:.1f} s")
    print(f"selected lambda {selected_lambda:.2f}; hue {selection['selected']['lowpass_boundary_hue_step_degrees']:.4f} deg; chroma residual {selection['selected']['lowpass_band_chroma_error']:.8f}")
    result = render(be, config, sdr_cache, hdr_cache, gain_field, spatial_field, damped, args.output_dir)
    elapsed = time.perf_counter() - total
    print(f"render {result['seconds']:.1f} s ({result['seconds'] / count:.3f} s/frame); total {elapsed:.1f} s")
    payload = {
        "schema": "openmatte-hdr-hue-damped-chroma-field/v1",
        "backend": be.name,
        "frames": count,
        "stride": args.stride,
        "split": {"fold_modulo": FOLD_MODULO, "holdout_remainder": HOLDOUT_REMAINDER, "training_indices": training_indices, "holdout_indices": holdout_indices},
        "cache": {"gigabytes": cache.gigabytes, "reused": decode_seconds == 0.0, "source_frame_hashes": manifest["source_frame_hashes"]},
        "timing_seconds": {"decode_cache": decode_seconds, "gain_pass": gain["seconds"], "spatial_fit": chroma["seconds"], "lambda_selection": selection["seconds"], "render": result["seconds"], "total": elapsed},
        "gain": {key: value for key, value in gain.items() if not key.endswith("_profile_stops")},
        "spatial_field": {
            "parameter_count": chroma["parameter_count"],
            "frames_used": chroma["frames_used"],
            "fit_seconds": chroma["seconds"],
            "z_shot": [chroma["z_shot"].real, chroma["z_shot"].imag],
            "clamped_fraction": chroma["clamped_fraction"],
            "band_chroma_error_before": chroma["band_chroma_error_before"],
            "coefficients_highest_order_first": {name: [[value.real, value.imag] for value in values] for name, values in chroma["coefficients"].items()},
        },
        "selection": selection,
        "hue_damping": {"selected_lambda": selected_lambda, "additional_fitted_controls": 1, "formula": "abs(z) * exp(i * lambda * arg(z))", "preserves_saturation_factor": True},
        "result": result,
        "guarantees": {"temporal_state": False, "ai": False, "lut": False, "mmr": False, "hdr_centre_regenerated": False, "intensity_modified_by_chroma_step": False, "chroma_detail_modified": False},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hue_damped_chroma_field.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "HUE_DAMPING_POC.md").write_text(markdown_report(payload), encoding="utf-8")
    print(args.output_dir / "HUE_DAMPING_POC.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
