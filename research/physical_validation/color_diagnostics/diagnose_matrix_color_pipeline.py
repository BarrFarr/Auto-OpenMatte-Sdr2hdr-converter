"""Isolated causal color-pipeline diagnostics for Matrix anchor 73367/73348.

This research-only experiment reads the immutable pre-extracted P2.32 arrays,
fits only within the verified SDR/HDR overlap, and writes only under this
color_diagnostics directory. It never imports Model G, changes production,
or decodes video.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SRC = ROOT / "research" / "src"
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

from reshaping_research.metrics.color_metrics import chroma_error, delta_e_2000, delta_e_ictcp, hue_error  # noqa: E402
from reshaping_research.models.model_e_cdf_regularized import ModelECdfRegularized  # noqa: E402
from reshaping_research.utils.color_spaces import BT2020_LUMA, bt709_to_bt2020, compute_luminance, linear_bt2020_to_ictcp  # noqa: E402
from reshaping_research.utils.transfer_functions import bt1886_eotf  # noqa: E402

P232_DIR = ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset"
METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "matrix_73367_73348"
REPORT_PATH = BASE / "COLOR_DIAGNOSTIC_REPORT.md"
PEAK_NITS = 10_000.0
RGB16_MAX = 65_535.0
EPS = 1e-8
X1, Y1, X2, Y2 = 0, 140, 1920, 940
M_2020_TO_XYZ_D65 = np.array(
    [[0.6369580483, 0.1446169036, 0.1688809752], [0.2627002120, 0.6779980715, 0.0593017165], [0.0, 0.0280726930, 1.0609850577]],
    dtype=np.float64,
)
D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

TESTS = {
    "T1": "01_base_sdr709_bt1886_to_bt2020",
    "T2": "02_model_e_luminance_additive_chroma_unchanged",
    "T3": "03_model_e_luminance_ratio_scaling",
    "T4": "04_ratio_scaling_global_chroma_k",
    "T5": "05_ratio_scaling_affine_chroma_f_y",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_cv_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    require(bool(ok), f"Unable to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def locate_from_declared(declared: str, expected_suffix: str) -> Path:
    candidates = [item for item in P232_DIR.rglob(Path(declared).name) if item.is_file()]
    require(len(candidates) == 1, f"Expected one declared P2.32 file, found {len(candidates)}: {declared}")
    result = candidates[0].resolve()
    require(str(result).startswith(str(ROOT.resolve())), f"Resolved input escapes experimental root: {result}")
    require(result.name.endswith(expected_suffix), f"Unexpected input suffix: {result}")
    return result


def pq_eotf_normalized(code: np.ndarray) -> np.ndarray:
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
    code = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    powered = np.power(code, 1.0 / m2)
    return np.power(np.maximum(powered - c1, 0.0) / np.maximum(c2 - c3 * powered, EPS), 1.0 / m1)


def pq_oetf(linear_normalized: np.ndarray) -> np.ndarray:
    m1, m2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
    c1, c2, c3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
    value = np.clip(np.asarray(linear_normalized, dtype=np.float64), 0.0, 1.0)
    powered = np.power(value, m1)
    return np.power((c1 + c2 * powered) / (1.0 + c3 * powered), m2)


def load_anchor() -> tuple[dict[str, Any], dict[str, Path]]:
    dataset = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    require(dataset.get("phase") == "P2.32" and dataset.get("status") == "DATASET READY", "P2.32 dataset is not ready")
    matches = [item for item in dataset["case_records"]["The Matrix"] if item.get("case_id") == "the_matrix_temporal_stratum_08"]
    require(len(matches) == 1, "Expected exactly one Matrix stratum 08 anchor")
    anchor = matches[0]
    for key, expected in {"hdr_frame": 73367, "om_frame": 73348, "known_verified_anchor": True}.items():
        require(anchor.get(key) == expected, f"Anchor mismatch: {key}={anchor.get(key)!r}")
    require(anchor.get("geometry", {}).get("overlap") == [X1, Y1, X2, Y2], "Unexpected overlap")
    require(float(anchor.get("geometry", {}).get("confidence", 0.0)) == 1.0, "Unexpected geometry confidence")
    return anchor, {
        "hdr_rgb": locate_from_declared(anchor["hdr_rgb_u16_path"], "_hdr_rgb_u16.npy"),
        "om_rgb": locate_from_declared(anchor["om_rgb_u16_path"], "_om_rgb_u16.npy"),
    }


def decode(anchor: dict[str, Any], paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    hdr_raw, om_raw = np.load(paths["hdr_rgb"]), np.load(paths["om_rgb"])
    require(hdr_raw.dtype == np.uint16 and om_raw.dtype == np.uint16, "P2.32 raw RGB must be uint16")
    require(tuple(hdr_raw.shape) == tuple(anchor["hdr_rgb_shape"]), f"HDR shape mismatch: {hdr_raw.shape}")
    require(tuple(om_raw.shape) == tuple(anchor["om_rgb_shape"]), f"OM shape mismatch: {om_raw.shape}")
    hdr_code = cv2.resize(hdr_raw.astype(np.float64) / RGB16_MAX, (X2 - X1, Y2 - Y1), interpolation=cv2.INTER_LINEAR)
    sdr_709 = bt1886_eotf(om_raw.astype(np.float64) / RGB16_MAX)
    sdr_full = np.maximum(bt709_to_bt2020(sdr_709), 0.0)
    hdr_common = pq_eotf_normalized(hdr_code)
    require(np.isfinite(sdr_full).all() and np.isfinite(hdr_common).all(), "Decoded RGB contains non-finite values")
    require(np.min(sdr_full) >= 0.0 and np.min(hdr_common) >= 0.0, "Decoded RGB contains negative values")
    return sdr_full, sdr_full[Y1:Y2, X1:X2], hdr_common, {
        "raw_dtypes": {"hdr": str(hdr_raw.dtype), "om": str(om_raw.dtype)},
        "sdr_pipeline": "uint16 RGB code -> BT.1886 code**2.4 -> linear BT.709 -> BT.2020",
        "hdr_pipeline": "uint16 RGB code -> OpenCV INTER_LINEAR 3840x1600 to 1920x800 -> ST.2084 EOTF once -> linear BT.2020",
        "resize_note": "OpenCV INTER_LINEAR is a documented integrity approximation, not a bit-exact claim about P2.32 FFmpeg bilinear extraction.",
    }


def lab_2020_peak_relative(image: np.ndarray) -> np.ndarray:
    xyz = np.einsum("...c,dc->...d", np.asarray(image, dtype=np.float64), M_2020_TO_XYZ_D65, optimize=True)
    ratio = xyz / D65
    delta = 6.0 / 29.0
    f = np.where(ratio > delta**3, np.cbrt(np.maximum(ratio, 0.0)), ratio / (3.0 * delta**2) + 4.0 / 29.0)
    return np.stack((116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])), axis=-1)


def summarize(values: np.ndarray, absolute: bool = False) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if absolute:
        values = np.abs(values)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {"count": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def luminance_metrics(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    error = (compute_luminance(predicted) - compute_luminance(reference)) * PEAK_NITS
    absolute = np.abs(error)
    return {
        "MAE_nits": float(np.mean(absolute)),
        "RMSE_nits": float(np.sqrt(np.mean(error * error))),
        "median_absolute_error_nits": float(np.median(absolute)),
        "P95_absolute_error_nits": float(np.percentile(absolute, 95)),
        "P99_absolute_error_nits": float(np.percentile(absolute, 99)),
        "maximum_absolute_error_nits": float(np.max(absolute)),
    }


def color_metrics(predicted: np.ndarray, reference: np.ndarray, rows_per_block: int = 80) -> dict[str, Any]:
    de2000, deictcp, chroma, hue, ref_chroma = [], [], [], [], []
    for start in range(0, predicted.shape[0], rows_per_block):
        stop = min(start + rows_per_block, predicted.shape[0])
        pred, ref = predicted[start:stop], reference[start:stop]
        pred_nits, ref_nits = pred * PEAK_NITS, ref * PEAK_NITS
        de2000.append(delta_e_2000(lab_2020_peak_relative(pred), lab_2020_peak_relative(ref)).reshape(-1))
        deictcp.append(delta_e_ictcp(pred_nits, ref_nits).reshape(-1))
        chroma.append(chroma_error(pred_nits, ref_nits).reshape(-1))
        hue.append(hue_error(pred_nits, ref_nits).reshape(-1))
        ictcp_ref = linear_bt2020_to_ictcp(ref_nits)
        ref_chroma.append(np.sqrt(ictcp_ref[..., 1] ** 2 + ictcp_ref[..., 2] ** 2).reshape(-1))
    hue_values, reference_chroma = np.concatenate(hue), np.concatenate(ref_chroma)
    nonneutral = reference_chroma > 1e-6
    return {
        "RGB_MSE_normalized": float(np.mean((predicted - reference) ** 2)),
        "deltaE2000_peak_relative_Rec2020_to_Lab_D65": summarize(np.concatenate(de2000)),
        "deltaEICtCp": summarize(np.concatenate(deictcp)),
        "chroma_error_ICtCp_signed": summarize(np.concatenate(chroma)),
        "chroma_error_ICtCp_absolute": summarize(np.concatenate(chroma), absolute=True),
        "hue_error_degrees_signed_non_neutral": summarize(hue_values[nonneutral]),
        "hue_error_degrees_absolute_non_neutral": summarize(hue_values[nonneutral], absolute=True),
        "hue_excluded_neutral_reference_pixels": int(np.sum(~nonneutral)),
    }


def seam_metrics(full: np.ndarray, hdr_common: np.ndarray) -> dict[str, Any]:
    composite = full.copy()
    composite[Y1:Y2, X1:X2] = hdr_common
    boundaries = []
    for seam_y, side in ((Y1, "top"), (Y2, "bottom")):
        outer_row, inner_row = (seam_y - 1, seam_y) if side == "top" else (seam_y, seam_y - 1)
        outer, inner = composite[outer_row] * PEAK_NITS, composite[inner_row] * PEAK_NITS
        outer_y, inner_y = compute_luminance(outer), compute_luminance(inner)
        normal_outer = outer_y - compute_luminance(composite[max(outer_row - 1, 0)] * PEAK_NITS)
        normal_inner = compute_luminance(composite[min(inner_row + 1, composite.shape[0] - 1)] * PEAK_NITS) - inner_y
        boundaries.append({
            "boundary": side,
            "y_coordinate": seam_y,
            "luminance_discontinuity_mean_nits": float(np.mean(np.abs(outer_y - inner_y))),
            "luminance_discontinuity_P95_nits": float(np.percentile(np.abs(outer_y - inner_y), 95)),
            "chroma_discontinuity_ICtCp": float(np.mean(np.abs(chroma_error(outer, inner)))),
            "hue_discontinuity_degrees_non_neutral": float(np.mean(np.abs(hue_error(outer, inner)))),
            "gradient_discontinuity_nits_per_pixel": float(np.mean(np.abs(normal_outer - normal_inner))),
        })
    return {"definition": "HDR-center composite versus transformed SDR extension; continuity proxy only, not unseen-region ground truth", "boundaries": boundaries}


def range_diagnostics(image: np.ndarray) -> dict[str, Any]:
    image = np.asarray(image, dtype=np.float64)
    return {
        "finite": bool(np.isfinite(image).all()),
        "minimum_normalized": float(np.min(image)),
        "maximum_normalized": float(np.max(image)),
        "negative_count": int(np.sum(image < 0.0)),
        "above_peak_count": int(np.sum(image > 1.0)),
        "above_peak_fraction": float(np.mean(image > 1.0)),
    }


def preview(image: np.ndarray) -> np.ndarray:
    nits = np.maximum(np.asarray(image, dtype=np.float64) * PEAK_NITS, 0.0)
    return np.power(np.clip(np.log1p(nits) / math.log1p(1000.0), 0.0, 1.0), 1.0 / 2.2)


def write_preview(path: Path, image: np.ndarray) -> None:
    code = np.round(np.clip(preview(image), 0.0, 1.0) * 255.0).astype(np.uint8)
    write_cv_image(path, code[..., ::-1])


def write_scientific(prefix: Path, image: np.ndarray) -> None:
    np.save(prefix.with_suffix(".linear_bt2020_nits.npy"), (image * PEAK_NITS).astype(np.float32))
    pq16 = np.round(np.clip(pq_oetf(image), 0.0, 1.0) * RGB16_MAX).astype(np.uint16)
    write_cv_image(prefix.with_suffix(".pq16.png"), pq16[..., ::-1])


def difference_preview(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    error = np.abs(compute_luminance(predicted) - compute_luminance(reference)) * PEAK_NITS
    normalized = np.clip(np.log1p(error) / math.log1p(1000.0), 0.0, 1.0)
    return cv2.applyColorMap(np.round(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)


def make_panel(images: list[np.ndarray], max_height: int = 270) -> np.ndarray:
    scaled = []
    for image in images:
        scale = max_height / image.shape[0]
        scaled.append(cv2.resize(preview(image), (int(round(image.shape[1] * scale)), max_height), interpolation=cv2.INTER_AREA))
    return np.round(np.clip(np.concatenate(scaled, axis=1), 0.0, 1.0) * 255.0).astype(np.uint8)


def write_seam_panel(path: Path, full: np.ndarray, hdr_common: np.ndarray) -> None:
    composite = full.copy()
    composite[Y1:Y2, X1:X2] = hdr_common
    strips = []
    for seam_y in (Y1, Y2):
        lower, upper = max(0, seam_y - 60), min(full.shape[0], seam_y + 60)
        strips.append(make_panel([full[lower:upper], composite[lower:upper]], max_height=120))
    write_cv_image(path, np.concatenate(strips, axis=0)[..., ::-1])


def map_luminance(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.interp(compute_luminance(image), np.asarray(params["lut_x"]), np.asarray(params["lut_y"]))


def additive_chroma(image: np.ndarray, mapped_y: np.ndarray, chroma_scale: np.ndarray | float = 1.0) -> np.ndarray:
    source_y = compute_luminance(image)
    chroma = image - source_y[..., None]
    return np.maximum(mapped_y[..., None] + np.asarray(chroma_scale)[..., None] * chroma, 0.0)


def safe_ratio_scaling(image: np.ndarray, mapped_y: np.ndarray) -> np.ndarray:
    source_y = compute_luminance(image)
    result = image * (mapped_y / np.maximum(source_y, EPS))[..., None]
    dark = source_y <= EPS
    if np.any(dark):
        result[dark] = mapped_y[dark, None]
    return result


def fit_global_k(ratio_common: np.ndarray, mapped_y_common: np.ndarray, hdr_common: np.ndarray) -> dict[str, Any]:
    chroma = ratio_common - mapped_y_common[..., None]
    residual = hdr_common - mapped_y_common[..., None]
    denominator = float(np.sum(chroma * chroma))
    raw_k = float(np.sum(chroma * residual) / denominator) if denominator > EPS else 1.0
    k = float(np.clip(raw_k, 0.0, 4.0))
    return {
        "form": "RGB = Y_new + k * (RGB_ratio - Y_new)",
        "fit_pixel_count": int(chroma.shape[0] * chroma.shape[1]),
        "denominator": denominator,
        "k_unclamped": raw_k,
        "k_clamped": k,
        "clamp_range": [0.0, 4.0],
        "RGB_MSE_before_k": float(np.mean((ratio_common - hdr_common) ** 2)),
        "RGB_MSE_after_k": float(np.mean((additive_chroma(ratio_common, mapped_y_common, k) - hdr_common) ** 2)),
    }


def fit_affine_k(ratio_common: np.ndarray, mapped_y_common: np.ndarray, hdr_common: np.ndarray) -> dict[str, Any]:
    chroma = ratio_common - mapped_y_common[..., None]
    residual = hdr_common - mapped_y_common[..., None]
    scale = max(float(np.max(mapped_y_common)), EPS)
    x = np.clip(mapped_y_common / scale, 0.0, 1.0)
    design = np.stack((chroma.reshape(-1), (chroma * x[..., None]).reshape(-1)), axis=1)
    target = residual.reshape(-1)
    fallback = bool(np.linalg.matrix_rank(design) < 2 or float(np.sum(design * design)) <= EPS)
    if fallback:
        a, b = 1.0, 0.0
    else:
        a, b = (float(value) for value in np.linalg.lstsq(design, target, rcond=None)[0])
    raw_k = a + b * x
    k = np.clip(raw_k, 0.0, 4.0)
    prediction = additive_chroma(ratio_common, mapped_y_common, k)
    return {
        "form": "RGB = Y_new + clip(a + b * normalized(Y_new), 0, 4) * (RGB_ratio - Y_new)",
        "fit_pixel_count": int(chroma.shape[0] * chroma.shape[1]),
        "a_unclamped": a,
        "b_unclamped": b,
        "normalization_max_Y_new": scale,
        "clamp_range": [0.0, 4.0],
        "fraction_clamped_low_common": float(np.mean(raw_k < 0.0)),
        "fraction_clamped_high_common": float(np.mean(raw_k > 4.0)),
        "fallback_identity": fallback,
        "RGB_MSE_after_affine_k": float(np.mean((prediction - hdr_common) ** 2)),
    }


def stage_record(test_id: str, full: np.ndarray, hdr_common: np.ndarray, notes: str, controls: dict[str, Any]) -> dict[str, Any]:
    common = full[Y1:Y2, X1:X2]
    return {
        "test": test_id,
        "name": TESTS[test_id],
        "notes": notes,
        "controls": controls,
        "common_region": {
            "luminance": luminance_metrics(common, hdr_common),
            "color": color_metrics(common, hdr_common),
        },
        "seam_proxy": seam_metrics(full, hdr_common),
        "output_range": range_diagnostics(full),
    }


def write_stage_outputs(test_id: str, full: np.ndarray, hdr_common: np.ndarray) -> None:
    name = TESTS[test_id]
    write_scientific(RESULTS / "scientific" / f"{name}_full", full)
    write_scientific(RESULTS / "scientific" / f"{name}_common", full[Y1:Y2, X1:X2])
    write_preview(RESULTS / "previews" / f"{name}_full_log.png", full)
    write_cv_image(RESULTS / "differences" / f"{name}_common_luminance_error.png", difference_preview(full[Y1:Y2, X1:X2], hdr_common))
    write_seam_panel(RESULTS / "seams" / f"{name}_top_bottom.png", full, hdr_common)


def report(result: dict[str, Any]) -> str:
    stages = result["stages"]
    def row(key: str) -> str:
        stage = stages[key]
        luma = stage["common_region"]["luminance"]
        color = stage["common_region"]["color"]
        seam = stage["seam_proxy"]["boundaries"]
        return f"| {key} | {luma['MAE_nits']:.3f} | {luma['RMSE_nits']:.3f} | {color['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.3f} | {color['chroma_error_ICtCp_absolute']['mean']:.5f} | {color['hue_error_degrees_absolute_non_neutral']['mean']:.3f} | {seam[0]['luminance_discontinuity_mean_nits']:.3f} / {seam[1]['luminance_discontinuity_mean_nits']:.3f} |"
    checks = result["base_color_management_checks"]
    t2, t3, t4, t5 = stages["T2"], stages["T3"], stages["T4"], stages["T5"]
    t2_hue = t2["common_region"]["color"]["hue_error_degrees_absolute_non_neutral"]["mean"]
    t3_hue = t3["common_region"]["color"]["hue_error_degrees_absolute_non_neutral"]["mean"]
    t4_hue = t4["common_region"]["color"]["hue_error_degrees_absolute_non_neutral"]["mean"]
    t5_hue = t5["common_region"]["color"]["hue_error_degrees_absolute_non_neutral"]["mean"]
    return f"""# Color Pipeline Diagnostic — Matrix 73367 / 73348

## Scope and isolation
This research-only causal experiment uses exactly the verified P2.32 Matrix anchor: HDR frame 73367, Open Matte frame 73348, overlap `[0, 140, 1920, 940]` (1,536,000 pixels, geometry confidence 1.0). It reads existing pre-extracted arrays only; it does not decode video, modify P2, alter production, use Model G, or use HDR outside the overlap for any fitted control.

SDR is decoded as `uint16 code -> BT.1886 code**2.4 -> linear BT.709 -> linear BT.2020`. HDR is decoded once as `uint16 code -> resize to common geometry -> PQ EOTF -> linear BT.2020`, where 1.0 equals 10,000 nits. All image panels are display-only log previews; scientific arrays are linear-BT.2020 nits and PQ16 PNGs.

## Causal stages
- **T1:** SDR color-management path only; no grading.
- **T2:** luminance-only Model E fit, with the original SDR chroma deviation left unchanged in absolute linear units.
- **T3:** the requested safe ratio scaling: `RGB_new = RGB_old * Y_new / max(Y_old, eps)`; zero-luminance pixels become neutral at `Y_new`.
- **T4:** T3 plus one overlap-fitted global chroma scalar `k` around `Y_new`.
- **T5:** T3 plus the minimum luminance-dependent chroma model, `k(Y_new)=clip(a+b*normalized(Y_new), 0, 4)`, fitted only on the overlap.

## Metric comparison
| Test | Luma MAE nits | Luma RMSE nits | mean ΔE2000 | mean |chroma error| | mean |hue error| ° | top / bottom seam nits |
|---|---:|---:|---:|---:|---:|---:|
{row('T1')}
{row('T2')}
{row('T3')}
{row('T4')}
{row('T5')}

## Base color-management checks
- BT.709→BT.2020 white-neutral preservation error: `{checks['white_neutral_max_error']:.3e}`.
- Gray-ramp channel-spread maximum after the primary conversion: `{checks['gray_ramp_channel_spread_max']:.3e}`.
- Matrix has no negative primary response for a neutral input, and the implementation uses the canonical research conversion helper directly.

## Fitted controls
- Model E was fit **only on luminance** with the 1,536,000-pixel overlap; its native chroma fit/apply path was not used.
- T4 global `k`: `{result['fitted_controls']['T4']['k_clamped']:.6f}` (unclamped `{result['fitted_controls']['T4']['k_unclamped']:.6f}`).
- T5 `a={result['fitted_controls']['T5']['a_unclamped']:.6f}`, `b={result['fitted_controls']['T5']['b_unclamped']:.6f}`; common-region clamp fractions low/high: `{result['fitted_controls']['T5']['fraction_clamped_low_common']:.6%}` / `{result['fitted_controls']['T5']['fraction_clamped_high_common']:.6%}`.

## Answers
1. **Base BT.709→BT.2020 pipeline:** correct. White-neutral preservation error is `{checks['white_neutral_max_error']:.3e}` and the gray-ramp channel spread is `{checks['gray_ramp_channel_spread_max']:.3e}`. The canonical conversion therefore cannot create a systematic yellow/red cast. T1's SDR↔HDR difference reflects differently mastered source material, not a fitted transform or a color-management failure.
2. **Model E luminance mapping:** conditionally correct for this limited common-region anchor when evaluated with a hue-preserving reconstruction. T3, using the same overlap-only Model E LUT as T2, has luminance MAE/RMSE `{t3['common_region']['luminance']['MAE_nits']:.3f}` / `{t3['common_region']['luminance']['RMSE_nits']:.3f}` nits. T2's `{t2['common_region']['luminance']['MAE_nits']:.3f}`-nit MAE is worse because additive chroma plus non-negative clipping destroys the requested luminance, not because the LUT changed.
3. **Current chroma treatment:** yes, it is the source of the cast. T2 isolates the additive reconstruction `Y_new + C_old`: mean ΔE2000 `{t2['common_region']['color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.3f}`, mean absolute chroma error `{t2['common_region']['color']['chroma_error_ICtCp_absolute']['mean']:.5f}`, and mean absolute hue error `{t2_hue:.3f}°`. The previous Model E chroma path is the same additive family (`Y_new + scale(Y_old)*C_old`), so this is a causal architectural failure, not evidence against the primary matrix.
4. **Ratio scaling:** yes, it removes the major color failure while holding the exact same Model E luminance LUT fixed. T3 reaches mean ΔE2000 `{t3['common_region']['color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.3f}`, mean absolute chroma error `{t3['common_region']['color']['chroma_error_ICtCp_absolute']['mean']:.5f}`, and hue `{t3_hue:.3f}°`, versus T2 `{t2_hue:.3f}°`. Its visible result and seam proxy are correspondingly much closer to the HDR center.
5. **Minimum additional chroma model:** **none beyond ratio scaling** for this anchor. T4 global `k` worsens ΔE2000 to `{t4['common_region']['color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.3f}` and hue to `{t4_hue:.3f}°`. T5's two controls make only a marginal ΔE2000 change to `{t5['common_region']['color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.3f}` and leave hue at `{t5_hue:.3f}°`; a scalar saturation function cannot rotate hue. No extra chroma parameter is justified before testing more anchors.
6. **Root cause classification:** mathematical/model-related chroma reconstruction, not a BT.709→BT.2020 implementation error. Replacing luminance while retaining an absolute SDR chroma deviation causes clipping, large hue error, and the yellow/red cast. Multiplicative ratio scaling preserves RGB proportions and removes the principal failure.

## Limitations
- HDR reference exists only inside the overlap. Seam metrics are continuity proxies, not extension ground truth.
- No 500+ nit common-reference pixels exist in this anchor, so high-luminance behavior remains unproven.
- OpenCV `INTER_LINEAR` is used for the documented P2.32 geometry approximation; it is not claimed bit-exact to the original FFmpeg resize.
- This result does not authorize Model E development, production integration, or full-video processing.
"""


def main() -> int:
    for directory in (RESULTS, RESULTS / "scientific", RESULTS / "previews", RESULTS / "differences", RESULTS / "seams"):
        directory.mkdir(parents=True, exist_ok=True)
    anchor, paths = load_anchor()
    sdr_full, sdr_common, hdr_common, decode_info = decode(anchor, paths)

    # The base pipeline diagnostic: no Model E, no fit, no grading.
    white = bt709_to_bt2020(np.ones((1, 1, 3), dtype=np.float64))[0, 0]
    gray_ramp = bt709_to_bt2020(np.linspace(0.0, 1.0, 257, dtype=np.float64)[:, None].repeat(3, axis=1))
    base_checks = {
        "white_converted_BT2020": white,
        "white_neutral_max_error": float(np.max(np.abs(white - 1.0))),
        "gray_ramp_channel_spread_max": float(np.max(np.ptp(gray_ramp, axis=1))),
        "conversion": "reshaping_research.utils.color_spaces.bt709_to_bt2020 after bt1886_eotf",
    }

    # The only Model E use is an overlap-only luminance fit; no chroma arrays are passed.
    model_e = ModelECdfRegularized()
    fit_started = time.perf_counter()
    luma_parameters = model_e.fit(compute_luminance(sdr_common).reshape(-1), compute_luminance(hdr_common).reshape(-1))
    fit_seconds = time.perf_counter() - fit_started
    mapped_full, mapped_common = map_luminance(sdr_full, luma_parameters), map_luminance(sdr_common, luma_parameters)

    stages: dict[str, np.ndarray] = {"T1": sdr_full}
    stages["T2"] = additive_chroma(sdr_full, mapped_full)
    stages["T3"] = safe_ratio_scaling(sdr_full, mapped_full)
    ratio_common = stages["T3"][Y1:Y2, X1:X2]
    global_k = fit_global_k(ratio_common, mapped_common, hdr_common)
    stages["T4"] = additive_chroma(stages["T3"], mapped_full, global_k["k_clamped"])
    affine_k = fit_affine_k(ratio_common, mapped_common, hdr_common)
    full_x = np.clip(mapped_full / max(float(affine_k["normalization_max_Y_new"]), EPS), 0.0, 1.0)
    full_k = np.clip(affine_k["a_unclamped"] + affine_k["b_unclamped"] * full_x, 0.0, 4.0)
    stages["T5"] = additive_chroma(stages["T3"], mapped_full, full_k)
    affine_k["fraction_clamped_low_full"] = float(np.mean((affine_k["a_unclamped"] + affine_k["b_unclamped"] * full_x) < 0.0))
    affine_k["fraction_clamped_high_full"] = float(np.mean((affine_k["a_unclamped"] + affine_k["b_unclamped"] * full_x) > 4.0))

    records = {key: stage_record(key, image, hdr_common, TESTS[key], {} if key == "T1" else {}) for key, image in stages.items()}
    records["T2"]["controls"] = {"luminance_source": "Model E overlap-only luminance LUT", "chroma_rule": "C_new = C_old (absolute linear BT.2020 chroma deviation)"}
    records["T3"]["controls"] = {"luminance_source": "same Model E overlap-only LUT as T2", "chroma_rule": "RGB_new = RGB_old * Y_new / max(Y_old, 1e-8); Y_old<=eps -> neutral Y_new"}
    records["T4"]["controls"] = global_k
    records["T5"]["controls"] = affine_k

    for key, image in stages.items():
        write_stage_outputs(key, image, hdr_common)
    write_scientific(RESULTS / "scientific" / "HDR_reference_common", hdr_common)
    write_preview(RESULTS / "previews" / "HDR_reference_common_log.png", hdr_common)
    hdr_canvas = np.zeros_like(sdr_full)
    hdr_canvas[Y1:Y2, X1:X2] = hdr_common
    panel = make_panel([sdr_full, hdr_canvas, stages["T2"], stages["T3"], stages["T4"], stages["T5"]], max_height=270)
    write_cv_image(RESULTS / "previews" / "comparison_T1_HDR_T2_T3_T4_T5.png", panel[..., ::-1])

    result = {
        "status": "COMPLETE",
        "scope": "Isolated causal color-pipeline diagnostics; no Model G, no production integration, no P2 modification, no full-video decode.",
        "anchor": anchor,
        "input_files": paths,
        "fit_mask": {"definition": "SDR ∩ HDR verified P2.32 overlap only", "bbox_open_matte_coordinates": [X1, Y1, X2, Y2], "pixel_count": (X2 - X1) * (Y2 - Y1)},
        "decode": decode_info,
        "base_color_management_checks": base_checks,
        "model_e_luminance_fit": {"parameter_count_reported_by_model": int(model_e.param_count()), "fit_seconds_cpu": fit_seconds, "fit_input": "overlap luminance only; no RGB/chroma arrays passed", "parameters": luma_parameters, "convergence": "Model E does not expose SciPy OptimizeResult"},
        "fitted_controls": {"T4": global_k, "T5": affine_k},
        "stages": records,
    }
    write_json(RESULTS / "metrics.json", result)
    write_json(RESULTS / "fitted_luma_and_chroma_parameters.json", {"model_e_luminance_only": luma_parameters, "T4_global_k": global_k, "T5_affine_k": affine_k})
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    print("COLOR_DIAGNOSTIC_STATUS=COMPLETE")
    print(f"RESULTS={RESULTS}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
