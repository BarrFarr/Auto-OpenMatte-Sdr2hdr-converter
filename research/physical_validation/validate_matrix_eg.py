"""Single-frame physical validation of research Models E and G.

This harness is intentionally isolated from the production pipeline and P2 artifacts.
It uses only the verified P2.32 Matrix anchor 73367/73348 and writes outputs
under research/physical_validation/. It never decodes a movie.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SRC = ROOT / "research" / "src"
SRC = ROOT / "src"
for path in (RESEARCH_SRC, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reshaping_research.metrics.color_metrics import (  # noqa: E402
    chroma_error,
    delta_e_2000,
    delta_e_ictcp,
    hue_error,
)
from reshaping_research.models.model_e_cdf_regularized import ModelECdfRegularized  # noqa: E402
from reshaping_research.models.model_g_hybrid import ModelGHybrid  # noqa: E402

P232_DIR = ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset"
METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
RESULTS = ROOT / "research" / "physical_validation" / "results" / "matrix_73367_73348"
REPORT_PATH = ROOT / "research" / "physical_validation" / "PHYSICAL_VALIDATION_REPORT.md"
PEAK_NITS = 10_000.0
RGB16_MAX = 65_535.0
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)
M_709_TO_2020 = np.array(
    [[0.6274039, 0.3292830, 0.0433131], [0.0690972, 0.9195404, 0.0113624], [0.0163916, 0.0880132, 0.8955952]],
    dtype=np.float64,
)
M_2020_TO_XYZ_D65 = np.array(
    [[0.6369580483, 0.1446169036, 0.1688809752], [0.2627002120, 0.6779980715, 0.0593017165], [0.0, 0.0280726930, 1.0609850577]],
    dtype=np.float64,
)
D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_cv_image(path: Path, image: np.ndarray) -> None:
    """Write an OpenCV-supported image without passing a Unicode path to OpenCV."""
    ok, encoded = cv2.imencode(path.suffix, image)
    require(bool(ok), f"Unable to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def locate_from_declared(declared: str, expected_suffix: str) -> Path:
    """Resolve P2.32 files by manifest basename inside this experimental root only."""
    basename = Path(declared).name
    candidates = [candidate for candidate in P232_DIR.rglob(basename) if candidate.is_file()]
    require(len(candidates) == 1, f"Expected exactly one P2.32 file named {basename!r}; found {len(candidates)}")
    result = candidates[0].resolve()
    require(str(result).startswith(str(ROOT.resolve())), f"Resolved file escapes experimental root: {result}")
    require(result.name.endswith(expected_suffix), f"Unexpected file suffix for {result}")
    return result


def pq_eotf_normalized(code: np.ndarray) -> np.ndarray:
    """ST.2084 EOTF; output 1.0 corresponds to PEAK_NITS."""
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    code = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    power = np.power(code, 1.0 / m2)
    numerator = np.maximum(power - c1, 0.0)
    denominator = np.maximum(c2 - c3 * power, 1e-12)
    return np.power(numerator / denominator, 1.0 / m1)


def pq_oetf(linear_normalized: np.ndarray) -> np.ndarray:
    m1 = 2610.0 / 16384.0
    m2 = 2523.0 / 4096.0 * 128.0
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 4096.0 * 32.0
    c3 = 2392.0 / 4096.0 * 32.0
    luminance = np.clip(np.asarray(linear_normalized, dtype=np.float64), 0.0, 1.0)
    powered = np.power(luminance, m1)
    return np.power((c1 + c2 * powered) / (1.0 + c3 * powered), m2)


def linearize_sdr_709(code: np.ndarray) -> np.ndarray:
    return np.power(np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0), 2.4)


def luminance(rgb_linear_2020: np.ndarray) -> np.ndarray:
    return np.einsum("...c,c->...", rgb_linear_2020, LUMA_2020, optimize=True)


def bt2020_to_lab_peak_relative(rgb_linear_2020: np.ndarray) -> np.ndarray:
    """Convert peak-relative linear BT.2020 RGB to CIE Lab D65 for ΔE2000.

    This is explicitly a peak-relative color-difference diagnostic, not a display
    calibration claim. Inputs are normalized so 1.0 equals 10,000 nits.
    """
    xyz = np.einsum("...c,dc->...d", np.asarray(rgb_linear_2020, dtype=np.float64), M_2020_TO_XYZ_D65, optimize=True)
    ratio = xyz / D65
    delta = 6.0 / 29.0
    f = np.where(ratio > delta**3, np.cbrt(np.maximum(ratio, 0.0)), ratio / (3.0 * delta**2) + 4.0 / 29.0)
    return np.stack((116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])), axis=-1)


def summary(values: np.ndarray, absolute: bool = False) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if absolute:
        values = np.abs(values)
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {"count": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def luminance_metrics(predicted_norm: np.ndarray, reference_norm: np.ndarray) -> dict[str, Any]:
    prediction = luminance(predicted_norm) * PEAK_NITS
    reference = luminance(reference_norm) * PEAK_NITS
    error = prediction - reference
    absolute = np.abs(error)
    output: dict[str, Any] = {
        "global": {
            "count": int(error.size),
            "MAE_nits": float(np.mean(absolute)),
            "RMSE_nits": float(np.sqrt(np.mean(error * error))),
            "median_absolute_error_nits": float(np.median(absolute)),
            "P95_absolute_error_nits": float(np.percentile(absolute, 95)),
            "P99_absolute_error_nits": float(np.percentile(absolute, 99)),
            "maximum_absolute_error_nits": float(np.max(absolute)),
        },
        "by_reference_luminance_nits": {},
    }
    for low, high, name in ((0.0, 10.0, "0-10"), (10.0, 100.0, "10-100"), (100.0, 500.0, "100-500"), (500.0, 1000.0, "500-1000"), (1000.0, float("inf"), ">1000")):
        mask = (reference >= low) & (reference < high if math.isfinite(high) else np.ones_like(reference, dtype=bool))
        selected = absolute[mask]
        output["by_reference_luminance_nits"][name] = {
            "range_nits": [low, None if not math.isfinite(high) else high],
            "count": int(selected.size),
            "MAE_nits": float(np.mean(selected)) if selected.size else None,
            "RMSE_nits": float(np.sqrt(np.mean(error[mask] ** 2))) if selected.size else None,
            "median_absolute_error_nits": float(np.median(selected)) if selected.size else None,
            "P95_absolute_error_nits": float(np.percentile(selected, 95)) if selected.size else None,
        }
    return output


def color_metrics(predicted_norm: np.ndarray, reference_norm: np.ndarray, rows_per_block: int = 80) -> dict[str, Any]:
    """Chunk perceptual calculations to keep physical validation memory bounded."""
    de2000_parts: list[np.ndarray] = []
    deictcp_parts: list[np.ndarray] = []
    chroma_parts: list[np.ndarray] = []
    hue_parts: list[np.ndarray] = []
    for start in range(0, predicted_norm.shape[0], rows_per_block):
        stop = min(start + rows_per_block, predicted_norm.shape[0])
        pred = predicted_norm[start:stop]
        ref = reference_norm[start:stop]
        de2000_parts.append(delta_e_2000(bt2020_to_lab_peak_relative(pred), bt2020_to_lab_peak_relative(ref)).reshape(-1))
        pred_nits = pred * PEAK_NITS
        ref_nits = ref * PEAK_NITS
        deictcp_parts.append(delta_e_ictcp(pred_nits, ref_nits).reshape(-1))
        chroma_parts.append(chroma_error(pred_nits, ref_nits).reshape(-1))
        hue_parts.append(hue_error(pred_nits, ref_nits).reshape(-1))
    de2000 = np.concatenate(de2000_parts)
    deictcp = np.concatenate(deictcp_parts)
    chroma = np.concatenate(chroma_parts)
    hue = np.concatenate(hue_parts)
    return {
        "deltaE2000_peak_relative_Rec2020_to_Lab_D65": summary(de2000, absolute=False),
        "deltaEICtCp": summary(deictcp, absolute=False),
        "chroma_error_ICtCp_signed": summary(chroma, absolute=False),
        "chroma_error_ICtCp_absolute": summary(chroma, absolute=True),
        "hue_error_degrees_signed": summary(hue, absolute=False),
        "hue_error_degrees_absolute": summary(hue, absolute=True),
    }


def output_range_diagnostics(image: np.ndarray) -> dict[str, Any]:
    image = np.asarray(image, dtype=np.float64)
    return {
        "finite": bool(np.isfinite(image).all()),
        "minimum_normalized": float(np.nanmin(image)),
        "maximum_normalized": float(np.nanmax(image)),
        "negative_count_after_model_clip": int(np.sum(image < 0.0)),
        "above_peak_count": int(np.sum(image > 1.0)),
        "above_peak_fraction": float(np.mean(image > 1.0)),
    }


def seam_metrics(model_full: np.ndarray, hdr_common: np.ndarray, y1: int, y2: int) -> dict[str, Any]:
    """Measure composite seam proxies; no extension ground truth is claimed."""
    composite = model_full.copy()
    composite[y1:y2] = hdr_common
    rows: list[dict[str, Any]] = []
    for seam_y, side in ((y1, "top"), (y2, "bottom")):
        outer_row, inner_row = (seam_y - 1, seam_y) if side == "top" else (seam_y, seam_y - 1)
        outer = composite[outer_row] * PEAK_NITS
        inner = composite[inner_row] * PEAK_NITS
        outer_y = luminance(outer)
        inner_y = luminance(inner)
        normal_gradient_outer = outer_y - luminance(composite[max(outer_row - 1, 0)] * PEAK_NITS)
        normal_gradient_inner = luminance(composite[min(inner_row + 1, composite.shape[0] - 1)] * PEAK_NITS) - inner_y
        hue = hue_error(outer, inner)
        chroma = chroma_error(outer, inner)
        rows.append({
            "boundary": side,
            "y_coordinate": int(seam_y),
            "luminance_discontinuity_nits": float(np.mean(np.abs(outer_y - inner_y))),
            "luminance_discontinuity_P95_nits": float(np.percentile(np.abs(outer_y - inner_y), 95)),
            "chroma_discontinuity_ICtCp": float(np.mean(np.abs(chroma))),
            "hue_discontinuity_degrees": float(np.mean(np.abs(hue))),
            "gradient_discontinuity_nits_per_pixel": float(np.mean(np.abs(normal_gradient_outer - normal_gradient_inner))),
        })
    return {"definition": "HDR-center composite versus model-predicted extension; proxy only, because no HDR extension reference exists", "boundaries": rows}


def write_rgb16(path: Path, rgb_code: np.ndarray) -> None:
    code = np.round(np.clip(rgb_code, 0.0, 1.0) * RGB16_MAX).astype(np.uint16)
    write_cv_image(path, code[..., ::-1])


def write_hdr_scientific(prefix: Path, linear_norm: np.ndarray, output: dict[str, Any]) -> None:
    np.save(prefix.with_suffix(".linear_bt2020_nits.npy"), (linear_norm * PEAK_NITS).astype(np.float32))
    write_rgb16(prefix.with_suffix(".pq16.png"), pq_oetf(linear_norm))
    exr = prefix.with_suffix(".linear_bt2020_nits.exr")
    try:
        write_cv_image(exr, (linear_norm * PEAK_NITS).astype(np.float32)[..., ::-1])
        output["exr"] = str(exr)
    except (RuntimeError, cv2.error) as exc:
        output["exr"] = f"not available: {exc}"[:500]


def preview(linear_norm: np.ndarray) -> np.ndarray:
    nits = np.maximum(np.asarray(linear_norm, dtype=np.float64) * PEAK_NITS, 0.0)
    display = np.log1p(nits) / math.log1p(1000.0)
    return np.power(np.clip(display, 0.0, 1.0), 1.0 / 2.2)


def write_preview(path: Path, image: np.ndarray) -> None:
    code = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    write_cv_image(path, code[..., ::-1])


def difference_preview(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    error = np.abs(luminance(predicted) - luminance(reference)) * PEAK_NITS
    normalized = np.clip(np.log1p(error) / math.log1p(1000.0), 0.0, 1.0)
    return cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def make_panel(images: list[np.ndarray], max_height: int = 360) -> np.ndarray:
    scaled: list[np.ndarray] = []
    for image in images:
        scale = max_height / image.shape[0]
        scaled.append(cv2.resize(image, (int(round(image.shape[1] * scale)), max_height), interpolation=cv2.INTER_AREA))
    return np.round(np.clip(np.concatenate(scaled, axis=1), 0.0, 1.0) * 255.0).astype(np.uint8)


def write_seam_panel(path: Path, model_full: np.ndarray, hdr_common: np.ndarray, y1: int, y2: int) -> None:
    composite = model_full.copy()
    composite[y1:y2] = hdr_common
    strips = []
    for seam_y in (y1, y2):
        lower = max(0, seam_y - 60)
        upper = min(model_full.shape[0], seam_y + 60)
        strips.append(make_panel([preview(model_full[lower:upper]), preview(composite[lower:upper])], max_height=120))
    write_cv_image(path, np.concatenate(strips, axis=0)[..., ::-1])


def load_anchor() -> tuple[dict[str, Any], dict[str, Path]]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    require(metrics.get("phase") == "P2.32" and metrics.get("status") == "DATASET READY", "P2.32 dataset is not marked DATASET READY")
    matches = [item for item in metrics["case_records"]["The Matrix"] if item.get("case_id") == "the_matrix_temporal_stratum_08"]
    require(len(matches) == 1, f"Expected one Matrix anchor 08, found {len(matches)}")
    anchor = matches[0]
    expected = {"hdr_frame": 73367, "om_frame": 73348, "known_verified_anchor": True}
    for key, value in expected.items():
        require(anchor.get(key) == value, f"Anchor mismatch for {key}: {anchor.get(key)!r} != {value!r}")
    geometry = anchor.get("geometry", {})
    require(geometry.get("overlap") == [0, 140, 1920, 940], f"Unexpected overlap: {geometry.get('overlap')}")
    require(float(geometry.get("confidence", 0.0)) == 1.0, f"Unexpected geometry confidence: {geometry.get('confidence')}")
    paths = {
        "hdr_rgb": locate_from_declared(anchor["hdr_rgb_u16_path"], "_hdr_rgb_u16.npy"),
        "om_rgb": locate_from_declared(anchor["om_rgb_u16_path"], "_om_rgb_u16.npy"),
        "sdr_nits": locate_from_declared(anchor["sdr_luminance_nits_path"], "_sdr_luminance_nits.npy"),
        "hdr_nits": locate_from_declared(anchor["hdr_luminance_nits_path"], "_hdr_luminance_nits.npy"),
    }
    return anchor, paths


def decode_real(anchor: dict[str, Any], paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    hdr_raw = np.load(paths["hdr_rgb"])
    om_raw = np.load(paths["om_rgb"])
    saved_sdr_nits = np.load(paths["sdr_nits"])
    saved_hdr_nits = np.load(paths["hdr_nits"])
    require(hdr_raw.dtype == np.uint16 and om_raw.dtype == np.uint16, f"Raw RGB dtype must be uint16: {hdr_raw.dtype}/{om_raw.dtype}")
    require(tuple(hdr_raw.shape) == tuple(anchor["hdr_rgb_shape"]), f"HDR shape mismatch: {hdr_raw.shape}")
    require(tuple(om_raw.shape) == tuple(anchor["om_rgb_shape"]), f"OM shape mismatch: {om_raw.shape}")
    x1, y1, x2, y2 = anchor["geometry"]["overlap"]
    shape = (y2 - y1, x2 - x1)
    require(saved_sdr_nits.shape == shape and saved_hdr_nits.shape == shape, f"Saved luminance shapes must be {shape}")
    require(np.issubdtype(saved_sdr_nits.dtype, np.floating) and np.issubdtype(saved_hdr_nits.dtype, np.floating), "Saved luminance must be floating point")
    require(np.isfinite(saved_sdr_nits).all() and np.isfinite(saved_hdr_nits).all(), "Saved luminance contains non-finite values")
    require(np.min(saved_sdr_nits) >= 0 and np.min(saved_hdr_nits) >= 0, "Saved luminance contains negative values")
    om_code = om_raw.astype(np.float64) / RGB16_MAX
    hdr_code_full = hdr_raw.astype(np.float64) / RGB16_MAX
    hdr_code = cv2.resize(hdr_code_full, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    sdr_full = np.einsum("...j,ij->...i", linearize_sdr_709(om_code), M_709_TO_2020, optimize=True)
    sdr_full = np.maximum(sdr_full, 0.0)
    sdr_common = sdr_full[y1:y2, x1:x2]
    hdr_common = pq_eotf_normalized(hdr_code)
    reconstructed_sdr_nits = luminance(sdr_common) * PEAK_NITS
    reconstructed_hdr_nits = luminance(hdr_common) * PEAK_NITS
    consistency = {
        "sdr_reconstructed_vs_saved_MAE_nits": float(np.mean(np.abs(reconstructed_sdr_nits - saved_sdr_nits))),
        "sdr_reconstructed_vs_saved_P95_nits": float(np.percentile(np.abs(reconstructed_sdr_nits - saved_sdr_nits), 95)),
        "hdr_reconstructed_vs_saved_MAE_nits": float(np.mean(np.abs(reconstructed_hdr_nits - saved_hdr_nits))),
        "hdr_reconstructed_vs_saved_P95_nits": float(np.percentile(np.abs(reconstructed_hdr_nits - saved_hdr_nits), 95)),
        "resize_backend": "OpenCV INTER_LINEAR; P2.32 original extraction used FFmpeg bilinear, so this is an integrity check rather than bit-exact equality",
    }
    return sdr_full, sdr_common, hdr_common, {"raw_dtypes": {"hdr": str(hdr_raw.dtype), "om": str(om_raw.dtype), "sdr_luminance": str(saved_sdr_nits.dtype), "hdr_luminance": str(saved_hdr_nits.dtype)}, "consistency": consistency}


def fit_and_apply(name: str, model: Any, sdr_full: np.ndarray, sdr_common: np.ndarray, hdr_common: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    sdr_y = luminance(sdr_common).reshape(-1)
    hdr_y = luminance(hdr_common).reshape(-1)
    tracemalloc.start()
    fit_started = time.perf_counter()
    params = model.fit(sdr_y, hdr_y, sdr_common.reshape(-1, 3), hdr_common.reshape(-1, 3))
    fit_seconds = time.perf_counter() - fit_started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    apply_started = time.perf_counter()
    predicted_full = model.apply(sdr_full, params)
    apply_seconds = time.perf_counter() - apply_started
    x1, y1, x2, y2 = [0, 140, 1920, 940]
    predicted_common = predicted_full[y1:y2, x1:x2]
    residual = predicted_common - hdr_common
    regularization = None
    convergence = "not exposed by research model"
    if name == "ModelE":
        coeffs = np.asarray(params["correction_coeffs"], dtype=np.float64)
        sig = np.asarray(params["sigmoid_params"], dtype=np.float64)
        regularization = float(model.smoothness_lambda * (np.sum(coeffs**2) + np.sum(sig[::3] ** 2)))
        convergence = "SciPy L-BFGS-B runs internally; OptimizeResult is not exposed by Model E"
    else:
        convergence = "analytical/statistical fit; no iterative optimizer or convergence object"
    details = {
        "model": model.name(),
        "parameter_count": int(model.param_count()),
        "parameters": params,
        "fit_time_seconds_cpu": fit_seconds,
        "apply_time_seconds_cpu": apply_seconds,
        "fit_python_tracemalloc_peak_bytes": int(peak),
        "objective_common_rgb_MSE_normalized": float(np.mean(residual**2)),
        "objective_common_luminance_MSE_nits_squared": float(np.mean(((luminance(predicted_common) - luminance(hdr_common)) * PEAK_NITS) ** 2)),
        "regularization_term": regularization,
        "convergence": convergence,
        "warnings": [],
        "output_range": output_range_diagnostics(predicted_full),
        "common_region_luminance": luminance_metrics(predicted_common, hdr_common),
        "common_region_color": color_metrics(predicted_common, hdr_common),
        "seam_proxy": seam_metrics(predicted_full, hdr_common, y1, y2),
    }
    if details["output_range"]["above_peak_count"]:
        details["warnings"].append("Model output contains values above 10,000 nits; PQ PNG previews clip them, while scientific NPY preserves them.")
    return predicted_full, details


def make_synthetic_scene(height: int = 576, width: int = 1024) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Create deterministic full HDR ground truth then derive a full SDR input."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    xn, yn = x / max(width - 1, 1), y / max(height - 1, 1)
    base = 0.002 + 0.10 * yn + 0.18 * xn
    glow = 0.85 * np.exp(-(((xn - 0.74) / 0.12) ** 2 + ((yn - 0.27) / 0.10) ** 2))
    shadow = -0.09 * np.exp(-(((xn - 0.22) / 0.18) ** 2 + ((yn - 0.72) / 0.16) ** 2))
    texture = 0.025 * np.sin(22 * xn + 7 * yn) + 0.015 * np.cos(35 * yn)
    lum = np.clip(base + glow + shadow + texture, 0.0002, 0.95)
    red = lum * (1.0 + 0.32 * np.sin(4 * np.pi * xn) * (0.3 + yn))
    green = lum * (1.0 + 0.22 * np.cos(3 * np.pi * yn))
    blue = lum * (1.0 + 0.38 * np.sin(3 * np.pi * (xn + yn)))
    hdr = np.maximum(np.stack((red, green, blue), axis=-1), 0.0)
    hdr = np.clip(hdr, 0.0, 1.0)
    # Controlled SDR creation: global compression plus mild gamut desaturation.
    sdr = hdr / (hdr + 0.075)
    sdr_y = luminance(sdr)[..., None]
    sdr = np.clip(sdr_y + 0.74 * (sdr - sdr_y), 0.0, 1.0)
    return hdr, sdr, (int(round(height * 0.20)), int(round(height * 0.80)))


def synthetic_hidden_region_test() -> dict[str, Any]:
    hdr_full, sdr_full, (y1, y2) = make_synthetic_scene()
    hdr_common, sdr_common = hdr_full[y1:y2], sdr_full[y1:y2]
    result: dict[str, Any] = {"definition": "Full synthetic HDR ground truth -> derived full SDR -> only central HDR crop supplied for fit", "geometry": {"full_shape": list(hdr_full.shape), "central_hdr_crop_y": [y1, y2]}, "models": {}}
    for name, model in (("ModelE", ModelECdfRegularized()), ("ModelG", ModelGHybrid())):
        params = model.fit(luminance(sdr_common).reshape(-1), luminance(hdr_common).reshape(-1), sdr_common.reshape(-1, 3), hdr_common.reshape(-1, 3))
        predicted = model.apply(sdr_full, params)
        hidden_mask = np.ones(hdr_full.shape[:2], dtype=bool)
        hidden_mask[y1:y2] = False
        result["models"][name] = {
            "parameter_count": int(model.param_count()),
            "parameters": params,
            "common_region": {"luminance": luminance_metrics(predicted[y1:y2], hdr_common), "color": color_metrics(predicted[y1:y2], hdr_common)},
            "hidden_region": {"luminance": luminance_metrics(predicted[hidden_mask].reshape(-1, 1, 3), hdr_full[hidden_mask].reshape(-1, 1, 3)), "color": color_metrics(predicted[hidden_mask].reshape(-1, 1, 3), hdr_full[hidden_mask].reshape(-1, 1, 3))},
            "output_range": output_range_diagnostics(predicted),
        }
        write_hdr_scientific(RESULTS / "scientific" / f"synthetic_{name}_full", predicted, result["models"][name])
    panel = make_panel([preview(sdr_full), preview(hdr_full), preview(ModelECdfRegularized().apply(sdr_full, result["models"]["ModelE"]["parameters"])), preview(ModelGHybrid().apply(sdr_full, result["models"]["ModelG"]["parameters"]))], max_height=288)
    write_cv_image(RESULTS / "previews" / "synthetic_hidden_region_comparison.png", panel[..., ::-1])
    return result


def cuda_status() -> dict[str, Any]:
    try:
        import cupy as cp
        device = cp.cuda.runtime.getDeviceProperties(0)
        name = device["name"].decode() if isinstance(device["name"], bytes) else str(device["name"])
        return {"available": True, "cupy_version": cp.__version__, "device_count": int(cp.cuda.runtime.getDeviceCount()), "device_name": name, "note": "Research Models E/G are NumPy/SciPy CPU implementations; no CUDA fit/apply path exists, so no CUDA model benchmark was run."}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def markdown_report(result: dict[str, Any]) -> str:
    e = result["real_models"]["ModelE"]
    g = result["real_models"]["ModelG"]
    synth_e = result["synthetic_hidden_region"]["models"]["ModelE"]
    synth_g = result["synthetic_hidden_region"]["models"]["ModelG"]
    def global_mae(block: dict[str, Any]) -> float:
        return float(block["common_region_luminance"]["global"]["MAE_nits"])
    def hidden_mae(block: dict[str, Any]) -> float:
        return float(block["hidden_region"]["luminance"]["global"]["MAE_nits"])
    return f"""# Physical Validation Report — Matrix 73367 / 73348

## Scope and safety
This isolated experiment uses one pre-extracted P2.32 Matrix anchor only. It did not decode a film, modify P2 artifacts, run P2.34, run a production render, or integrate either model into production. Model fitting only receives the SDR/HDR common region; the full SDR Open Matte is used only after fitting.

## Inputs and geometry
- HDR frame: `{result['anchor']['hdr_frame']}`; Open Matte frame: `{result['anchor']['om_frame']}`.
- Verified overlap in Open Matte coordinates: `{result['anchor']['geometry']['overlap']}`; geometry confidence `{result['anchor']['geometry']['confidence']}`.
- Input files were programmatically resolved within the experimental root and are recorded in `metrics.json`.
- Full Open Matte: 1920×1080. HDR was scaled from 3840×1600 to the 1920×800 common region.

## Color pipeline
- SDR source: `rgb48le`-derived `uint16` RGB code values, BT.709 primaries/matrix, modeled with BT.1886-style `code**2.4` EOTF.
- SDR fit/apply RGB: linear BT.2020 after the documented 709→2020 matrix.
- HDR reference: `rgb48le`-derived `uint16` RGB code values, BT.2020 primaries, ST.2084/PQ transfer; PQ EOTF is applied once.
- Model domain: linear BT.2020 RGB; normalized HDR `1.0 = 10,000 nits`.
- Scientific arrays are linear BT.2020 nits (`.npy`); scientific PNGs are PQ-coded 16-bit. Display previews are logarithmically tone-mapped and are not HDR scientific outputs.
- Reconstructed-vs-saved P2.32 luminance consistency is reported in `metrics.json`. The OpenCV bilinear resize is not expected to be bit-identical to the original FFmpeg bilinear resize.

## Model fit
| Model | Parameters | CPU fit s | CPU apply s | Common MAE (nits) | Common RMSE (nits) |
|---|---:|---:|---:|---:|---:|
| Model E | {e['parameter_count']} | {e['fit_time_seconds_cpu']:.3f} | {e['apply_time_seconds_cpu']:.3f} | {global_mae(e):.4f} | {e['common_region_luminance']['global']['RMSE_nits']:.4f} |
| Model G | {g['parameter_count']} | {g['fit_time_seconds_cpu']:.3f} | {g['apply_time_seconds_cpu']:.3f} | {global_mae(g):.4f} | {g['common_region_luminance']['global']['RMSE_nits']:.4f} |

Full fitted parameter values, objectives, regularization, warnings, output-range diagnostics, luminance ranges, ΔE2000, ΔEICtCp, chroma, hue, and seam proxies are in `metrics.json` and `fitted_parameters.json`.

## Common-region quality
The metrics above are **common-region reconstruction quality**, not evidence for objective correctness in the unseen Open Matte extension. ΔE2000 is computed as a documented peak-relative Rec.2020→XYZ D65→Lab diagnostic; ΔEICtCp is also reported for HDR-native color comparison.

## Seam analysis
`10_ModelE_Seam.png` and `11_ModelG_Seam.png` show each model prediction and an HDR-center composite around the top/bottom crop edges. Numeric seam values measure HDR-center versus predicted-extension discontinuities. Because Matrix has no full HDR Open Matte reference, these are continuity proxies only, not ground-truth extension errors.

| Model | Top seam mean / P95 luma (nits) | Bottom seam mean / P95 luma (nits) | Top / bottom hue discontinuity (°) |
|---|---:|---:|---:|
| Model E | {e['seam_proxy']['boundaries'][0]['luminance_discontinuity_nits']:.3f} / {e['seam_proxy']['boundaries'][0]['luminance_discontinuity_P95_nits']:.3f} | {e['seam_proxy']['boundaries'][1]['luminance_discontinuity_nits']:.3f} / {e['seam_proxy']['boundaries'][1]['luminance_discontinuity_P95_nits']:.3f} | {e['seam_proxy']['boundaries'][0]['hue_discontinuity_degrees']:.3f} / {e['seam_proxy']['boundaries'][1]['hue_discontinuity_degrees']:.3f} |
| Model G | {g['seam_proxy']['boundaries'][0]['luminance_discontinuity_nits']:.3f} / {g['seam_proxy']['boundaries'][0]['luminance_discontinuity_P95_nits']:.3f} | {g['seam_proxy']['boundaries'][1]['luminance_discontinuity_nits']:.3f} / {g['seam_proxy']['boundaries'][1]['luminance_discontinuity_P95_nits']:.3f} | {g['seam_proxy']['boundaries'][0]['hue_discontinuity_degrees']:.3f} / {g['seam_proxy']['boundaries'][1]['hue_discontinuity_degrees']:.3f} |

## Visual findings
The corrected 8-bit display panels use a log-to-1,000-nit preview transform and are only for review; `.pq16.png` and `.linear_bt2020_nits.npy` remain the scientific deliverables. Direct review of `comparison_full_A_B_C_D.png` shows a substantial yellow/red chromatic shift and contrast exaggeration in the full-frame extensions for **both** models relative to the neutral HDR center; the crop boundaries are visibly apparent in `10_ModelE_Seam.png` and `11_ModelG_Seam.png`. Model E has lower numeric discontinuities, but neither transformed Open Matte is visually coherent enough to be accepted as an HDR extension on this anchor. This visual result is qualitative; no physical HDR Open Matte reference exists outside the crop.

## Synthetic hidden-region test
A deterministic full HDR Open Matte was created, converted to SDR, and then restricted to a central HDR crop for fitting. This creates known full-frame ground truth.

| Model | Synthetic common MAE (nits) | Synthetic hidden-region MAE (nits) | Hidden RMSE (nits) | Hidden mean ΔE2000 |
|---|---:|---:|---:|---:|
| Model E | {synth_e['common_region']['luminance']['global']['MAE_nits']:.4f} | {hidden_mae(synth_e):.4f} | {synth_e['hidden_region']['luminance']['global']['RMSE_nits']:.4f} | {synth_e['hidden_region']['color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.4f} |
| Model G | {synth_g['common_region']['luminance']['global']['MAE_nits']:.4f} | {hidden_mae(synth_g):.4f} | {synth_g['hidden_region']['luminance']['global']['RMSE_nits']:.4f} | {synth_g['hidden_region']['color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.4f} |

This controlled test is supportive only: its SDR creation function is known and does not establish transfer to real unseen imagery.

## CUDA and performance
CUDA status: `{result['cuda']['available']}`. {result['cuda'].get('note', result['cuda'].get('error', ''))} CPU timing above is the only model timing measured.

## Final decision A–F
**A — Common-region adequacy.** Model E is adequate only for **luminance** reconstruction in this limited low/mid-range anchor: MAE is {global_mae(e):.4f} nits and P95 is {e['common_region_luminance']['global']['P95_absolute_error_nits']:.4f} nits. It does not adequately reproduce common-region RGB appearance: the corrected panel shows a strong warm chromatic shift, and the anchor has no samples at or above 500 nits. Therefore Model E fails an overall physical-image acceptance criterion despite its luminance score.

**B — Value of Model G's extra parameters.** No. Model G's {g['parameter_count']} parameters are not justified here versus Model E's {e['parameter_count']}: it has worse MAE/RMSE/P95 ({global_mae(g):.4f}/{g['common_region_luminance']['global']['RMSE_nits']:.4f}/{g['common_region_luminance']['global']['P95_absolute_error_nits']:.4f} nits), longer CPU fit/apply time, and slightly higher mean ΔE2000 ({g['common_region_color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.4f} vs {e['common_region_color']['deltaE2000_peak_relative_Rec2020_to_Lab_D65']['mean']:.4f}).

**C — Coherent grading outside HDR.** No for this anchor. The corrected full-frame and seam panels visibly show yellow/red chromatic shift, contrast exaggeration, and crop-boundary mismatch in both extensions. Model E is less discontinuous numerically but is not visually acceptable as an HDR extension; lack of HDR ground truth outside the overlap remains an additional limitation.

**D — Observed failure modes.** Both models exhibit global warm/yellow-red grading drift outside the reference crop, visible seams, large hue error, and crop-boundary hue discontinuity. Model G additionally has materially larger 10–100 and 100–500 nit errors. The anchor contains no 500+ nit reference pixels, leaving shoulder/highlight behavior untested.

**E — Minimum sufficient complexity.** Twelve parameters (Model E) are the minimum sufficient tested complexity only for common-region reconstruction on this single anchor; neither tested complexity is sufficient for visually acceptable physical extension.

**F — Next action.** Do not advance either model toward production or full-video processing. Model E may be retained solely as the lower-complexity baseline for a separately authorized diagnostic real-anchor study aimed at correcting the extension color/continuity failure; do not alter P2 artifacts.

## Limitations and recommendation
- The Matrix anchor is one temporally selected frame, not independently verified scene ground truth.
- The fitting mask is exactly the verified SDR∩HDR overlap (1,536,000 pixels); neither model sees HDR information outside it during fitting.
- Full fitted parameter values and fitting metadata are preserved in `fitted_parameters.json` and `metrics.json`; Model E uses 12 fitted controls, while Model G uses 29.
- No HDR ground truth exists for the physical Open Matte extension; the result can only claim visual/continuity assessment there.
- Model E exposes no optimizer convergence object; Model G uses separate analytical/statistical components rather than one global RGB loss.
- OpenEXR could not be encoded by this OpenCV build. Scientific outputs are instead full-precision linear-BT.2020-nit `.npy` arrays and 16-bit PQ PNGs.
- Any output above 10,000 nits is retained in scientific arrays and clipped only for PQ PNG serialization.
- The recommendation is research-only: Model E may proceed to another isolated anchor, with no production integration.
"""


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scientific").mkdir(exist_ok=True)
    (RESULTS / "previews").mkdir(exist_ok=True)
    anchor, input_paths = load_anchor()
    sdr_full, sdr_common, hdr_common, decode_info = decode_real(anchor, input_paths)
    x1, y1, x2, y2 = anchor["geometry"]["overlap"]
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "experimental_root": str(ROOT),
        "anchor": anchor,
        "input_files": input_paths,
        "fit_mask": {"definition": "SDR ∩ HDR verified P2.32 overlap only", "bbox_open_matte_coordinates": [x1, y1, x2, y2], "pixel_count": int((x2-x1)*(y2-y1))},
        "color_pipeline": {"sdr_transfer": "BT.709 code -> power 2.4 (BT.1886-style)", "sdr_primaries": "BT.709", "sdr_matrix": "BT.709-to-BT.2020 3x3", "hdr_transfer": "ST.2084 PQ EOTF", "hdr_primaries": "BT.2020", "model_representation": "linear BT.2020 RGB; HDR normalized where 1=10000 nits", "raw_representation": "P2.32 rgb48le-derived uint16 RGB; not YCbCr", "scientific_output": "linear BT.2020 nits NPY plus PQ 16-bit PNG"},
        "decode_validation": decode_info,
        "cuda": cuda_status(),
        "real_models": {},
    }
    write_rgb16(RESULTS / "01_SDR_OpenMatte.png", np.load(input_paths["om_rgb"]).astype(np.float64) / RGB16_MAX)
    write_hdr_scientific(RESULTS / "scientific" / "02_HDR_Reference", hdr_common, result)
    write_rgb16(RESULTS / "03_SDR_Common.png", np.load(input_paths["om_rgb"])[y1:y2, x1:x2].astype(np.float64) / RGB16_MAX)
    predicted: dict[str, np.ndarray] = {}
    for short, model in (("ModelE", ModelECdfRegularized()), ("ModelG", ModelGHybrid())):
        full, details = fit_and_apply(short, model, sdr_full, sdr_common, hdr_common)
        predicted[short] = full
        result["real_models"][short] = details
        number = "04" if short == "ModelE" else "05"
        crop_number = "06" if short == "ModelE" else "07"
        diff_number = "08" if short == "ModelE" else "09"
        seam_number = "10" if short == "ModelE" else "11"
        write_hdr_scientific(RESULTS / "scientific" / f"{number}_{short}_HDR_OpenMatte", full, details)
        write_hdr_scientific(RESULTS / "scientific" / f"{crop_number}_{short}_Common_Crop", full[y1:y2, x1:x2], details)
        write_cv_image(RESULTS / f"{diff_number}_{short}_Difference.png", difference_preview(full[y1:y2, x1:x2], hdr_common))
        write_seam_panel(RESULTS / f"{seam_number}_{short}_Seam.png", full, hdr_common, y1, y2)
    write_preview(RESULTS / "previews" / "01_SDR_OpenMatte_display.png", sdr_full)
    write_preview(RESULTS / "previews" / "02_HDR_Reference_display.png", hdr_common)
    write_preview(RESULTS / "previews" / "04_ModelE_HDR_OpenMatte_display.png", predicted["ModelE"])
    write_preview(RESULTS / "previews" / "05_ModelG_HDR_OpenMatte_display.png", predicted["ModelG"])
    hdr_canvas = np.zeros_like(sdr_full)
    hdr_canvas[y1:y2, x1:x2] = hdr_common
    panel = make_panel([preview(sdr_full), preview(hdr_canvas), preview(predicted["ModelE"]), preview(predicted["ModelG"])])
    write_cv_image(RESULTS / "previews" / "comparison_full_A_B_C_D.png", panel[..., ::-1])
    synthetic = synthetic_hidden_region_test()
    result["synthetic_hidden_region"] = synthetic
    write_json(RESULTS / "metrics.json", result)
    write_json(RESULTS / "fitted_parameters.json", {name: block["parameters"] for name, block in result["real_models"].items()})
    REPORT_PATH.write_text(markdown_report(result), encoding="utf-8")
    print(f"PHYSICAL_VALIDATION_STATUS={result['status']}")
    print(f"RESULTS={RESULTS}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
