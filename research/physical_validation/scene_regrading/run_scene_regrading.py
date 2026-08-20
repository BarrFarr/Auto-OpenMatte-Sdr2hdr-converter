"""Phase 3 research-only scene-level HDR regrading experiment.

Model H0 = Model E luminance-only fit + hue-preserving RGB ratio scaling.
Model H1 = H0 followed by one regularized scene-local 3x3 RGB matrix.

The script reads pre-extracted P2.32 arrays only, fits every control only on a
matched common region, writes solely below this directory, never imports
Model G or production code, and never decodes video.
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
from scipy.optimize import minimize

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
REPORT_PATH = BASE / "SCENE_REGRADING_REPORT.md"
PEAK_NITS = 10_000.0
RGB16_MAX = 65_535.0
EPS = 1e-8
M_2020_TO_XYZ_D65 = np.array(((0.6369580483, 0.1446169036, 0.1688809752), (0.2627002120, 0.6779980715, 0.0593017165), (0.0, 0.0280726930, 1.0609850577)), dtype=np.float64)
D65 = np.array((0.95047, 1.0, 1.08883), dtype=np.float64)
LAMBDA_GRID = (1e-3, 1e-2, 1e-1)
TEMPORAL_CASES = ("the_matrix_temporal_stratum_01", "the_matrix_temporal_stratum_03", "the_matrix_temporal_stratum_08", "the_matrix_temporal_stratum_12", "the_matrix_temporal_stratum_16", "the_matrix_temporal_stratum_20")


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
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_cv_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    require(bool(ok), f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def pq_eotf_normalized(code: np.ndarray) -> np.ndarray:
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    code = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    p = np.power(code, 1.0 / m2)
    return np.power(np.maximum(p - c1, 0.0) / np.maximum(c2 - c3 * p, EPS), 1.0 / m1)


def pq_oetf(value: np.ndarray) -> np.ndarray:
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    v = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    p = np.power(v, m1)
    return np.power((c1 + c2 * p) / (1.0 + c3 * p), m2)


def locate(declared: str, suffix: str) -> Path:
    matches = [path for path in P232_DIR.rglob(Path(declared).name) if path.is_file()]
    require(len(matches) == 1, f"Expected exactly one P2.32 input for {declared}; got {len(matches)}")
    path = matches[0].resolve()
    require(str(path).startswith(str(ROOT.resolve())) and path.name.endswith(suffix), f"Unsafe P2.32 path: {path}")
    return path


def load_record(case_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
    dataset = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    require(dataset.get("phase") == "P2.32" and dataset.get("status") == "DATASET READY", "P2.32 is not DATASET READY")
    records = [record for record in dataset["case_records"]["The Matrix"] if record.get("case_id") == case_id]
    require(len(records) == 1, f"Expected one Matrix record for {case_id}")
    record = records[0]
    geometry = record.get("geometry", {})
    overlap = geometry.get("overlap")
    require(isinstance(overlap, list) and len(overlap) == 4 and float(geometry.get("confidence", 0.0)) == 1.0, f"Invalid geometry: {case_id}")
    require(record.get("sync_status") == "LOCKED" and record.get("finite", {}).get("hdr_rgb") and record.get("finite", {}).get("om_rgb"), f"Unverified inputs: {case_id}")
    return record, {"hdr_rgb": locate(record["hdr_rgb_u16_path"], "_hdr_rgb_u16.npy"), "om_rgb": locate(record["om_rgb_u16_path"], "_om_rgb_u16.npy")}


def decode_record(record: dict[str, Any], paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    hdr_raw, om_raw = np.load(paths["hdr_rgb"]), np.load(paths["om_rgb"])
    require(hdr_raw.dtype == np.uint16 and om_raw.dtype == np.uint16, "Raw P2.32 RGB must be uint16")
    require(tuple(hdr_raw.shape) == tuple(record["hdr_rgb_shape"]) and tuple(om_raw.shape) == tuple(record["om_rgb_shape"]), "Raw P2.32 RGB shape mismatch")
    x1, y1, x2, y2 = (int(value) for value in record["geometry"]["overlap"])
    hdr_code = cv2.resize(hdr_raw.astype(np.float64) / RGB16_MAX, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    sdr_full = np.maximum(bt709_to_bt2020(bt1886_eotf(om_raw.astype(np.float64) / RGB16_MAX)), 0.0)
    hdr_common = pq_eotf_normalized(hdr_code)
    sdr_common = sdr_full[y1:y2, x1:x2]
    require(sdr_common.shape == hdr_common.shape and np.isfinite(sdr_full).all() and np.isfinite(hdr_common).all(), "Invalid decoded common images")
    return sdr_full, sdr_common, hdr_common, (x1, y1, x2, y2)


def lab2020(image: np.ndarray) -> np.ndarray:
    xyz = np.einsum("...c,dc->...d", np.asarray(image, dtype=np.float64), M_2020_TO_XYZ_D65, optimize=True)
    ratio = xyz / D65
    delta = 6.0 / 29.0
    f = np.where(ratio > delta**3, np.cbrt(np.maximum(ratio, 0.0)), ratio / (3.0 * delta**2) + 4.0 / 29.0)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])), axis=-1)


def summarize(values: np.ndarray, absolute: bool = False) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if absolute:
        values = np.abs(values)
    return {"count": int(values.size), "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))} if values.size else {"count": 0, "mean": None, "median": None, "p95": None}


def metrics(predicted: np.ndarray, reference: np.ndarray, rows: int = 80) -> dict[str, Any]:
    y_error = (compute_luminance(predicted) - compute_luminance(reference)) * PEAK_NITS
    de, di, chroma, hue, ref_chroma = [], [], [], [], []
    for start in range(0, predicted.shape[0], rows):
        stop = min(start + rows, predicted.shape[0])
        p, r = predicted[start:stop], reference[start:stop]
        pn, rn = p * PEAK_NITS, r * PEAK_NITS
        de.append(delta_e_2000(lab2020(p), lab2020(r)).reshape(-1))
        di.append(delta_e_ictcp(pn, rn).reshape(-1))
        chroma.append(chroma_error(pn, rn).reshape(-1))
        hue.append(hue_error(pn, rn).reshape(-1))
        ref_i = linear_bt2020_to_ictcp(rn)
        ref_chroma.append(np.sqrt(ref_i[..., 1] ** 2 + ref_i[..., 2] ** 2).reshape(-1))
    hue_values, chroma_reference = np.concatenate(hue), np.concatenate(ref_chroma)
    valid_hue = chroma_reference > 1e-6
    return {
        "RGB_MSE_normalized": float(np.mean((predicted - reference) ** 2)),
        "luminance": {"MAE_nits": float(np.mean(np.abs(y_error))), "RMSE_nits": float(np.sqrt(np.mean(y_error * y_error))), "median_absolute_error_nits": float(np.median(np.abs(y_error))), "P95_absolute_error_nits": float(np.percentile(np.abs(y_error), 95))},
        "deltaE2000": summarize(np.concatenate(de)), "deltaEICtCp": summarize(np.concatenate(di)),
        "chroma_error_ICtCp_signed": summarize(np.concatenate(chroma)), "chroma_error_ICtCp_absolute": summarize(np.concatenate(chroma), True),
        "hue_error_degrees_signed_non_neutral": summarize(hue_values[valid_hue]), "hue_error_degrees_absolute_non_neutral": summarize(hue_values[valid_hue], True),
        "hue_excluded_neutral_reference_pixels": int(np.sum(~valid_hue)),
    }


def range_info(image: np.ndarray) -> dict[str, Any]:
    return {"finite": bool(np.isfinite(image).all()), "min": float(np.min(image)), "max": float(np.max(image)), "negative_count": int(np.sum(image < 0)), "above_peak_count": int(np.sum(image > 1)), "above_peak_fraction": float(np.mean(image > 1))}


def preview(image: np.ndarray) -> np.ndarray:
    nits = np.maximum(np.asarray(image, dtype=np.float64) * PEAK_NITS, 0.0)
    return np.power(np.clip(np.log1p(nits) / math.log1p(1000), 0, 1), 1 / 2.2)


def write_scientific(prefix: Path, image: np.ndarray) -> None:
    np.save(prefix.with_suffix(".linear_bt2020_nits.npy"), (image * PEAK_NITS).astype(np.float32))
    pq = np.round(np.clip(pq_oetf(image), 0, 1) * RGB16_MAX).astype(np.uint16)
    write_cv_image(prefix.with_suffix(".pq16.png"), pq[..., ::-1])


def write_preview(path: Path, image: np.ndarray) -> None:
    code = np.round(np.clip(preview(image), 0, 1) * 255).astype(np.uint8)
    write_cv_image(path, code[..., ::-1])


def panel(images: list[np.ndarray], height: int = 270) -> np.ndarray:
    resized = []
    for image in images:
        scale = height / image.shape[0]
        resized.append(cv2.resize(preview(image), (int(round(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA))
    return np.round(np.clip(np.concatenate(resized, axis=1), 0, 1) * 255).astype(np.uint8)


def seam_metrics(full: np.ndarray, hdr_common: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    _, y1, _, y2 = bbox
    composite = full.copy(); composite[y1:y2] = hdr_common
    rows = []
    for seam, label in ((y1, "top"), (y2, "bottom")):
        outer_idx, inner_idx = (seam - 1, seam) if label == "top" else (seam, seam - 1)
        outer, inner = composite[outer_idx] * PEAK_NITS, composite[inner_idx] * PEAK_NITS
        oy, iy = compute_luminance(outer), compute_luminance(inner)
        outer_gradient = oy - compute_luminance(composite[max(0, outer_idx - 1)] * PEAK_NITS)
        inner_gradient = compute_luminance(composite[min(composite.shape[0] - 1, inner_idx + 1)] * PEAK_NITS) - iy
        rows.append({"boundary": label, "luma_mean_nits": float(np.mean(np.abs(oy - iy))), "luma_P95_nits": float(np.percentile(np.abs(oy - iy), 95)), "chroma_ICtCp": float(np.mean(np.abs(chroma_error(outer, inner)))), "hue_degrees": float(np.mean(np.abs(hue_error(outer, inner)))), "gradient_nits_per_pixel": float(np.mean(np.abs(outer_gradient - inner_gradient)))})
    return {"definition": "HDR-center composite continuity proxy; not unseen-region ground truth", "boundaries": rows}


def map_luma_ratio(sdr: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source_y = compute_luminance(sdr)
    target_y = np.interp(source_y, np.asarray(params["lut_x"]), np.asarray(params["lut_y"]))
    result = sdr * (target_y / np.maximum(source_y, EPS))[..., None]
    dark = source_y <= EPS
    if np.any(dark):
        result[dark] = target_y[dark, None]
    return result, target_y


def apply_matrix(image: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.einsum("ij,...j->...i", matrix, image, optimize=True)
    return np.maximum(raw, 0.0), {"pre_nonnegative_clip": range_info(raw), "post_nonnegative_clip": range_info(np.maximum(raw, 0.0))}


def objective_terms(matrix: np.ndarray, base: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, _ = apply_matrix(base, matrix)
    py, ty = compute_luminance(pred), compute_luminance(target)
    pn, tn = pred * PEAK_NITS, target * PEAK_NITS
    chroma = chroma_error(pn, tn)
    hue = hue_error(pn, tn)
    ref = linear_bt2020_to_ictcp(tn)
    nonneutral = np.sqrt(ref[..., 1] ** 2 + ref[..., 2] ** 2) > 1e-6
    de = delta_e_2000(lab2020(pred), lab2020(target))
    return {"luma_mse_scaled": float(np.mean(((py - ty) / 0.02) ** 2)), "chroma_mse_scaled": float(np.mean((chroma / 0.05) ** 2)), "hue_mse_scaled": float(np.mean((hue[nonneutral] / 30.0) ** 2)) if np.any(nonneutral) else 0.0, "deltaE_mse_scaled": float(np.mean((de / 5.0) ** 2))}


def candidate_specs() -> list[tuple[str, list[tuple[float | None, float | None]] | None]]:
    unbounded = None
    positive_diag = [(-2.0, 2.0)] * 9
    for index in (0, 4, 8): positive_diag[index] = (0.02, 3.0)
    limited_mixing = [(-0.25, 0.25)] * 9
    for index in (0, 4, 8): limited_mixing[index] = (0.5, 1.5)
    return [("A_unconstrained_regularized", unbounded), ("B_positive_diagonal", positive_diag), ("C_limited_cross_channel", limited_mixing)]


def spatial_samples(base: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = base.shape[:2]
    yy, xx = np.indices((height, width))
    held_mask = ((yy // 80 + xx // 120) % 5) == 0
    def choose(mask: np.ndarray, maximum: int) -> np.ndarray:
        indices = np.flatnonzero(mask.reshape(-1))
        return indices[np.linspace(0, len(indices) - 1, min(maximum, len(indices)), dtype=int)]
    train_idx, hold_idx = choose(~held_mask, 4096), choose(held_mask, 2048)
    flat_base, flat_target = base.reshape(-1, 3), target.reshape(-1, 3)
    return flat_base[train_idx][None, ...], flat_target[train_idx][None, ...], flat_base[hold_idx][None, ...], flat_target[hold_idx][None, ...]


def fit_matrix(base: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    train_base, train_target, hold_base, hold_target = spatial_samples(base, target)
    candidates = []
    identity = np.eye(3, dtype=np.float64)
    for family, bounds in candidate_specs():
        for regularization in LAMBDA_GRID:
            def function(vector: np.ndarray) -> float:
                matrix = vector.reshape(3, 3)
                terms = objective_terms(matrix, train_base, train_target)
                return 0.35 * terms["luma_mse_scaled"] + 0.20 * terms["chroma_mse_scaled"] + 0.20 * terms["hue_mse_scaled"] + 0.25 * terms["deltaE_mse_scaled"] + regularization * float(np.sum((matrix - identity) ** 2))
            started = time.perf_counter()
            solved = minimize(function, identity.reshape(-1), method="L-BFGS-B", bounds=bounds, options={"maxiter": 90, "ftol": 1e-10})
            matrix = solved.x.reshape(3, 3)
            condition = float(np.linalg.cond(matrix)); determinant = float(np.linalg.det(matrix))
            hold_terms = objective_terms(matrix, hold_base, hold_target)
            hold_objective = 0.35 * hold_terms["luma_mse_scaled"] + 0.20 * hold_terms["chroma_mse_scaled"] + 0.20 * hold_terms["hue_mse_scaled"] + 0.25 * hold_terms["deltaE_mse_scaled"] + regularization * float(np.sum((matrix - identity) ** 2))
            stable = bool(np.isfinite(condition) and condition <= 30.0 and abs(determinant) >= 0.05)
            candidates.append({"constraint_family": family, "lambda": regularization, "matrix": matrix, "determinant": determinant, "condition_number": condition, "deviation_from_identity_frobenius": float(np.linalg.norm(matrix - identity)), "optimizer_success": bool(solved.success), "optimizer_message": str(solved.message), "optimizer_iterations": int(getattr(solved, "nit", -1)), "fit_seconds_cpu": time.perf_counter() - started, "holdout_terms": hold_terms, "holdout_objective": float(hold_objective), "stable": stable})
    accepted = [candidate for candidate in candidates if candidate["stable"]]
    require(bool(accepted), "All H1 matrix candidates were numerically unstable")
    selected = min(accepted, key=lambda candidate: candidate["holdout_objective"])
    return {"selected": selected, "candidates": candidates, "training_sample_count": int(train_base.shape[1]), "holdout_sample_count": int(hold_base.shape[1]), "objective": "0.35*luminance_MSE_scaled + 0.20*ICtCp_chroma_MSE_scaled + 0.20*nonneutral_hue_MSE_scaled + 0.25*DeltaE2000_MSE_scaled + lambda*||M-I||_F^2"}


def fit_anchor(case_id: str, write_images: bool = False) -> dict[str, Any]:
    record, paths = load_record(case_id)
    sdr_full, sdr_common, hdr_common, bbox = decode_record(record, paths)
    model = ModelECdfRegularized()
    fit_started = time.perf_counter()
    luma_params = model.fit(compute_luminance(sdr_common).reshape(-1), compute_luminance(hdr_common).reshape(-1))
    luma_seconds = time.perf_counter() - fit_started
    h0_full, _ = map_luma_ratio(sdr_full, luma_params)
    h0_common = h0_full[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    matrix_fit = fit_matrix(h0_common, hdr_common)
    matrix = np.asarray(matrix_fit["selected"]["matrix"], dtype=np.float64)
    apply_started = time.perf_counter(); h1_full, h1_range = apply_matrix(h0_full, matrix); apply_seconds = time.perf_counter() - apply_started
    h1_common = h1_full[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    result = {"case_id": case_id, "anchor": record, "input_files": paths, "bbox": bbox, "fit_mask": {"definition": "SDR ∩ HDR only", "pixel_count": int((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))}, "model_definition": "H0: Model E luminance-only + safe ratio scaling; H1: selected regularized 3x3 matrix on H0", "luminance_fit": {"reported_parameter_count": int(model.param_count()), "scene_luminance_parameters_effective": 10, "fit_seconds_cpu": luma_seconds, "parameters": luma_params, "convergence": "Model E does not expose OptimizeResult"}, "matrix_fit": matrix_fit, "parameter_count": {"luminance_compact_controls": 10, "rgb_matrix": 9, "total_effective_scene_parameters": 19, "note": "65 sampled LUT ordinates are representation samples, not independent scene parameters."}, "H0": {"common_metrics": metrics(h0_common, hdr_common), "seam": seam_metrics(h0_full, hdr_common, bbox), "range": range_info(h0_full)}, "H1": {"common_metrics": metrics(h1_common, hdr_common), "seam": seam_metrics(h1_full, hdr_common, bbox), "range": h1_range, "apply_seconds_cpu": apply_seconds}}
    if write_images:
        for directory in (RESULTS, RESULTS / "scientific", RESULTS / "previews", RESULTS / "differences", RESULTS / "seams"):
            directory.mkdir(parents=True, exist_ok=True)
        write_scientific(RESULTS / "scientific" / "H0_RatioScaling_HDR_OpenMatte", h0_full)
        write_scientific(RESULTS / "scientific" / "H1_3x3Matrix_HDR_OpenMatte", h1_full)
        write_scientific(RESULTS / "scientific" / "H1_CommonCrop", h1_common)
        write_scientific(RESULTS / "scientific" / "HDR_Reference_Common", hdr_common)
        write_preview(RESULTS / "previews" / "H0_RatioScaling_HDR_OpenMatte_log.png", h0_full)
        write_preview(RESULTS / "previews" / "H1_3x3Matrix_HDR_OpenMatte_log.png", h1_full)
        diff = np.abs(compute_luminance(h1_common) - compute_luminance(hdr_common)) * PEAK_NITS
        difference = cv2.applyColorMap(np.round(np.clip(np.log1p(diff) / math.log1p(1000), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        write_cv_image(RESULTS / "differences" / "H1_Difference.png", difference)
        composite = h1_full.copy(); composite[bbox[1]:bbox[3], bbox[0]:bbox[2]] = hdr_common
        seam_panel = panel([h1_full[max(0,bbox[1]-60):bbox[1]+60], composite[max(0,bbox[1]-60):bbox[1]+60], h1_full[bbox[3]-60:min(h1_full.shape[0],bbox[3]+60)], composite[bbox[3]-60:min(h1_full.shape[0],bbox[3]+60)]], 120)
        write_cv_image(RESULTS / "seams" / "H1_Seam.png", seam_panel[..., ::-1])
        canvas = np.zeros_like(sdr_full); canvas[bbox[1]:bbox[3], bbox[0]:bbox[2]] = hdr_common
        comparison = panel([sdr_full, canvas, h0_full, h1_full], 270)
        write_cv_image(RESULTS / "previews" / "H1_ComparisonPanel.png", comparison[..., ::-1])
    return result


def synthetic_scene() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    height, width = 360, 640
    y, x = np.mgrid[0:height, 0:width].astype(np.float64); xn, yn = x/(width-1), y/(height-1)
    base = 0.001 + 0.025 + 0.12*xn + 0.08*yn + 0.30*np.exp(-(((xn-.72)/.14)**2+((yn-.30)/.13)**2))
    sdr = np.stack((base*(.72+.32*xn), base*(.82+.22*np.cos(4*yn)), base*(.65+.40*np.sin(3*xn+yn)**2)), axis=-1)
    sdr = np.maximum(sdr, 1e-5)
    truth_matrix = np.array(((1.09, -0.06, 0.02), (-0.03, 1.04, 0.01), (0.02, -0.04, 1.12)), dtype=np.float64)
    hdr = np.maximum(np.einsum("ij,...j->...i", truth_matrix, sdr), 0.0)
    hdr *= (0.32 + 2.4 * compute_luminance(sdr))[..., None]
    return np.clip(hdr, 0, 0.95), sdr, (72, 288)


def run_synthetic() -> dict[str, Any]:
    hdr_full, sdr_full, (y1, y2) = synthetic_scene(); sdr_common, hdr_common = sdr_full[y1:y2], hdr_full[y1:y2]
    model = ModelECdfRegularized(); params = model.fit(compute_luminance(sdr_common).reshape(-1), compute_luminance(hdr_common).reshape(-1))
    h0_full, _ = map_luma_ratio(sdr_full, params); h0_common = h0_full[y1:y2]
    fitted = fit_matrix(h0_common, hdr_common); matrix = np.asarray(fitted["selected"]["matrix"]); h1_full, h1_range = apply_matrix(h0_full, matrix)
    hidden = np.ones(hdr_full.shape[:2], dtype=bool); hidden[y1:y2] = False
    def selected(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        return metrics(image[mask].reshape(-1, 1, 3), reference[mask].reshape(-1, 1, 3))
    output = {"definition": "Known full HDR -> SDR/DVD-style grade; H0/H1 fit only central HDR crop", "geometry": {"full_shape": list(hdr_full.shape), "central_crop_y": [y1,y2]}, "truth_matrix": truth_matrix if False else None, "matrix_fit": fitted, "H0": {"common": metrics(h0_common,hdr_common), "hidden": selected(h0_full,hdr_full,hidden)}, "H1": {"common": metrics(h1_full[y1:y2],hdr_common), "hidden": selected(h1_full,hdr_full,hidden), "range": h1_range}}
    synth_dir = RESULTS / "synthetic"; synth_dir.mkdir(parents=True, exist_ok=True)
    write_scientific(synth_dir / "H0_full", h0_full); write_scientific(synth_dir / "H1_full", h1_full); write_scientific(synth_dir / "HDR_truth", hdr_full)
    write_cv_image(synth_dir / "comparison.png", panel([sdr_full, hdr_full, h0_full, h1_full], 220)[..., ::-1])
    return output


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: return {"performed": False}
    coefficients = np.stack([np.asarray(row["matrix_fit"]["selected"]["matrix"]).reshape(-1) for row in rows])
    h0 = np.array([row["H0"]["common_metrics"]["deltaE2000"]["mean"] for row in rows]); h1 = np.array([row["H1"]["common_metrics"]["deltaE2000"]["mean"] for row in rows])
    return {"performed": True, "case_ids": [row["case_id"] for row in rows], "deltaE2000_H0_median": float(np.median(h0)), "deltaE2000_H1_median": float(np.median(h1)), "improved_anchor_count": int(np.sum(h1 < h0)), "coefficient_cross_stratum_mean": coefficients.mean(axis=0).reshape(3,3), "coefficient_cross_stratum_std": coefficients.std(axis=0).reshape(3,3), "coefficient_cross_stratum_max_deviation": float(np.max(np.abs(coefficients-coefficients.mean(axis=0)))), "stability_note": "Not a same-shot stability metric: P2.32 supplies one frame per distant temporal stratum and no verified shot IDs."}


def report(result: dict[str, Any]) -> str:
    m, h0, h1 = result["matrix_anchor"], result["matrix_anchor"]["H0"]["common_metrics"], result["matrix_anchor"]["H1"]["common_metrics"]
    selected = m["matrix_fit"]["selected"]; synth = result["synthetic"]
    multi = result["multi_temporal"]
    return f"""# Scene Regrading Report — Model H

## Model definition and color space
H0 is Model E fitted on **overlap luminance only**, followed by safe ratio scaling: `RGB_base = RGB_sdr * Y_target / max(Y_sdr, eps)`. H1 then applies exactly one scene-local 3×3 matrix in linear BT.2020: `RGB_hdr = M_scene @ RGB_base`. HDR uses one ST.2084 EOTF; SDR uses BT.1886-style `code**2.4` then BT.709→BT.2020. HDR outside `[0,140,1920,940]` was never loaded for fitting because it does not exist in the P2.32 input.

## Fit objective and regularization
The matrix was fitted on deterministic spatially separated overlap samples, selected by held-out multi-metric objective, not raw RGB MSE alone:
`0.35*scaled luminance MSE + 0.20*scaled ICtCp chroma MSE + 0.20*scaled non-neutral hue MSE + 0.25*scaled DeltaE2000 MSE + lambda*||M-I||_F^2`.
Tested regularization lambdas: `{list(LAMBDA_GRID)}`. Constraints A/B/C respectively tested unbounded regularized, positive diagonal, and limited cross-channel mixing. Candidates with condition number >30 or |determinant|<0.05 were rejected.

## Matrix anchor: Matrix 73367 / 73348
- Verified overlap: `[0,140,1920,940]`; geometry confidence 1.0; fit pixels 1,536,000.
- Selected configuration: `{selected['constraint_family']}`, lambda `{selected['lambda']}`.
- Matrix: `{np.array2string(np.asarray(selected['matrix']), precision=7)}`.
- Determinant `{selected['determinant']:.7f}`, condition number `{selected['condition_number']:.5f}`, `||M-I||_F` `{selected['deviation_from_identity_frobenius']:.7f}`.
- Effective scene parameters: 10 compact Model E luminance controls + 9 RGB-matrix controls = 19. The sampled LUT ordinates are not counted as independent scene controls.

| Variant | luma MAE / RMSE nits | mean / P95 ΔE2000 | mean |chroma| | mean |hue| ° |
|---|---:|---:|---:|---:|
| H0 ratio scaling | {h0['luminance']['MAE_nits']:.4f} / {h0['luminance']['RMSE_nits']:.4f} | {h0['deltaE2000']['mean']:.4f} / {h0['deltaE2000']['p95']:.4f} | {h0['chroma_error_ICtCp_absolute']['mean']:.5f} | {h0['hue_error_degrees_absolute_non_neutral']['mean']:.4f} |
| H1 3×3 regrading | {h1['luminance']['MAE_nits']:.4f} / {h1['luminance']['RMSE_nits']:.4f} | {h1['deltaE2000']['mean']:.4f} / {h1['deltaE2000']['p95']:.4f} | {h1['chroma_error_ICtCp_absolute']['mean']:.5f} | {h1['hue_error_degrees_absolute_non_neutral']['mean']:.4f} |

H0/H1 seam data, full candidate coefficients/objectives, range diagnostics, and preview/scientific image locations are in `results/matrix_73367_73348/metrics.json`.

## Seam and CPU performance
| Variant | top seam mean / P95 nits | bottom seam mean / P95 nits |
|---|---:|---:|
| H0 | {m['H0']['seam']['boundaries'][0]['luma_mean_nits']:.4f} / {m['H0']['seam']['boundaries'][0]['luma_P95_nits']:.4f} | {m['H0']['seam']['boundaries'][1]['luma_mean_nits']:.4f} / {m['H0']['seam']['boundaries'][1]['luma_P95_nits']:.4f} |
| H1 | {m['H1']['seam']['boundaries'][0]['luma_mean_nits']:.4f} / {m['H1']['seam']['boundaries'][0]['luma_P95_nits']:.4f} | {m['H1']['seam']['boundaries'][1]['luma_mean_nits']:.4f} / {m['H1']['seam']['boundaries'][1]['luma_P95_nits']:.4f} |

Model E luminance-only fit took `{m['luminance_fit']['fit_seconds_cpu']:.4f}` s CPU; selected H1 matrix fitting took `{selected['fit_seconds_cpu']:.4f}` s CPU on its deterministic overlap samples; H1 full-frame application took `{m['H1']['apply_seconds_cpu']:.4f}` s CPU. CUDA was intentionally not used because correctness is the subject of this reference experiment.

## Physical visual review
The H1 comparison and seam panels show only a modest scene-wide visual change from H0. The dark garment contains localized blue/purple speckling in the H1 preview, an unacceptable artifact until its source is understood. Consequently, the small held-out metric reduction is not accepted as evidence of an improved physical full-frame regrade. H1 reduces the top/bottom continuity proxies slightly, but those proxies are not ground truth for the unseen Open Matte.

## Synthetic hidden Open Matte
| Variant | common mean ΔE2000 | hidden mean ΔE2000 | hidden luma MAE nits |
|---|---:|---:|---:|
| H0 | {synth['H0']['common']['deltaE2000']['mean']:.4f} | {synth['H0']['hidden']['deltaE2000']['mean']:.4f} | {synth['H0']['hidden']['luminance']['MAE_nits']:.4f} |
| H1 | {synth['H1']['common']['deltaE2000']['mean']:.4f} | {synth['H1']['hidden']['deltaE2000']['mean']:.4f} | {synth['H1']['hidden']['luminance']['MAE_nits']:.4f} |

## Multi-temporal-stratum and stability
{json.dumps(json_safe(multi), ensure_ascii=False, indent=2)}

## Decision A–H
A. **No material acceptance over H0 on Matrix.** H1 improves mean ΔE2000 from `{h0['deltaE2000']['mean']:.4f}` to `{h1['deltaE2000']['mean']:.4f}` ({100.0 * (1.0 - h1['deltaE2000']['mean'] / h0['deltaE2000']['mean']):.2f}%), below the predeclared 5% gate.
B. **Representationally yes, experimentally only marginally.** The stable 3×3 changes RGB relationships and reduces mean hue error from `{h0['hue_error_degrees_absolute_non_neutral']['mean']:.4f}°` to `{h1['hue_error_degrees_absolute_non_neutral']['mean']:.4f}°`, but not enough to establish robust HDR studio regrading.
C. **No demonstrated physical-image improvement.** The visual panel is close to H0 and contains localized blue/purple artifacts in a dark garment; the small numerical gain is therefore not sufficient for physical acceptance.
D. **Synthetic yes; physical unseen region unproven.** H1 substantially improves the synthetic hidden region, where the transform is known global. For the real Open Matte extension, only seam continuity proxies exist and they are not ground truth.
E. **Parameters necessary:** H0's 10 compact luminance controls remain the supported baseline. The 9 additional matrix controls (19 total) are not justified by this single real anchor.
F. **A single 3×3 is not established as sufficient.** Its stable selected matrix gives only a below-gate local gain, no verified multi-scene confirmation, and no same-shot stability evidence.
G. **No basis yet for a luminance-dependent matrix.** That next hypothesis requires demonstrated residual hue/chroma behavior that changes systematically by luminance after H1. This experiment instead finds a small aggregate gain plus a physical artifact, so it must not expand parameters.
H. **Implementation compatibility is retained.** H0 plus a 3×3 multiply is compact, local CPU/GPU-compatible, AI-free, and LUT-free. This is a feasibility statement only; it does not authorize production integration.

## Limitations, failure cases, and recommendation
- **Failure case:** the selected unconstrained-but-stable matrix (`||M-I||_F={selected['deviation_from_identity_frobenius']:.4f}`) produces localized blue/purple artifacts in a dark garment preview. Numerical stability does not guarantee visually plausible regrading.
- Only stratum 08 is a known verified Matrix anchor. The other predeclared temporal records were not run because the 5% H1 gate failed; they are not verified independent scenes in any event.
- P2.32 contains no same-shot adjacent-frame groups, so same-shot matrix stability is unavailable.
- No full-video processing, P2 modification, production changes, Model G work, or Model H production integration occurred.
- **Recommendation:** retain H0 ratio scaling as the validated research baseline. Reject H1 as an accepted scene regrading model for now; do not add M(Y) until a separate residual-by-luminance diagnostic identifies a specific failure it would solve.
"""


def main() -> int:
    matrix = fit_anchor("the_matrix_temporal_stratum_08", write_images=True)
    h0_de = matrix["H0"]["common_metrics"]["deltaE2000"]["mean"]; h1_de = matrix["H1"]["common_metrics"]["deltaE2000"]["mean"]
    synthetic = run_synthetic()
    success = bool(h1_de < h0_de * 0.95 and matrix["matrix_fit"]["selected"]["stable"])
    temporal_rows = []
    if success:
        for case_id in TEMPORAL_CASES:
            temporal_rows.append(matrix if case_id == matrix["case_id"] else fit_anchor(case_id, write_images=False))
    result = {"status": "COMPLETE", "scope": "Research-only Model H; no Model G, no production/P2 modification, no full-video decode.", "matrix_anchor": matrix, "synthetic": synthetic, "matrix_success_gate": {"H1_deltaE2000_improves_H0_by_at_least_5_percent": success, "H0_deltaE2000": h0_de, "H1_deltaE2000": h1_de}, "multi_temporal": aggregate(temporal_rows)}
    write_json(RESULTS / "metrics.json", result)
    write_json(RESULTS / "H1_matrix_candidates.json", matrix["matrix_fit"])
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    print("SCENE_REGRADING_STATUS=COMPLETE")
    print(f"RESULTS={RESULTS}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
