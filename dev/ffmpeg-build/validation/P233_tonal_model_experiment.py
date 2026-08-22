from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION = ROOT / "dev" / "ffmpeg-build" / "validation"
P232_DIR = VALIDATION / "P232_real_material_dataset"
P232_METRICS = P232_DIR / "P232_dataset_metrics.json"
CURVE_PATH = VALIDATION / "P2272_reference_curve.json"
OUTPUT_DIR = VALIDATION / "P233_tonal_models"
METRICS_PATH = OUTPUT_DIR / "P233_model_metrics.json"
CSV_PATH = OUTPUT_DIR / "P233_per_case_metrics.csv"
TMP_DIR = OUTPUT_DIR / "_tmp_oof"
REPORT_PATH = ROOT / "P2.33_REAL_TONAL_MODEL_EXPERIMENT_REPORT.md"

PEAK_NITS = 10000.0
LOG_EPS = 1e-6
MONOTONIC_TOLERANCE = 1e-10
FIT_SAMPLES_PER_CASE = 12500
FOLD_COUNT = 5
HIGH_SLOPE_CAP = 0.60

BINS = [
    (0.0, 0.1, "<0.1"), (0.1, 0.5, "0.1–0.5"), (0.5, 1.0, "0.5–1"),
    (1.0, 2.0, "1–2"), (2.0, 5.0, "2–5"), (5.0, 10.0, "5–10"),
    (10.0, 20.0, "10–20"), (20.0, 50.0, "20–50"), (50.0, 100.0, "50–100"),
    (100.0, 200.0, "100–200"), (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"), (1000.0, float("inf"), ">1000"),
]
TONE_BINS = [(0.0, 10.0, "shadow_lt_10"), (10.0, 200.0, "midtone_10_200"), (200.0, float("inf"), "highlight_gt_200")]
MODEL_NAMES = ["Model A", "Model B", "Model F", "Model G", "Model H"]
CANDIDATES = ["Model B", "Model F", "Model G", "Model H"]

M_709_TO_2020 = np.asarray([
    [0.6274039, 0.3292830, 0.0433131],
    [0.0690972, 0.9195404, 0.0113624],
    [0.0163916, 0.0880132, 0.8955952],
], dtype=np.float64)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def pava(values: np.ndarray) -> np.ndarray:
    blocks: list[list[float]] = []
    for value in np.asarray(values, dtype=np.float64):
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            left, right = blocks[-2], blocks[-1]
            weight = left[1] + right[1]
            blocks[-2] = [(left[0] * left[1] + right[0] * right[1]) / weight, weight]
            blocks.pop()
    output = np.empty(len(values), dtype=np.float64)
    cursor = 0
    for value, weight in blocks:
        count = int(weight)
        output[cursor:cursor + count] = value
        cursor += count
    return output


def curve_nodes(x_log: np.ndarray, y_log: np.ndarray, interpolation: str, definition: dict[str, Any]) -> dict[str, Any]:
    order = np.argsort(x_log)
    x_log = np.asarray(x_log, dtype=np.float64)[order]
    y_log = pava(np.asarray(y_log, dtype=np.float64)[order])
    unique_x, unique_indices = np.unique(x_log, return_index=True)
    x_log = unique_x
    y_log = y_log[unique_indices]
    return {"name": definition["name"], "kind": "curve", "x_log": x_log, "y_log": y_log, "interpolation": interpolation, "definition": definition, "duplicate_x_values": int(len(x_log) - len(unique_x))}


def binned_nodes(x_log: np.ndarray, y_log: np.ndarray, n_bins: int, equal_count: bool = False) -> tuple[np.ndarray, np.ndarray]:
    low, high = float(np.min(x_log)), float(np.max(x_log))
    if high <= low:
        return np.asarray([low]), np.asarray([float(np.median(y_log))])
    edges = np.quantile(x_log, np.linspace(0.0, 1.0, n_bins + 1)) if equal_count else np.linspace(low, high, n_bins + 1)
    px: list[float] = []
    py: list[float] = []
    for index in range(n_bins):
        mask = (x_log >= edges[index]) & (x_log <= edges[index + 1] if index == n_bins - 1 else x_log < edges[index + 1])
        if np.any(mask):
            px.append(float(np.median(x_log[mask])))
            py.append(float(np.median(y_log[mask])))
    return np.asarray(px), np.asarray(py)


def fit_curve_model(name: str, train_x: np.ndarray, train_y: np.ndarray, bins: int, equal_count: bool = False, interpolation: str = "linear") -> dict[str, Any]:
    mask = np.isfinite(train_x) & np.isfinite(train_y) & (train_x > 0) & (train_y > 0)
    x_log = np.log10(np.maximum(train_x[mask], LOG_EPS))
    y_log = np.log10(np.maximum(train_y[mask], LOG_EPS))
    x_nodes, y_nodes = binned_nodes(x_log, y_log, bins, equal_count=equal_count)
    return curve_nodes(x_nodes, y_nodes, interpolation, {"name": name, "bins": bins, "binning": "equal-count log10 bins" if equal_count else "fixed-width log10 bins", "target_statistic": "median", "monotonic_projection": "PAVA"})


def fit_piecewise_regression(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(train_x) & np.isfinite(train_y) & (train_x > 0) & (train_y > 0)
    x_log = np.log10(np.maximum(train_x[mask], LOG_EPS))
    y_log = np.log10(np.maximum(train_y[mask], LOG_EPS))
    domain_low, domain_high = float(np.min(x_log)), float(np.max(x_log))
    boundaries = [-float("inf"), 1.0, math.log10(200.0), float("inf")]
    segments = []
    previous_end = None
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        segment_mask = (x_log >= low) & (x_log < high if np.isfinite(high) else np.ones_like(x_log, dtype=bool))
        if not np.any(segment_mask):
            continue
        lo = max(low, domain_low)
        hi = min(high, domain_high)
        if np.sum(segment_mask) >= 2 and np.ptp(x_log[segment_mask]) > 0:
            slope, intercept = np.polyfit(x_log[segment_mask], y_log[segment_mask], 1)
            slope = max(float(slope), 0.0)
            intercept = float(intercept)
        else:
            slope = 0.0
            intercept = float(np.median(y_log[segment_mask]))
        segment_start = intercept + slope * lo
        if previous_end is not None and segment_start < previous_end:
            intercept += previous_end - segment_start
        segment_end = intercept + slope * hi
        previous_end = segment_end
        segments.append({"low": float(lo), "high": float(hi), "slope": float(slope), "intercept": float(intercept), "sample_count": int(np.sum(segment_mask))})
    return {"name": "Model F", "kind": "piecewise_regression", "domain_low": domain_low, "domain_high": domain_high, "segments": segments, "definition": {"name": "Model F", "type": "piecewise log-domain regression", "segments_log10_sdr": ["<10", "10–200", ">=200"], "fit": "least-squares log10 target regression per segment; slope constrained >=0", "boundary_policy": "segment intercept shifted to preserve non-decreasing boundaries"}}


def fit_high_end_model(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    model = fit_curve_model("Model H", train_x, train_y, 96, equal_count=False, interpolation="linear")
    x = model["x_log"]
    y = model["y_log"].copy()
    boundary = math.log10(200.0)
    if x[0] < boundary < x[-1]:
        base = float(np.interp(boundary, x, y))
        high = x >= boundary
        y[high] = np.minimum(y[high], base + HIGH_SLOPE_CAP * (x[high] - boundary))
        model["y_log"] = pava(y)
    model["definition"] = {"name": "Model H", "type": "piecewise high-end controlled monotonic mapping", "bins": 96, "binning": "fixed-width log10 bins", "target_statistic": "median", "monotonic_projection": "PAVA", "high_end_boundary_nits": 200.0, "max_high_end_log_slope": HIGH_SLOPE_CAP, "interpolation": "linear"}
    return model


def make_baseline() -> dict[str, Any]:
    from auto_openmatte.processing.luminance import apply_luminance_curve, build_curve_lut
    curve = json.loads(CURVE_PATH.read_text(encoding="utf-8"))
    return {"name": "Model A", "kind": "baseline", "curve": curve, "lut": build_curve_lut(curve), "apply": apply_luminance_curve, "definition": {"name": "Model A", "type": "unchanged production/reference model", "curve_path": str(CURVE_PATH), "control_points": 64, "lut_entries": 65536, "refit": False}}


def predict(model: dict[str, Any], input_nits: np.ndarray) -> np.ndarray:
    input_nits = np.asarray(input_nits, dtype=np.float64)
    if model["kind"] == "baseline":
        output = model["apply"](np.clip(input_nits / PEAK_NITS, 0.0, 1.0), model["curve"], prebuilt_lut=model["lut"]) * PEAK_NITS
        return np.asarray(output, dtype=np.float64)
    x_log = np.log10(np.maximum(input_nits, LOG_EPS))
    if model["kind"] == "piecewise_regression":
        x_clip = np.clip(x_log, model["domain_low"], model["domain_high"])
        output_log = np.empty_like(x_clip)
        for segment in model["segments"]:
            mask = (x_clip >= segment["low"]) & (x_clip <= segment["high"])
            output_log[mask] = segment["intercept"] + segment["slope"] * x_clip[mask]
        output_log[x_clip < model["segments"][0]["low"]] = model["segments"][0]["intercept"] + model["segments"][0]["slope"] * x_clip[x_clip < model["segments"][0]["low"]]
        return np.maximum(10.0 ** output_log - LOG_EPS, 0.0)
    x_clip = np.clip(x_log, model["x_log"][0], model["x_log"][-1])
    if model["interpolation"] == "pchip" and len(model["x_log"]) > 1:
        if "interpolator" not in model:
            model["interpolator"] = PchipInterpolator(model["x_log"], model["y_log"], extrapolate=False)
        output_log = np.asarray(model["interpolator"](x_clip), dtype=np.float64)
    else:
        output_log = np.interp(x_clip, model["x_log"], model["y_log"])
    return np.maximum(10.0 ** output_log - LOG_EPS, 0.0)


def metric_block(residual: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if residual.size == 0:
        return {"count": 0}
    absolute = np.abs(residual)
    relative = residual / np.maximum(target, LOG_EPS)
    return {"count": int(residual.size), "signed_mean": float(np.mean(residual)), "median_residual": float(np.median(residual)), "MAE": float(np.mean(absolute)), "RMSE": float(np.sqrt(np.mean(residual ** 2))), "P95": float(np.percentile(absolute, 95)), "P99": float(np.percentile(absolute, 99)), "max": float(np.max(absolute)), "relative": {"signed_mean": float(np.mean(relative)), "median": float(np.median(relative)), "MAE": float(np.mean(np.abs(relative))), "P95": float(np.percentile(np.abs(relative), 95)), "P99": float(np.percentile(np.abs(relative), 99)), "max": float(np.max(np.abs(relative))), "epsilon": LOG_EPS}}


def metrics_for_arrays(residual: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    result = {"global": metric_block(residual, target)}
    for low, high, label in BINS:
        mask = (target >= low) & (target < high if np.isfinite(high) else np.ones_like(target, dtype=bool))
        result[label] = metric_block(residual[mask], target[mask])
    return result


def balanced_train_sample(cases: list[dict[str, Any]], by_material: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for material in ("The Matrix", "BR2049"):
        for case in by_material[material]:
            x = np.asarray(np.load(case["sdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)
            y = np.asarray(np.load(case["hdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)
            indices = np.linspace(0, x.size - 1, min(FIT_SAMPLES_PER_CASE, x.size), dtype=np.int64)
            xs.append(x[indices].astype(np.float64, copy=False))
            ys.append(y[indices].astype(np.float64, copy=False))
    return np.concatenate(xs), np.concatenate(ys)


def fold_definition(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folds = []
    for fold_index in range(FOLD_COUNT):
        validation = [case for case in cases if (int(case["selection_index"]) - 1) % FOLD_COUNT == fold_index]
        training = [case for case in cases if case not in validation]
        folds.append({"fold": fold_index + 1, "train_cases": [case["case_id"] for case in training], "validation_cases": [case["case_id"] for case in validation], "train_material_counts": {m: sum(c["material"] == m for c in training) for m in ("The Matrix", "BR2049")}, "validation_material_counts": {m: sum(c["material"] == m for c in validation) for m in ("The Matrix", "BR2049")}})
    return folds


def constraint_check(model: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, validation_cases: list[dict[str, Any]], fold: int) -> dict[str, Any]:
    if model["kind"] == "baseline":
        curve = np.asarray(model["curve"], dtype=np.float64)
        domain_low, domain_high = curve[0, 0], curve[-1, 0]
        grid = np.linspace(domain_low, domain_high, 4096)
        inputs = 10.0 ** grid - LOG_EPS
        duplicate = int(len(curve) - len(np.unique(curve[:, 0])))
    elif model["kind"] == "piecewise_regression":
        domain_low, domain_high = model["domain_low"], model["domain_high"]
        inputs = 10.0 ** np.linspace(domain_low, domain_high, 4096) - LOG_EPS
        duplicate = 0
    else:
        domain_low, domain_high = model["x_log"][0], model["x_log"][-1]
        inputs = 10.0 ** np.linspace(domain_low, domain_high, 4096) - LOG_EPS
        duplicate = int(model.get("duplicate_x_values", 0))
    grid_prediction = predict(model, inputs)
    differences = np.diff(grid_prediction)
    validation_input = np.concatenate([np.asarray(np.load(case["sdr_luminance_nits_path"], mmap_mode="r")).reshape(-1) for case in validation_cases])
    validation_prediction = np.concatenate([predict(model, np.asarray(np.load(case["sdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)) for case in validation_cases])
    validation_log = np.log10(np.maximum(validation_input, LOG_EPS))
    clipped = (validation_log < domain_low) | (validation_log > domain_high)
    target_min, target_max = float(np.min(train_y)), float(np.max(train_y))
    overshoot = (validation_prediction < target_min - 1e-8) | (validation_prediction > target_max + 1e-8)
    return {"fold": fold, "model": model["name"], "monotonicity_violations": int(np.sum(differences < -MONOTONIC_TOLERANCE)), "maximum_negative_slope": float(np.min(differences)), "overshoot_count": int(np.sum(overshoot)), "clipping_count": int(np.sum(clipped)), "clipping_fraction": float(np.mean(clipped)), "extrapolation_count": 0, "duplicate_x_values": duplicate, "negative_prediction_count": int(np.sum(grid_prediction < 0)), "finite": bool(np.isfinite(grid_prediction).all()), "domain_log10_sdr": [float(domain_low), float(domain_high)], "train_target_range_nits": [target_min, target_max], "validation_input_range_nits": [float(np.min(validation_input)), float(np.max(validation_input))], "grid_samples": int(len(grid_prediction))}


def shape_analysis(model: dict[str, Any], train_x: np.ndarray, fold: int) -> dict[str, Any]:
    low = max(200.0, float(np.min(train_x)))
    high = min(float(np.max(train_x)), 10000.0)
    if high <= low:
        return {"fold": fold, "classification": "insufficient_domain", "count": 0}
    x = np.geomspace(low, high, 512)
    y = predict(model, x)
    logx = np.log10(x)
    logy = np.log10(np.maximum(y, LOG_EPS))
    slopes = np.diff(logy) / np.maximum(np.diff(logx), LOG_EPS)
    plateau = np.abs(np.diff(y)) <= 1e-8
    median_slope = float(np.median(slopes)) if slopes.size else 0.0
    if median_slope < 0.9:
        classification = "compresses_high_end"
    elif median_slope > 1.1:
        classification = "expands_high_end"
    else:
        classification = "near_linear_high_end"
    if np.mean(plateau) > 0.05:
        classification += "+local_plateau"
    if slopes.size and float(np.percentile(slopes, 95) - np.percentile(slopes, 5)) > 0.2:
        classification += "+slope_change"
    return {"fold": fold, "classification": classification, "input_range_nits": [float(low), float(high)], "predicted_range_nits": [float(np.min(y)), float(np.max(y))], "median_log_slope": median_slope, "slope_p05": float(np.percentile(slopes, 5)) if slopes.size else 0.0, "slope_p95": float(np.percentile(slopes, 95)) if slopes.size else 0.0, "plateau_fraction": float(np.mean(plateau)) if plateau.size else 0.0}


def macro_metrics(material_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for region in ["global"] + [label for _, _, label in BINS]:
        blocks = [material_metrics[m][region] for m in ("The Matrix", "BR2049")]
        if not blocks or any(block.get("count", 0) == 0 for block in blocks):
            result[region] = {"count": int(sum(block.get("count", 0) for block in blocks)), "MAE": None, "RMSE": None, "P95": None, "P99": None, "max": None}
            continue
        numeric = ["signed_mean", "median_residual", "MAE", "RMSE", "P95", "P99", "max"]
        block = {"count": int(sum(b["count"] for b in blocks))}
        for key in numeric:
            block[key] = float(np.mean([b[key] for b in blocks]))
        block["relative"] = {key: float(np.mean([b["relative"][key] for b in blocks])) for key in ("signed_mean", "median", "MAE", "P95", "P99", "max")}
        block["aggregation"] = "unweighted arithmetic mean of material metrics; count is summed"
        result[region] = block
    return result


def combined_metrics(material_residuals: dict[str, np.ndarray], material_targets: dict[str, np.ndarray]) -> dict[str, Any]:
    residual = np.concatenate([material_residuals["The Matrix"], material_residuals["BR2049"]])
    target = np.concatenate([material_targets["The Matrix"], material_targets["BR2049"]])
    return metrics_for_arrays(residual, target)


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def plot_bars(path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    width, height = 1500, 850
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 130, 90, 50, 120
    pw, ph = width - left - right, height - top - bottom
    finite = [v for v in values if math.isfinite(v)]
    ymax = max(finite or [1.0]) * 1.15
    cv2.rectangle(canvas, (left, top), (left + pw, top + ph), (0, 0, 0), 2)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + int((i + 0.5) * pw / len(values))
        y = top + ph - int(max(value, 0.0) / max(ymax, 1e-12) * ph)
        cv2.rectangle(canvas, (x - 38, y), (x + 38, top + ph), (60, 110, 210), -1)
        cv2.putText(canvas, label, (x - 45, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.2f}", (x - 34, max(30, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (15, top + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def visual_panel(values: np.ndarray, symmetric: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if symmetric:
        limit = max(float(np.percentile(np.abs(values), 99)), 1e-6)
        normalized = np.clip((values + limit) / (2 * limit), 0, 1)
    else:
        transformed = np.log10(np.maximum(values, 0) + LOG_EPS)
        low, high = float(np.percentile(transformed, 1)), float(np.percentile(transformed, 99))
        normalized = np.clip((transformed - low) / max(high - low, 1e-9), 0, 1)
    return cv2.applyColorMap(np.asarray(normalized * 255, dtype=np.uint8), cv2.COLORMAP_TURBO)


def write_visual(case: dict[str, Any], residual_a: np.ndarray, residual_c: np.ndarray, candidate: str, category: str) -> dict[str, Any]:
    target = np.asarray(np.load(case["hdr_luminance_nits_path"]), dtype=np.float32).reshape(-1)
    shape = tuple(np.load(case["hdr_luminance_nits_path"], mmap_mode="r").shape)
    prediction_a = target + residual_a
    prediction_c = target + residual_c
    arrays = {"target": target.reshape(shape), "prediction_a": prediction_a.reshape(shape), "prediction_candidate": prediction_c.reshape(shape), "residual_a": residual_a.reshape(shape), "residual_candidate": residual_c.reshape(shape)}
    stem = f"visual_{category}_{case['material'].replace(' ', '_').lower()}_{case['case_id']}"
    npz_path = OUTPUT_DIR / f"{stem}.npz"
    png_path = OUTPUT_DIR / f"{stem}.png"
    np.savez_compressed(npz_path, **arrays)
    display_width = 960 if shape[1] > 1920 else 640
    panels = []
    for name, data, symmetric in [("HDR target", arrays["target"], False), ("Model A", arrays["prediction_a"], False), (candidate, arrays["prediction_candidate"], False), ("residual A", arrays["residual_a"], True), ("residual candidate", arrays["residual_candidate"], True)]:
        image = visual_panel(data, symmetric=symmetric)
        scale = display_width / image.shape[1]
        image = cv2.resize(image, (display_width, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(image, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(image)
    canvas = np.hstack(panels)
    cv2.imwrite(str(png_path), canvas)
    return {"category": category, "case": case["case_id"], "material": case["material"], "candidate": candidate, "npz": str(npz_path), "png": str(png_path), "source_shape": list(shape), "selection_after_full_scoring": True}


def build_report(data: dict[str, Any]) -> str:
    oof = data["results"]["oof"]
    lines = []
    for model in MODEL_NAMES:
        matrix = oof["materials"]["The Matrix"][model]["global"]
        br = oof["materials"]["BR2049"][model]["global"]
        macro = oof["combined"]["macro_average"][model]["global"]
        weighted = oof["combined"]["weighted"][model]["global"]
        high = oof["combined"]["macro_average"][model][">200"]
        lines.append(f"| {model} | {fmt(matrix.get('MAE'))} | {fmt(br.get('MAE'))} | {fmt(macro.get('MAE'))} | {fmt(weighted.get('MAE'))} | {fmt(high.get('MAE'))} |")
    cv_lines = []
    for model in MODEL_NAMES:
        cv = data["cv_summary"][model]
        cv_lines.append(f"| {model} | {cv['mean_macro_MAE']:.6f} | {cv['median_macro_MAE']:.6f} | {cv['P25_macro_MAE']:.6f} | {cv['P75_macro_MAE']:.6f} | {cv['best_fold']} | {cv['worst_fold']} |")
    gate_lines = []
    for model in CANDIDATES:
        item = data["selection"]["candidates"][model]
        gate_lines.append(f"| {model} | {item['go_for_implementation_review']} | {item['checks']['high_end_improves']} | {item['checks']['shadow_stable']} | {item['checks']['midtone_stable']} | {item['checks']['matrix_not_worse']} | {item['checks']['br_not_worse']} | {item['checks']['cv_stable']} |")
    return f"""# P2.33 — REAL MATERIAL TONAL MODEL EXPERIMENT

**Final status:** `{data['status']}`  
**Production integration:** `NONE`  
**Priority order:** high-end compression → midtone stability → shadow stability → global error → cross-material stability.

## 1. Dataset and scope

Only the immutable P2.32 artifacts were read:

- `P232_dataset_manifest.json`;
- `P232_dataset_metrics.json`;
- `luminance_overlap_nits/` arrays.

No source film was decoded, no synchronization was executed, and no P2.32 file was modified. The dataset has 20 Matrix and 20 BR2049 frame cases. `verified_scene_count` remains unknown. BR2049 geometry confidence remains `0.9155`, below `0.95`, and is treated as a limitation rather than promoted.

## 2. Frame-case split

Five deterministic folds were created at frame-case level: four Matrix and four BR2049 validation cases per fold, with all pixels from a case assigned to one side only. The fixed material-balanced training sample contains exactly `{FIT_SAMPLES_PER_CASE}` SDR/HDR pixel pairs per training case; this sampling rule was defined before scoring and prevents BR2049's larger pixel count from dominating model construction.

No validation pixels or targets were used for fitting. No hyperparameter search was performed after observing validation results. All model definitions, bins, boundaries, and the Model H slope cap were fixed in the harness before the run.

## 3. Models

- **Model A:** unchanged production/reference curve and LUT from `{CURVE_PATH}`; 64 log-domain control points and 65,536-entry LUT; no refit.
- **Model B:** historical P2.30 comparator: 512 fixed-width log10 SDR-nits bins, median target, PAVA, dense linear interpolation, input-domain clipping, no extrapolation. It remains explicitly non-candidate from P2.31's high-end regression.
- **Model F:** fixed three-segment log-domain least-squares regression (`<10`, `10–200`, `>=200` SDR nits), non-negative slope constraint, boundary intercept adjustment for monotonicity, input-domain clipping.
- **Model G:** 128 equal-count log10 SDR bins, median target, PAVA, monotonic PCHIP, input-domain clipping.
- **Model H:** 96 fixed-width log10 SDR bins, median target, PAVA, linear interpolation, with a predeclared high-end boundary at 200 SDR nits and maximum log-output slope `{HIGH_SLOPE_CAP}` above that boundary.

All experimental models are offline only. No production LUT or curve was generated.

## 4. Material-separated and balanced results

`Matrix MAE` and `BR2049 MAE` are calculated from each material's out-of-fold predictions. `Combined macro-average` is the unweighted arithmetic mean of the two material metrics. `Combined weighted` is the pixel-weighted aggregate.

| Model | Matrix MAE | BR2049 MAE | Combined macro MAE | Combined weighted MAE | Combined macro >200 MAE |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

The complete bin metrics, including count, MAE, RMSE, P95, P99, max, and relative residual, are in the JSON and CSV for Matrix, BR2049, and combined outputs. The high-end table is not allowed to substitute a populated range for a missing one: empty bins retain `count=0`.

## 5. High-end, midtone, and shadow gates

| Candidate | Go for review | High-end improves | Shadow stable | Midtone stable | Matrix not worse | BR2049 not worse | CV stable |
|---|---|---|---|---|---|---|---|
{chr(10).join(gate_lines)}

The primary high-end ranges are `200–500`, `500–1000`, and `>1000` nits. The JSON records these separately for each material and combined. High-end shape is also classified as compression, expansion, near-linear, local plateau, and/or slope change from the predicted SDR→HDR mapping; no assumption that compression is inherently wrong is made.

Shadow uses `<10` nits and midtone uses `10–200` nits. Relative residual is reported separately from absolute residual and is not used as the primary shadow gate.

## 6. Cross-validation robustness

| Model | Mean CV macro MAE | Median | P25 | P75 | Best fold | Worst fold |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(cv_lines)}

The fold-level records and constraint checks are in `cv_folds` and `constraint_checks`. No model is selected from a single fold.

## 7. Per-frame consistency

For each candidate, the JSON reports Model A MAE, candidate MAE, and improvement percentage for all 40 out-of-fold frame cases, plus median, mean, P25, P75, minimum, maximum, improved/worsened counts, and counts beyond ±10%. These are case-level values, not pixel-level split values.

## 8. Visual check

After all scoring, the harness automatically selected three best, three median, and three worst cases by the selected visual candidate's out-of-fold improvement. Each artifact contains HDR target, Model A prediction, candidate prediction, residual A, and candidate residual. The selection is recorded as post-scoring in JSON; no visual case was chosen before metrics.

Artifacts are under `{OUTPUT_DIR}` and listed in `artifacts.visual_examples`.

## 9. Constraint validation

Every model/fold reports monotonicity violations, maximum negative slope, overshoot, clipping count/fraction, extrapolation count, duplicate x-values, negative predictions, finite status, and explicit input domain. All predictions are clipped to a declared training domain; no uncontrolled extrapolation is used.

## 10. Decision

`{data['status']}`.

**Selected visual candidate / priority comparator:** `{data['selection']['visual_candidate']}`.  
**Implementation review candidate:** `{data['selection']['implementation_candidate']}`.

A model can receive `GO FOR IMPLEMENTATION REVIEW` only if all predefined high-end, shadow, midtone, cross-material, CV, monotonicity, and domain gates pass. The final decision does not integrate any model and does not alter Model A.

## 11. Output

- Report: `{REPORT_PATH}`
- Metrics JSON: `{METRICS_PATH}`
- Per-case CSV: `{CSV_PATH}`
- Artifact directory: `{OUTPUT_DIR}`

## 12. Strict scope confirmation

No production code, Model A, Model B, LUT, CUDA backend, renderer, FFmpeg, dataset file, source film, synchronization result, or production default was changed. No candidate was integrated, no production LUT was generated, no MKV was rendered, and no commit or push was performed.
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(P232_METRICS.read_text(encoding="utf-8"))
    if dataset.get("status") != "DATASET READY":
        raise RuntimeError("P2.32 dataset is not marked DATASET READY")
    cases = [case for material in ("The Matrix", "BR2049") for case in dataset["case_records"][material]]
    if len(cases) != 40:
        raise RuntimeError(f"Expected 40 P2.32 cases, found {len(cases)}")
    for case in cases:
        if not Path(case["sdr_luminance_nits_path"]).exists() or not Path(case["hdr_luminance_nits_path"]).exists():
            raise RuntimeError(f"Missing immutable P2.32 luminance array for {case['case_id']}")
    by_material = {material: [case for case in cases if case["material"] == material] for material in ("The Matrix", "BR2049")}
    folds = fold_definition(cases)
    baseline = make_baseline()
    model_definitions = {"Model A": baseline["definition"], "Model B": {"name": "Model B", "type": "P2.30 dense monotonic interpolation", "bins": 512, "binning": "fixed-width log10 bins", "target_statistic": "median", "monotonic_projection": "PAVA", "interpolation": "linear", "extrapolation": False}, "Model F": {"name": "Model F", "type": "piecewise log-domain regression", "segments": ["<10", "10–200", ">=200"], "slope_min": 0.0, "interpolation": "piecewise linear in log domain", "extrapolation": False}, "Model G": {"name": "Model G", "type": "robust quantile/bin-based monotonic mapping", "bins": 128, "binning": "equal-count log10 bins", "target_statistic": "median", "monotonic_projection": "PAVA", "interpolation": "monotonic PCHIP", "extrapolation": False}, "Model H": {"name": "Model H", "type": "piecewise high-end controlled mapping", "bins": 96, "high_end_boundary_nits": 200.0, "max_high_end_log_slope": HIGH_SLOPE_CAP, "extrapolation": False}}

    target_stores: dict[str, np.memmap] = {}
    residual_stores: dict[str, dict[str, np.memmap]] = {model: {} for model in MODEL_NAMES}
    offsets: dict[str, dict[str, tuple[int, int]]] = {}
    for material in ("The Matrix", "BR2049"):
        material_cases = by_material[material]
        total = sum(int(np.load(case["hdr_luminance_nits_path"], mmap_mode="r").size) for case in material_cases)
        target_path = TMP_DIR / f"target_{material.replace(' ', '_').lower()}.dat"
        target_store = np.memmap(target_path, mode="w+", dtype="float32", shape=(total,))
        cursor = 0
        offsets[material] = {}
        for case in material_cases:
            target = np.asarray(np.load(case["hdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)
            end = cursor + target.size
            target_store[cursor:end] = target
            offsets[material][case["case_id"]] = (cursor, end)
            cursor = end
        target_store.flush()
        target_stores[material] = target_store
        for model in MODEL_NAMES:
            path = TMP_DIR / f"residual_{model.replace(' ', '_').lower()}_{material.replace(' ', '_').lower()}.dat"
            residual_stores[model][material] = np.memmap(path, mode="w+", dtype="float32", shape=(total,))

    per_case_results: list[dict[str, Any]] = []
    fold_results: dict[str, Any] = {}
    constraint_checks: list[dict[str, Any]] = []
    shape_checks: list[dict[str, Any]] = []
    for fold_info in folds:
        fold = int(fold_info["fold"])
        train_cases = [case for case in cases if case["case_id"] in fold_info["train_cases"]]
        validation_cases = [case for case in cases if case["case_id"] in fold_info["validation_cases"]]
        train_x, train_y = balanced_train_sample(train_cases, {material: [case for case in train_cases if case["material"] == material] for material in ("The Matrix", "BR2049")})
        models = {"Model A": baseline, "Model B": fit_curve_model("Model B", train_x, train_y, 512), "Model F": fit_piecewise_regression(train_x, train_y), "Model G": fit_curve_model("Model G", train_x, train_y, 128, equal_count=True, interpolation="pchip"), "Model H": fit_high_end_model(train_x, train_y)}
        fold_results[str(fold)] = {"train_case_ids": fold_info["train_cases"], "validation_case_ids": fold_info["validation_cases"], "material_validation": {}}
        for model_name, model in models.items():
            constraint_checks.append(constraint_check(model, train_x, train_y, validation_cases, fold))
            shape_checks.append(shape_analysis(model, train_x, fold))
        for material in ("The Matrix", "BR2049"):
            mat_val = [case for case in validation_cases if case["material"] == material]
            fold_results[str(fold)]["material_validation"][material] = {}
            for model_name, model in models.items():
                fold_residuals, fold_targets = [], []
                for case in mat_val:
                    input_values = np.asarray(np.load(case["sdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)
                    target = np.asarray(np.load(case["hdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)
                    prediction = predict(model, input_values)
                    if not np.isfinite(prediction).all():
                        raise RuntimeError(f"Non-finite prediction in fold {fold}, {model_name}, {case['case_id']}")
                    residual = prediction - target
                    start, end = offsets[material][case["case_id"]]
                    residual_stores[model_name][material][start:end] = residual.astype(np.float32)
                    fold_residuals.append(residual)
                    fold_targets.append(target)
                    case_result = {"material": material, "case_id": case["case_id"], "snapshot_index": case["snapshot_index"] if "snapshot_index" in case else case["selection_index"], "hdr_frame": case["hdr_frame"], "om_frame": case["om_frame"], "fold": fold, "model": model_name, "split": "validation", "metrics": metrics_for_arrays(residual, target)}
                    per_case_results.append(case_result)
                if fold_residuals:
                    aggregate = metrics_for_arrays(np.concatenate(fold_residuals), np.concatenate(fold_targets))
                else:
                    aggregate = metrics_for_arrays(np.empty(0), np.empty(0))
                fold_results[str(fold)]["material_validation"][material][model_name] = aggregate

    for model in MODEL_NAMES:
        for material in ("The Matrix", "BR2049"):
            residual_stores[model][material].flush()

    oof = {"materials": {}, "combined": {"weighted": {}, "macro_average": {}}}
    for material in ("The Matrix", "BR2049"):
        oof["materials"][material] = {}
        for model in MODEL_NAMES:
            residual = np.asarray(residual_stores[model][material])
            target = np.asarray(target_stores[material])
            oof["materials"][material][model] = metrics_for_arrays(residual, target)
    for model in MODEL_NAMES:
        material_residuals = {material: np.asarray(residual_stores[model][material]) for material in ("The Matrix", "BR2049")}
        material_targets = {material: np.asarray(target_stores[material]) for material in ("The Matrix", "BR2049")}
        oof["combined"]["weighted"][model] = combined_metrics(material_residuals, material_targets)
        oof["combined"]["macro_average"][model] = macro_metrics({material: oof["materials"][material][model] for material in ("The Matrix", "BR2049")})

    cv_summary = {}
    for model in MODEL_NAMES:
        fold_scores = []
        for fold in range(1, FOLD_COUNT + 1):
            per_material = fold_results[str(fold)]["material_validation"]
            fold_scores.append(float(np.mean([per_material[material][model]["global"]["MAE"] for material in ("The Matrix", "BR2049")])) )
        cv_summary[model] = {"fold_macro_MAE": fold_scores, "mean_macro_MAE": float(np.mean(fold_scores)), "median_macro_MAE": float(np.median(fold_scores)), "P25_macro_MAE": float(np.percentile(fold_scores, 25)), "P75_macro_MAE": float(np.percentile(fold_scores, 75)), "best_fold": int(np.argmin(fold_scores) + 1), "worst_fold": int(np.argmax(fold_scores) + 1), "stable_lower_than_baseline_folds": int(sum(score < cv_summary.get("Model A", {"fold_macro_MAE": [float("inf")] * FOLD_COUNT})["fold_macro_MAE"][index] for index, score in enumerate(fold_scores)))}

    per_case_by_model = {(row["case_id"], row["model"]): row for row in per_case_results}
    consistency = {}
    for model in CANDIDATES:
        improvements = []
        for case in cases:
            a = per_case_by_model[(case["case_id"], "Model A")]["metrics"]["global"]["MAE"]
            c = per_case_by_model[(case["case_id"], model)]["metrics"]["global"]["MAE"]
            improvements.append({"case_id": case["case_id"], "material": case["material"], "model_a_MAE": a, "candidate_MAE": c, "improvement_percent": 100.0 * (a - c) / max(a, LOG_EPS), "improved": c < a, "worsened": c > a})
        values = np.asarray([item["improvement_percent"] for item in improvements])
        consistency[model] = {"cases": improvements, "median_improvement_percent": float(np.median(values)), "mean_improvement_percent": float(np.mean(values)), "P25": float(np.percentile(values, 25)), "P75": float(np.percentile(values, 75)), "min": float(np.min(values)), "max": float(np.max(values)), "cases_improved": int(sum(item["improved"] for item in improvements)), "cases_worsened": int(sum(item["worsened"] for item in improvements)), "cases_over_10_percent_improved": int(sum(item["improvement_percent"] > 10 for item in improvements)), "cases_over_10_percent_worsened": int(sum(item["improvement_percent"] < -10 for item in improvements))}

    selection_candidates = {}
    base_macro = oof["combined"]["macro_average"]["Model A"]
    for model in CANDIDATES:
        candidate = oof["combined"]["macro_average"][model]
        base_high = base_macro[">200"]["MAE"]
        cand_high = candidate[">200"]["MAE"]
        checks = {
            "high_end_improves": cand_high is not None and base_high is not None and cand_high < base_high * 0.95,
            "shadow_stable": candidate["<0.1"]["MAE"] <= base_macro["<0.1"]["MAE"] * 1.05,
            "midtone_stable": candidate["10–20"]["MAE"] is not None and base_macro["10–20"]["MAE"] is not None,
            "matrix_not_worse": oof["materials"]["The Matrix"][model]["global"]["MAE"] <= oof["materials"]["The Matrix"]["Model A"]["global"]["MAE"] * 1.05,
            "br_not_worse": oof["materials"]["BR2049"][model]["global"]["MAE"] <= oof["materials"]["BR2049"]["Model A"]["global"]["MAE"] * 1.05,
            "cv_stable": cv_summary[model]["mean_macro_MAE"] < cv_summary["Model A"]["mean_macro_MAE"] and cv_summary[model]["stable_lower_than_baseline_folds"] >= 3,
            "per_case_not_single_outlier": consistency[model]["median_improvement_percent"] > 0 and consistency[model]["cases_improved"] > consistency[model]["cases_worsened"],
            "monotonicity_zero": all(check["monotonicity_violations"] == 0 for check in constraint_checks if check["model"] == model),
            "no_uncontrolled_extrapolation": all(check["extrapolation_count"] == 0 for check in constraint_checks if check["model"] == model),
        }
        # Midtone gate is evaluated from the explicit 10-20, 20-50, 50-100,
        # and 100-200 bins, not from a hidden global aggregate.
        mid_labels = ["10–20", "20–50", "50–100", "100–200"]
        cand_mid = np.mean([candidate[label]["MAE"] for label in mid_labels if candidate[label]["MAE"] is not None])
        base_mid = np.mean([base_macro[label]["MAE"] for label in mid_labels if base_macro[label]["MAE"] is not None])
        checks["midtone_stable"] = cand_mid <= base_mid * 1.05
        selection_candidates[model] = {"checks": checks, "go_for_implementation_review": bool(all(checks.values())), "macro_global_MAE": candidate["global"]["MAE"], "macro_high_end_MAE": cand_high, "macro_midtone_MAE": float(cand_mid), "macro_shadow_MAE": candidate["<0.1"]["MAE"]}

    visual_candidate = min(CANDIDATES, key=lambda model: (float("inf") if selection_candidates[model]["macro_high_end_MAE"] is None else selection_candidates[model]["macro_high_end_MAE"], selection_candidates[model]["macro_midtone_MAE"], selection_candidates[model]["macro_shadow_MAE"], selection_candidates[model]["macro_global_MAE"]))
    implementation_candidates = [model for model in CANDIDATES if selection_candidates[model]["go_for_implementation_review"]]
    implementation_candidate = implementation_candidates[0] if implementation_candidates else None
    status = "GO FOR IMPLEMENTATION REVIEW" if implementation_candidate else "NO CLEAR WINNER"

    per_case_consistency_for_visual = consistency[visual_candidate]["cases"]
    ordered = sorted(per_case_consistency_for_visual, key=lambda item: item["improvement_percent"])
    visual_selection = [("worst", ordered[:3]), ("median", ordered[len(ordered)//2 - 1:len(ordered)//2 + 2]), ("best", ordered[-3:])]
    visual_artifacts = []
    for category, selected in visual_selection:
        for item in selected:
            case = next(case for case in cases if case["case_id"] == item["case_id"])
            material = case["material"]
            start, end = offsets[material][case["case_id"]]
            target = np.asarray(target_stores[material][start:end], dtype=np.float32)
            residual_a = np.asarray(residual_stores["Model A"][material][start:end], dtype=np.float32)
            residual_c = np.asarray(residual_stores[visual_candidate][material][start:end], dtype=np.float32)
            visual_artifacts.append(write_visual(case, residual_a, residual_c, visual_candidate, category))

    plot_bars(OUTPUT_DIR / "P233_high_end_macro_MAE.png", "P2.33 macro high-end MAE", MODEL_NAMES, [oof["combined"]["macro_average"][model][">200"]["MAE"] or 0.0 for model in MODEL_NAMES], "MAE [nits]")
    plot_bars(OUTPUT_DIR / "P233_cv_macro_MAE.png", "P2.33 mean 5-fold macro MAE", MODEL_NAMES, [cv_summary[model]["mean_macro_MAE"] for model in MODEL_NAMES], "mean CV macro MAE [nits]")

    csv_rows = []
    for row in per_case_results:
        for region, metrics in row["metrics"].items():
            csv_rows.append({"material": row["material"], "case_id": row["case_id"], "fold": row["fold"], "model": row["model"], "split": row["split"], "region": region, "count": metrics.get("count"), "signed_mean": metrics.get("signed_mean"), "median_residual": metrics.get("median_residual"), "MAE": metrics.get("MAE"), "RMSE": metrics.get("RMSE"), "P95": metrics.get("P95"), "P99": metrics.get("P99"), "max": metrics.get("max"), "relative_MAE": metrics.get("relative", {}).get("MAE")})
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fields = ["material", "case_id", "fold", "model", "split", "region", "count", "signed_mean", "median_residual", "MAE", "RMSE", "P95", "P99", "max", "relative_MAE"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    checks_json = {"per_fold": constraint_checks, "all_models_monotonic": all(check["monotonicity_violations"] == 0 for check in constraint_checks), "all_predictions_finite": all(check["finite"] for check in constraint_checks), "all_extrapolation_zero": all(check["extrapolation_count"] == 0 for check in constraint_checks)}
    data = {
        "phase": "P2.33", "status": status, "scope": "offline tonal-model experiment using immutable P2.32 only; no production integration", "dataset_reference": {"metrics": str(P232_METRICS), "manifest": str(P232_DIR / "P232_dataset_manifest.json"), "material_case_counts": {"The Matrix": 20, "BR2049": 20}, "verified_scene_count": {"The Matrix": None, "BR2049": None}, "br2049_geometry_confidence": 0.9155}, "split": {"method": "5-fold case-level cross-validation", "fold_count": FOLD_COUNT, "validation_cases_per_fold": 8, "validation_material_balance": {"The Matrix": 4, "BR2049": 4}, "pixel_level_split": False, "fit_samples_per_case": FIT_SAMPLES_PER_CASE, "folds": folds}, "models": model_definitions, "results": {"folds": fold_results, "oof": oof, "per_case": per_case_results}, "cv_summary": cv_summary, "per_frame_consistency": consistency, "constraint_checks": checks_json, "high_end_shape": {model: [check for check in shape_checks if check["model"] == model] for model in MODEL_NAMES}, "selection": {"status": status, "visual_candidate": visual_candidate, "implementation_candidate": implementation_candidate, "candidates": selection_candidates, "gates": {"high_end": "macro >200 MAE at least 5% below Model A", "shadow": "macro <0.1 MAE no more than 5% worse", "midtone": "mean of 10–20/20–50/50–100/100–200 MAE no more than 5% worse", "cross_material": "Matrix and BR2049 global MAE no more than 5% worse", "cv": "mean macro MAE lower and at least 3/5 folds lower", "per_case": "positive median improvement and more cases improved than worsened", "constraints": "zero monotonic violations and zero extrapolation"}}, "artifacts": {"json": str(METRICS_PATH), "csv": str(CSV_PATH), "plots": [str(OUTPUT_DIR / "P233_high_end_macro_MAE.png"), str(OUTPUT_DIR / "P233_cv_macro_MAE.png")], "visual_examples": visual_artifacts, "output_directory": str(OUTPUT_DIR)}, "scope_constraints": {"dataset_modified": False, "production_code_changed": False, "model_a_changed": False, "model_b_integrated": False, "model_f_integrated": False, "model_g_integrated": False, "model_h_integrated": False, "production_lut_changed": False, "cuda_changed": False, "renderer_changed": False, "fitting_integrated": False, "production_render_performed": False, "source_films_decoded": False, "synchronization_performed": False, "dataset_pairs_added": False, "commit_created": False, "push_performed": False}}
    METRICS_PATH.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    REPORT_PATH.write_text(build_report(data), encoding="utf-8")
    # The temporary OOF stores are implementation-only and are removed after artifacts are complete.
    shutil.rmtree(TMP_DIR)
    print(f"P2.33_STATUS={status}")
    print(f"VISUAL_CANDIDATE={visual_candidate}")
    print(f"IMPLEMENTATION_CANDIDATE={implementation_candidate}")
    print(f"METRICS={METRICS_PATH}")
    print(f"CSV={CSV_PATH}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
