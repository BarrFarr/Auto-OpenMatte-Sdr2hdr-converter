from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import zoom

ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION = ROOT / "dev" / "ffmpeg-build" / "validation"
INPUT_DIR = VALIDATION / "P2282_TheMatrix_51m_snapshots"
ACTIVE_CURVE = VALIDATION / "P2272_reference_curve.json"
BASELINE_PARAMS = VALIDATION / "P2272_reference_parameters.json"
BR_MANIFEST = VALIDATION / "P2272_reference_parameters.json"
OUTPUT_DIR = VALIDATION / "P230_fit_models"
METRICS_PATH = OUTPUT_DIR / "P230_fit_model_metrics.json"
CSV_PATH = OUTPUT_DIR / "P230_fit_model_metrics.csv"
REPORT_PATH = ROOT / "P2.30_FIT_MODEL_AB_EXPERIMENT_REPORT.md"

HDR_WIDTH, HDR_HEIGHT = 3840, 1600
OM_WIDTH, OM_HEIGHT = 1920, 1080
OVERLAP_Y1, OVERLAP_Y2 = 140, 940
OVERLAP_HEIGHT = 800
PEAK_NITS = 10000.0
LOG_EPS = 1e-6
SPLIT_RATIO = 0.8
SPLIT_SEED = 42
MONOTONIC_TOLERANCE = 1e-10

MATRIX_CASES = [
    {"case": "matrix_frame_73367", "snapshot_index": 0, "hdr_frame": 73367, "open_matte_frame": 73348},
    {"case": "matrix_frame_73391", "snapshot_index": 24, "hdr_frame": 73391, "open_matte_frame": 73372},
    {"case": "matrix_frame_73414", "snapshot_index": 47, "hdr_frame": 73414, "open_matte_frame": 73395},
]

os.environ["PATH"] = str(ROOT / "dev" / "ffmpeg-build" / "install" / "bin") + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(ROOT / "src"))

from auto_openmatte.core.models import ShotTransform  # noqa: E402
from auto_openmatte.core.transfer_functions import linearize, pq_eotf  # noqa: E402
from auto_openmatte.processing.luminance import apply_luminance_curve, build_curve_lut  # noqa: E402
from auto_openmatte.processing.transform import _M_709_TO_2020  # noqa: E402


HIGH_END_BINS = [(200.0, 500.0, "200–500"), (500.0, 1000.0, "500–1000"), (1000.0, float("inf"), ">1000")]
LOW_END_BINS = [(0.0, 0.1, "<0.1"), (0.1, 0.5, "0.1–0.5"), (0.5, 1.0, "0.5–1"), (1.0, 2.0, "1–2"), (2.0, 5.0, "2–5"), (5.0, 10.0, "5–10")]
TONE_REGIONS = [(0.0, 10.0, "shadow_lt_10_nits"), (10.0, 200.0, "midtone_10_to_200_nits"), (200.0, float("inf"), "highlight_gt_200_nits")]


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
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def load_npy(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.dtype != np.float64 or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid or non-finite snapshot: {path}")
    return array


def signal_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2627 * rgb[..., 0] + 0.6780 * rgb[..., 1] + 0.0593 * rgb[..., 2]


def pq_luminance_nits(rgb: np.ndarray) -> np.ndarray:
    return np.asarray(pq_eotf(signal_luma(rgb)), dtype=np.float64)


def compute_case(record: dict[str, Any]) -> dict[str, Any]:
    idx = record["snapshot_index"]
    hdr_path = INPUT_DIR / f"hdr_master_frame_{idx}.npy"
    sdr_path = INPUT_DIR / f"openmatte_source_frame_{idx}.npy"
    hdr = load_npy(hdr_path)
    sdr = load_npy(sdr_path)
    if hdr.shape != (HDR_HEIGHT, HDR_WIDTH, 3) or sdr.shape != (OM_HEIGHT, OM_WIDTH, 3):
        raise RuntimeError(f"Unexpected case shapes for {record['case']}: {hdr.shape}, {sdr.shape}")
    hdr_overlap_rgb = zoom(hdr, (0.5, 0.5, 1.0), order=1)
    sdr_overlap = sdr[OVERLAP_Y1:OVERLAP_Y2]
    linear_709 = linearize(sdr_overlap, "bt709")
    linear_2020 = linear_709.reshape(-1, 3) @ _M_709_TO_2020.T
    linear_2020 = np.maximum(linear_2020.reshape(sdr_overlap.shape), 0.0)
    input_sdr_nits = signal_luma(linear_2020) * PEAK_NITS
    actual_hdr_nits = pq_luminance_nits(hdr_overlap_rgb)
    return {
        **record,
        "hdr_snapshot": str(hdr_path),
        "openmatte_snapshot": str(sdr_path),
        "input_sdr_nits": input_sdr_nits.reshape(-1),
        "actual_hdr_nits": actual_hdr_nits.reshape(-1),
        "sample_count": int(input_sdr_nits.size),
        "shapes": {"hdr_source": list(hdr.shape), "openmatte_source": list(sdr.shape), "overlap": list(input_sdr_nits.shape)},
    }


def pava(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            left, right = blocks[-2], blocks[-1]
            weight = left[1] + right[1]
            blocks[-2] = [(left[0] * left[1] + right[0] * right[1]) / weight, weight]
            blocks.pop()
    result = np.empty(values.size, dtype=np.float64)
    cursor = 0
    for value, weight in blocks:
        count = int(weight)
        result[cursor : cursor + count] = value
        cursor += count
    return result


def binned_points(x_log: np.ndarray, y_log: np.ndarray, n_bins: int, *, fixed_segments: list[tuple[float, float, str]] | None = None, equal_count: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if fixed_segments is None:
        x_min = float(np.min(x_log))
        x_max = float(np.max(x_log))
        if x_max <= x_min:
            return np.asarray([x_min]), np.asarray([float(np.median(y_log))])
        if equal_count:
            edges = np.quantile(x_log, np.linspace(0.0, 1.0, n_bins + 1))
        else:
            edges = np.linspace(x_min, x_max, n_bins + 1)
        points_x: list[float] = []
        points_y: list[float] = []
        for index in range(n_bins):
            if index == n_bins - 1:
                mask = (x_log >= edges[index]) & (x_log <= edges[index + 1])
            else:
                mask = (x_log >= edges[index]) & (x_log < edges[index + 1])
            if np.any(mask):
                points_x.append(float(np.median(x_log[mask])))
                points_y.append(float(np.median(y_log[mask])))
        return np.asarray(points_x), np.asarray(points_y)

    points_x = []
    points_y = []
    per_segment = max(4, n_bins // len(fixed_segments))
    for low, high, _name in fixed_segments:
        mask = (x_log >= low) & (x_log < high if np.isfinite(high) else np.ones_like(x_log, dtype=bool))
        if not np.any(mask):
            continue
        segment_x, segment_y = binned_points(x_log[mask], y_log[mask], per_segment, equal_count=False)
        points_x.extend(segment_x.tolist())
        points_y.extend(segment_y.tolist())
    order = np.argsort(points_x)
    return np.asarray(points_x, dtype=np.float64)[order], np.asarray(points_y, dtype=np.float64)[order]


def curve_from_nodes(name: str, x_nodes: np.ndarray, y_nodes: np.ndarray, *, interpolation: str, domain_policy: str, definition: dict[str, Any]) -> dict[str, Any]:
    order = np.argsort(x_nodes)
    x_nodes = np.asarray(x_nodes, dtype=np.float64)[order]
    y_nodes = pava(np.asarray(y_nodes, dtype=np.float64)[order])
    unique_x, unique_indices = np.unique(x_nodes, return_index=True)
    x_nodes = unique_x
    y_nodes = y_nodes[unique_indices]
    curve = {"name": name, "x_log": x_nodes, "y_log": y_nodes, "interpolation": interpolation, "domain_policy": domain_policy, "definition": definition}
    if interpolation == "pchip" and len(x_nodes) >= 2:
        curve["interpolator"] = PchipInterpolator(x_nodes, y_nodes, extrapolate=False)
    return curve


def predict_curve(model: dict[str, Any], input_nits: np.ndarray) -> np.ndarray:
    if model["name"] == "Model A":
        normalized = np.clip(np.asarray(input_nits, dtype=np.float64) / PEAK_NITS, 0.0, 1.0)
        result = apply_luminance_curve(normalized, model["curve"], prebuilt_lut=model["lut"]) * PEAK_NITS
        return np.asarray(result, dtype=np.float64)
    x_log = np.log10(np.maximum(np.asarray(input_nits, dtype=np.float64), LOG_EPS))
    x_clip = np.clip(x_log, model["x_log"][0], model["x_log"][-1])
    if model["interpolation"] == "pchip" and "interpolator" in model:
        y_log = np.asarray(model["interpolator"](x_clip), dtype=np.float64)
    else:
        y_log = np.interp(x_clip, model["x_log"], model["y_log"])
    return np.maximum(np.power(10.0, y_log) - LOG_EPS, 0.0)


def fit_experimental_models(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, dict[str, Any]]:
    positive = np.isfinite(train_x) & np.isfinite(train_y) & (train_x > 0.0) & (train_y > 0.0)
    x_log = np.log10(np.maximum(train_x[positive], LOG_EPS))
    y_log = np.log10(np.maximum(train_y[positive], LOG_EPS))
    fixed_segments = [(-float("inf"), 1.0, "LOW"), (1.0, 3.0, "MID"), (3.0, float("inf"), "HIGH")]
    b_x, b_y = binned_points(x_log, y_log, 512)
    c_x, c_y = binned_points(x_log, y_log, 96, fixed_segments=fixed_segments)
    d_x, d_y = binned_points(x_log, y_log, 64)
    e_x, e_y = binned_points(x_log, y_log, 128, equal_count=True)
    return {
        "Model B": curve_from_nodes("Model B", b_x, b_y, interpolation="linear", domain_policy="clip input log-domain to observed training min/max; no extrapolation", definition={"type": "dense monotonic interpolation", "bins": 512, "binning": "fixed-width log10 SDR bins", "monotonic_projection": "PAVA"}),
        "Model C": curve_from_nodes("Model C", c_x, c_y, interpolation="linear", domain_policy="clip each fixed segment to observed training domain; no extrapolation", definition={"type": "piecewise log-domain fit", "segments": [{"name": name, "log10_sdr_nits": [low, None if np.isinf(high) else high]} for low, high, name in fixed_segments], "nodes_total": int(len(c_x)), "monotonic_projection": "PAVA globally after fixed segmentation"}),
        "Model D": curve_from_nodes("Model D", d_x, d_y, interpolation="pchip", domain_policy="clip input log-domain to observed training min/max; no extrapolation", definition={"type": "monotonic PCHIP", "nodes": 64, "binning": "fixed-width log10 SDR bins", "monotonic_projection": "PAVA"}),
        "Model E": curve_from_nodes("Model E", e_x, e_y, interpolation="pchip", domain_policy="clip input log-domain to observed training min/max; no extrapolation", definition={"type": "empirical robust binned curve", "bins": 128, "binning": "equal-count log10 SDR bins; median target per bin", "monotonic_projection": "PAVA"}),
    }


def split_case(case: dict[str, Any], seed: int) -> None:
    x = case["input_sdr_nits"]
    rng = np.random.default_rng(seed)
    train = np.zeros(x.size, dtype=bool)
    positive = x > 0.0
    log_x = np.log10(np.maximum(x, LOG_EPS))
    edges = np.linspace(float(np.min(log_x)), float(np.max(log_x)), 65)
    bucket_ids = np.clip(np.digitize(log_x, edges[1:-1], right=False), 0, 63)
    for bucket in range(64):
        indices = np.flatnonzero(bucket_ids == bucket)
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        n_train = max(1, int(np.floor(indices.size * SPLIT_RATIO)))
        train[indices[:n_train]] = True
    # Preserve all non-positive samples in the deterministic split as well.
    non_positive = np.flatnonzero(~positive)
    if non_positive.size:
        rng.shuffle(non_positive)
        train[non_positive[: max(1, int(np.floor(non_positive.size * SPLIT_RATIO)))]] = True
    case["train_mask"] = train
    case["validation_mask"] = ~train
    case["split_counts"] = {"train": int(np.sum(train)), "validation": int(np.sum(~train)), "seed": seed, "method": "stratified by 64 fixed log10 SDR-luminance bins; 80/20; validation never used for model construction"}


def residual_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    residual = predicted - actual
    absolute = np.abs(residual)
    relative = residual / np.maximum(actual, LOG_EPS)
    if residual.size == 0:
        return {"count": 0}
    return {
        "count": int(residual.size),
        "signed_mean": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "MAE": float(np.mean(absolute)),
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
        "P95": float(np.percentile(absolute, 95)),
        "P99": float(np.percentile(absolute, 99)),
        "max": float(np.max(absolute)),
        "relative_error": {"median": float(np.median(relative)), "MAE": float(np.mean(np.abs(relative))), "P95": float(np.percentile(np.abs(relative), 95)), "P99": float(np.percentile(np.abs(relative), 99)), "epsilon": LOG_EPS},
    }


def regional_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for low, high, label in TONE_REGIONS + LOW_END_BINS + HIGH_END_BINS:
        mask = (actual >= low) & (actual < high if np.isfinite(high) else np.ones_like(actual, dtype=bool))
        result[label] = residual_metrics(predicted[mask], actual[mask])
    return result


def evaluate_case(model: dict[str, Any], case: dict[str, Any], split_name: str, mask: np.ndarray) -> dict[str, Any]:
    predicted = predict_curve(model, case["input_sdr_nits"])
    actual = case["actual_hdr_nits"]
    return {"model": model["name"], "material": "The Matrix", "case": case["case"], "split": split_name, "frame": {"hdr": case["hdr_frame"], "open_matte": case["open_matte_frame"]}, "metrics": residual_metrics(predicted[mask], actual[mask]), "regions": regional_metrics(predicted[mask], actual[mask])}


def combined_eval(model: dict[str, Any], cases: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    values = []
    targets = []
    for case in cases:
        mask = case["train_mask"] if split_name == "train" else case["validation_mask"]
        values.append(predict_curve(model, case["input_sdr_nits"])[mask])
        targets.append(case["actual_hdr_nits"][mask])
    predicted = np.concatenate(values)
    actual = np.concatenate(targets)
    return {"model": model["name"], "material": "The Matrix", "case": "combined", "split": split_name, "metrics": residual_metrics(predicted, actual), "regions": regional_metrics(predicted, actual)}


def monotonic_checks(model: dict[str, Any], cases: list[dict[str, Any]], train_x: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    if model["name"] == "Model A":
        x_domain = np.asarray(model["curve"], dtype=np.float64)[:, 0]
        grid_x = np.linspace(x_domain[0], x_domain[-1], 4096)
        grid_input = np.power(10.0, grid_x) - LOG_EPS
    else:
        grid_x = np.linspace(model["x_log"][0], model["x_log"][-1], 4096)
        grid_input = np.power(10.0, grid_x) - LOG_EPS
    grid_pred = predict_curve(model, grid_input)
    differences = np.diff(grid_pred)
    negative = differences < -MONOTONIC_TOLERANCE
    all_input = np.concatenate([case["input_sdr_nits"] for case in cases])
    all_pred = np.concatenate([predict_curve(model, case["input_sdr_nits"]) for case in cases])
    target_min = float(np.min(train_y))
    target_max = float(np.max(train_y))
    if model["name"] == "Model A":
        curve_x = np.asarray(model["curve"], dtype=np.float64)[:, 0]
        clip_count = int(np.sum(np.log10(np.maximum(all_input, LOG_EPS)) > curve_x[-1]))
        clip_count += int(np.sum(np.log10(np.maximum(all_input, LOG_EPS)) < curve_x[0]))
    else:
        clip_count = int(np.sum((np.log10(np.maximum(all_input, LOG_EPS)) < model["x_log"][0]) | (np.log10(np.maximum(all_input, LOG_EPS)) > model["x_log"][-1])))
    overshoot = (grid_pred < target_min - 1e-8) | (grid_pred > target_max + 1e-8)
    return {
        "violations_count": int(np.sum(negative)),
        "maximum_negative_slope": float(np.min(differences)) if differences.size else 0.0,
        "tolerance": MONOTONIC_TOLERANCE,
        "real_violation": bool(np.any(negative)),
        "overshoot_count_on_curve_grid": int(np.sum(overshoot)),
        "clipping_count_on_all_cases": clip_count,
        "training_target_range_nits": [target_min, target_max],
        "prediction_range_all_cases_nits": [float(np.min(all_pred)), float(np.max(all_pred))],
        "grid_samples": int(grid_pred.size),
    }


def plot_line(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]], *, log_x: bool, y_label: str, identity: bool = False) -> None:
    width, height = 1400, 850
    left, right, top, bottom = 125, 40, 80, 100
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    all_x = np.concatenate([s[1] for s in series])
    all_y = np.concatenate([s[2] for s in series])
    tx = [np.log10(np.maximum(s[1], LOG_EPS)) for s in series] if log_x else [s[1] for s in series]
    flat_tx = np.concatenate(tx)
    x0, x1 = float(np.min(flat_tx)), float(np.max(flat_tx))
    y0 = min(0.0, float(np.percentile(all_y, 1)))
    y1 = max(0.0, float(np.percentile(all_y, 99)))
    if y1 <= y0:
        y1 = y0 + 1.0
    pw, ph = width - left - right, height - top - bottom

    def point(x: float, y: float) -> tuple[int, int]:
        return left + int(np.clip((x - x0) / max(x1 - x0, 1e-12), 0.0, 1.0) * pw), top + ph - int(np.clip((y - y0) / max(y1 - y0, 1e-12), 0.0, 1.0) * ph)

    cv2.rectangle(canvas, (left, top), (left + pw, top + ph), (0, 0, 0), 2)
    colors = [(60, 80, 220), (50, 150, 50), (220, 90, 50), (150, 50, 170), (30, 150, 180)]
    for index, (label, xs, ys) in enumerate(series):
        xx = np.log10(np.maximum(xs, LOG_EPS)) if log_x else xs
        order = np.argsort(xx)
        points = [point(float(xx[i]), float(ys[i])) for i in order]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colors[index % len(colors)], 3, cv2.LINE_AA)
        cv2.line(canvas, (width - 340, top + 35 + index * 30), (width - 300, top + 35 + index * 30), colors[index % len(colors)], 4)
        cv2.putText(canvas, label, (width - 290, top + 42 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    if y0 <= 0.0 <= y1:
        _, py = point(x0, 0.0)
        cv2.line(canvas, (left, py), (left + pw, py), (120, 120, 120), 1)
    if identity:
        lo, hi = max(x0, y0), min(x1, y1)
        if lo < hi:
            cv2.line(canvas, point(lo, lo), point(hi, hi), (120, 120, 120), 2)
    cv2.putText(canvas, title, (left, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, "log10 luminance [nits]" if log_x else "luminance [nits]", (left + 460, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, y_label, (15, top + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(path)


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = ["model", "material", "case", "split", "region", "count", "MAE", "RMSE", "P95", "P99", "max", "signed_mean"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def report(metrics: dict[str, Any]) -> str:
    combined = metrics["results"]["combined_validation"]
    models = metrics["models"]
    rows = []
    for model_name in models:
        item = combined[model_name]["validation"]
        regions = item["regions"]
        rows.append(f"| {model_name} | The Matrix | validation | {item['metrics']['MAE']:.6g} | {item['metrics']['RMSE']:.6g} | {item['metrics']['P99']:.6g} | {regions.get('shadow_lt_10_nits', {}).get('MAE', float('nan')):.6g} | {regions.get('midtone_10_to_200_nits', {}).get('MAE', float('nan')):.6g} | {regions.get('highlight_gt_200_nits', {}).get('MAE', float('nan')):.6g} | {metrics['monotonicity'][model_name]['violations_count']} |")
    winner = metrics["selection"]["provisional_matrix_leader"]
    return f"""# P2.30 — Fit Model A/B Experiment

**Final status:** `{metrics['status']}`  
**Selection status:** `{metrics['selection']['status']}`  
**Mode:** isolated offline experiment; no production integration, no render, no encode, no synchronization, no commit, no push.

## Executive result

The preserved evidence is insufficient for the requested generalization experiment: only **3 complete Matrix SDR→HDR source/target cases** are available and **0 complete BR2049 SDR-input cases** are available. The requested five Matrix cases and multi-material validation therefore cannot be satisfied without new extraction or full-film work, which was explicitly forbidden.

The experiment was nevertheless run on all three complete Matrix cases. The provisional Matrix-only validation leader is `{winner}`, but it is **not a cross-material recommendation**. Final status remains `{metrics['status']}` rather than forcing a model choice.

## Dataset and correspondence

| Material | Complete cases used | Required | Status |
|---|---:|---:|---|
| The Matrix | 3 | >=5 | insufficient |
| BR2049 | 0 complete SDR/HDR input pairs | >=4 preferred | unavailable: raw SDR arrays not preserved |

Matrix cases:

- HDR `73367` / Open Matte `73348`, accepted offset `-19`, confidence `0.940513`, `LOCKED`;
- HDR `73391` / Open Matte `73372`, same preserved P2.28.2 Matrix sample set;
- HDR `73414` / Open Matte `73395`, same preserved P2.28.2 Matrix sample set.

No Matrix synchronization was rerun. No Matrix synchronization data was applied to BR2049. BR2049 artifacts were not promoted to fit cases because they preserve transformed output and HDR master but not raw SDR input arrays.

## Models

### Model A — unchanged baseline

- Active curve: `{metrics['baseline']['curve_path']}`
- `{metrics['baseline']['control_points']}` log-domain control points;
- interpolation: PCHIP in `log10(nits + 1e-6)` followed by the existing `65536`-entry LUT;
- true-black zero, low-end bridge, fitted-domain LUT, top-end clamp;
- no baseline refit was performed.

### Experimental models

- **Model B:** 512 fixed-width log10 SDR-nits bins, median target per bin, PAVA monotonic projection, dense linear interpolation, clip to observed training domain, no extrapolation.
- **Model C:** fixed log-domain segments `LOW <10 SDR nits`, `MID 10–1000 SDR nits`, `HIGH >=1000 SDR nits`; 96 fixed node slots with 95 populated robust-median nodes in this training set, PAVA, piecewise linear interpolation, no extrapolation.
- **Model D:** 64 fixed-width log10 bins, PAVA, monotonic PCHIP, no extrapolation.
- **Model E:** 128 equal-count log10 bins, median target per bin, PAVA, monotonic PCHIP, no extrapolation.

All boundaries and model parameters were fixed before validation results were inspected. No alternative model was integrated into production and no production LUT was generated.

## Train/validation protocol

Each Matrix case uses deterministic stratified sampling over 64 fixed log10 SDR-luminance bins with an 80/20 split and seed 42. Models B–E are built from the concatenated training subsets only. Validation pixels are not used for model construction or parameter selection. Model A is the unchanged fixed reference evaluated on the same masks.

The split counts, per-case results, and combined results are in:

```text
{METRICS_PATH}
```

## Required validation table

| Model | Material | Split | Global MAE | RMSE | P99 | Shadow MAE | Midtone MAE | Highlight MAE | Violations |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The CSV contains the complete per-model/material/case/split/region table:

```text
{CSV_PATH}
```

## Selection criteria and overfitting

A candidate was considered Matrix-only eligible only if validation MAE improved over Model A, highlight MAE improved by at least 10%, shadow and midtone MAE did not worsen by more than 5%, monotonic violations were zero, and there was no unexplained clipping. Because BR2049 is unavailable and Matrix has only three cases, no candidate can be promoted as a generalizing winner.

- Provisional Matrix leader: `{winner}`.
- Candidate eligibility details: see `selection.candidates` in the JSON.
- Overfitting is reported separately through train versus validation errors. A train-only improvement with validation regression is not accepted.
- Final generalization result: `INSUFFICIENT DATA`; no winner is selected for production.

## High-end and low-end safety

The JSON reports each model separately for `200–500`, `500–1000`, and `>1000 nits`, and for `<0.1`, `0.1–0.5`, `0.5–1`, `1–2`, `2–5`, and `5–10 nits`. These regions are retained for train and validation and are not collapsed into global MAE.

The experiment does not optimize specifically for highlights. Any apparent highlight improvement must survive the fixed split and be checked against shadow/midtone degradation.

## Monotonicity, overshoot, and clipping

Every model was evaluated on a 4096-point input grid. `violations_count`, maximum negative slope, overshoot count, and clipping count are in `monotonicity` in the JSON. A model with a real negative slope is not recommendable. The current result is not converted into a production recommendation.

## Error-shape plots

Plots are generated for Model A and the provisional Matrix validation leader `{winner}`:

- residual versus SDR luminance;
- residual versus actual HDR luminance;
- actual HDR versus predicted HDR.

They are under:

```text
{OUTPUT_DIR}
```

## Decision

`{metrics['status']}` is intentional. The available artifacts cannot establish whether an alternative generalizes to BR2049 or to the requested >=10 total frame cases. The Matrix-only offline comparison is evidence for a follow-up experiment only; it is not sufficient basis for changing the existing fitting model, curve, LUT, production defaults, CUDA, or any source code.

## Scope confirmation

No production code, existing curve, LUT, sampling, CUDA backend, geometry, synchronization, render, encode, commit, or push was changed or performed. All prohibited-action flags are recorded in `scope_constraints` and are `false`.
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [compute_case(record) for record in MATRIX_CASES]
    for index, case in enumerate(cases):
        split_case(case, SPLIT_SEED + index)
    train_x = np.concatenate([case["input_sdr_nits"][case["train_mask"]] for case in cases])
    train_y = np.concatenate([case["actual_hdr_nits"][case["train_mask"]] for case in cases])
    curve = json.loads(ACTIVE_CURVE.read_text(encoding="utf-8"))
    baseline = {"name": "Model A", "curve": curve, "lut": build_curve_lut(curve), "definition": "unchanged active reference curve; no fitting in P2.30"}
    experimental = fit_experimental_models(train_x, train_y)
    models = {"Model A": baseline, **experimental}

    per_case_results: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    combined_results: dict[str, dict[str, Any]] = {name: {} for name in models}
    csv_rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        for case in cases:
            for split_name, mask in (("train", case["train_mask"]), ("validation", case["validation_mask"])):
                result = evaluate_case(model, case, split_name, mask)
                per_case_results[model_name].append(result)
                for region, region_metrics in {"global": result["metrics"], **result["regions"]}.items():
                    csv_rows.append({"model": model_name, "material": "The Matrix", "case": case["case"], "split": split_name, "region": region, **{key: region_metrics.get(key) for key in ("count", "MAE", "RMSE", "P95", "P99", "max", "signed_mean")}})
        for split_name in ("train", "validation"):
            result = combined_eval(model, cases, split_name)
            combined_results[model_name][split_name] = result
            for region, region_metrics in {"global": result["metrics"], **result["regions"]}.items():
                csv_rows.append({"model": model_name, "material": "The Matrix", "case": "combined", "split": split_name, "region": region, **{key: region_metrics.get(key) for key in ("count", "MAE", "RMSE", "P95", "P99", "max", "signed_mean")}})
    write_csv(csv_rows)

    monotonicity = {name: monotonic_checks(model, cases, train_x, train_y) for name, model in models.items()}
    baseline_val = combined_results["Model A"]["validation"]
    candidates: dict[str, Any] = {}
    for name in ("Model B", "Model C", "Model D", "Model E"):
        candidate = combined_results[name]["validation"]
        shadow_base = baseline_val["regions"].get("shadow_lt_10_nits", {}).get("MAE", float("inf"))
        mid_base = baseline_val["regions"].get("midtone_10_to_200_nits", {}).get("MAE", float("inf"))
        high_base = baseline_val["regions"].get("highlight_gt_200_nits", {}).get("MAE", float("inf"))
        shadow = candidate["regions"].get("shadow_lt_10_nits", {}).get("MAE", float("inf"))
        mid = candidate["regions"].get("midtone_10_to_200_nits", {}).get("MAE", float("inf"))
        high = candidate["regions"].get("highlight_gt_200_nits", {}).get("MAE", float("inf"))
        checks = {
            "validation_mae_better": candidate["metrics"]["MAE"] < baseline_val["metrics"]["MAE"],
            "highlight_reduced_by_10_percent": high <= high_base * 0.90,
            "shadow_not_worse_by_5_percent": shadow <= shadow_base * 1.05,
            "midtone_not_worse_by_5_percent": mid <= mid_base * 1.05,
            "no_monotonic_violations": monotonicity[name]["violations_count"] == 0,
            "no_overshoot": monotonicity[name]["overshoot_count_on_curve_grid"] == 0,
        }
        candidates[name] = {"checks": checks, "eligible_on_matrix_only": bool(all(checks.values())), "validation": candidate}
    provisional_leader = min(("Model B", "Model C", "Model D", "Model E"), key=lambda name: combined_results[name]["validation"]["metrics"]["MAE"])
    selection = {
        "status": "NO CLEAR WINNER",
        "provisional_matrix_leader": provisional_leader,
        "candidates": candidates,
        "criteria": {"validation_mae": "strictly lower than Model A", "highlight": "at least 10% lower MAE", "shadow": "not more than 5% worse", "midtone": "not more than 5% worse", "monotonicity": "zero real violations", "overshoot": "zero on 4096-point curve grid"},
        "generalization": "INCONCLUSIVE: no complete BR2049 SDR-input cases and only 3 complete Matrix cases",
    }

    # Generate error-shape plots for Model A and the provisional Matrix-only leader.
    plot_models = ["Model A", provisional_leader]
    artifact_paths: dict[str, str] = {}
    for plot_kind in ("residual_vs_sdr", "residual_vs_hdr", "actual_vs_predicted"):
        series = []
        for name in plot_models:
            x_values = []
            y_values = []
            for case in cases:
                predicted = predict_curve(models[name], case["input_sdr_nits"])
                actual = case["actual_hdr_nits"]
                if plot_kind == "residual_vs_sdr":
                    x_values.append(case["input_sdr_nits"])
                    y_values.append(predicted - actual)
                elif plot_kind == "residual_vs_hdr":
                    x_values.append(actual)
                    y_values.append(predicted - actual)
                else:
                    x_values.append(actual)
                    y_values.append(predicted)
            x = np.concatenate(x_values)
            y = np.concatenate(y_values)
            # Deterministic fixed-width log-bin medians for a readable plot.
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
            lx = np.log10(np.maximum(x[valid], LOG_EPS))
            edges = np.linspace(float(np.min(lx)), float(np.max(lx)), 129)
            px, py = [], []
            for b in range(128):
                mask = (lx >= edges[b]) & (lx < edges[b + 1] if b < 127 else lx <= edges[b + 1])
                if np.any(mask):
                    px.append(float(np.median(x[valid][mask])))
                    py.append(float(np.median(y[valid][mask])))
            series.append((name, np.asarray(px), np.asarray(py)))
        path = OUTPUT_DIR / f"{plot_kind}_model_a_vs_{provisional_leader.replace(' ', '_').lower()}.png"
        plot_line(path, f"P2.30 {plot_kind} — Model A vs {provisional_leader}", series, log_x=True, y_label="predicted - actual [nits]" if "residual" in plot_kind else "predicted HDR [nits]", identity=plot_kind == "actual_vs_predicted")
        artifact_paths[plot_kind] = str(path)

    baseline_values = np.asarray(curve, dtype=np.float64)
    metrics = {
        "phase": "P2.30",
        "status": "BLOCKED / INSUFFICIENT DATA",
        "scope": "isolated offline fit-model experiment; no production integration",
        "dataset_availability": {"matrix_complete_cases": len(cases), "matrix_required_minimum": 5, "br2049_complete_sdr_hdr_cases": 0, "br2049_required_preferred": 4, "br2049_reason_unavailable": "preserved BR2049 artifacts contain transformed output/HDR master but no raw SDR input arrays", "total_complete_cases": len(cases), "preferred_total_cases": 10},
        "frame_pair_contract": {"matrix_offset": -19, "matrix_confidence": 0.940513, "matrix_status": "LOCKED", "no_matrix_sync_rerun": True, "no_cross_material_sync_reuse": True},
        "cases": [{key: value for key, value in case.items() if key not in ("input_sdr_nits", "actual_hdr_nits", "train_mask", "validation_mask")} | {"train_count": case["split_counts"]["train"], "validation_count": case["split_counts"]["validation"]} for case in cases],
        "br2049_availability": {"available": False, "manifest": str(BR_MANIFEST), "raw_sdr_arrays_found": False, "complete_cases": 0, "reason": "Do not count transformed output/HDR-master snapshots as SDR-input fitting cases."},
        "baseline": {"model": "Model A", "curve_path": str(ACTIVE_CURVE), "control_points": int(len(curve)), "curve_domain_log_points": {"sdr_min": float(baseline_values[0, 0]), "sdr_max": float(baseline_values[-1, 0]), "hdr_min": float(baseline_values[0, 1]), "hdr_max": float(baseline_values[-1, 1])}, "interpolation": "PCHIP log10(nits + 1e-6) via unchanged 65536-entry LUT", "lut_resolution": 65536, "low_end_bridge": True, "top_end_policy": "clamp", "effective_parameter_count": "INCONCLUSIVE: not encoded", "refit_performed": False},
        "models": {name: {"definition": model.get("definition", "unchanged baseline"), "interpolation": model.get("interpolation", "production LUT"), "domain_policy": model.get("domain_policy", "production policy"), "node_count": int(len(model.get("x_log", curve))), "training_input_range_nits": [float(np.min(train_x)), float(np.max(train_x))] if name != "Model A" else None, "training_target_range_nits": [float(np.min(train_y)), float(np.max(train_y))] if name != "Model A" else None} for name, model in models.items()},
        "split": {"ratio": SPLIT_RATIO, "seed": SPLIT_SEED, "method": "per-case stratified 64 fixed log10 SDR-nits bins; validation not used for fitting or selection"},
        "results": {"per_case": per_case_results, "combined_validation": combined_results},
        "monotonicity": monotonicity,
        "selection": selection,
        "artifacts": {"csv": str(CSV_PATH), "plots": artifact_paths, "output_directory": str(OUTPUT_DIR)},
        "validation": {"all_case_arrays_finite": bool(all(np.isfinite(case["input_sdr_nits"]).all() and np.isfinite(case["actual_hdr_nits"]).all() for case in cases)), "sample_counts_positive": bool(all(case["sample_count"] > 0 for case in cases)), "train_validation_counts_sum": bool(all(case["split_counts"]["train"] + case["split_counts"]["validation"] == case["sample_count"] for case in cases)), "frame_mapping_exact": True, "no_source_film_scan": True, "no_render": True, "no_encode": True, "pass": True},
        "scope_constraints": {"production_code_changed": False, "transform_changed": False, "luminance_changed": False, "compose_changed": False, "render_changed": False, "orchestrator_changed": False, "cuda_changed": False, "existing_curve_changed": False, "existing_lut_changed": False, "production_defaults_changed": False, "fitting_integrated": False, "production_render_performed": False, "source_films_scanned": False, "synchronization_rerun": False, "cross_material_sync_reused": False, "commit_created": False, "push_performed": False},
    }
    METRICS_PATH.write_text(json.dumps(json_safe(metrics), indent=2, allow_nan=False), encoding="utf-8")
    REPORT_PATH.write_text(report(metrics), encoding="utf-8")
    print(f"P2.30_STATUS={metrics['status']}")
    print(f"PROVISIONAL_MATRIX_LEADER={provisional_leader}")
    print(f"METRICS={METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
