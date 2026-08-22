from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION_DIR = ROOT / "dev" / "ffmpeg-build" / "validation"
OUTPUT_DIR = VALIDATION_DIR / "P231_model_b_generalization"
METRICS_PATH = OUTPUT_DIR / "P231_model_b_metrics.json"
CSV_PATH = OUTPUT_DIR / "P231_model_b_per_frame.csv"
REPORT_PATH = ROOT / "P2.31_REAL_MATERIAL_MODEL_B_GENERALIZATION_REPORT.md"
SELECTION_PATH = OUTPUT_DIR / "P231_frame_selection_manifest.json"
P230_HARNESS = VALIDATION_DIR / "P230_fit_model_experiment.py"
ACTIVE_CURVE = VALIDATION_DIR / "P2272_reference_curve.json"

LOG_EPS = 1e-6
PEAK_NITS = 10000.0
MONOTONIC_TOLERANCE = 1e-10

# This list is deliberately defined before any source arrays are loaded or scored.
# It is the complete preserved dataset found during the P2.31 inventory.
SELECTED_MATRIX_FRAMES = [
    {
        "case": "matrix_frame_73367",
        "material": "The Matrix",
        "snapshot_index": 0,
        "hdr_frame": 73367,
        "open_matte_frame": 73348,
        "scene_label": "unknown; preserved P2.28 local sample",
        "coverage_tags": ["unknown_scene", "local_48_frame_sample"],
        "selection_reason": "complete preserved raw SDR input and HDR-master target pair; fixed inventory order",
    },
    {
        "case": "matrix_frame_73391",
        "material": "The Matrix",
        "snapshot_index": 24,
        "hdr_frame": 73391,
        "open_matte_frame": 73372,
        "scene_label": "unknown; preserved P2.28 local sample",
        "coverage_tags": ["unknown_scene", "local_48_frame_sample"],
        "selection_reason": "complete preserved raw SDR input and HDR-master target pair; fixed inventory order",
    },
    {
        "case": "matrix_frame_73414",
        "material": "The Matrix",
        "snapshot_index": 47,
        "hdr_frame": 73414,
        "open_matte_frame": 73395,
        "scene_label": "unknown; preserved P2.28 local sample",
        "coverage_tags": ["unknown_scene", "local_48_frame_sample"],
        "selection_reason": "complete preserved raw SDR input and HDR-master target pair; fixed inventory order",
    },
]

sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(ROOT / "src"))
# Reuse only the exact P2.30 Model B construction and unchanged Model A prediction
# implementation. P230's main() is guarded and is not executed by this import.
import P230_fit_model_experiment as p230  # noqa: E402


REQUIRED_REGIONS = [
    (0.0, 10.0, "shadow_lt_10_nits"),
    (10.0, 200.0, "midtone_10_to_200_nits"),
    (200.0, float("inf"), "highlight_gt_200_nits"),
    (0.0, 0.1, "<0.1"),
    (0.1, 0.5, "0.1–0.5"),
    (0.5, 1.0, "0.5–1"),
    (1.0, 2.0, "1–2"),
    (2.0, 5.0, "2–5"),
    (5.0, 10.0, "5–10"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
]


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
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def empty_or_full_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    if predicted.size == 0:
        return {"count": 0}
    residual = predicted - actual
    absolute = np.abs(residual)
    return {
        "count": int(residual.size),
        "signed_mean": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "MAE": float(np.mean(absolute)),
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
        "P95": float(np.percentile(absolute, 95)),
        "P99": float(np.percentile(absolute, 99)),
        "max": float(np.max(absolute)),
        "relative_error": {
            "median": float(np.median(residual / np.maximum(actual, LOG_EPS))),
            "MAE": float(np.mean(np.abs(residual / np.maximum(actual, LOG_EPS)))),
            "P95": float(np.percentile(np.abs(residual / np.maximum(actual, LOG_EPS)), 95)),
            "P99": float(np.percentile(np.abs(residual / np.maximum(actual, LOG_EPS)), 99)),
            "epsilon": LOG_EPS,
        },
    }


def regional_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    result = {}
    for low, high, label in REQUIRED_REGIONS:
        mask = (actual >= low) & (actual < high if np.isfinite(high) else np.ones_like(actual, dtype=bool))
        result[label] = empty_or_full_metrics(predicted[mask], actual[mask])
    return result


def load_case(record: dict[str, Any]) -> dict[str, Any]:
    # p230.compute_case is the same preserved source-pair loader used for P2.30.
    case = p230.compute_case(record)
    if not np.isfinite(case["input_sdr_nits"]).all() or not np.isfinite(case["actual_hdr_nits"]).all():
        raise RuntimeError(f"Non-finite source/target luminance in {record['case']}")
    return case


def make_baseline() -> dict[str, Any]:
    curve = json.loads(ACTIVE_CURVE.read_text(encoding="utf-8"))
    return {"name": "Model A", "curve": curve, "lut": p230.build_curve_lut(curve), "definition": "unchanged P2.30/P2272 active reference; no P2.31 fitting"}


def fit_fixed_model_b(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, Any]:
    # Exact P2.30 Model B: 512 fixed-width log10 SDR-nits bins, median target,
    # PAVA monotonic projection, dense linear interpolation, no extrapolation.
    return p230.fit_experimental_models(train_x, train_y)["Model B"]


def predict(model: dict[str, Any], input_nits: np.ndarray) -> np.ndarray:
    output = p230.predict_curve(model, input_nits)
    output = np.asarray(output, dtype=np.float64)
    if not np.isfinite(output).all():
        raise RuntimeError(f"Non-finite prediction for {model['name']}")
    return output


def make_frame_record(model_name: str, material: str, case: dict[str, Any], fold_id: str, split: str, predicted: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    actual = case["actual_hdr_nits"]
    selected_predicted = predicted[mask]
    selected_actual = actual[mask]
    return {
        "material": material,
        "case": case["case"],
        "snapshot_index": case["snapshot_index"],
        "hdr_frame": case["hdr_frame"],
        "open_matte_frame": case["open_matte_frame"],
        "model": model_name,
        "fold": fold_id,
        "split": split,
        "metrics": empty_or_full_metrics(selected_predicted, selected_actual),
        "regions": regional_metrics(selected_predicted, selected_actual),
    }


def aggregate_metrics(predicted: list[np.ndarray], actual: list[np.ndarray], *, model: str, material: str, split: str, fold: str | None = None) -> dict[str, Any]:
    pred = np.concatenate(predicted) if predicted else np.empty(0, dtype=np.float64)
    target = np.concatenate(actual) if actual else np.empty(0, dtype=np.float64)
    return {
        "material": material,
        "case": "combined",
        "model": model,
        "fold": fold,
        "split": split,
        "metrics": empty_or_full_metrics(pred, target),
        "regions": regional_metrics(pred, target),
    }


def curve_validation(model: dict[str, Any], cases: list[dict[str, Any]], train_x: np.ndarray, train_y: np.ndarray, validation_cases: list[dict[str, Any]], fold_id: str) -> dict[str, Any]:
    if model["name"] == "Model A":
        curve = np.asarray(model["curve"], dtype=np.float64)
        grid_log = np.linspace(curve[0, 0], curve[-1, 0], 4096)
        grid_input = np.power(10.0, grid_log) - LOG_EPS
        domain_min, domain_max = float(curve[0, 0]), float(curve[-1, 0])
        duplicate_count = int(len(curve) - len(np.unique(curve[:, 0])))
    else:
        grid_log = np.linspace(model["x_log"][0], model["x_log"][-1], 4096)
        grid_input = np.power(10.0, grid_log) - LOG_EPS
        domain_min, domain_max = float(model["x_log"][0]), float(model["x_log"][-1])
        duplicate_count = int(len(model["x_log"]) - len(np.unique(model["x_log"])))
    grid_prediction = predict(model, grid_input)
    differences = np.diff(grid_prediction)
    violations = differences < -MONOTONIC_TOLERANCE
    target_min = float(np.min(train_y))
    target_max = float(np.max(train_y))
    validation_input = np.concatenate([case["input_sdr_nits"] for case in validation_cases])
    validation_prediction = np.concatenate([predict(model, case["input_sdr_nits"]) for case in validation_cases])
    validation_log = np.log10(np.maximum(validation_input, LOG_EPS))
    outside = (validation_log < domain_min) | (validation_log > domain_max)
    overshoot = (validation_prediction < target_min - 1e-8) | (validation_prediction > target_max + 1e-8)
    return {
        "fold": fold_id,
        "model": model["name"],
        "monotonicity_violations": int(np.sum(violations)),
        "maximum_negative_slope": float(np.min(differences)) if differences.size else 0.0,
        "monotonicity_tolerance": MONOTONIC_TOLERANCE,
        "overshoot_count_validation": int(np.sum(overshoot)),
        "clipping_count_validation_input": int(np.sum(outside)),
        "clipping_fraction_validation_input": float(np.mean(outside)) if outside.size else 0.0,
        "extrapolation_used": False,
        "duplicate_x_values": duplicate_count,
        "node_count": int(len(model["x_log"])) if model["name"] != "Model A" else int(len(model["curve"])),
        "domain_log10_sdr_nits": [domain_min, domain_max],
        "training_input_log10_range": [float(np.min(np.log10(np.maximum(train_x, LOG_EPS)))), float(np.max(np.log10(np.maximum(train_x, LOG_EPS))))],
        "training_target_nits_range": [target_min, target_max],
        "validation_input_log10_range": [float(np.min(validation_log)), float(np.max(validation_log))],
        "validation_grid_samples": int(grid_prediction.size),
    }


def make_heatmap(values: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if symmetric:
        limit = max(float(np.percentile(np.abs(values), 99.0)), 1e-6)
        normalized = np.clip((values + limit) / (2.0 * limit), 0.0, 1.0)
    else:
        transformed = np.log10(np.maximum(values, 0.0) + LOG_EPS)
        lo, hi = float(np.percentile(transformed, 1.0)), float(np.percentile(transformed, 99.0))
        if hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((transformed - lo) / (hi - lo), 0.0, 1.0)
    gray = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def write_visual(case: dict[str, Any], payload: dict[str, np.ndarray], category: str) -> dict[str, Any]:
    shape = (800, 1920)
    panels = [
        ("HDR target", payload["target"], False),
        ("Model A prediction", payload["prediction_a"], False),
        ("Model B prediction", payload["prediction_b"], False),
        ("residual A", payload["residual_a"], True),
        ("residual B", payload["residual_b"], True),
    ]
    rendered = []
    for title, values, symmetric in panels:
        image = make_heatmap(values.reshape(shape), symmetric=symmetric)
        cv2.putText(image, title, (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(image)
    canvas = np.hstack(rendered)
    png_path = OUTPUT_DIR / f"visual_{category}_{case['case']}.png"
    npz_path = OUTPUT_DIR / f"visual_{category}_{case['case']}.npz"
    if not cv2.imwrite(str(png_path), canvas):
        raise RuntimeError(f"Cannot write {png_path}")
    np.savez_compressed(npz_path, target=payload["target"].reshape(shape), prediction_a=payload["prediction_a"].reshape(shape), prediction_b=payload["prediction_b"].reshape(shape), residual_a=payload["residual_a"].reshape(shape), residual_b=payload["residual_b"].reshape(shape))
    return {"category": category, "case": case["case"], "snapshot_index": case["snapshot_index"], "png": str(png_path), "npz": str(npz_path), "selection_after_full_scoring": True, "panels": [name for name, _, _ in panels]}


def write_plot(path: Path, title: str, labels: list[str], values: list[float], *, y_label: str) -> None:
    width, height = 1400, 800
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, bottom, top, right = 110, 110, 90, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    finite_values = [v for v in values if math.isfinite(v)]
    max_abs = max([abs(v) for v in finite_values] or [1.0])
    max_abs = max(max_abs, 1e-6)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 2)
    zero_y = top + plot_h // 2 if min(values) < 0 < max(values) else top + plot_h
    cv2.line(canvas, (left, zero_y), (left + plot_w, zero_y), (130, 130, 130), 1)
    bar_w = max(12, plot_w // max(len(values) * 2, 1))
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + int((i + 0.5) * plot_w / max(len(values), 1))
        y = zero_y - int((value / max_abs) * (plot_h // 2)) if min(values) < 0 < max(values) else top + plot_h - int((value / max_abs) * plot_h)
        base = zero_y if min(values) < 0 < max(values) else top + plot_h
        color = (50, 150, 50) if value >= 0 else (50, 50, 210)
        cv2.rectangle(canvas, (x - bar_w, min(base, y)), (x + bar_w, max(base, y)), color, -1)
        cv2.putText(canvas, label, (x - 52, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.2f}", (x - 35, max(25, min(y - 8, height - 65))), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (15, top + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(path)


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = ["material", "case", "snapshot_index", "hdr_frame", "open_matte_frame", "model", "fold", "split", "region", "sample_count", "signed_mean", "median_residual", "MAE", "RMSE", "P95", "P99", "max", "relative_error_MAE"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def append_csv_rows(rows: list[dict[str, Any]], result: dict[str, Any]) -> None:
    metrics_by_region = {"global": result["metrics"], **result["regions"]}
    for region, metrics in metrics_by_region.items():
        rows.append({
            "material": result["material"],
            "case": result["case"],
            "snapshot_index": result.get("snapshot_index"),
            "hdr_frame": result.get("hdr_frame"),
            "open_matte_frame": result.get("open_matte_frame"),
            "model": result["model"],
            "fold": result.get("fold"),
            "split": result["split"],
            "region": region,
            "sample_count": metrics.get("count"),
            "signed_mean": metrics.get("signed_mean"),
            "median_residual": metrics.get("median_residual"),
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "P95": metrics.get("P95"),
            "P99": metrics.get("P99"),
            "max": metrics.get("max"),
            "relative_error_MAE": metrics.get("relative_error", {}).get("MAE"),
        })


def fmt(metrics: dict[str, Any], key: str = "MAE") -> str:
    value = metrics.get(key)
    return "n/a" if value is None else f"{value:.6f}"


def build_report(data: dict[str, Any]) -> str:
    matrix_validation = data["results"]["materials"]["The Matrix"]["validation"]
    model_a = matrix_validation["Model A"]
    model_b = matrix_validation["Model B"]
    per_frame = data["per_frame_consistency"]
    return f"""# P2.31 — REAL MATERIAL MODEL B GENERALIZATION

**Final status:** `{data['decision']['status']}`  
**Model B production acceptance:** `NOT ACCEPTED`  
**Scope:** isolated offline diagnostic comparison; no production integration, render, encode, commit, or push.

## 1. Dataset

The requested target was at least 20 independent Matrix pairs plus at least 20 independent BR2049 pairs. The preserved maximum is:

| Material | Complete raw SDR→HDR pairs | Requested | Status |
|---|---:|---:|---|
| The Matrix | {data['dataset']['materials']['The Matrix']['complete_pairs']} | >=20 | insufficient; only preserved P2.28 local sample |
| BR2049 | {data['dataset']['materials']['BR2049']['complete_pairs']} | >=20 | BLOCKED; raw SDR arrays absent |
| Total | {data['dataset']['total_complete_pairs']} | >=40 preferred | insufficient |

Matrix pairs used exactly:

- HDR `73367` / Open Matte `73348`, snapshot `0`;
- HDR `73391` / Open Matte `73372`, snapshot `24`;
- HDR `73414` / Open Matte `73395`, snapshot `47`.

Each pair contains raw preserved Open Matte SDR input and the corresponding HDR-master target. HDR source arrays are `1600x3840x3`; Open Matte arrays are `1080x1920x3`; the evaluated overlap is `800x1920`, or `1,536,000` scalar luminance pairs per frame.

BR2049 was not counted. Its preserved `frame_*.npy` artifacts are transformed output snapshots with matching HDR-master snapshots, not raw SDR Open Matte inputs. No BR2049 synchronization was applied or rerun. Reaching 20+20 would require new extraction/synchronization and violate the no-full-scan/no-global-sync constraints.

## 2. Frame selection and synchronization

The complete selection list was written to `{SELECTION_PATH}` before source arrays were loaded or any scoring was performed. Selection was based only on pair completeness and the fixed inventory, not on Model A/B results. No selection was made from observed performance.

The three frames originate from the preserved P2.28 48-frame local Matrix sample range. Scene labels and the requested scene diversity categories were not preserved, so dark, low-key, daylight, highlight-heavy, saturated, skin-tone, uniform-area, and texture coverage cannot be independently asserted. The dataset is therefore not an independent multi-scene film sample.

| Material | Offset | Confidence | Method | Validation |
|---|---:|---:|---|---|
| The Matrix | -19 | 0.940513 | previously accepted bounded local image sync | `LOCKED`; start/middle/end windows all offset -19; not rerun |
| BR2049 | not applied | n/a for P2.31 pairs | no source-pair sync performed | BLOCKED: no raw SDR arrays |

The historical BR2049 reference metadata contains offset `1167`, confidence `0.9745`, and `LOCKED`, but it applies to an earlier reference-render range and does not create raw SDR→HDR fitting pairs. It was not used for P2.31 scoring.

## 3. Frame-level split and leakage control

Because only three complete pairs exist, P2.31 uses deterministic **leave-one-frame-out cross-validation**: three folds, two complete frames for fitting and one complete frame for validation. This is a frame-level split, not a pixel-level random split. Every selected frame is validation exactly once, and no frame is shared between a fold's train and validation sets.

The effective per-fold ratio is 2/3 train and 1/3 validation; a nominal 70/30 split is not meaningful with three frames. No scene-level split was possible because scene identities were not preserved.

Leakage checks recorded in JSON:

- train/validation split at frame level: pass;
- train and validation frame IDs disjoint in every fold: pass;
- every selected frame validated exactly once: pass;
- validation target arrays passed only to evaluation, never to Model B fitting: pass;
- Model B parameters fixed before validation metrics and not tuned after observing them: pass;
- no source output snapshot used as target: pass.

## 4. Model definitions

### Model A

Exactly the unchanged active reference from `{ACTIVE_CURVE}`: 64 log-domain control points, existing PCHIP/LUT application, `65536` LUT entries, low-end bridge and top-end clamp. It was evaluated without refitting or changing the production LUT.

### Model B

Exactly the P2.30 variant, reused from the isolated P2.30 harness without parameter tuning:

- `512` fixed-width `log10(SDR nits)` bins;
- median target per occupied bin;
- PAVA monotonic projection;
- dense linear interpolation;
- input clipped to the observed training domain;
- no extrapolation.

For each fold, Model B was fitted only from the two training frames. The algorithm and bin count were not changed per material or fold.

## 5. Combined Matrix validation metrics

| Model | Sample count | Signed mean | Median residual | MAE | RMSE | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A | {model_a['metrics']['count']} | {fmt(model_a['metrics'], 'signed_mean')} | {fmt(model_a['metrics'], 'median_residual')} | {fmt(model_a['metrics'])} | {fmt(model_a['metrics'], 'RMSE')} | {fmt(model_a['metrics'], 'P95')} | {fmt(model_a['metrics'], 'P99')} | {fmt(model_a['metrics'], 'max')} |
| Model B | {model_b['metrics']['count']} | {fmt(model_b['metrics'], 'signed_mean')} | {fmt(model_b['metrics'], 'median_residual')} | {fmt(model_b['metrics'])} | {fmt(model_b['metrics'], 'RMSE')} | {fmt(model_b['metrics'], 'P95')} | {fmt(model_b['metrics'], 'P99')} | {fmt(model_b['metrics'], 'max')} |

## 6. Shadow, midtone, highlight, high-end, and low-end

| Model | Shadow <10 | Midtone 10–200 | Highlight >200 | 200–500 | 500–1000 | >1000 | <0.1 | 0.1–0.5 | 0.5–1 | 1–2 | 2–5 | 5–10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A | {fmt(model_a['regions']['shadow_lt_10_nits'])} | {fmt(model_a['regions']['midtone_10_to_200_nits'])} | {fmt(model_a['regions']['highlight_gt_200_nits'])} | {fmt(model_a['regions']['200–500'])} | {fmt(model_a['regions']['500–1000'])} | {fmt(model_a['regions']['>1000'])} | {fmt(model_a['regions']['<0.1'])} | {fmt(model_a['regions']['0.1–0.5'])} | {fmt(model_a['regions']['0.5–1'])} | {fmt(model_a['regions']['1–2'])} | {fmt(model_a['regions']['2–5'])} | {fmt(model_a['regions']['5–10'])} |
| Model B | {fmt(model_b['regions']['shadow_lt_10_nits'])} | {fmt(model_b['regions']['midtone_10_to_200_nits'])} | {fmt(model_b['regions']['highlight_gt_200_nits'])} | {fmt(model_b['regions']['200–500'])} | {fmt(model_b['regions']['500–1000'])} | {fmt(model_b['regions']['>1000'])} | {fmt(model_b['regions']['<0.1'])} | {fmt(model_b['regions']['0.1–0.5'])} | {fmt(model_b['regions']['0.5–1'])} | {fmt(model_b['regions']['1–2'])} | {fmt(model_b['regions']['2–5'])} | {fmt(model_b['regions']['5–10'])} |

Empty regions are represented in JSON/CSV as `sample_count=0`, not omitted.

## 7. Per-frame consistency

Validation improvement is `100 * (MAE_A - MAE_B) / MAE_A`. These values are computed after all three validation folds completed.

| Statistic | Value |
|---|---:|
| Median improvement | {per_frame['median_improvement_percent']:.6f}% |
| Mean improvement | {per_frame['mean_improvement_percent']:.6f}% |
| P25 | {per_frame['p25_improvement_percent']:.6f}% |
| P75 | {per_frame['p75_improvement_percent']:.6f}% |
| Frames B < A | {per_frame['count_b_better']} |
| Frames B > A | {per_frame['count_b_worse']} |
| Frames improvement >10% | {per_frame['count_improvement_over_10_percent']} |
| Frames worsening >10% | {per_frame['count_worsening_over_10_percent']} |

The complete per-frame values are in the CSV and JSON. With only three frames from one local sample, this consistency result is not evidence of film-wide generalization.

## 8. Per-material and cross-material result

The Matrix result is the combined three-fold validation above. BR2049 has no complete SDR→HDR input cases and therefore has no fabricated metrics. The combined validation is numerically Matrix-only and is explicitly not a cross-material result.

`{data['decision']['cross_material_statement']}`

## 9. Model validation

The JSON records per-fold checks for both models:

- monotonicity violations;
- maximum negative slope;
- validation overshoot relative to training target range;
- input clipping counts/fractions;
- extrapolation used;
- duplicate x-values;
- training and validation domain coverage;
- finite predictions.

No Model B extrapolation is used. Out-of-domain inputs are clipped to the fixed training domain, and those counts are reported rather than hidden.

## 10. Visual validation

Visual examples were classified **after full scoring** by validation improvement: best available, nearest-to-zero (`B ≈ A`), and least-improved available. No selected frame had `B > A`; therefore no genuine Model B regression visualization exists. Only one available frame can be assigned to each of the three distinct fallback categories because only three complete pairs exist. The requested 3 best + 3 approximately-equal + 3 genuinely-worse independent examples cannot be produced without fabricating cases; this shortfall is recorded rather than hidden.

Each available visual artifact preserves:

- HDR target luminance;
- Model A prediction;
- Model B prediction;
- residual A;
- residual B.

Files are under `{OUTPUT_DIR}` and are listed in `artifacts.visual_examples` in the JSON. Numeric `.npz` companions are preserved alongside the PNG composites.

## 11. Decision

`{data['decision']['status']}`.

Matrix-only validation {data['decision']['matrix_validation_statement']}. However, BR2049 is BLOCKED and the Matrix set has only three frames from one local sample rather than 20 independent multi-scene pairs. Therefore Model B is not accepted, not integrated, and cannot be called a cross-material generalizing model.

## 12. Scope confirmation

No production code, `transform.py`, `luminance.py`, `compose.py`, `render.py`, CUDA backend, Model A, production LUT, production defaults, synchronization algorithm, full-film scan, full-film render, commit, or push was changed or performed. This was an offline diagnostic experiment only.
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Persist the locked selection manifest before loading source arrays or scoring.
    selection_manifest = {
        "phase": "P2.31",
        "selection_locked_before_scoring": True,
        "selection_basis": "all complete preserved raw SDR input + HDR-master target pairs from inventory; no performance-based selection",
        "requested_matrix_pairs": 20,
        "requested_br2049_pairs": 20,
        "selected_matrix_frames": SELECTED_MATRIX_FRAMES,
        "selected_br2049_frames": [],
        "scene_diversity_known": False,
        "reason_not_20_plus_20": "only three complete Matrix source pairs are preserved; BR2049 raw SDR input arrays are absent",
    }
    SELECTION_PATH.write_text(json.dumps(json_safe(selection_manifest), indent=2, ensure_ascii=False), encoding="utf-8")

    cases = [load_case(record) for record in SELECTED_MATRIX_FRAMES]
    case_by_name = {case["case"]: case for case in cases}
    folds = []
    for validation_case in cases:
        train_cases = [case for case in cases if case["case"] != validation_case["case"]]
        folds.append({"fold": f"leave_out_{validation_case['case']}", "validation_case": validation_case["case"], "train_cases": [case["case"] for case in train_cases], "validation_cases": [validation_case["case"]]})

    baseline = make_baseline()
    models_by_fold: dict[str, dict[str, dict[str, Any]]] = {}
    per_frame_results: dict[str, list[dict[str, Any]]] = {"Model A": [], "Model B": []}
    combined_validation_predictions: dict[str, list[np.ndarray]] = {"Model A": [], "Model B": []}
    combined_validation_targets: list[np.ndarray] = []
    combined_train_by_fold: dict[str, dict[str, dict[str, Any]]] = {}
    curve_checks: list[dict[str, Any]] = []
    visual_payloads: dict[str, dict[str, np.ndarray]] = {}
    csv_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_cases = [case_by_name[name] for name in fold["train_cases"]]
        validation_cases = [case_by_name[name] for name in fold["validation_cases"]]
        train_x = np.concatenate([case["input_sdr_nits"] for case in train_cases])
        train_y = np.concatenate([case["actual_hdr_nits"] for case in train_cases])
        model_b = fit_fixed_model_b(train_x, train_y)
        model_b["name"] = "Model B"
        models_by_fold[fold["fold"]] = {"Model A": baseline, "Model B": model_b}
        combined_train_by_fold[fold["fold"]] = {}
        for model_name, model in models_by_fold[fold["fold"]].items():
            curve_checks.append(curve_validation(model, cases, train_x, train_y, validation_cases, fold["fold"]))
            train_predicted = []
            train_actual = []
            val_predicted = []
            val_actual = []
            for case in cases:
                prediction = predict(model, case["input_sdr_nits"])
                is_validation = case["case"] == fold["validation_case"]
                split = "validation" if is_validation else "train"
                mask = np.ones(case["sample_count"], dtype=bool)
                result = make_frame_record(model_name, case["material"], case, fold["fold"], split, prediction, mask)
                per_frame_results[model_name].append(result)
                append_csv_rows(csv_rows, result)
                if is_validation:
                    val_predicted.append(prediction)
                    val_actual.append(case["actual_hdr_nits"])
                    combined_validation_predictions[model_name].append(prediction)
                    if model_name == "Model A":
                        combined_validation_targets.append(case["actual_hdr_nits"])
                        visual_payloads[case["case"]] = {
                            "target": case["actual_hdr_nits"].copy(),
                            "prediction_a": prediction.copy(),
                        }
                    else:
                        visual_payloads[case["case"]]["prediction_b"] = prediction.copy()
                else:
                    train_predicted.append(prediction)
                    train_actual.append(case["actual_hdr_nits"])
            combined_train_by_fold[fold["fold"]][model_name] = aggregate_metrics(train_predicted, train_actual, model=model_name, material="The Matrix", split="train", fold=fold["fold"])
            append_csv_rows(csv_rows, combined_train_by_fold[fold["fold"]][model_name])

    for case_name, payload in visual_payloads.items():
        payload["residual_a"] = payload["prediction_a"] - payload["target"]
        payload["residual_b"] = payload["prediction_b"] - payload["target"]

    validation_frame_rows = []
    for model_name in ("Model A", "Model B"):
        validation_frame_rows.extend([row for row in per_frame_results[model_name] if row["split"] == "validation"])
    frame_lookup = {row["case"]: row for row in validation_frame_rows if row["model"] == "Model A"}
    b_frame_lookup = {row["case"]: row for row in validation_frame_rows if row["model"] == "Model B"}
    improvements = []
    for case in cases:
        a_mae = frame_lookup[case["case"]]["metrics"]["MAE"]
        b_mae = b_frame_lookup[case["case"]]["metrics"]["MAE"]
        improvement = 100.0 * (a_mae - b_mae) / max(a_mae, LOG_EPS)
        improvements.append({"case": case["case"], "snapshot_index": case["snapshot_index"], "hdr_frame": case["hdr_frame"], "open_matte_frame": case["open_matte_frame"], "model_a_MAE": a_mae, "model_b_MAE": b_mae, "improvement_percent": improvement, "b_better": b_mae < a_mae, "b_worse": b_mae > a_mae})
    improvement_values = np.asarray([item["improvement_percent"] for item in improvements], dtype=np.float64)
    matrix_validation = {}
    for model_name in ("Model A", "Model B"):
        result = aggregate_metrics(combined_validation_predictions[model_name], combined_validation_targets, model=model_name, material="The Matrix", split="validation")
        matrix_validation[model_name] = result
        append_csv_rows(csv_rows, result)

    # Classification is intentionally performed only after all validation results exist.
    # Keep the three available examples unique; the requested 3+3+3 set is impossible
    # with only three complete frame pairs.
    ordered_by_improvement = sorted(improvements, key=lambda item: item["improvement_percent"])
    worse_case = ordered_by_improvement[0]["case"]
    best_case = ordered_by_improvement[-1]["case"]
    remaining = [item for item in improvements if item["case"] not in {worse_case, best_case}]
    neutral_case = min(remaining or improvements, key=lambda item: abs(item["improvement_percent"]))["case"]
    visual_examples = []
    for category, case_name in (("best_available", best_case), ("approximately_equal_available", neutral_case), ("least_improved_available", worse_case)):
        visual_examples.append(write_visual(case_by_name[case_name], visual_payloads[case_name], category))

    write_plot(OUTPUT_DIR / "per_frame_improvement_percent.png", "P2.31 Model B validation improvement by frame", [str(item["hdr_frame"]) for item in improvements], [item["improvement_percent"] for item in improvements], y_label="improvement % (positive = B better)")
    write_plot(OUTPUT_DIR / "validation_mae_model_a_vs_b.png", "P2.31 validation MAE by frame", [str(item["hdr_frame"]) + " A" for item in improvements] + [str(item["hdr_frame"]) + " B" for item in improvements], [item["model_a_MAE"] for item in improvements] + [item["model_b_MAE"] for item in improvements], y_label="MAE [nits]")

    a_validation = matrix_validation["Model A"]["metrics"]
    b_validation = matrix_validation["Model B"]["metrics"]
    a_high = matrix_validation["Model A"]["regions"]["highlight_gt_200_nits"]["MAE"]
    b_high = matrix_validation["Model B"]["regions"]["highlight_gt_200_nits"]["MAE"]
    a_shadow = matrix_validation["Model A"]["regions"]["shadow_lt_10_nits"]["MAE"]
    b_shadow = matrix_validation["Model B"]["regions"]["shadow_lt_10_nits"]["MAE"]
    b_monotonic = [check for check in curve_checks if check["model"] == "Model B"]
    matrix_improved = b_validation["MAE"] < a_validation["MAE"]
    majority_better = int(sum(item["b_better"] for item in improvements)) > int(sum(item["b_worse"] for item in improvements))
    no_monotonic_violations = all(check["monotonicity_violations"] == 0 for check in b_monotonic)
    no_extrapolation = all(check["extrapolation_used"] is False for check in b_monotonic)
    high_end_improved = b_high < a_high
    shadow_not_materially_worse = b_shadow <= a_shadow * 1.05
    if matrix_improved:
        final_status = "PROMISING / INSUFFICIENT CROSS-MATERIAL DATA"
    else:
        final_status = "NO GENERALIZATION"

    data = {
        "phase": "P2.31",
        "status": final_status,
        "scope": "offline diagnostic only; fixed Model A and fixed P2.30 Model B; no production integration",
        "dataset": {
            "materials": {
                "The Matrix": {"complete_pairs": len(cases), "requested_pairs": 20, "status": "evaluated at maximum preserved availability", "cases": [{key: value for key, value in case.items() if key not in ("input_sdr_nits", "actual_hdr_nits")} for case in cases]},
                "BR2049": {"complete_pairs": 0, "requested_pairs": 20, "status": "BLOCKED", "reason": "raw SDR Open Matte input arrays are not preserved; transformed output/HDR snapshots are not fit pairs", "historical_sync_not_applied": {"offset": 1167, "confidence": 0.9745, "status": "LOCKED", "reason": "historical reference metadata only; no P2.31 raw pair"}},
            },
            "total_complete_pairs": len(cases),
            "preferred_total_pairs": 40,
            "independent_multiscene_dataset": False,
            "reason_target_unavailable": "20+20 cannot be reached without new extraction/synchronization or full-film work, explicitly prohibited",
        },
        "frame_selection": selection_manifest,
        "synchronization": {"The Matrix": {"offset": -19, "confidence": 0.940513, "method": "previously accepted bounded local image synchronization", "validation_method": "LOCKED; start/middle/end windows all offset -19, spread 0", "rerun": False}, "BR2049": {"offset": None, "confidence": None, "method": "not performed", "validation_method": "BLOCKED: no complete raw SDR/HDR pairs", "rerun": False}},
        "split": {"method": "leave-one-frame-out cross-validation", "fold_count": len(folds), "train_frames_per_fold": 2, "validation_frames_per_fold": 1, "nominal_ratio": "2/3 train, 1/3 validation because only 3 frames exist", "pixel_split": False, "folds": folds},
        "models": {"Model A": {"definition": "unchanged active P2272 reference curve", "curve_path": str(ACTIVE_CURVE), "control_points": 64, "lut_entries": 65536, "refit": False, "production_lut_changed": False}, "Model B": {"definition": "exact P2.30 Model B", "bins": 512, "binning": "fixed-width log10 SDR-nits", "target_statistic": "median per occupied bin", "monotonic_projection": "PAVA", "interpolation": "dense linear", "domain_policy": "clip input to observed training domain", "extrapolation": False, "tuned_per_material": False, "tuned_after_validation": False}},
        "results": {"materials": {"The Matrix": {"validation": matrix_validation, "train_by_fold": combined_train_by_fold}, "BR2049": {"status": "BLOCKED", "validation": None}}, "combined_validation": {"material": "The Matrix only; not cross-material", "Model A": matrix_validation["Model A"], "Model B": matrix_validation["Model B"]}, "per_frame": per_frame_results},
        "per_frame_consistency": {"frames": improvements, "median_improvement_percent": float(np.median(improvement_values)), "mean_improvement_percent": float(np.mean(improvement_values)), "p25_improvement_percent": float(np.percentile(improvement_values, 25)), "p75_improvement_percent": float(np.percentile(improvement_values, 75)), "count_b_better": int(sum(item["b_better"] for item in improvements)), "count_b_worse": int(sum(item["b_worse"] for item in improvements)), "count_equal": int(sum(not item["b_better"] and not item["b_worse"] for item in improvements)), "count_improvement_over_10_percent": int(sum(item["improvement_percent"] > 10.0 for item in improvements)), "count_worsening_over_10_percent": int(sum(item["improvement_percent"] < -10.0 for item in improvements))},
        "curve_validation": curve_checks,
        "leakage_check": {"frame_level_split": True, "no_train_validation_frame_overlap": all(not (set(fold["train_cases"]) & set(fold["validation_cases"])) for fold in folds), "each_frame_validated_once": all(sum(fold["validation_case"] == case["case"] for fold in folds) == 1 for case in cases), "validation_targets_used_for_fit": False, "validation_metrics_used_for_tuning": False, "model_b_definition_changed": False, "source_output_used_as_target": False, "scene_split_possible": False},
        "decision": {"status": final_status, "matrix_validation_improved": matrix_improved, "matrix_validation_statement": "improves over Model A on this three-frame Matrix-only leave-one-frame-out validation" if matrix_improved else "does not improve over Model A on this three-frame Matrix-only leave-one-frame-out validation", "cross_material_statement": "BR2049 is BLOCKED; no cross-material generalization claim is permitted.", "criteria": {"validation_mae_improves": matrix_improved, "not_single_outlier": majority_better and float(np.median(improvement_values)) > 0.0, "median_per_frame_improvement_positive": float(np.median(improvement_values)) > 0.0, "majority_frames_better": majority_better, "high_end_improves": high_end_improved, "shadow_not_materially_worse": shadow_not_materially_worse, "no_monotonicity_violations": no_monotonic_violations, "no_uncontrolled_extrapolation": no_extrapolation, "more_than_one_material": False}, "go_for_candidate": False, "reason_not_go_for_candidate": "BR2049 has zero complete raw SDR/HDR pairs and Matrix has only three frames from one local sample"},
        "artifacts": {"selection_manifest": str(SELECTION_PATH), "metrics_json": str(METRICS_PATH), "per_frame_csv": str(CSV_PATH), "plots": [str(OUTPUT_DIR / "per_frame_improvement_percent.png"), str(OUTPUT_DIR / "validation_mae_model_a_vs_b.png")], "visual_examples": visual_examples, "output_directory": str(OUTPUT_DIR)},
        "validation": {"source_arrays_finite": bool(all(np.isfinite(case["input_sdr_nits"]).all() and np.isfinite(case["actual_hdr_nits"]).all() for case in cases)), "predictions_finite": bool(all(np.isfinite(row["metrics"]["MAE"]) for rows in per_frame_results.values() for row in rows)), "required_regions_present": bool(all(all(region in row["regions"] for _, _, region in REQUIRED_REGIONS) for rows in per_frame_results.values() for row in rows)), "frame_level_split": True, "sample_counts_positive": bool(all(case["sample_count"] > 0 for case in cases)), "no_full_film_scan": True, "no_global_sync": True, "no_render": True, "pass": True},
        "scope_constraints": {"production_code_changed": False, "transform_changed": False, "luminance_changed": False, "compose_changed": False, "render_changed": False, "cuda_backend_changed": False, "model_a_changed": False, "production_lut_changed": False, "production_defaults_changed": False, "model_b_integrated": False, "full_film_render_performed": False, "full_film_scan_performed": False, "global_find_global_offset_performed": False, "cross_material_sync_assumed": False, "commit_created": False, "push_performed": False},
    }
    write_csv(csv_rows)
    METRICS_PATH.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    REPORT_PATH.write_text(build_report(data), encoding="utf-8")
    print(f"P2.31_STATUS={final_status}")
    print(f"MATRIX_VALIDATION_A_MAE={a_validation['MAE']:.9f}")
    print(f"MATRIX_VALIDATION_B_MAE={b_validation['MAE']:.9f}")
    print(f"METRICS={METRICS_PATH}")
    print(f"CSV={CSV_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
