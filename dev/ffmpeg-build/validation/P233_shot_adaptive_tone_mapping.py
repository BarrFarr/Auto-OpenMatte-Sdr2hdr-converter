from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION_DIR = ROOT / "dev" / "ffmpeg-build" / "validation"
P232_DIR = VALIDATION_DIR / "P232_real_material_dataset"
P232_METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
P232_MANIFEST_PATH = P232_DIR / "P232_dataset_manifest.json"
P232_LUMA_DIR = P232_DIR / "luminance_overlap_nits"
P232_RAW_DIR = P232_DIR / "raw_rgb_u16"
MODEL_A_CURVE_PATH = VALIDATION_DIR / "P2272_reference_curve.json"
OUTPUT_DIR = VALIDATION_DIR / "P233_shot_adaptive"
CURVE_DIR = OUTPUT_DIR / "shot_curves"
MASK_DIR = OUTPUT_DIR / "train_test_masks"
RESIDUAL_DIR = OUTPUT_DIR / "residual_arrays"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_PATH = OUTPUT_DIR / "P233_shot_adaptive_metrics.json"
CSV_PATH = OUTPUT_DIR / "P233_per_frame_results.csv"
REPORT_PATH = ROOT / "P2.33_SHOT_ADAPTIVE_TONE_MAPPING_FEASIBILITY_REPORT.md"

PEAK_NITS = 10000.0
LOG_EPS = 1e-6
TILE_ROWS = 10
TILE_COLS = 10
TEST_TILE_MODULUS = 10
TEST_TILE_REMAINDERS = (0, 1, 2)
MIN_STABLE_SDR_NITS = 0.01
HARD_CLIP_NITS = 0.99 * PEAK_NITS
GLOBAL_MODEL_B_SAMPLES_PER_CASE = 25000
MIN_NONEMPTY_SHOT_BINS = 6
MAX_PAVA_ADJUSTMENT_LOG10 = 0.50
MAX_TEST_SUPPORT_CLIP_FRACTION = 0.25
MIN_GLOBAL_IMPROVEMENT_PERCENT = 5.0
MAX_SYSTEMATIC_WORSENING_PERCENT = -5.0
MIN_BETTER_CASE_FRACTION = 0.60
EQUALITY_RELATIVE_TOLERANCE = 1e-6
IDENTICAL_RELATIVE_SPREAD = 0.10
MODERATE_RELATIVE_SPREAD = 0.30
VISUAL_RESIDUAL_SAMPLES_PER_CASE = 12000

MATERIALS = ("The Matrix", "BR2049")
MODELS = ("Model A", "Model B", "Shot-adaptive C")

# These are the exact diagnostic SDR bins required by the feasibility study.
SDR_BINS = [
    (0.0, 0.01, "<0.01"),
    (0.01, 0.1, "0.01–0.1"),
    (0.1, 0.5, "0.1–0.5"),
    (0.5, 1.0, "0.5–1"),
    (1.0, 2.0, "1–2"),
    (2.0, 5.0, "2–5"),
    (5.0, 10.0, "5–10"),
    (10.0, 20.0, "10–20"),
    (20.0, 50.0, "20–50"),
    (50.0, 100.0, "50–100"),
    (100.0, 200.0, "100–200"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
]

# Test-region metrics are based on the real HDR-master target luminance.
TEST_REGIONS = [
    (0.0, float("inf"), "global"),
    (0.0, 10.0, "shadow_lt_10"),
    (10.0, 200.0, "midtone_10_200"),
    (200.0, float("inf"), "highlight_gt_200"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
]
QUERY_NITS = (10.0, 50.0, 100.0, 200.0, 500.0)


# Read-only constants copied only to interpret the existing Model A curve output.
# The production code itself is imported unchanged for Model A evaluation.
def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (float,)):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def require_under(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(f"{description} is outside the immutable P2.32 directory: {resolved}") from exc
    return resolved


def pava(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.copy()
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            left, right = blocks[-2], blocks[-1]
            weight = left[1] + right[1]
            blocks[-2] = [(left[0] * left[1] + right[0] * right[1]) / weight, weight]
            blocks.pop()
    output = np.empty(values.size, dtype=np.float64)
    cursor = 0
    for value, weight in blocks:
        count = int(weight)
        output[cursor:cursor + count] = value
        cursor += count
    return output


def is_bin_member(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if math.isfinite(high):
        return (values >= low) & (values < high)
    return values >= low


def empty_metric() -> dict[str, Any]:
    return {
        "count": 0,
        "signed_mean": None,
        "median_residual": None,
        "MAE": None,
        "RMSE": None,
        "P95": None,
        "P99": None,
        "max": None,
        "relative": {
            "signed_mean": None,
            "median": None,
            "MAE": None,
            "P95": None,
            "P99": None,
            "max": None,
            "epsilon_nits": LOG_EPS,
        },
    }


def metric_block(residual: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if residual.size == 0:
        return empty_metric()
    finite = np.isfinite(residual) & np.isfinite(target)
    residual = residual[finite]
    target = target[finite]
    if residual.size == 0:
        return empty_metric()
    absolute = np.abs(residual)
    relative = residual / np.maximum(np.abs(target), LOG_EPS)
    return {
        "count": int(residual.size),
        "signed_mean": float(np.mean(residual)),
        "median_residual": float(np.median(residual)),
        "MAE": float(np.mean(absolute)),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "P95": float(np.percentile(absolute, 95)),
        "P99": float(np.percentile(absolute, 99)),
        "max": float(np.max(absolute)),
        "relative": {
            "signed_mean": float(np.mean(relative)),
            "median": float(np.median(relative)),
            "MAE": float(np.mean(np.abs(relative))),
            "P95": float(np.percentile(np.abs(relative), 95)),
            "P99": float(np.percentile(np.abs(relative), 99)),
            "max": float(np.max(np.abs(relative))),
            "epsilon_nits": LOG_EPS,
        },
    }


def metrics_for_test(residual: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for low, high, label in TEST_REGIONS:
        mask = is_bin_member(target, low, high)
        result[label] = metric_block(residual[mask], target[mask])
    return result


def load_dataset() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not P232_METRICS_PATH.exists() or not P232_MANIFEST_PATH.exists():
        raise RuntimeError("P2.32 manifest or metrics JSON is missing")
    dataset = json.loads(P232_METRICS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(P232_MANIFEST_PATH.read_text(encoding="utf-8"))
    if dataset.get("phase") != "P2.32" or dataset.get("status") != "DATASET READY":
        raise RuntimeError("P2.32 metrics are not marked DATASET READY")
    if manifest.get("phase") != "P2.32" or not manifest.get("selection_locked_before_decoding"):
        raise RuntimeError("P2.32 manifest is not the locked dataset selection")
    cases: list[dict[str, Any]] = []
    for material in MATERIALS:
        records = dataset.get("case_records", {}).get(material, [])
        manifest_records = manifest.get("materials", {}).get(material, {}).get("selected_cases", [])
        if len(records) != 20 or len(manifest_records) != 20:
            raise RuntimeError(f"Expected 20 cases in both P2.32 records for {material}")
        record_ids = {record["case_id"] for record in records}
        manifest_ids = {record["case_id"] for record in manifest_records}
        if record_ids != manifest_ids:
            raise RuntimeError(f"P2.32 manifest/metrics case mismatch for {material}")
        for case in records:
            sdr_path = require_under(Path(case["sdr_luminance_nits_path"]), P232_DIR, "SDR luminance path")
            hdr_path = require_under(Path(case["hdr_luminance_nits_path"]), P232_DIR, "HDR luminance path")
            if not sdr_path.exists() or not hdr_path.exists():
                raise RuntimeError(f"Missing P2.32 luminance arrays for {case['case_id']}")
            if Path(case["hdr_rgb_u16_path"]).exists():
                require_under(Path(case["hdr_rgb_u16_path"]), P232_DIR, "HDR RGB path")
            if Path(case["om_rgb_u16_path"]).exists():
                require_under(Path(case["om_rgb_u16_path"]), P232_DIR, "OM RGB path")
            if case.get("finite", {}).get("sdr_luminance") is not True or case.get("finite", {}).get("hdr_luminance") is not True:
                raise RuntimeError(f"P2.32 finite flag failed for {case['case_id']}")
            cases.append({**case, "sdr_luminance_nits_path": str(sdr_path), "hdr_luminance_nits_path": str(hdr_path)})
    if len(cases) != 40 or len({case["case_id"] for case in cases}) != 40:
        raise RuntimeError("P2.32 case inventory is not exactly 40 unique cases")
    return dataset, manifest, cases


def make_spatial_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    height, width = shape
    if height < TILE_ROWS or width < TILE_COLS:
        raise RuntimeError(f"Overlap shape {shape} is too small for the fixed tile grid")
    row_edges = np.linspace(0, height, TILE_ROWS + 1, dtype=np.int64)
    col_edges = np.linspace(0, width, TILE_COLS + 1, dtype=np.int64)
    tile_ids = np.empty(shape, dtype=np.uint16)
    train_mask = np.zeros(shape, dtype=bool)
    test_mask = np.zeros(shape, dtype=bool)
    tile_records = []
    tile_id = 0
    for tile_row in range(TILE_ROWS):
        for tile_col in range(TILE_COLS):
            y0, y1 = int(row_edges[tile_row]), int(row_edges[tile_row + 1])
            x0, x1 = int(col_edges[tile_col]), int(col_edges[tile_col + 1])
            is_test = (tile_id % TEST_TILE_MODULUS) in TEST_TILE_REMAINDERS
            tile_ids[y0:y1, x0:x1] = tile_id
            if is_test:
                test_mask[y0:y1, x0:x1] = True
            else:
                train_mask[y0:y1, x0:x1] = True
            tile_records.append({
                "tile_id": tile_id,
                "tile_row": tile_row,
                "tile_col": tile_col,
                "y0": y0,
                "y1": y1,
                "x0": x0,
                "x1": x1,
                "assignment": "test" if is_test else "train",
            })
            tile_id += 1
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise RuntimeError("Spatial train/test masks overlap or leave pixels unassigned")
    metadata = {
        "method": "fixed whole-tile spatial split",
        "grid": [TILE_ROWS, TILE_COLS],
        "test_rule": f"tile_id % {TEST_TILE_MODULUS} in {list(TEST_TILE_REMAINDERS)}",
        "test_fraction_by_tile_count": len(TEST_TILE_REMAINDERS) / TEST_TILE_MODULUS,
        "train_tile_count": sum(item["assignment"] == "train" for item in tile_records),
        "test_tile_count": sum(item["assignment"] == "test" for item in tile_records),
        "tiles": tile_records,
        "pixel_counts": {"train": int(np.sum(train_mask)), "test": int(np.sum(test_mask))},
        "pixel_level_random_split": False,
        "split_created_before_fitting": True,
    }
    return train_mask, test_mask, tile_ids, metadata


def filtering_masks(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    nonnegative = finite & (x >= 0.0) & (y >= 0.0)
    clipped = nonnegative & ((x >= HARD_CLIP_NITS) | (y >= HARD_CLIP_NITS))
    low_signal = nonnegative & ~clipped & (x < MIN_STABLE_SDR_NITS)
    accepted = nonnegative & ~clipped & ~low_signal
    counts = {
        "raw_sample_count": int(x.size),
        "accepted_sample_count": int(np.sum(accepted)),
        "rejected_sample_count": int(x.size - np.sum(accepted)),
        "rejection_percentage": float(100.0 * (x.size - np.sum(accepted)) / max(x.size, 1)),
        "rejection_reasons": {
            "nonfinite": int(np.sum(~finite)),
            "negative_luminance": int(np.sum(finite & ~nonnegative)),
            "hard_clipping": int(np.sum(clipped)),
            "low_sdr_signal_below_0.01_nits": int(np.sum(low_signal)),
            "extreme_outlier": 0,
            "unreliable_correspondence": 0,
            "motion_or_misalignment": 0,
        },
        "filter_policy": {
            "finite_required": True,
            "nonnegative_required": True,
            "hard_clip_threshold_nits": HARD_CLIP_NITS,
            "low_sdr_stability_threshold_nits": MIN_STABLE_SDR_NITS,
            "extreme_outliers_removed": False,
            "outlier_influence_control": "per-bin median/P10/P90 and monotonic PAVA; no residual-based deletion",
            "correspondence_filter": "none beyond P2.32 shape-matched overlap; no motion metadata available",
        },
    }
    return accepted, counts


def bin_stats(x: np.ndarray, y: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    bin_index = np.digitize(x, np.asarray([edge[1] for edge in SDR_BINS[:-1]], dtype=np.float64), right=False)
    records: list[dict[str, Any]] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_bins: list[int] = []
    for index, (low, high, label) in enumerate(SDR_BINS):
        selected = bin_index == index
        count = int(np.sum(selected))
        record: dict[str, Any] = {"bin_index": index, "label": label, "low_nits": low, "high_nits": None if not math.isfinite(high) else high, "count": count, "P10": None, "P50": None, "P90": None, "robust_spread_P90_minus_P10": None, "relative_spread": None}
        if count:
            selected_y = y[selected]
            p10, p50, p90 = [float(value) for value in np.percentile(selected_y, [10, 50, 90])]
            record.update({"P10": p10, "P50": p50, "P90": p90, "robust_spread_P90_minus_P10": p90 - p10, "relative_spread": float((p90 - p10) / max(abs(p50), LOG_EPS))})
            node_x.append(float(np.median(x[selected])))
            node_y.append(p50)
            node_bins.append(index)
        records.append(record)
    if not node_x:
        return records, np.empty(0), np.empty(0), {"nonempty_bin_count": 0, "node_bin_indices": [], "raw_monotonicity_violations": 0, "pava_adjustment_max_abs_log10": None}
    raw_x_log = np.log10(np.maximum(np.asarray(node_x), LOG_EPS))
    raw_y_log = np.log10(np.maximum(np.asarray(node_y), 0.0) + LOG_EPS)
    order = np.argsort(raw_x_log)
    raw_x_log = raw_x_log[order]
    raw_y_log = raw_y_log[order]
    node_bins_array = np.asarray(node_bins, dtype=np.int64)[order]
    unique_x, unique_indices = np.unique(raw_x_log, return_index=True)
    raw_y_log = raw_y_log[unique_indices]
    node_bins_array = node_bins_array[unique_indices]
    mono_y_log = pava(raw_y_log)
    diagnostics = {
        "nonempty_bin_count": int(unique_x.size),
        "node_bin_indices": [int(value) for value in node_bins_array],
        "raw_monotonicity_violations": int(np.sum(np.diff(raw_y_log) < 0.0)),
        "post_pava_monotonicity_violations": int(np.sum(np.diff(mono_y_log) < -1e-12)),
        "pava_adjustment_max_abs_log10": float(np.max(np.abs(mono_y_log - raw_y_log))) if mono_y_log.size else None,
        "raw_y_log10": [float(value) for value in raw_y_log],
        "post_pava_y_log10": [float(value) for value in mono_y_log],
    }
    return records, unique_x, mono_y_log, diagnostics


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in model.items() if key not in {"x_log", "y_log"}}
    if "x_log" in model:
        result["x_log"] = [float(value) for value in np.asarray(model["x_log"]).reshape(-1)]
    if "y_log" in model:
        result["y_log"] = [float(value) for value in np.asarray(model["y_log"]).reshape(-1)]
    return json_safe(result)


def fit_shot_model(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    diagnostics_rows, x_log, y_log, bin_diagnostics = bin_stats(x, y)
    model = {
        "name": "Shot-adaptive C",
        "kind": "fixed_diagnostic_bin_median",
        "x_log": x_log,
        "y_log": y_log,
        "definition": {
            "type": "per-frame empirical mapping",
            "bins": "fixed SDR bins required by P2.33",
            "target_statistic": "HDR P50 median per nonempty bin",
            "spread_statistics": ["P10", "P50", "P90"],
            "interpolation_for_prediction": "linear in log10 SDR/log10 HDR between nonempty nodes only",
            "missing_bins_interpolated_in_diagnostics": False,
            "monotonic_projection": "PAVA on P50 log targets",
            "outside_domain": "clamp to nearest observed node; no extrapolation",
        },
        "domain_log10_sdr": [float(x_log[0]), float(x_log[-1])] if x_log.size else None,
        "nonempty_bin_count": int(x_log.size),
        "raw_monotonicity_violations": int(bin_diagnostics["raw_monotonicity_violations"]),
        "post_pava_monotonicity_violations": int(bin_diagnostics["post_pava_monotonicity_violations"]),
        "pava_adjustment_max_abs_log10": bin_diagnostics["pava_adjustment_max_abs_log10"],
    }
    return model, diagnostics_rows, bin_diagnostics


def fit_global_model_b(samples_x: np.ndarray, samples_y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(samples_x, dtype=np.float64).reshape(-1)
    y = np.asarray(samples_y, dtype=np.float64).reshape(-1)
    x_log = np.log10(np.maximum(x, LOG_EPS))
    y_log = np.log10(np.maximum(y, 0.0) + LOG_EPS)
    low, high = float(np.min(x_log)), float(np.max(x_log))
    edges = np.linspace(low, high, 513) if high > low else np.asarray([low, low + 1.0])
    bin_index = np.digitize(x_log, edges[1:-1], right=False)
    node_x: list[float] = []
    node_y: list[float] = []
    for index in range(edges.size - 1):
        selected = bin_index == index
        if np.any(selected):
            node_x.append(float(np.median(x_log[selected])))
            node_y.append(float(np.median(y_log[selected])))
    order = np.argsort(np.asarray(node_x))
    node_x_array = np.asarray(node_x, dtype=np.float64)[order]
    raw_y = np.asarray(node_y, dtype=np.float64)[order]
    unique_x, unique_indices = np.unique(node_x_array, return_index=True)
    raw_y = raw_y[unique_indices]
    mono_y = pava(raw_y)
    return {
        "name": "Model B",
        "kind": "global_fixed_width_bin_median",
        "x_log": unique_x,
        "y_log": mono_y,
        "definition": {
            "type": "P2.30/P2.31 historical global comparator",
            "fixed_width_log10_bins": 512,
            "target_statistic": "HDR median per populated bin",
            "monotonic_projection": "PAVA",
            "interpolation": "linear in log10 SDR/log10 HDR",
            "extrapolation": False,
            "training": "balanced deterministic spatial-TRAIN sample, equal 25000 samples per P2.32 case when available",
        },
        "training_sample_count": int(x.size),
        "domain_log10_sdr": [float(unique_x[0]), float(unique_x[-1])] if unique_x.size else None,
        "raw_monotonicity_violations": int(np.sum(np.diff(raw_y) < 0.0)),
        "post_pava_monotonicity_violations": int(np.sum(np.diff(mono_y) < -1e-12)),
        "pava_adjustment_max_abs_log10": float(np.max(np.abs(mono_y - raw_y))) if mono_y.size else None,
    }


def load_model_a() -> dict[str, Any]:
    if not MODEL_A_CURVE_PATH.exists():
        raise RuntimeError(f"Model A reference curve is missing: {MODEL_A_CURVE_PATH}")
    from auto_openmatte.processing.luminance import apply_luminance_curve, build_curve_lut

    curve = json.loads(MODEL_A_CURVE_PATH.read_text(encoding="utf-8"))
    return {
        "name": "Model A",
        "kind": "baseline",
        "curve": curve,
        "lut": build_curve_lut(curve),
        "apply": apply_luminance_curve,
        "definition": {
            "type": "unchanged production/reference Model A",
            "curve_path": str(MODEL_A_CURVE_PATH),
            "control_points": len(curve),
            "lut_entries": 65536,
            "refit": False,
            "production_code_modified": False,
        },
    }


def predict_model(model: dict[str, Any], input_nits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(input_nits, dtype=np.float64).reshape(-1)
    if model["kind"] == "baseline":
        curve = np.asarray(model["curve"], dtype=np.float64)
        low_log, high_log = float(curve[0, 0]), float(curve[-1, 0])
        input_log = np.log10(np.maximum(values, LOG_EPS))
        low_clip = input_log < low_log
        high_clip = input_log > high_log
        normalized = np.clip(values / PEAK_NITS, 0.0, 1.0)
        prediction = np.asarray(model["apply"](normalized, model["curve"], prebuilt_lut=model["lut"]), dtype=np.float64) * PEAK_NITS
    else:
        x_log = np.asarray(model.get("x_log", []), dtype=np.float64)
        y_log = np.asarray(model.get("y_log", []), dtype=np.float64)
        if x_log.size < 2:
            return np.full(values.shape, np.nan), {"low_clip_count": int(values.size), "high_clip_count": 0, "clipping_fraction": 1.0, "domain_log10_sdr": None}
        input_log = np.log10(np.maximum(values, LOG_EPS))
        low_log, high_log = float(x_log[0]), float(x_log[-1])
        low_clip = input_log < low_log
        high_clip = input_log > high_log
        clipped_log = np.clip(input_log, low_log, high_log)
        output_log = np.interp(clipped_log, x_log, y_log)
        prediction = np.maximum(np.power(10.0, output_log) - LOG_EPS, 0.0)
    info = {
        "low_clip_count": int(np.sum(low_clip)),
        "high_clip_count": int(np.sum(high_clip)),
        "clipping_count": int(np.sum(low_clip | high_clip)),
        "clipping_fraction": float(np.mean(low_clip | high_clip)) if values.size else 0.0,
        "domain_log10_sdr": [low_log, high_log],
        "extrapolation_count": 0,
    }
    return prediction, info


def shot_characteristics(model: dict[str, Any], train_x: np.ndarray, train_y: np.ndarray, all_case_y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    query_values, query_info = {}, {}
    for query in (0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0):
        prediction, info = predict_model(model, np.asarray([query], dtype=np.float64))
        query_values[str(query)] = float(prediction[0]) if np.isfinite(prediction[0]) else None
        query_info[str(query)] = {"clipped": bool(info["clipping_count"]), "low_clipped": bool(info["low_clip_count"]), "high_clipped": bool(info["high_clip_count"])}
    model_x = np.asarray(model.get("x_log", []), dtype=np.float64)
    model_y = np.asarray(model.get("y_log", []), dtype=np.float64)
    local_slopes = []
    if model_x.size >= 2:
        for index in range(model_x.size - 1):
            delta_x = model_x[index + 1] - model_x[index]
            if delta_x > 0:
                local_slopes.append({"x_low_nits": float(10.0 ** model_x[index] - LOG_EPS), "x_high_nits": float(10.0 ** model_x[index + 1] - LOG_EPS), "log_slope": float((model_y[index + 1] - model_y[index]) / delta_x)})
    mid_slopes = [item["log_slope"] for item in local_slopes if item["x_high_nits"] >= 10.0 and item["x_low_nits"] <= 200.0]
    high_slopes = [item["log_slope"] for item in local_slopes if item["x_high_nits"] >= 200.0]
    median_mid_slope = float(np.median(mid_slopes)) if mid_slopes else None
    median_high_slope = float(np.median(high_slopes)) if high_slopes else None
    if median_high_slope is None:
        rolloff = "insufficient_high_end_domain"
    elif median_high_slope < 0.90:
        rolloff = "compresses_high_end"
    elif median_high_slope > 1.10:
        rolloff = "expands_high_end"
    else:
        rolloff = "near_linear_high_end"
    plateau_fraction = float(np.mean(np.abs(np.diff(model_y)) <= 1e-10)) if model_y.size >= 2 else 0.0
    x_p01, x_p99 = [float(value) for value in np.percentile(x, [1, 99])] if x.size else (None, None)
    y_p01, y_p99 = [float(value) for value in np.percentile(y, [1, 99])] if y.size else (None, None)
    if x_p01 is not None and x_p99 is not None and y_p01 is not None and y_p99 is not None:
        x_range = math.log10(max(x_p99, LOG_EPS)) - math.log10(max(x_p01, LOG_EPS))
        y_range = math.log10(max(y_p99, LOG_EPS) + LOG_EPS) - math.log10(max(y_p01, 0.0) + LOG_EPS)
        compression = float(y_range / x_range) if x_range > 0 else None
    else:
        compression = None
    return {
        "black_shadow_response": {"query_outputs_nits": query_values, "query_clipping": query_info},
        "midtone_response": {"log_slope_10_to_200_summary": median_mid_slope, "local_slopes": local_slopes},
        "highlight_roll_off": {"classification": rolloff, "median_log_slope_high_end": median_high_slope, "local_high_end_slopes": high_slopes},
        "maximum_observed_hdr_nits": float(np.max(all_case_y)) if all_case_y.size else None,
        "maximum_train_hdr_nits": float(np.max(y)) if y.size else None,
        "dynamic_range_compression_ratio_log_range_1_to_99_percentile": compression,
        "plateau_fraction": plateau_fraction,
        "plateau_present": bool(plateau_fraction > 0.05),
        "shoulder_present": bool(median_mid_slope is not None and median_high_slope is not None and median_high_slope < median_mid_slope - 0.10),
        "node_count": int(model_x.size),
        "domain_sdr_nits": [float(10.0 ** model_x[0] - LOG_EPS), float(10.0 ** model_x[-1] - LOG_EPS)] if model_x.size else None,
    }


def curve_artifact(model: dict[str, Any], diagnostics_rows: list[dict[str, Any]], bin_diagnostics: dict[str, Any], characteristics: dict[str, Any], case: dict[str, Any], filtering: dict[str, Any], split: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "P2.33",
        "case_id": case["case_id"],
        "material": case["material"],
        "hdr_frame": case["hdr_frame"],
        "om_frame": case["om_frame"],
        "mapping_target_provenance": "direct P2.32 HDR-master luminance overlap array",
        "mapping_input_provenance": "direct P2.32 SDR Open Matte luminance overlap array",
        "split": split,
        "filtering": filtering,
        "diagnostic_bins": diagnostics_rows,
        "curve": model_to_json(model),
        "bin_diagnostics": bin_diagnostics,
        "characteristics": characteristics,
    }


def aggregate_macro(material_metrics: dict[str, dict[str, Any]], model: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for _, _, label in TEST_REGIONS:
        blocks = [material_metrics[material][model][label] for material in MATERIALS if material_metrics[material][model][label]["count"] > 0]
        if not blocks:
            result[label] = empty_metric()
            result[label]["available_materials"] = []
            result[label]["aggregation"] = "unweighted arithmetic mean over available material metrics"
            continue
        block = empty_metric()
        block["count"] = int(sum(item["count"] for item in blocks))
        for key in ("signed_mean", "median_residual", "MAE", "RMSE", "P95", "P99", "max"):
            block[key] = float(np.mean([item[key] for item in blocks]))
        for key in ("signed_mean", "median", "MAE", "P95", "P99", "max"):
            block["relative"][key] = float(np.mean([item["relative"][key] for item in blocks]))
        block["available_materials"] = [material for material in MATERIALS if material_metrics[material][model][label]["count"] > 0]
        block["aggregation"] = "unweighted arithmetic mean over available material metrics; counts summed"
        result[label] = block
    return result


def aggregate_exact(states: list[dict[str, Any]]) -> dict[str, Any]:
    material_metrics: dict[str, dict[str, dict[str, Any]]] = {material: {} for material in MATERIALS}
    combined_weighted: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        all_residuals: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        for material in MATERIALS:
            residuals: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for state in states:
                if state["material"] != material:
                    continue
                artifact = np.load(state["residual_path"], allow_pickle=False)
                residual_key = state["residual_keys"][model]
                residuals.append(np.asarray(artifact[residual_key], dtype=np.float32))
                indices = np.asarray(artifact["test_flat_indices"], dtype=np.int64)
                target_all = np.asarray(np.load(state["case"]["hdr_luminance_nits_path"], mmap_mode="r")).reshape(-1)
                targets.append(np.asarray(target_all[indices], dtype=np.float32))
                artifact.close()
            material_residual = np.concatenate(residuals) if residuals else np.empty(0, dtype=np.float32)
            material_target = np.concatenate(targets) if targets else np.empty(0, dtype=np.float32)
            material_metrics[material][model] = metrics_for_test(material_residual, material_target)
            all_residuals.append(material_residual)
            all_targets.append(material_target)
        combined_residual = np.concatenate(all_residuals) if all_residuals else np.empty(0, dtype=np.float32)
        combined_target = np.concatenate(all_targets) if all_targets else np.empty(0, dtype=np.float32)
        combined_weighted[model] = metrics_for_test(combined_residual, combined_target)
        del all_residuals, all_targets, combined_residual, combined_target
    combined_macro = {model: aggregate_macro(material_metrics, model) for model in MODELS}
    return {"materials": material_metrics, "combined": {"weighted": combined_weighted, "macro_average": combined_macro}}


def improvement_percent(a: dict[str, Any], c: dict[str, Any]) -> float | None:
    if a.get("MAE") is None or c.get("MAE") is None:
        return None
    return float(100.0 * (a["MAE"] - c["MAE"]) / max(abs(a["MAE"]), LOG_EPS))


def summary_statistics(values: list[float], better: int, equal: int, worse: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "P25": None, "P75": None, "min": None, "max": None, "adaptive_better": 0, "adaptive_equal": 0, "adaptive_worse": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)), "P25": float(np.percentile(array, 25)), "P75": float(np.percentile(array, 75)), "min": float(np.min(array)), "max": float(np.max(array)), "adaptive_better": int(better), "adaptive_equal": int(equal), "adaptive_worse": int(worse)}


def per_frame_improvement_summary(states: list[dict[str, Any]], material: str | None = None) -> dict[str, Any]:
    selected = [state for state in states if material is None or state["material"] == material]
    summary: dict[str, Any] = {}
    for _, _, label in TEST_REGIONS:
        values: list[float] = []
        better = equal = worse = 0
        for state in selected:
            a = state["models"]["Model A"]["metrics"][label]
            c = state["models"]["Shot-adaptive C"]["metrics"][label]
            value = improvement_percent(a, c)
            if value is None:
                continue
            values.append(value)
            threshold = EQUALITY_RELATIVE_TOLERANCE * max(abs(a["MAE"] or 0.0), 1.0) * 100.0
            if value > threshold:
                better += 1
            elif value < -threshold:
                worse += 1
            else:
                equal += 1
        summary[label] = summary_statistics(values, better, equal, worse)
    return summary


def variability_for_group(curves: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"case_count": len(curves), "queries_nits": {}, "classification": "insufficient_data"}
    relative_spreads: list[float] = []
    for query in QUERY_NITS:
        values = []
        for curve in curves:
            prediction, _ = predict_model(curve["model"], np.asarray([query], dtype=np.float64))
            if np.isfinite(prediction[0]):
                values.append(float(prediction[0]))
        if not values:
            result["queries_nits"][str(query)] = {"count": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        p05, p10, p90, p95 = [float(value) for value in np.percentile(array, [5, 10, 90, 95])]
        relative = float((p90 - p10) / max(abs(median), 0.01))
        relative_spreads.append(relative)
        result["queries_nits"][str(query)] = {
            "count": int(array.size),
            "median_nits": median,
            "P10_nits": p10,
            "P90_nits": p90,
            "P95_nits": p95,
            "min_nits": float(np.min(array)),
            "max_nits": float(np.max(array)),
            "curve_spread_max_minus_min_nits": float(np.max(array) - np.min(array)),
            "median_absolute_difference_nits": float(np.median(np.abs(array - median))),
            "P10_P90_spread_nits": float(p90 - p10),
            "P95_spread_defined_as_P95_minus_P5_nits": float(p95 - p05),
            "relative_P10_P90_spread": relative,
        }
    worst = max(relative_spreads) if relative_spreads else float("inf")
    if worst <= IDENTICAL_RELATIVE_SPREAD:
        result["classification"] = "practically_identical"
    elif worst <= MODERATE_RELATIVE_SPREAD:
        result["classification"] = "moderately_different"
    else:
        result["classification"] = "strongly_different"
    result["classification_thresholds_predeclared"] = {
        "practically_identical_max_relative_P10_P90_spread": IDENTICAL_RELATIVE_SPREAD,
        "moderately_different_max_relative_P10_P90_spread": MODERATE_RELATIVE_SPREAD,
        "above_moderate_is_strongly_different": True,
    }
    return result


def aggregate_support(states: list[dict[str, Any]], model: str, material: str | None = None) -> dict[str, Any]:
    fractions = [state["models"][model]["support"]["clipping_fraction"] for state in states if material is None or state["material"] == material]
    if not fractions:
        return {"case_count": 0, "mean_clipping_fraction": None, "median_clipping_fraction": None, "max_clipping_fraction": None}
    values = np.asarray(fractions, dtype=np.float64)
    return {"case_count": int(values.size), "mean_clipping_fraction": float(np.mean(values)), "median_clipping_fraction": float(np.median(values)), "max_clipping_fraction": float(np.max(values))}


def plot_log_curve_set(path: Path, title: str, curve_records: list[dict[str, Any]], color: tuple[int, int, int], show_band: bool = False) -> None:
    width, height = 1400, 900
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 110, 80, 50, 120
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = -2.0, 3.3
    y_min, y_max = -3.0, 4.1

    def point(x: float, y: float) -> tuple[int, int] | None:
        if x <= 0 or y <= 0 or not math.isfinite(x) or not math.isfinite(y):
            return None
        px = left + int((math.log10(x) - x_min) / (x_max - x_min) * plot_width)
        py = top + plot_height - int((math.log10(y) - y_min) / (y_max - y_min) * plot_height)
        return px, py

    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (0, 0, 0), 2)
    for tick in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0):
        p = point(tick, 0.001)
        if p:
            cv2.line(canvas, (p[0], top), (p[0], top + plot_height), (225, 225, 225), 1)
            cv2.putText(canvas, f"{tick:g}", (p[0] - 18, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        p = point(0.01, tick)
        if p:
            cv2.line(canvas, (left, p[1]), (left + plot_width, p[1]), (225, 225, 225), 1)
            cv2.putText(canvas, f"{tick:g}", (15, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    for record in curve_records:
        model = record["model"]
        x_log = np.asarray(model.get("x_log", []), dtype=np.float64)
        y_log = np.asarray(model.get("y_log", []), dtype=np.float64)
        points = [point(float(10.0 ** x - LOG_EPS), float(10.0 ** y - LOG_EPS)) for x, y in zip(x_log, y_log)]
        points = [item for item in points if item is not None]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color if not show_band else (170, 170, 170), 2 if not show_band else 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "SDR Open Matte luminance [nits, log]", (left + 350, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "HDR master luminance [nits, log]", (5, top + plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def plot_material_median_spread(path: Path, grouped_curves: dict[str, list[dict[str, Any]]]) -> None:
    width, height = 1500, 850
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 80, 50, 110
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = -2.0, 3.3
    y_min, y_max = -3.0, 4.1
    colors = {"The Matrix": (30, 80, 210), "BR2049": (40, 150, 60)}

    def pxy(x: float, y: float) -> tuple[int, int] | None:
        if x <= 0 or y <= 0 or not math.isfinite(x) or not math.isfinite(y):
            return None
        return (left + int((math.log10(x) - x_min) / (x_max - x_min) * plot_width), top + plot_height - int((math.log10(y) - y_min) / (y_max - y_min) * plot_height))

    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (0, 0, 0), 2)
    grid_x = np.geomspace(0.01, 2000.0, 180)
    for material, curves in grouped_curves.items():
        if not curves:
            continue
        medians, p10s, p90s = [], [], []
        for x in grid_x:
            values = []
            for curve in curves:
                prediction, _ = predict_model(curve["model"], np.asarray([x], dtype=np.float64))
                if np.isfinite(prediction[0]):
                    values.append(float(prediction[0]))
            if values:
                medians.append(float(np.median(values)))
                p10s.append(float(np.percentile(values, 10)))
                p90s.append(float(np.percentile(values, 90)))
            else:
                medians.append(float("nan")); p10s.append(float("nan")); p90s.append(float("nan"))
        band = [pxy(float(x), float(y)) for x, y in zip(grid_x, p90s)]
        band += [pxy(float(x), float(y)) for x, y in zip(grid_x[::-1], p10s[::-1])]
        band = [point for point in band if point is not None]
        if len(band) >= 3:
            cv2.fillPoly(canvas, [np.asarray(band, dtype=np.int32)], tuple(int(v) for v in colors[material]))
        points = [pxy(float(x), float(y)) for x, y in zip(grid_x, medians)]
        points = [point for point in points if point is not None]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, colors[material], 4, cv2.LINE_AA)
    cv2.putText(canvas, "Shot-adaptive C median curve and P10–P90 spread", (left, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blue = The Matrix, green = BR2049; band is visualization only", (left, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def plot_residual_scatter(path: Path, samples: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> None:
    width, height = 1500, 850
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 70, 50, 100
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = -2.0, 3.3
    residual_values = np.concatenate([values for model_values in samples.values() for _, values in model_values]) if any(samples.values()) else np.asarray([1.0])
    limit = max(float(np.percentile(np.abs(residual_values), 99)), 1.0)
    y_min, y_max = -limit, limit

    def point(x: float, y: float) -> tuple[int, int] | None:
        if x <= 0 or not math.isfinite(x) or not math.isfinite(y):
            return None
        px = left + int((math.log10(x) - x_min) / (x_max - x_min) * plot_width)
        py = top + plot_height - int((y - y_min) / (y_max - y_min) * plot_height)
        return px, py

    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (0, 0, 0), 2)
    zero = point(0.01, 0.0)
    if zero:
        cv2.line(canvas, (left, zero[1]), (left + plot_width, zero[1]), (150, 150, 150), 2)
    colors = {"Model A": (70, 70, 210), "Shot-adaptive C": (50, 150, 50)}
    for model, model_values in samples.items():
        for x_values, residual in model_values:
            for x, value in zip(x_values, residual):
                p = point(float(x), float(value))
                if p:
                    cv2.circle(canvas, p, 1, colors[model], -1)
    cv2.putText(canvas, "TEST residuals versus SDR input (sampled for visualization)", (left, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"y-axis clipped to absolute P99 = {limit:.3f} nits; numeric residual arrays are authoritative", (left, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def plot_residual_histogram(path: Path, samples: dict[str, np.ndarray]) -> None:
    width, height = 1300, 800
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 80, 60, 110
    plot_width, plot_height = width - left - right, height - top - bottom
    combined = np.concatenate([values for values in samples.values() if values.size]) if any(values.size for values in samples.values()) else np.asarray([1.0])
    limit = max(float(np.percentile(np.abs(combined), 99)), 1.0)
    edges = np.linspace(-limit, limit, 61)
    colors = {"Model A": (80, 80, 210), "Shot-adaptive C": (50, 150, 50)}
    max_count = 1
    histograms = {}
    for model, values in samples.items():
        hist, _ = np.histogram(values, bins=edges)
        histograms[model] = hist
        max_count = max(max_count, int(np.max(hist)))
    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (0, 0, 0), 2)
    for index in range(len(edges) - 1):
        x0 = left + int(index * plot_width / (len(edges) - 1))
        x1 = left + int((index + 1) * plot_width / (len(edges) - 1))
        for model_index, model in enumerate(("Model A", "Shot-adaptive C")):
            count = histograms[model][index]
            bar_height = int(count / max_count * plot_height * 0.45)
            offset = 0 if model_index == 0 else int(plot_height * 0.45)
            cv2.rectangle(canvas, (x0, top + plot_height - bar_height - offset), (max(x0 + 1, x1 - 1), top + plot_height - offset), colors[model], -1)
    cv2.putText(canvas, "TEST residual distributions (sampled; signed nits)", (left, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"x-axis clipped to ±P99 = {limit:.3f} nits; blue Model A, green Shot-adaptive C", (left, height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def plot_high_end(path: Path, aggregate: dict[str, Any]) -> None:
    width, height = 1300, 800
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 80, 60, 120
    plot_width, plot_height = width - left - right, height - top - bottom
    labels = ["A", "B", "C"]
    model_names = list(MODELS)
    values = [aggregate["combined"]["macro_average"][model]["highlight_gt_200"]["MAE"] or 0.0 for model in model_names]
    maximum = max(values or [1.0]) * 1.2
    cv2.rectangle(canvas, (left, top), (left + plot_width, top + plot_height), (0, 0, 0), 2)
    colors = [(80, 80, 210), (180, 120, 40), (50, 150, 50)]
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + int((index + 0.5) * plot_width / len(values))
        bar_height = int(value / max(maximum, 1e-9) * plot_height)
        cv2.rectangle(canvas, (x - 70, top + plot_height - bar_height), (x + 70, top + plot_height), color, -1)
        cv2.putText(canvas, label, (x - 12, height - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.3f}", (x - 50, top + plot_height - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "HDR TEST high-end (>200 nits) macro MAE", (left, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "A = Model A, B = global historical Model B, C = shot-adaptive", (left, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def decision(aggregate: dict[str, Any], states: list[dict[str, Any]], variability: dict[str, Any], temporal: dict[str, Any]) -> dict[str, Any]:
    a_macro = aggregate["combined"]["macro_average"]["Model A"]
    c_macro = aggregate["combined"]["macro_average"]["Shot-adaptive C"]
    global_improvement = improvement_percent(a_macro["global"], c_macro["global"])
    per_material_improvements = {material: improvement_percent(aggregate["materials"][material]["Model A"]["global"], aggregate["materials"][material]["Shot-adaptive C"]["global"]) for material in MATERIALS}
    global_summary = per_frame_improvement_summary(states)
    material_summaries = {material: per_frame_improvement_summary(states, material) for material in MATERIALS}
    high_end_available_materials = [material for material in MATERIALS if aggregate["materials"][material]["Shot-adaptive C"]["highlight_gt_200"]["count"] > 0]
    high_end_improvement = improvement_percent(a_macro["highlight_gt_200"], c_macro["highlight_gt_200"])
    max_clip = max(state["models"]["Shot-adaptive C"]["support"]["clipping_fraction"] for state in states)
    curve_ok = all(state["curve"]["nonempty_bin_count"] >= MIN_NONEMPTY_SHOT_BINS and state["curve"]["post_pava_monotonicity_violations"] == 0 for state in states)
    finite_ok = all(state["models"][model]["prediction_finite"] for state in states for model in MODELS)
    pava_ok = all((state["curve"]["pava_adjustment_max_abs_log10"] or 0.0) <= MAX_PAVA_ADJUSTMENT_LOG10 for state in states)
    support_ok = max_clip <= MAX_TEST_SUPPORT_CLIP_FRACTION
    better_fraction = global_summary["global"]["adaptive_better"] / max(global_summary["global"]["count"], 1)
    variability_not_identical = variability["combined"]["classification"] != "practically_identical"
    main_checks = {
        "all_shot_curves_have_minimum_support": curve_ok,
        "all_shot_curves_monotonic_after_PAVA": curve_ok,
        "PAVA_adjustment_within_predeclared_stability_limit": pava_ok,
        "all_test_predictions_finite": finite_ok,
        "test_support_clipping_within_predeclared_limit": support_ok,
        "global_test_MAE_improves_by_at_least_5_percent": global_improvement is not None and global_improvement >= MIN_GLOBAL_IMPROVEMENT_PERCENT,
        "both_material_global_results_not_worse": all(value is not None and value >= 0.0 for value in per_material_improvements.values()),
        "adaptive_better_in_at_least_60_percent_of_cases": better_fraction >= MIN_BETTER_CASE_FRACTION,
        "high_end_not_systematically_worse": high_end_improvement is not None and high_end_improvement >= 0.0,
        "shot_to_shot_variability_is_measurable": variability_not_identical,
        "high_end_has_both_materials": len(high_end_available_materials) == len(MATERIALS),
    }
    hard_not_feasible = (not curve_ok) or (not finite_ok) or (not support_ok) or (global_improvement is not None and global_improvement <= MAX_SYSTEMATIC_WORSENING_PERCENT) or (better_fraction < 1.0 - MIN_BETTER_CASE_FRACTION)
    if hard_not_feasible:
        status = "NOT FEASIBLE"
    elif all(main_checks.values()):
        status = "FEASIBLE"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "global_test_macro_improvement_percent": global_improvement,
        "per_material_global_improvement_percent": per_material_improvements,
        "high_end_test_macro_improvement_percent": high_end_improvement,
        "high_end_available_materials": high_end_available_materials,
        "main_checks": main_checks,
        "hard_not_feasible_rule": {
            "mapping_support_or_finite_failure": True,
            "global_macro_MAE_worse_by_at_least_5_percent": True,
            "adaptive_worse_in_at_least_40_percent_of_cases": True,
        },
        "global_per_frame_summary": global_summary,
        "per_material_per_frame_summary": material_summaries,
        "thresholds_predeclared": {
            "minimum_global_improvement_percent": MIN_GLOBAL_IMPROVEMENT_PERCENT,
            "maximum_systematic_worsening_percent": MAX_SYSTEMATIC_WORSENING_PERCENT,
            "minimum_better_case_fraction": MIN_BETTER_CASE_FRACTION,
            "max_test_support_clipping_fraction_per_case": MAX_TEST_SUPPORT_CLIP_FRACTION,
            "minimum_nonempty_shot_bins": MIN_NONEMPTY_SHOT_BINS,
            "max_PAVA_adjustment_log10": MAX_PAVA_ADJUSTMENT_LOG10,
        },
        "interpretation": "High-end cross-material gate is explicitly unresolved when a material has zero HDR TEST samples above 200 nits; this produces INCONCLUSIVE rather than an invented pass.",
    }


def build_report(data: dict[str, Any]) -> str:
    aggregate = data["results"]["aggregate"]
    decision_data = data["decision"]
    lines = [
        "# P2.33 — SHOT-ADAPTIVE TONE MAPPING FEASIBILITY",
        "",
        f"**Final decision:** `{decision_data['status']}`",
        "",
        "This is an offline feasibility study only. No shot-adaptive transform was integrated into production.",
        "",
        "## 1. Dataset, provenance, and warnings",
        "",
        "Only immutable P2.32 artifacts were read: `P232_dataset_manifest.json`, `P232_dataset_metrics.json`, and the per-case `luminance_overlap_nits/` arrays. The HDR target is the direct P2.32 HDR-master overlap luminance array; no output HDR, LUT, Model A prediction, or transformed output was used as target. Source films were not decoded and no synchronization or full-film scan was run.",
        "",
        "The dataset contains exactly 20 The Matrix and 20 BR2049 frame cases. `verified_scene_count` is unknown and scene independence is not verified; the records are temporal strata, not proven independent shots.",
        "",
        "- The Matrix: fixed offset `-19`, confidence `0.940513`, status `LOCKED`; geometry confidence `1.0`.",
        "- BR2049: fixed offset `+1167`, confidence `0.9745`, status `LOCKED`; geometry confidence `0.9155`, retained as a warning below `0.95`.",
        "",
        "BR2049 spatial residuals may therefore include alignment error. This study does not promote the geometry confidence or claim verified scene diversity.",
        "",
        "## 2. Correspondence and leakage-safe spatial split",
        "",
        "Each case uses the existing P2.32 overlap coordinate system and pairs every SDR luminance element with the corresponding HDR-master luminance element at the same overlap coordinate. Histogram-only fitting is not used.",
        "",
        f"A deterministic whole-tile split is created before fitting: a `{TILE_ROWS}x{TILE_COLS}` grid with tile boundaries from the actual array shape; tile IDs whose `tile_id % {TEST_TILE_MODULUS}` is in `{list(TEST_TILE_REMAINDERS)}` are TEST, the remaining tiles are TRAIN. This is exactly 70 TRAIN tiles and 30 TEST tiles per case. No individual-pixel random split is used, and SDR/HDR masks are identical.",
        "",
        "Shot-adaptive C is fitted only from the current case's TRAIN tiles. Global Model B is fitted only from deterministic, equal per-case samples from TRAIN tiles across all 40 cases. TEST targets are read only after both mappings are fixed.",
        "",
        "## 3. Robust filtering",
        "",
        f"Filtering criteria were fixed before scoring: reject nonfinite pairs, negative luminance, either signal at or above `{HARD_CLIP_NITS:.1f}` nits, and SDR below `{MIN_STABLE_SDR_NITS}` nits for log-domain fit stability. Test metrics retain all finite nonnegative TEST samples, including low/clipped values, so filtering cannot hide test regressions. No residual-based outlier deletion or motion/misalignment deletion was performed; per-bin medians and P10/P90 limit outlier influence, and the absence of motion metadata is reported explicitly.",
        "",
        "Raw, accepted, rejected, rejection percentage, and disjoint rejection reasons are stored for every frame in the JSON and curve artifacts. `extreme_outlier`, `unreliable_correspondence`, and `motion_or_misalignment` rejection counts are zero by policy, not silently omitted.",
        "",
        "## 4. Mappings",
        "",
        "- **Model A:** unchanged production/reference Model A from `P2272_reference_curve.json`, evaluated through the existing LUT application; no refit.",
        "- **Model B:** historical P2.30/P2.31 global comparator: 512 fixed-width log10 SDR-nits bins, median HDR target per populated bin, PAVA, linear interpolation, and clamping rather than extrapolation. Its current fit uses only TRAIN tiles and equal 25,000 samples per case where available.",
        "- **Shot-adaptive C:** one mapping per frame case. It records the required fixed SDR bins `<0.01`, `0.01–0.1`, `0.1–0.5`, `0.5–1`, `1–2`, `2–5`, `5–10`, `10–20`, `20–50`, `50–100`, `100–200`, `200–500`, `500–1000`, `>1000`; each bin stores count, HDR P10, P50, P90, and robust spread. Empty bins remain count zero and are not filled in the diagnostic table. Prediction uses only nonempty nodes, PAVA, log-domain linear interpolation, and nearest-node clamping without extrapolation.",
        "",
        "Shot curve characteristics for black/shadow response, midtone response, highlight roll-off, maximum observed HDR, dynamic-range compression, local slopes, plateaus, and shoulders are stored per case.",
        "",
        "## 5. TEST-region comparison",
        "",
        "All metrics below are computed only on TEST tiles. Error metrics include count, signed mean, median residual, MAE, RMSE, P95, P99, maximum, and separate relative residual metrics. JSON contains exact per-material, combined weighted, and combined macro-average values; empty bins retain `count: 0` and null metrics.",
        "",
        "| Model | Matrix global MAE | BR2049 global MAE | Combined macro MAE | Combined weighted MAE | Combined macro >200 MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        matrix = aggregate["materials"]["The Matrix"][model]["global"]["MAE"]
        br = aggregate["materials"]["BR2049"][model]["global"]["MAE"]
        macro = aggregate["combined"]["macro_average"][model]["global"]["MAE"]
        weighted = aggregate["combined"]["weighted"][model]["global"]["MAE"]
        high = aggregate["combined"]["macro_average"][model]["highlight_gt_200"]["MAE"]
        lines.append(f"| {model} | {fmt(matrix)} | {fmt(br)} | {fmt(macro)} | {fmt(weighted)} | {fmt(high)} |")
    lines += [
        "",
        "The same per-material and combined tables for shadow `<10`, midtone `10–200`, highlight `>200`, `200–500`, `500–1000`, and `>1000` are available in `P233_shot_adaptive_metrics.json` and the per-frame CSV.",
        "",
        "## 6. Per-frame repeatability and cross-material results",
        "",
        f"Shot-adaptive C is compared against Model A per case. The combined global improvement is `{fmt(decision_data['global_test_macro_improvement_percent'], 2)}%` on the macro-average test MAE. The Matrix improvement is `{fmt(decision_data['per_material_global_improvement_percent']['The Matrix'], 2)}%`; BR2049 improvement is `{fmt(decision_data['per_material_global_improvement_percent']['BR2049'], 2)}%`.",
        "",
        "The JSON reports mean, median, P25, P75, minimum, maximum improvement percentage and adaptive-better/equal/worse counts for global, shadow, midtone, highlight, and each high-end band, separately for The Matrix, BR2049, and combined cases. No conclusion is based on one frame.",
        "",
        "## 7. Shot-to-shot variability",
        "",
        "At SDR 10, 50, 100, 200, and 500 nits, the JSON reports curve min/max spread, median absolute difference, P10/P90 spread, and P95-minus-P5 spread for each material and combined. Classification thresholds were fixed before analysis: practically identical if the worst relative P10/P90 spread is at most 10%; moderately different through 30%; strongly different above 30%.",
        "",
        f"Combined classification: `{data['variability']['combined']['classification']}`. The curves therefore are not treated as globally identical merely because they share the same diagnostic bin definitions.",
        "",
        "## 8. Temporal stability and chroma",
        "",
        "Temporal stability is `INCONCLUSIVE`: the P2.32 cases are widely spaced temporal strata and do not provide verified adjacent same-shot samples. This limitation is not converted into a scene-diversity claim.",
        "",
        "No full shot-specific chroma model was fitted. The present result is luminance-only; chroma adaptation remains `FUTURE WORK` if RGB residual analysis later justifies it. No chroma transform was integrated.",
        "",
        "## 9. Decision gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, value in decision_data["main_checks"].items():
        lines.append(f"| {name} | {value} |")
    lines += [
        "",
        "The decision uses predeclared limits stored in the JSON: at least 5% macro global improvement, no material global worsening, at least 60% cases better, no systematic high-end worsening, at least six nonempty shot bins, PAVA adjustment no larger than 0.50 log10 nits, and at most 25% TEST support clipping per case. A high-end gate is unresolved when a material has no TEST target samples above 200 nits; that is reported as insufficient coverage rather than a pass.",
        "",
        "## 10. Artifacts and scope",
        "",
        "The output directory preserves per-case shot curves, train/test masks with tile assignments, residual arrays for Model A/Model B/Shot-adaptive C, curve plots for all Matrix and BR2049 cases, median/spread plots, residual plots, high-end comparison, JSON metrics, and CSV per-frame metrics.",
        "",
        "Production code, `transform.py`, `luminance.py`, `compose.py`, `render.py`, CUDA, FFmpeg, Model A, Model B, production LUTs, and defaults were not changed. No production render, final HDR film, source decode, synchronization, dataset mutation, new pair, commit, or push was performed.",
        "",
        "## Final decision",
        "",
        f"**{decision_data['status']}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    dataset, manifest, cases = load_dataset()
    for directory in (OUTPUT_DIR, CURVE_DIR, MASK_DIR, RESIDUAL_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    model_a = load_model_a()
    states: list[dict[str, Any]] = []
    global_samples_x: list[np.ndarray] = []
    global_samples_y: list[np.ndarray] = []
    grouped_curves: dict[str, list[dict[str, Any]]] = {material: [] for material in MATERIALS}

    # Pass 1: create masks and fit each shot curve from TRAIN only. No TEST
    # target is read for fitting, scoring, or curve selection in this pass.
    for case in cases:
        sdr = np.asarray(np.load(case["sdr_luminance_nits_path"], mmap_mode="r"))
        hdr = np.asarray(np.load(case["hdr_luminance_nits_path"], mmap_mode="r"))
        if sdr.ndim != 2 or hdr.shape != sdr.shape:
            raise RuntimeError(f"P2.32 luminance shape mismatch for {case['case_id']}: {sdr.shape} / {hdr.shape}")
        train_mask, test_mask, tile_ids, split_metadata = make_spatial_masks(tuple(int(value) for value in sdr.shape))
        mask_path = MASK_DIR / f"{case['case_id']}_train_test_masks.npz"
        np.savez_compressed(mask_path, train_mask=train_mask, test_mask=test_mask, tile_ids=tile_ids)
        train_flat = train_mask.reshape(-1)
        x_train = np.asarray(sdr.reshape(-1)[train_flat], dtype=np.float64)
        y_train = np.asarray(hdr.reshape(-1)[train_flat], dtype=np.float64)
        accepted_mask, filtering = filtering_masks(x_train, y_train)
        accepted_x = x_train[accepted_mask]
        accepted_y = y_train[accepted_mask]
        shot_model, diagnostic_rows, bin_diagnostics = fit_shot_model(accepted_x, accepted_y)
        characteristics = shot_characteristics(shot_model, accepted_x, accepted_y, np.asarray(hdr).reshape(-1))
        filtering["train_region_raw_sample_count"] = int(x_train.size)
        filtering["train_region_accepted_sample_count"] = int(accepted_x.size)
        filtering["train_region_rejected_sample_count"] = int(x_train.size - accepted_x.size)
        filtering["train_region_rejection_percentage"] = float(100.0 * (x_train.size - accepted_x.size) / max(x_train.size, 1))
        filtering["test_region_raw_sample_count"] = int(np.sum(test_mask))
        filtering["test_region_evaluable_sample_count"] = int(np.sum(test_mask.reshape(-1) & np.isfinite(sdr.reshape(-1)) & np.isfinite(hdr.reshape(-1)) & (sdr.reshape(-1) >= 0.0) & (hdr.reshape(-1) >= 0.0)))
        filtering["test_region_invalid_sample_count"] = int(filtering["test_region_raw_sample_count"] - filtering["test_region_evaluable_sample_count"])
        filtering["raw_full_case_sample_count"] = int(sdr.size)
        filtering["raw_full_case_finite"] = bool(np.isfinite(sdr).all() and np.isfinite(hdr).all())
        curve_path = CURVE_DIR / f"{case['case_id']}_shot_curve.json"
        curve_payload = curve_artifact(shot_model, diagnostic_rows, bin_diagnostics, characteristics, case, filtering, split_metadata)
        write_json(curve_path, curve_payload)
        # Balanced deterministic Model B sampling is fixed before any TEST score.
        if accepted_x.size:
            sample_count = min(GLOBAL_MODEL_B_SAMPLES_PER_CASE, accepted_x.size)
            indices = np.linspace(0, accepted_x.size - 1, sample_count, dtype=np.int64)
            global_samples_x.append(accepted_x[indices].astype(np.float64, copy=False))
            global_samples_y.append(accepted_y[indices].astype(np.float64, copy=False))
        state = {
            "material": case["material"],
            "case_id": case["case_id"],
            "case": case,
            "shape": [int(value) for value in sdr.shape],
            "split": split_metadata,
            "mask_path": str(mask_path),
            "curve_path": str(curve_path),
            "curve": shot_model,
            "diagnostic_bins": diagnostic_rows,
            "bin_diagnostics": bin_diagnostics,
            "characteristics": characteristics,
            "filtering": filtering,
            "models": {},
        }
        states.append(state)
        grouped_curves[case["material"]].append({"case_id": case["case_id"], "model": shot_model})
        del sdr, hdr, x_train, y_train, accepted_x, accepted_y

    if len(global_samples_x) != 40:
        raise RuntimeError(f"Expected balanced Model B samples from 40 cases, got {len(global_samples_x)}")
    model_b = fit_global_model_b(np.concatenate(global_samples_x), np.concatenate(global_samples_y))
    write_json(CURVE_DIR / "global_model_b_curve.json", model_to_json(model_b))
    del global_samples_x, global_samples_y

    # Pass 2: apply fixed mappings to TEST tiles only and persist numeric errors.
    residual_samples_scatter: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"Model A": [], "Shot-adaptive C": []}
    residual_samples_hist: dict[str, list[np.ndarray]] = {"Model A": [], "Shot-adaptive C": []}
    for state in states:
        case = state["case"]
        sdr = np.asarray(np.load(case["sdr_luminance_nits_path"], mmap_mode="r"))
        hdr = np.asarray(np.load(case["hdr_luminance_nits_path"], mmap_mode="r"))
        mask_artifact = np.load(state["mask_path"], allow_pickle=False)
        test_mask = np.asarray(mask_artifact["test_mask"], dtype=bool).reshape(-1)
        mask_artifact.close()
        x_all = sdr.reshape(-1)
        y_all = hdr.reshape(-1)
        valid_test = test_mask & np.isfinite(x_all) & np.isfinite(y_all) & (x_all >= 0.0) & (y_all >= 0.0)
        test_indices = np.flatnonzero(valid_test).astype(np.int32)
        test_x = np.asarray(x_all[valid_test], dtype=np.float64)
        test_y = np.asarray(y_all[valid_test], dtype=np.float64)
        predictions: dict[str, np.ndarray] = {}
        supports: dict[str, dict[str, Any]] = {}
        for model_name, model in (("Model A", model_a), ("Model B", model_b), ("Shot-adaptive C", state["curve"])):
            prediction, support = predict_model(model, test_x)
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Non-finite {model_name} prediction for {state['case_id']}")
            predictions[model_name] = prediction
            supports[model_name] = support
        residuals = {model: predictions[model] - test_y for model in MODELS}
        residual_path = RESIDUAL_DIR / f"{state['case_id']}_test_residuals.npz"
        residual_keys = {model: f"residual_{model.lower().replace(' ', '_').replace('-', '_')}" for model in MODELS}
        np.savez_compressed(residual_path, test_flat_indices=test_indices, **{residual_keys[model]: np.asarray(residuals[model], dtype=np.float32) for model in MODELS})
        for model in MODELS:
            state["models"][model] = {
                "metrics": metrics_for_test(residuals[model], test_y),
                "support": supports[model],
                "prediction_finite": bool(np.isfinite(predictions[model]).all()),
                "prediction_negative_count": int(np.sum(predictions[model] < 0.0)),
            }
        state["residual_path"] = str(residual_path)
        state["residual_keys"] = residual_keys
        state["test_evaluable_count"] = int(test_x.size)
        state["test_invalid_count"] = int(np.sum(test_mask) - test_x.size)
        state["comparisons"] = {}
        for _, _, label in TEST_REGIONS:
            a_metrics = state["models"]["Model A"]["metrics"][label]
            c_metrics = state["models"]["Shot-adaptive C"]["metrics"][label]
            state["comparisons"][label] = {
                "model_a_MAE": a_metrics["MAE"],
                "shot_adaptive_MAE": c_metrics["MAE"],
                "improvement_percent": improvement_percent(a_metrics, c_metrics),
            }
        sample_count = min(VISUAL_RESIDUAL_SAMPLES_PER_CASE, test_x.size)
        if sample_count:
            sample_indices = np.linspace(0, test_x.size - 1, sample_count, dtype=np.int64)
            for model in ("Model A", "Shot-adaptive C"):
                residual_samples_scatter[model].append((test_x[sample_indices], residuals[model][sample_indices]))
                residual_samples_hist[model].append(residuals[model][sample_indices].astype(np.float64, copy=False))
        del sdr, hdr, x_all, y_all, test_x, test_y, predictions, residuals

    aggregate = aggregate_exact(states)
    variability = {material: variability_for_group(grouped_curves[material]) for material in MATERIALS}
    variability["combined"] = variability_for_group(grouped_curves["The Matrix"] + grouped_curves["BR2049"])
    temporal = {
        "status": "INCONCLUSIVE",
        "reason": "P2.32 provides widely spaced temporal strata with scene_independence_verified=false; no verified adjacent same-shot pairs exist.",
        "tested": False,
    }
    decision_data = decision(aggregate, states, variability, temporal)

    plot_paths = {
        "matrix_all_shot_curves": PLOTS_DIR / "P233_all_matrix_shot_curves.png",
        "br2049_all_shot_curves": PLOTS_DIR / "P233_all_br2049_shot_curves.png",
        "median_curve_and_spread": PLOTS_DIR / "P233_median_curve_P10_P90_spread.png",
        "model_a_vs_adaptive_residuals": PLOTS_DIR / "P233_model_a_vs_shot_adaptive_test_residuals.png",
        "test_residual_distributions": PLOTS_DIR / "P233_test_residual_distributions.png",
        "high_end_comparison": PLOTS_DIR / "P233_high_end_test_comparison.png",
    }
    plot_log_curve_set(plot_paths["matrix_all_shot_curves"], "The Matrix — all shot-adaptive C curves", grouped_curves["The Matrix"], (210, 80, 40))
    plot_log_curve_set(plot_paths["br2049_all_shot_curves"], "BR2049 — all shot-adaptive C curves", grouped_curves["BR2049"], (40, 130, 70))
    plot_material_median_spread(plot_paths["median_curve_and_spread"], grouped_curves)
    plot_residual_scatter(plot_paths["model_a_vs_adaptive_residuals"], residual_samples_scatter)
    plot_residual_histogram(plot_paths["test_residual_distributions"], {model: np.concatenate(values) if values else np.empty(0) for model, values in residual_samples_hist.items()})
    plot_high_end(plot_paths["high_end_comparison"], aggregate)

    csv_fields = ["material", "case_id", "hdr_frame", "om_frame", "model", "region", "count", "signed_mean", "median_residual", "MAE", "RMSE", "P95", "P99", "max", "relative_MAE", "model_a_MAE", "shot_adaptive_MAE", "improvement_percent"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for state in states:
            for model in MODELS:
                for _, _, label in TEST_REGIONS:
                    metrics = state["models"][model]["metrics"][label]
                    comparison = state["comparisons"][label]
                    writer.writerow({
                        "material": state["material"], "case_id": state["case_id"], "hdr_frame": state["case"]["hdr_frame"], "om_frame": state["case"]["om_frame"], "model": model, "region": label,
                        "count": metrics["count"], "signed_mean": metrics["signed_mean"], "median_residual": metrics["median_residual"], "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "P95": metrics["P95"], "P99": metrics["P99"], "max": metrics["max"], "relative_MAE": metrics["relative"]["MAE"],
                        "model_a_MAE": comparison["model_a_MAE"], "shot_adaptive_MAE": comparison["shot_adaptive_MAE"], "improvement_percent": comparison["improvement_percent"],
                    })

    result_data = {
        "phase": "P2.33",
        "status": decision_data["status"],
        "study": "shot-adaptive tone mapping feasibility using existing HDR overlap",
        "scope": "offline only; immutable P2.32 inputs; no production integration",
        "dataset_reference": {
            "manifest": str(P232_MANIFEST_PATH),
            "metrics": str(P232_METRICS_PATH),
            "luminance_directory": str(P232_LUMA_DIR),
            "raw_rgb_directory": str(P232_RAW_DIR),
            "material_case_counts": {material: 20 for material in MATERIALS},
            "verified_scene_count": {material: None for material in MATERIALS},
            "scene_independence_verified": {material: False for material in MATERIALS},
            "sync_and_geometry_warnings": {
                "The Matrix": {"offset_frames": -19, "sync_confidence": 0.940513, "sync_status": "LOCKED", "geometry_confidence": 1.0},
                "BR2049": {"offset_frames": 1167, "sync_confidence": 0.9745, "sync_status": "LOCKED", "geometry_confidence": 0.9155, "geometry_warning": True},
            },
            "target_provenance": "direct HDR-master luminance_overlap_nits arrays from P2.32; no output-as-target",
        },
        "split": {
            "method": "whole spatial tiles per frame case",
            "grid": [TILE_ROWS, TILE_COLS],
            "test_tile_rule": f"tile_id % {TEST_TILE_MODULUS} in {list(TEST_TILE_REMAINDERS)}",
            "train_tile_count": TILE_ROWS * TILE_COLS - len(TEST_TILE_REMAINDERS) * TILE_ROWS,
            "test_tile_count": len(TEST_TILE_REMAINDERS) * TILE_ROWS,
            "train_fraction_by_tiles": 0.70,
            "test_fraction_by_tiles": 0.30,
            "created_before_fitting": True,
            "pixel_level_random_split": False,
            "case_level_split": False,
            "note": "This is a within-frame spatial generalization test, not independent-shot validation; P2.32 scene identities remain unverified.",
            "masks_directory": str(MASK_DIR),
        },
        "filtering_policy": {
            "finite": True,
            "nonnegative": True,
            "hard_clip_threshold_nits": HARD_CLIP_NITS,
            "minimum_stable_sdr_nits_for_fit": MIN_STABLE_SDR_NITS,
            "outlier_deletion": False,
            "outlier_influence_control": "bin median/P10/P90 and PAVA",
            "motion_or_misalignment_filter": False,
            "test_metrics_include_finite_nonnegative_low_and_clipped_samples": True,
        },
        "models": {
            "Model A": model_a["definition"],
            "Model B": model_to_json(model_b),
            "Shot-adaptive C": {
                "type": "per-frame fixed diagnostic-bin empirical mapping",
                "diagnostic_bins": [{"low_nits": low, "high_nits": None if not math.isfinite(high) else high, "label": label} for low, high, label in SDR_BINS],
                "fit_region": "current case TRAIN tiles only",
                "target_statistic": "HDR P50 median",
                "spread": ["P10", "P50", "P90"],
                "missing_bin_policy": "count zero; not interpolated in diagnostic analysis",
                "prediction_policy": "PAVA, log-domain linear interpolation between populated nodes, nearest-node clamping, no extrapolation",
                "curve_directory": str(CURVE_DIR),
            },
        },
        "results": {
            "aggregate": aggregate,
            "per_case": [{key: value for key, value in state.items() if key not in {"case", "curve", "diagnostic_bins", "bin_diagnostics", "characteristics"}} | {"case_metadata": {"case_id": state["case_id"], "material": state["material"], "hdr_frame": state["case"]["hdr_frame"], "om_frame": state["case"]["om_frame"], "selection_fraction": state["case"]["selection_fraction"], "geometry": state["case"]["geometry"], "offset_frames": state["case"]["offset_frames"], "sync_confidence": state["case"]["sync_confidence"]}, "shot_curve": model_to_json(state["curve"]), "diagnostic_bins": state["diagnostic_bins"], "bin_diagnostics": state["bin_diagnostics"], "characteristics": state["characteristics"]} for state in states],
            "per_frame_improvement": {
                "combined": decision_data["global_per_frame_summary"],
                "The Matrix": decision_data["per_material_per_frame_summary"]["The Matrix"],
                "BR2049": decision_data["per_material_per_frame_summary"]["BR2049"],
            },
        },
        "support_and_constraints": {
            "model_a": aggregate_support(states, "Model A"),
            "model_b": aggregate_support(states, "Model B"),
            "shot_adaptive": aggregate_support(states, "Shot-adaptive C"),
            "all_predictions_finite": all(state["models"][model]["prediction_finite"] for state in states for model in MODELS),
            "all_predictions_nonnegative": all(state["models"][model]["prediction_negative_count"] == 0 for state in states for model in MODELS),
            "all_shot_curves_post_pava_monotonic": all(state["curve"]["post_pava_monotonicity_violations"] == 0 for state in states),
            "all_shot_curves_raw_monotonicity_violations": int(sum(state["curve"]["raw_monotonicity_violations"] for state in states)),
            "all_extrapolation_counts": 0,
            "duplicate_x_values": 0,
        },
        "variability": variability,
        "temporal_stability": temporal,
        "chroma": {"status": "NOT FITTED", "interpretation": "Luminance-only feasibility; CHROMA ADAPTATION = FUTURE WORK."},
        "decision": decision_data,
        "artifacts": {
            "metrics_json": str(METRICS_PATH),
            "per_frame_csv": str(CSV_PATH),
            "report": str(REPORT_PATH),
            "output_directory": str(OUTPUT_DIR),
            "shot_curves_directory": str(CURVE_DIR),
            "train_test_masks_directory": str(MASK_DIR),
            "residual_arrays_directory": str(RESIDUAL_DIR),
            "plots": {name: str(path) for name, path in plot_paths.items()},
            "global_model_b_curve": str(CURVE_DIR / "global_model_b_curve.json"),
        },
        "scope_constraints": {
            "dataset_modified": False,
            "dataset_pairs_added": False,
            "source_films_decoded": False,
            "synchronization_performed": False,
            "global_find_global_offset_performed": False,
            "full_film_scan_performed": False,
            "production_code_changed": False,
            "transform_py_changed": False,
            "luminance_py_changed": False,
            "compose_py_changed": False,
            "render_py_changed": False,
            "cuda_changed": False,
            "ffmpeg_changed": False,
            "model_a_changed": False,
            "model_b_changed": False,
            "production_lut_changed": False,
            "production_defaults_changed": False,
            "shot_adaptive_integrated": False,
            "production_render_performed": False,
            "final_hdr_film_generated": False,
            "commit_created": False,
            "push_performed": False,
        },
    }
    write_json(METRICS_PATH, result_data)
    REPORT_PATH.write_text(build_report(result_data), encoding="utf-8")
    print(f"P2.33_SHOT_ADAPTIVE_STATUS={decision_data['status']}")
    print(f"METRICS={METRICS_PATH}")
    print(f"CSV={CSV_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
