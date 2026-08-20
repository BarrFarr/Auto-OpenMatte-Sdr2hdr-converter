from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import zoom

from auto_openmatte.core.transfer_functions import linearize


ROOT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter")
VALIDATION_DIR = ROOT / "dev" / "ffmpeg-build" / "validation"
P232_DIR = VALIDATION_DIR / "P232_real_material_dataset"
P232_METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
P232_MANIFEST_PATH = P232_DIR / "P232_dataset_manifest.json"
MODEL_A_CURVE_PATH = VALIDATION_DIR / "P2272_reference_curve.json"
OUTPUT_DIR = VALIDATION_DIR / "P234_shot_level"
BOUNDARY_DIR = OUTPUT_DIR / "shot_boundaries"
MANIFEST_DIR = OUTPUT_DIR / "shot_manifests"
SAMPLE_DIR = OUTPUT_DIR / "temporal_samples"
CURVE_DIR = OUTPUT_DIR / "shot_curves"
RESIDUAL_DIR = OUTPUT_DIR / "residuals"
PLOT_DIR = OUTPUT_DIR / "plots"
METRICS_PATH = OUTPUT_DIR / "P234_shot_level_metrics.json"
CSV_PATH = OUTPUT_DIR / "P234_shot_results.csv"
REPORT_PATH = ROOT / "P2.34_TRUE_SHOT_LEVEL_ANALYSIS_REPORT.md"
FFMPEG = (ROOT / "dev" / "ffmpeg-build" / "install" / "bin" / "ffmpeg.exe").resolve()

FPS_NUM = 24000
FPS_DEN = 1001
FPS = FPS_NUM / FPS_DEN
PEAK_NITS = 10000.0
RGB16_MAX = 65535.0
LOG_EPS = 1e-6
LOCAL_HALF_WINDOW_SECONDS = 5.0
MAX_HALF_WINDOW_SECONDS = 5.0
PROXY_WIDTH = 320
BOUNDARY_THRESHOLD = 0.30
GRADUAL_THRESHOLD = 0.15
GRADUAL_MIN_RUN_FRAMES = 5
ANCHOR_NEAR_CUT_FRAMES = 12
MIN_SAMPLE_SEPARATION_FRAMES = 5
TEMPORAL_FRACTIONS = ((0.10, "10%"), (0.25, "25%"), (0.50, "50%"), (0.75, "75%"), (0.90, "90%"))
QUERY_NITS = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
FIT_PIXELS_PER_FRAME = 100000
RESIDUAL_PIXELS_PER_FRAME = 100000
HARD_CLIP_NITS = 0.99 * PEAK_NITS
MIN_STABLE_SDR_NITS = 0.01

# Fixed P2.33 diagnostic bins reused only as an empirical diagnostic convention.
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
TEST_REGIONS = [
    (0.0, float("inf"), "global"),
    (0.0, 10.0, "shadow_lt_10"),
    (10.0, 200.0, "midtone_10_200"),
    (200.0, float("inf"), "highlight_gt_200"),
    (200.0, 500.0, "200–500"),
    (500.0, 1000.0, "500–1000"),
    (1000.0, float("inf"), ">1000"),
]
MATERIALS = ("The Matrix", "BR2049")

# Predeclared decision thresholds. They are recorded in the output and are not
# changed after boundary detection or scoring.
MIN_USABLE_SHOTS = 6
MIN_USABLE_SHOTS_PER_MATERIAL = 2
MIN_PREDICTION_SPLITS = 12
MIN_GLOBAL_IMPROVEMENT_PERCENT = 5.0
MIN_BETTER_SPLIT_FRACTION = 0.60
MAX_RELATIVE_P95_CURVE_SPREAD = 0.30
MAX_MEDIAN_TEST_CLIPPING_FRACTION = 0.25
HIGH_END_WORSENING_TOLERANCE_PERCENT = -5.0

M_709_TO_2020 = np.asarray(
    [
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690972, 0.9195404, 0.0113624],
        [0.0163916, 0.0880132, 0.8955952],
    ],
    dtype=np.float64,
)
LUMA_2020 = np.asarray([0.2627, 0.6780, 0.0593], dtype=np.float64)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def frame_timestamp(frame: int) -> float:
    return float(frame / FPS)


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
        "relative": {"signed_mean": None, "median": None, "MAE": None, "P95": None, "P99": None, "max": None, "epsilon_nits": LOG_EPS},
    }


def metric_block(residual: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
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


def metrics_for_target(residual: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for low, high, label in TEST_REGIONS:
        mask = (target >= low) & (target < high if math.isfinite(high) else np.ones_like(target, dtype=bool))
        result[label] = metric_block(residual[mask], target[mask])
    return result


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
    result = np.empty(values.size, dtype=np.float64)
    cursor = 0
    for value, weight in blocks:
        count = int(weight)
        result[cursor:cursor + count] = value
        cursor += count
    return result


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in model.items() if key not in {"x_log", "y_log"}}
    result["x_log"] = [float(value) for value in np.asarray(model.get("x_log", []), dtype=np.float64)]
    result["y_log"] = [float(value) for value in np.asarray(model.get("y_log", []), dtype=np.float64)]
    return json_safe(result)


def require_under(path: Path, root: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{description} is outside P2.32: {resolved}") from exc
    return resolved


def load_cases() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    dataset = json.loads(P232_METRICS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(P232_MANIFEST_PATH.read_text(encoding="utf-8"))
    if dataset.get("phase") != "P2.32" or dataset.get("status") != "DATASET READY":
        raise RuntimeError("P2.32 metrics are not DATASET READY")
    cases: list[dict[str, Any]] = []
    for material in MATERIALS:
        records = dataset["case_records"][material]
        manifest_cases = manifest["materials"][material]["selected_cases"]
        if len(records) != 20 or len(manifest_cases) != 20:
            raise RuntimeError(f"Expected 20 P2.32 anchors for {material}")
        if {record["case_id"] for record in records} != {record["case_id"] for record in manifest_cases}:
            raise RuntimeError(f"Manifest/metrics anchor mismatch for {material}")
        for record in records:
            require_under(Path(record["sdr_luminance_nits_path"]), P232_DIR, "P2.32 SDR array")
            require_under(Path(record["hdr_luminance_nits_path"]), P232_DIR, "P2.32 HDR array")
            if not Path(record["sdr_luminance_nits_path"]).exists() or not Path(record["hdr_luminance_nits_path"]).exists():
                raise RuntimeError(f"Missing P2.32 luminance array for {record['case_id']}")
            cases.append(record)
    if len(cases) != 40:
        raise RuntimeError(f"Expected exactly 40 anchors, found {len(cases)}")
    return dataset, manifest, cases


def source_metadata(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "hdr_source": case["hdr_source"],
        "om_source": case["om_source"],
        "hdr_width": int(case["hdr_resolution"][0]),
        "hdr_height": int(case["hdr_resolution"][1]),
        "om_width": int(case["om_resolution"][0]),
        "om_height": int(case["om_resolution"][1]),
        "offset_frames": int(case["offset_frames"]),
        "offset_seconds": float(case["offset_seconds"]),
        "fps": case["fps"],
        "fps_float": float(case["fps_float"]),
        "sync_confidence": float(case["sync_confidence"]),
        "sync_status": case["sync_status"],
        "geometry": case["geometry"],
    }


def decode_gray_window(case: dict[str, Any], start_frame: int, end_frame: int) -> tuple[np.ndarray, dict[str, Any]]:
    if LOCAL_HALF_WINDOW_SECONDS > MAX_HALF_WINDOW_SECONDS:
        raise RuntimeError("Configured local window exceeds P2.34 hard limit")
    n_frames = max(0, int(end_frame - start_frame))
    source = Path(case["hdr_source"])
    width = PROXY_WIDTH
    height = max(2, int(round(case["hdr_resolution"][1] * width / case["hdr_resolution"][0] / 2.0) * 2))
    frame_bytes = width * height
    command = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{frame_timestamp(start_frame):.6f}", "-i", str(source),
        "-map", "0:v:0", "-vf", f"scale={width}:{height},format=gray",
        "-frames:v", str(n_frames), "-pix_fmt", "gray", "-vsync", "0", "-f", "rawvideo", "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    raw = completed.stdout
    decoded = min(n_frames, len(raw) // frame_bytes)
    metadata = {
        "command": command,
        "returncode": int(completed.returncode),
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
        "requested_start_frame": int(start_frame),
        "requested_end_frame": int(end_frame),
        "requested_frame_count": int(n_frames),
        "decoded_frame_count": int(decoded),
        "proxy_shape": [height, width],
        "seek_mode": "bounded fast input seek; local diagnostic only",
        "full_film_scan": False,
    }
    if completed.returncode != 0 or decoded < 2:
        raise RuntimeError(f"Bounded proxy decode failed for {case['case_id']}: {metadata}")
    frames = np.frombuffer(raw[:decoded * frame_bytes], dtype=np.uint8).reshape(decoded, height, width).astype(np.float32) / 255.0
    return frames, metadata


def edge_map(frame: np.ndarray) -> np.ndarray:
    return cv2.Canny(np.asarray(frame * 255.0, dtype=np.uint8), 32, 96).astype(np.float32) / 255.0


def boundary_score(frame_a: np.ndarray, frame_b: np.ndarray, edge_a: np.ndarray, edge_b: np.ndarray) -> tuple[float, float, float, float]:
    hist_a, _ = np.histogram(frame_a, bins=64, range=(0.0, 1.0))
    hist_b, _ = np.histogram(frame_b, bins=64, range=(0.0, 1.0))
    hist_a = hist_a.astype(np.float64) / max(float(hist_a.sum()), 1.0)
    hist_b = hist_b.astype(np.float64) / max(float(hist_b.sum()), 1.0)
    denom = hist_a + hist_b
    mask = denom > 0
    hist_score = float(min(1.0, np.sum((hist_a[mask] - hist_b[mask]) ** 2 / denom[mask]) / 2.0)) if np.any(mask) else 0.0
    lum_score = float(abs(np.mean(frame_a) - np.mean(frame_b)))
    a = edge_a.reshape(-1).astype(np.float64)
    b = edge_b.reshape(-1).astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denom_ncc = float(np.linalg.norm(a) * np.linalg.norm(b))
    ncc = float(np.dot(a, b) / denom_ncc) if denom_ncc > 1e-12 else 0.0
    edge_score = max(0.0, 1.0 - ncc)
    combined = 0.4 * hist_score + 0.2 * lum_score + 0.4 * edge_score
    return float(combined), hist_score, lum_score, edge_score


def detect_local_boundaries(case: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    anchor = int(case["hdr_frame"])
    half = int(round(LOCAL_HALF_WINDOW_SECONDS * FPS))
    total_frames = int(math.floor(float(case["hdr_source"]) != "" and case["hdr_timestamp_seconds"] * 0 + 1)) if False else None
    # P2.32 anchor timestamps are frame/FPS values; source duration is not in each record.
    # The selected anchor is far from the material ends, so the bounded interval is
    # clamped only against zero here; actual decode count is recorded explicitly.
    start_frame = max(0, anchor - half)
    end_frame = anchor + half + 1
    frames, decode_metadata = decode_gray_window(case, start_frame, end_frame)
    frame_ids = np.arange(start_frame, start_frame + frames.shape[0], dtype=np.int64)
    scores = []
    components = []
    for index in range(frames.shape[0] - 1):
        edge_a = edge_map(frames[index])
        edge_b = edge_map(frames[index + 1])
        combined, hist, lum, edge = boundary_score(frames[index], frames[index + 1], edge_a, edge_b)
        scores.append(combined)
        components.append({"frame_a": int(frame_ids[index]), "frame_b": int(frame_ids[index + 1]), "score": combined, "histogram": hist, "luminance": lum, "edge": edge, "threshold": BOUNDARY_THRESHOLD})
    scores_array = np.asarray(scores, dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []
    for index, score in enumerate(scores_array):
        if score < BOUNDARY_THRESHOLD:
            continue
        left = scores_array[index - 1] if index > 0 else 0.0
        right = scores_array[index + 1] if index + 1 < scores_array.size else 0.0
        if left < GRADUAL_THRESHOLD and right < GRADUAL_THRESHOLD:
            candidates.append({"frame": int(frame_ids[index + 1]), "timestamp_seconds": frame_timestamp(int(frame_ids[index + 1])), "boundary_score": float(score), "threshold": BOUNDARY_THRESHOLD, "type": "hard", "confidence": float(min(1.0, score / BOUNDARY_THRESHOLD))})
    index = 0
    while index < scores_array.size:
        if scores_array[index] < GRADUAL_THRESHOLD:
            index += 1
            continue
        run_start = index
        while index < scores_array.size and scores_array[index] >= GRADUAL_THRESHOLD:
            index += 1
        run_end = index
        if run_end - run_start < GRADUAL_MIN_RUN_FRAMES:
            continue
        segment = scores_array[run_start:run_end]
        peak_index = int(run_start + np.argmax(segment))
        candidate = {"frame": int(frame_ids[peak_index + 1]), "timestamp_seconds": frame_timestamp(int(frame_ids[peak_index + 1])), "boundary_score": float(np.max(segment)), "threshold": GRADUAL_THRESHOLD, "type": "gradual", "confidence": float(min(1.0, np.max(segment) / GRADUAL_THRESHOLD)), "run_start_frame": int(frame_ids[run_start]), "run_end_frame": int(frame_ids[run_end])}
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["frame"])
    boundaries: list[dict[str, Any]] = []
    for candidate in candidates:
        if boundaries and candidate["frame"] - boundaries[-1]["frame"] <= GRADUAL_MIN_RUN_FRAMES:
            if candidate["boundary_score"] > boundaries[-1]["boundary_score"]:
                boundaries[-1] = candidate
        else:
            boundaries.append(candidate)
    nearest_distance = min((abs(item["frame"] - anchor) for item in boundaries), default=None)
    anchor_near_cut = nearest_distance is not None and nearest_distance <= ANCHOR_NEAR_CUT_FRAMES
    left = [item for item in boundaries if item["frame"] <= anchor]
    right = [item for item in boundaries if item["frame"] > anchor]
    status = "LOCAL_SHOT_UNRESOLVED"
    shot_start = None
    shot_end = None
    if left and right:
        shot_start = int(left[-1]["frame"])
        shot_end = int(right[0]["frame"])
        if shot_end - shot_start >= 3:
            status = "LOCAL_SHOT_RESOLVED"
        else:
            status = "SHOT_TOO_SHORT"
    if anchor_near_cut:
        status = "ANCHOR_NEAR_CUT" if status == "LOCAL_SHOT_RESOLVED" else status
    result = {
        "case_id": case["case_id"],
        "material": case["material"],
        "anchor_hdr_frame": anchor,
        "anchor_om_frame": int(case["om_frame"]),
        "anchor_timestamp_seconds": frame_timestamp(anchor),
        "window": {"half_window_seconds": LOCAL_HALF_WINDOW_SECONDS, "start_frame": start_frame, "end_frame_exclusive": int(start_frame + frames.shape[0]), "requested_end_frame_exclusive": end_frame, "decoded_frame_count": int(frames.shape[0])},
        "boundaries": boundaries,
        "boundary_count": len(boundaries),
        "anchor_near_cut": bool(anchor_near_cut),
        "anchor_near_cut_guard_frames": ANCHOR_NEAR_CUT_FRAMES,
        "nearest_boundary_distance_frames": nearest_distance,
        "shot_status": status,
        "shot_start_frame": shot_start,
        "shot_end_frame_exclusive": shot_end,
        "shot_duration_frames": int(shot_end - shot_start) if shot_start is not None and shot_end is not None else None,
        "shot_duration_seconds": float((shot_end - shot_start) / FPS) if shot_start is not None and shot_end is not None else None,
        "anchor_position_in_shot": float((anchor - shot_start) / max(shot_end - shot_start, 1)) if shot_start is not None and shot_end is not None else None,
        "boundary_detection": {"algorithm": "deterministic local frame difference", "weights": {"histogram": 0.4, "mean_luminance": 0.2, "edge_ncc_change": 0.4}, "hard_threshold": BOUNDARY_THRESHOLD, "gradual_threshold": GRADUAL_THRESHOLD, "gradual_min_run_frames": GRADUAL_MIN_RUN_FRAMES, "full_film_scan": False},
        "decode": decode_metadata,
    }
    return result, np.asarray(scores_array, dtype=np.float32)


def select_temporal_samples(shot: dict[str, Any]) -> list[dict[str, Any]]:
    start = shot.get("shot_start_frame")
    end = shot.get("shot_end_frame_exclusive")
    if start is None or end is None or end <= start:
        return []
    duration = int(end - start)
    selected: list[dict[str, Any]] = []
    for fraction, label in TEMPORAL_FRACTIONS:
        desired = int(round(start + fraction * max(duration - 1, 0)))
        candidates = sorted(range(start, end), key=lambda frame: (abs(frame - desired), frame))
        choice = next((frame for frame in candidates if all(abs(frame - old["hdr_frame"]) >= MIN_SAMPLE_SEPARATION_FRAMES for old in selected)), None)
        if choice is not None:
            selected.append({"label": label, "fraction": fraction, "hdr_frame": int(choice), "om_frame": int(choice + shot["case_offset_frames"]), "hdr_timestamp_seconds": frame_timestamp(int(choice)), "om_timestamp_seconds": frame_timestamp(int(choice + shot["case_offset_frames"]))})
    selected.sort(key=lambda item: item["hdr_frame"])
    if len(selected) < 3 and duration >= 2 * MIN_SAMPLE_SEPARATION_FRAMES + 1:
        fallback = [int(round(start + fraction * max(duration - 1, 0))) for fraction in (0.10, 0.50, 0.90)]
        selected = []
        for frame, fraction in zip(fallback, (0.10, 0.50, 0.90)):
            if all(abs(frame - old["hdr_frame"]) >= MIN_SAMPLE_SEPARATION_FRAMES for old in selected):
                selected.append({"label": f"fallback_{int(fraction * 100)}%", "fraction": fraction, "hdr_frame": frame, "om_frame": int(frame + shot["case_offset_frames"]), "hdr_timestamp_seconds": frame_timestamp(frame), "om_timestamp_seconds": frame_timestamp(int(frame + shot["case_offset_frames"]))})
    return selected


def run_ffmpeg_frame(source: Path, width: int, height: int, timestamp: float) -> tuple[np.ndarray, dict[str, Any]]:
    expected_bytes = width * height * 3 * 2
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp:.9f}", "-i", str(source), "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb48le", "-vsync", "0", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=180, check=False)
    raw = completed.stdout
    metadata = {"command": command, "returncode": int(completed.returncode), "stderr": completed.stderr.decode("utf-8", errors="replace"), "stdout_bytes": int(len(raw)), "expected_bytes": int(expected_bytes), "stream": "0:v:0", "pixel_format": "rgb48le"}
    if completed.returncode != 0 or len(raw) < expected_bytes:
        raise RuntimeError(f"Frame extraction failed: {metadata}")
    values = np.frombuffer(raw[:expected_bytes], dtype="<u2").reshape((height, width, 3)).copy()
    return values, metadata


def source_luminance(rgb_u16: np.ndarray, transfer: str) -> np.ndarray:
    signal = np.asarray(rgb_u16, dtype=np.float64) / RGB16_MAX
    linear = np.stack([np.asarray(linearize(signal[..., channel], transfer, peak_nits=PEAK_NITS), dtype=np.float64) for channel in range(3)], axis=-1)
    if transfer == "bt709":
        linear = linear.reshape(-1, 3) @ M_709_TO_2020.T
        linear = np.maximum(linear.reshape(signal.shape), 0.0)
    return np.asarray(np.sum(linear * LUMA_2020, axis=-1) * PEAK_NITS, dtype=np.float32)


def make_overlap(hdr_u16: np.ndarray, om_u16: np.ndarray, case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [int(value) for value in case["geometry"]["overlap"]]
    om_overlap = om_u16[y1:y2, x1:x2]
    scale_x = float(case["geometry"]["scale_x"])
    scale_y = float(case["geometry"]["scale_y"])
    if scale_x != 1.0 or scale_y != 1.0:
        hdr_overlap = zoom(hdr_u16.astype(np.float32) / RGB16_MAX, (scale_y, scale_x, 1.0), order=1)
        hdr_overlap = np.clip(np.round(hdr_overlap * RGB16_MAX), 0.0, RGB16_MAX).astype(np.uint16)
    else:
        hdr_overlap = hdr_u16
    if hdr_overlap.shape[:2] != om_overlap.shape[:2]:
        raise RuntimeError(f"Temporal overlap shape mismatch for {case['case_id']}")
    return source_luminance(om_overlap, "bt709"), source_luminance(hdr_overlap, "smpte2084")


def filter_pairs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    nonnegative = finite & (x >= 0.0) & (y >= 0.0)
    clipped = nonnegative & ((x >= HARD_CLIP_NITS) | (y >= HARD_CLIP_NITS))
    low = nonnegative & ~clipped & (x < MIN_STABLE_SDR_NITS)
    accepted = nonnegative & ~clipped & ~low
    return accepted, {"raw_sample_count": int(x.size), "accepted_sample_count": int(np.sum(accepted)), "rejection_percentage": float(100.0 * np.sum(~accepted) / max(x.size, 1)), "rejection_reasons": {"nonfinite": int(np.sum(~finite)), "negative": int(np.sum(finite & ~nonnegative)), "hard_clipping": int(np.sum(clipped)), "low_sdr_below_0.01_nits": int(np.sum(low)), "extreme_outlier": 0, "motion_or_misalignment": 0}, "fit_filter_policy": "finite/nonnegative; reject >=9900 nits; reject SDR <0.01 nits; no residual-based deletion; per-bin median and PAVA limit influence"}


def deterministic_fit_sample(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = min(FIT_PIXELS_PER_FRAME, x.size)
    if count == x.size:
        return x, y
    indices = np.linspace(0, x.size - 1, count, dtype=np.int64)
    return x[indices], y[indices]


def fit_curve(x: np.ndarray, y: np.ndarray, name: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    accepted, filtering = filter_pairs(x, y)
    accepted_x = np.asarray(x).reshape(-1)[accepted].astype(np.float64, copy=False)
    accepted_y = np.asarray(y).reshape(-1)[accepted].astype(np.float64, copy=False)
    fit_x, fit_y = deterministic_fit_sample(accepted_x, accepted_y)
    bin_index = np.digitize(fit_x, np.asarray([item[1] for item in SDR_BINS[:-1]], dtype=np.float64), right=False)
    rows: list[dict[str, Any]] = []
    node_x: list[float] = []
    node_y: list[float] = []
    for index, (low, high, label) in enumerate(SDR_BINS):
        mask = bin_index == index
        count = int(np.sum(mask))
        row = {"label": label, "low_nits": low, "high_nits": None if not math.isfinite(high) else high, "count": count, "P10": None, "P50": None, "P90": None, "P90_minus_P10": None}
        if count:
            target = fit_y[mask]
            p10, p50, p90 = [float(value) for value in np.percentile(target, [10, 50, 90])]
            row.update({"P10": p10, "P50": p50, "P90": p90, "P90_minus_P10": p90 - p10})
            node_x.append(float(np.median(fit_x[mask])))
            node_y.append(p50)
        rows.append(row)
    if not node_x:
        model = {"name": name, "kind": "unavailable", "x_log": np.empty(0), "y_log": np.empty(0), "definition": {"type": "fixed diagnostic bins", "fit_sample_count": int(fit_x.size)}}
        diagnostics = {"node_count": 0, "raw_monotonicity_violations": 0, "post_pava_monotonicity_violations": 0, "pava_adjustment_max_abs_log10": None, "fit_sample_count": int(fit_x.size)}
        return model, rows, {"filtering": filtering, "diagnostics": diagnostics}
    x_log = np.log10(np.maximum(np.asarray(node_x), LOG_EPS))
    raw_y_log = np.log10(np.maximum(np.asarray(node_y), 0.0) + LOG_EPS)
    order = np.argsort(x_log)
    x_log = x_log[order]
    raw_y_log = raw_y_log[order]
    unique_x, unique_indices = np.unique(x_log, return_index=True)
    raw_y_log = raw_y_log[unique_indices]
    mono_y_log = pava(raw_y_log)
    model = {"name": name, "kind": "empirical_log_bin_median", "x_log": unique_x, "y_log": mono_y_log, "definition": {"type": "empirical SDR-to-HDR curve", "fixed_bins": [item[2] for item in SDR_BINS], "target_statistic": "HDR median/P50", "spread": ["P10", "P50", "P90"], "monotonic_projection": "PAVA", "interpolation": "linear in log10 domain", "outside_domain": "clamp; no extrapolation", "fit_sample_count": int(fit_x.size)}}
    diagnostics = {"node_count": int(unique_x.size), "raw_monotonicity_violations": int(np.sum(np.diff(raw_y_log) < 0.0)), "post_pava_monotonicity_violations": int(np.sum(np.diff(mono_y_log) < -1e-12)), "pava_adjustment_max_abs_log10": float(np.max(np.abs(mono_y_log - raw_y_log))), "fit_sample_count": int(fit_x.size), "domain_sdr_nits": [float(10.0 ** unique_x[0] - LOG_EPS), float(10.0 ** unique_x[-1] - LOG_EPS)]}
    return model, rows, {"filtering": filtering, "diagnostics": diagnostics}


def predict_curve(model: dict[str, Any], x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    x_log = np.asarray(model.get("x_log", []), dtype=np.float64)
    y_log = np.asarray(model.get("y_log", []), dtype=np.float64)
    if x_log.size < 2:
        return np.full(values.shape, np.nan), {"outside_domain_count": int(values.size), "clipping_fraction": 1.0, "extrapolation_count": 0, "domain_sdr_nits": None}
    input_log = np.log10(np.maximum(values, LOG_EPS))
    low, high = float(x_log[0]), float(x_log[-1])
    low_clip = input_log < low
    high_clip = input_log > high
    output = np.maximum(np.power(10.0, np.interp(np.clip(input_log, low, high), x_log, y_log)) - LOG_EPS, 0.0)
    return output, {"outside_domain_count": int(np.sum(low_clip | high_clip)), "low_clip_count": int(np.sum(low_clip)), "high_clip_count": int(np.sum(high_clip)), "clipping_fraction": float(np.mean(low_clip | high_clip)) if values.size else 0.0, "extrapolation_count": 0, "domain_sdr_nits": [float(10.0 ** low - LOG_EPS), float(10.0 ** high - LOG_EPS)]}


def load_model_a() -> dict[str, Any]:
    from auto_openmatte.processing.luminance import apply_luminance_curve, build_curve_lut
    curve = json.loads(MODEL_A_CURVE_PATH.read_text(encoding="utf-8"))
    return {"name": "Model A", "curve": curve, "lut": build_curve_lut(curve), "apply": apply_luminance_curve, "definition": {"type": "unchanged production/reference Model A", "curve_path": str(MODEL_A_CURVE_PATH), "control_points": len(curve), "lut_entries": 65536, "refit": False}}


def predict_model_a(model: dict[str, Any], x_nits: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    curve = np.asarray(model["curve"], dtype=np.float64)
    input_log = np.log10(np.maximum(np.asarray(x_nits, dtype=np.float64), LOG_EPS))
    low = float(curve[0, 0])
    high = float(curve[-1, 0])
    low_clip = input_log < low
    high_clip = input_log > high
    prediction = np.asarray(model["apply"](np.clip(np.asarray(x_nits, dtype=np.float64) / PEAK_NITS, 0.0, 1.0), model["curve"], prebuilt_lut=model["lut"]), dtype=np.float64) * PEAK_NITS
    return prediction, {"outside_domain_count": int(np.sum(low_clip | high_clip)), "low_clip_count": int(np.sum(low_clip)), "high_clip_count": int(np.sum(high_clip)), "clipping_fraction": float(np.mean(low_clip | high_clip)) if input_log.size else 0.0, "extrapolation_count": 0, "domain_sdr_nits": [float(10.0 ** low - LOG_EPS), float(10.0 ** high - LOG_EPS)]}


def sample_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "P90": None, "P95": None, "P99": None, "max": None, "variance": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "median": float(np.median(array)), "P90": float(np.percentile(array, 90)), "P95": float(np.percentile(array, 95)), "P99": float(np.percentile(array, 99)), "max": float(np.max(array)), "variance": float(np.var(array))}


def curve_value(model: dict[str, Any], query: float) -> float | None:
    output, _ = predict_curve(model, np.asarray([query], dtype=np.float64))
    return float(output[0]) if np.isfinite(output[0]) else None


def curve_variability(frame_models: list[dict[str, Any]], pooled_model: dict[str, Any]) -> dict[str, Any]:
    by_query: dict[str, Any] = {}
    pooled_differences = []
    for query in QUERY_NITS:
        values = [curve_value(model, query) for model in frame_models]
        values = [value for value in values if value is not None]
        pairwise = [abs(values[left] - values[right]) for left in range(len(values)) for right in range(left + 1, len(values))]
        pooled_value = curve_value(pooled_model, query)
        frame_to_pooled = [abs(value - pooled_value) for value in values] if pooled_value is not None else []
        pooled_differences.extend(frame_to_pooled)
        variance = sample_stats(values)
        p05 = float(np.percentile(values, 5)) if values else None
        p95 = float(np.percentile(values, 95)) if values else None
        p01 = float(np.percentile(values, 1)) if values else None
        p99 = float(np.percentile(values, 99)) if values else None
        variance.update({"query_nits": query, "P95_spread_P95_minus_P5": p95 - p05 if p95 is not None and p05 is not None else None, "P99_spread_P99_minus_P1": p99 - p01 if p99 is not None and p01 is not None else None, "relative_P95_spread": float((p95 - p05) / max(abs(np.median(values)), 0.01)) if values and p95 is not None and p05 is not None else None, "pairwise_absolute_difference": sample_stats(pairwise), "frame_to_pooled_absolute_difference": sample_stats(frame_to_pooled)})
        by_query[str(query)] = variance
    temporal = {}
    for query in QUERY_NITS:
        values = [curve_value(model, query) for model in frame_models]
        values = [value for value in values if value is not None]
        differences = np.diff(values) if len(values) >= 2 else np.asarray([])
        signs = np.sign(differences[np.abs(differences) > 1e-6])
        reversals = int(np.sum(signs[1:] != signs[:-1])) if signs.size >= 2 else 0
        temporal[str(query)] = {"sample_count": len(values), "differences": [float(value) for value in differences], "monotonic_non_decreasing": bool(differences.size > 0 and np.all(differences >= -1e-6)), "monotonic_non_increasing": bool(differences.size > 0 and np.all(differences <= 1e-6)), "direction_reversals": reversals}
    relative_spreads = [item["relative_P95_spread"] for item in by_query.values() if item["relative_P95_spread"] is not None]
    return {"queries": by_query, "frame_to_pooled_overall": sample_stats(pooled_differences), "temporal_monotonicity": temporal, "max_relative_P95_spread": max(relative_spreads) if relative_spreads else None, "median_relative_P95_spread": float(np.median(relative_spreads)) if relative_spreads else None}


def pooled_curve_from_samples(samples: list[dict[str, Any]], name: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    xs, ys = [], []
    for sample in samples:
        accepted, _ = filter_pairs(sample["sdr"], sample["hdr"])
        x = np.asarray(sample["sdr"]).reshape(-1)[accepted].astype(np.float64, copy=False)
        y = np.asarray(sample["hdr"]).reshape(-1)[accepted].astype(np.float64, copy=False)
        x, y = deterministic_fit_sample(x, y)
        xs.append(x)
        ys.append(y)
    if not xs:
        model, rows, info = fit_curve(np.empty(0), np.empty(0), name)
        return model, info, rows
    model, rows, info = fit_curve(np.concatenate(xs), np.concatenate(ys), name)
    info["pooled_frame_count"] = len(samples)
    info["pooled_fit_sample_count"] = int(sum(array.size for array in xs))
    return model, info, rows


def extract_sample(case: dict[str, Any], sample: dict[str, Any], shot_dir: Path) -> dict[str, Any]:
    hdr_rgb, hdr_decode = run_ffmpeg_frame(Path(case["hdr_source"]), int(case["hdr_resolution"][0]), int(case["hdr_resolution"][1]), sample["hdr_timestamp_seconds"])
    om_rgb, om_decode = run_ffmpeg_frame(Path(case["om_source"]), int(case["om_resolution"][0]), int(case["om_resolution"][1]), sample["om_timestamp_seconds"])
    sdr, hdr = make_overlap(hdr_rgb, om_rgb, case)
    if sdr.shape != hdr.shape or not np.isfinite(sdr).all() or not np.isfinite(hdr).all():
        raise RuntimeError(f"Invalid temporal sample for {case['case_id']} frame {sample['hdr_frame']}")
    sample_path = shot_dir / f"sample_{sample['label'].replace('%', 'pct')}_{sample['hdr_frame']}.npz"
    np.savez_compressed(sample_path, sdr_luminance_nits=sdr, hdr_luminance_nits=hdr)
    frame_model, frame_rows, frame_info = fit_curve(sdr, hdr, f"Frame-specific {sample['label']}")
    curve_path = CURVE_DIR / f"{case['case_id']}_frame_{sample['label'].replace('%', 'pct')}_{sample['hdr_frame']}.json"
    write_json(curve_path, {"phase": "P2.34", "case_id": case["case_id"], "material": case["material"], "sample": sample, "curve": model_to_json(frame_model), "diagnostic_bins": frame_rows, "fit_info": frame_info})
    sample.update({"sample_path": str(sample_path), "curve_path": str(curve_path), "array_shape": list(sdr.shape), "hdr_decode": hdr_decode, "om_decode": om_decode, "frame_curve": frame_model, "frame_curve_info": frame_info, "diagnostic_bins": frame_rows})
    sample["sdr"] = sdr
    sample["hdr"] = hdr
    return sample


def evaluate_prediction_split(case: dict[str, Any], shot: dict[str, Any], samples_by_label: dict[str, dict[str, Any]], train_labels: list[str], test_label: str, model_a: dict[str, Any]) -> dict[str, Any]:
    train_samples = [samples_by_label[label] for label in train_labels if label in samples_by_label]
    if test_label not in samples_by_label or len(train_samples) < 3:
        return {"status": "INSUFFICIENT_TEMPORAL_SAMPLES", "train_labels": train_labels, "test_label": test_label, "available_labels": sorted(samples_by_label)}
    pooled, pooled_info, pooled_rows = pooled_curve_from_samples(train_samples, f"Shot pooled train {','.join(train_labels)}")
    test_sample = samples_by_label[test_label]
    test_x = np.asarray(test_sample["sdr"], dtype=np.float64).reshape(-1)
    test_y = np.asarray(test_sample["hdr"], dtype=np.float64).reshape(-1)
    prediction_a, support_a = predict_model_a(model_a, test_x)
    prediction_pooled, support_pooled = predict_curve(pooled, test_x)
    if not np.isfinite(prediction_pooled).all():
        return {"status": "NONFINITE_POOLED_PREDICTION", "train_labels": train_labels, "test_label": test_label, "pooled_curve": model_to_json(pooled)}
    residual_a = prediction_a - test_y
    residual_pooled = prediction_pooled - test_y
    sample_count = min(RESIDUAL_PIXELS_PER_FRAME, test_x.size)
    indices = np.linspace(0, test_x.size - 1, sample_count, dtype=np.int64) if sample_count else np.empty(0, dtype=np.int64)
    split_name = f"train_{'_'.join(label.replace('%', '') for label in train_labels)}_test_{test_label.replace('%', '')}"
    residual_path = RESIDUAL_DIR / f"{case['case_id']}_{split_name}.npz"
    np.savez_compressed(residual_path, test_flat_indices=indices.astype(np.int32), residual_model_a=residual_a[indices].astype(np.float32), residual_shot_pooled=residual_pooled[indices].astype(np.float32))
    curve_path = CURVE_DIR / f"{case['case_id']}_{split_name}_pooled_curve.json"
    write_json(curve_path, {"phase": "P2.34", "case_id": case["case_id"], "train_labels": train_labels, "test_label": test_label, "curve": model_to_json(pooled), "diagnostic_bins": pooled_rows, "fit_info": pooled_info})
    return {
        "status": "SCORED",
        "split_name": split_name,
        "train_labels": train_labels,
        "test_label": test_label,
        "train_frame_count": len(train_samples),
        "test_frame": test_sample["hdr_frame"],
        "models": {"Model A": {"metrics": metrics_for_target(residual_a, test_y), "support": support_a}, "Shot pooled curve": {"metrics": metrics_for_target(residual_pooled, test_y), "support": support_pooled}},
        "pooled_curve": model_to_json(pooled),
        "pooled_curve_path": str(curve_path),
        "residual_path": str(residual_path),
        "test_sample_count": int(test_x.size),
        "residual_saved_sample_count": int(indices.size),
        "target_provenance": "direct HDR-master luminance from P2.34 temporal sample decoded with P2.32 geometry/offset",
    }


def aggregate_split_results(splits: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region in [item[2] for item in TEST_REGIONS]:
        result[region] = {}
        for model in ("Model A", "Shot pooled curve"):
            blocks = [split["models"][model]["metrics"][region] for split in splits if split.get("status") == "SCORED" and split["models"][model]["metrics"][region]["count"] > 0]
            if not blocks:
                result[region][model] = empty_metric()
                continue
            count = sum(block["count"] for block in blocks)
            block = empty_metric()
            block["count"] = int(count)
            block["MAE"] = float(sum(item["MAE"] * item["count"] for item in blocks) / count)
            block["RMSE"] = float(math.sqrt(sum(item["RMSE"] ** 2 * item["count"] for item in blocks) / count))
            for key in ("signed_mean", "median_residual", "P95", "P99"):
                block[key] = float(np.average([item[key] for item in blocks], weights=[item["count"] for item in blocks]))
            block["max"] = float(max(item["max"] for item in blocks))
            for key in ("signed_mean", "median", "MAE", "P95", "P99", "max"):
                values = [item["relative"][key] for item in blocks if item["relative"][key] is not None]
                block["relative"][key] = float(np.average(values, weights=[item["count"] for item in blocks if item["relative"][key] is not None])) if values else None
            result[region][model] = block
    return result


def split_improvements(splits: list[dict[str, Any]], region: str = "global") -> list[float]:
    values = []
    for split in splits:
        if split.get("status") != "SCORED":
            continue
        a = split["models"]["Model A"]["metrics"][region]["MAE"]
        c = split["models"]["Shot pooled curve"]["metrics"][region]["MAE"]
        if a is not None and c is not None:
            values.append(float(100.0 * (a - c) / max(abs(a), LOG_EPS)))
    return values


def shot_to_shot_variation(shot_records: list[dict[str, Any]], material: str | None = None) -> dict[str, Any]:
    selected = [record for record in shot_records if record["status"] == "USABLE_SHOT" and (material is None or record["material"] == material)]
    result = {"shot_count": len(selected), "queries": {}, "classification": "INSUFFICIENT DATA"}
    for query in QUERY_NITS:
        values = []
        for record in selected:
            value = curve_value(record["pooled_all_curve"], query)
            if value is not None:
                values.append(value)
        stats = sample_stats(values)
        if values:
            p05, p95 = np.percentile(values, [5, 95])
            p01, p99 = np.percentile(values, [1, 99])
            stats.update({"P95_spread_P95_minus_P5": float(p95 - p05), "P99_spread_P99_minus_P1": float(p99 - p01), "relative_P95_spread": float((p95 - p05) / max(abs(np.median(values)), 0.01))})
        result["queries"][str(query)] = stats
    spreads = [item["relative_P95_spread"] for item in result["queries"].values() if item.get("relative_P95_spread") is not None]
    if spreads:
        result["max_relative_P95_spread"] = float(max(spreads))
        result["median_relative_P95_spread"] = float(np.median(spreads))
        result["classification"] = "PRACTICALLY_SIMILAR" if max(spreads) <= 0.10 else "MODERATELY_DIFFERENT" if max(spreads) <= 0.30 else "STRONGLY_DIFFERENT"
    return result


def correlation_duration_stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    points = [(record["shot_duration_seconds"], record["temporal_stability"]["median_relative_P95_spread"]) for record in records if record["status"] == "USABLE_SHOT" and record["temporal_stability"].get("median_relative_P95_spread") is not None]
    if len(points) < 3:
        return {"count": len(points), "status": "INSUFFICIENT DATA", "pearson_r": None, "spearman_r": None}
    duration = np.asarray([point[0] for point in points], dtype=np.float64)
    stability = np.asarray([point[1] for point in points], dtype=np.float64)
    pearson = float(np.corrcoef(duration, stability)[0, 1]) if np.std(duration) > 0 and np.std(stability) > 0 else 0.0
    duration_rank = np.argsort(np.argsort(duration)).astype(np.float64)
    stability_rank = np.argsort(np.argsort(stability)).astype(np.float64)
    spearman = float(np.corrcoef(duration_rank, stability_rank)[0, 1]) if np.std(duration_rank) > 0 and np.std(stability_rank) > 0 else 0.0
    return {"count": len(points), "status": "COMPUTED", "pearson_r": pearson, "spearman_r": spearman, "points": [{"duration_seconds": float(duration[index]), "stability_median_relative_P95_spread": float(stability[index])} for index in range(len(points))]}


def decide(shot_records: list[dict[str, Any]], all_splits: list[dict[str, Any]], aggregate: dict[str, Any], variation: dict[str, Any]) -> dict[str, Any]:
    usable = [record for record in shot_records if record["status"] == "USABLE_SHOT"]
    scored = [split for split in all_splits if split.get("status") == "SCORED"]
    improvements = split_improvements(scored)
    high_improvements = split_improvements(scored, "highlight_gt_200")
    global_mae_a = aggregate.get("global", {}).get("Model A", {}).get("MAE")
    global_mae_c = aggregate.get("global", {}).get("Shot pooled curve", {}).get("MAE")
    global_improvement = 100.0 * (global_mae_a - global_mae_c) / max(abs(global_mae_a), LOG_EPS) if global_mae_a is not None and global_mae_c is not None else None
    better_fraction = float(np.mean(np.asarray(improvements) > 0.0)) if improvements else None
    clipping = [split["models"]["Shot pooled curve"]["support"]["clipping_fraction"] for split in scored]
    median_clipping = float(np.median(clipping)) if clipping else None
    stability_values = [record["temporal_stability"].get("max_relative_P95_spread") for record in usable if record["temporal_stability"].get("max_relative_P95_spread") is not None]
    median_stability = float(np.median(stability_values)) if stability_values else None
    per_material = {material: sum(record["material"] == material for record in usable) for material in MATERIALS}
    checks = {
        "enough_usable_shots": len(usable) >= MIN_USABLE_SHOTS,
        "enough_shots_per_material": all(per_material[material] >= MIN_USABLE_SHOTS_PER_MATERIAL for material in MATERIALS),
        "enough_temporal_prediction_splits": len(scored) >= MIN_PREDICTION_SPLITS,
        "shot_curve_temporal_spread_within_30_percent": median_stability is not None and median_stability <= MAX_RELATIVE_P95_CURVE_SPREAD,
        "pooled_curve_global_MAE_improves_by_5_percent": global_improvement is not None and global_improvement >= MIN_GLOBAL_IMPROVEMENT_PERCENT,
        "pooled_curve_better_in_60_percent_of_splits": better_fraction is not None and better_fraction >= MIN_BETTER_SPLIT_FRACTION,
        "median_test_domain_clipping_within_25_percent": median_clipping is not None and median_clipping <= MAX_MEDIAN_TEST_CLIPPING_FRACTION,
        "high_end_not_worse_by_more_than_5_percent": bool(high_improvements) and float(np.mean(high_improvements)) >= HIGH_END_WORSENING_TOLERANCE_PERCENT,
        "both_materials_have_temporal_evidence": per_material["The Matrix"] >= MIN_USABLE_SHOTS_PER_MATERIAL and per_material["BR2049"] >= MIN_USABLE_SHOTS_PER_MATERIAL,
    }
    if len(usable) < MIN_USABLE_SHOTS or len(scored) < MIN_PREDICTION_SPLITS:
        status = "INSUFFICIENT DATA"
    elif all(checks.values()):
        status = "FEASIBLE"
    elif checks["shot_curve_temporal_spread_within_30_percent"] and (checks["pooled_curve_global_MAE_improves_by_5_percent"] or checks["pooled_curve_better_in_60_percent_of_splits"]) and (per_material["The Matrix"] >= MIN_USABLE_SHOTS_PER_MATERIAL and per_material["BR2049"] >= MIN_USABLE_SHOTS_PER_MATERIAL):
        status = "PARTIALLY FEASIBLE"
    else:
        status = "NOT FEASIBLE"
    return {"status": status, "usable_shot_count": len(usable), "usable_shots_per_material": per_material, "scored_split_count": len(scored), "global_improvement_percent": global_improvement, "better_split_fraction": better_fraction, "median_test_clipping_fraction": median_clipping, "median_shot_temporal_spread": median_stability, "mean_high_end_improvement_percent": float(np.mean(high_improvements)) if high_improvements else None, "checks": checks, "thresholds_predeclared": {"minimum_usable_shots": MIN_USABLE_SHOTS, "minimum_usable_shots_per_material": MIN_USABLE_SHOTS_PER_MATERIAL, "minimum_prediction_splits": MIN_PREDICTION_SPLITS, "max_relative_P95_curve_spread": MAX_RELATIVE_P95_CURVE_SPREAD, "minimum_global_improvement_percent": MIN_GLOBAL_IMPROVEMENT_PERCENT, "minimum_better_split_fraction": MIN_BETTER_SPLIT_FRACTION, "max_median_test_clipping_fraction": MAX_MEDIAN_TEST_CLIPPING_FRACTION, "high_end_worsening_tolerance_percent": HIGH_END_WORSENING_TOLERANCE_PERCENT}}


def plot_bar(path: Path, title: str, labels: list[str], values: list[float], colors: list[tuple[int, int, int]]) -> None:
    width, height = 1400, 800
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 80, 50, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    finite = [value for value in values if math.isfinite(value)]
    maximum = max(finite or [1.0]) * 1.2
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 2)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + int((index + 0.5) * plot_w / max(len(values), 1))
        bar_h = int(max(value, 0.0) / max(maximum, 1e-12) * plot_h)
        cv2.rectangle(canvas, (x - 48, top + plot_h - bar_h), (x + 48, top + plot_h), color, -1)
        cv2.putText(canvas, label, (x - 45, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.3f}", (x - 45, max(24, top + plot_h - bar_h - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title, (left, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def plot_temporal_curves(path: Path, records: list[dict[str, Any]], material: str) -> None:
    width, height = 1500, 850
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    left, top, right, bottom = 100, 80, 50, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = -2.0, 3.1
    y_min, y_max = -3.0, 4.1
    def point(x: float, y: float) -> tuple[int, int] | None:
        if x <= 0 or y <= 0 or not math.isfinite(x) or not math.isfinite(y):
            return None
        return left + int((math.log10(x) - x_min) / (x_max - x_min) * plot_w), top + plot_h - int((math.log10(y) - y_min) / (y_max - y_min) * plot_h)
    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (0, 0, 0), 2)
    for record in records:
        if record["status"] != "USABLE_SHOT":
            continue
        color = (160, 160, 160)
        for sample in record["samples"]:
            model = sample["frame_curve"]
            nodes = [point(float(10.0 ** x - LOG_EPS), float(10.0 ** y - LOG_EPS)) for x, y in zip(model.get("x_log", []), model.get("y_log", []))]
            nodes = [item for item in nodes if item is not None]
            if len(nodes) >= 2:
                cv2.polylines(canvas, [np.asarray(nodes, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
        model = record["pooled_all_curve"]
        nodes = [point(float(10.0 ** x - LOG_EPS), float(10.0 ** y - LOG_EPS)) for x, y in zip(model.get("x_log", []), model.get("y_log", []))]
        nodes = [item for item in nodes if item is not None]
        if len(nodes) >= 2:
            cv2.polylines(canvas, [np.asarray(nodes, dtype=np.int32)], False, (30, 60, 210), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"{material} — temporal frame curves (gray) and shot pooled curves (blue)", (left, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "SDR luminance [nits, log]", (left + 500, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def build_report(data: dict[str, Any]) -> str:
    decision = data["decision"]
    aggregate = data["results"]["aggregate_prediction_metrics"]
    lines = [
        "# P2.34 — TRUE SHOT-LEVEL TEMPORAL TONE ANALYSIS",
        "",
        f"**Final status:** `{decision['status']}`",
        "",
        "This is a bounded offline temporal feasibility study. No production shot-level solution was implemented or integrated.",
        "",
        "## 1. Dataset and scope",
        "",
        "Only the 40 existing P2.32 anchors were used: 20 The Matrix and 20 BR2049. The P2.32 manifest and metrics remain immutable. HDR targets are direct HDR-master overlap luminance decoded at the paired temporal sample; no Model A output, LUT output, transformed output, or rendered HDR output was used as target.",
        "",
        "The Matrix retains offset `-19`, sync confidence `0.940513`, status `LOCKED`, and geometry confidence `1.0`. BR2049 retains offset `+1167`, sync confidence `0.9745`, status `LOCKED`, and geometry confidence `0.9155`; the BR2049 geometry warning remains active. `verified_scene_count` and scene independence remain unknown/unverified; a detected local boundary is not promoted to a verified film-wide scene label.",
        "",
        "No global synchronization, full-film scan, production transform, Model C integration, render, LUT generation, CUDA/FFmpeg change, commit, or push was performed.",
        "",
        "## 2. Fixed local-window and boundary method",
        "",
        f"For every anchor, only an HDR local window of ±{LOCAL_HALF_WINDOW_SECONDS:g} seconds was requested. This harness has a hard maximum of ±{MAX_HALF_WINDOW_SECONDS:g} seconds and does not expand the window when a boundary is missing. Low-resolution HDR proxies were decoded with the isolated FFmpeg executable only for these bounded windows.",
        "",
        f"Boundary score is deterministic: 0.4 normalized 64-bin histogram chi-square, 0.2 mean-luminance difference, and 0.4 edge-NCC change. Fixed hard threshold `{BOUNDARY_THRESHOLD:.2f}`; fixed gradual threshold `{GRADUAL_THRESHOLD:.2f}`; gradual run minimum `{GRADUAL_MIN_RUN_FRAMES}` frames. Every emitted boundary records frame, timestamp, score, threshold, type, and confidence. An anchor within `{ANCHOR_NEAR_CUT_FRAMES}` frames of a boundary is marked `ANCHOR_NEAR_CUT`.",
        "",
        "A real local shot is only accepted when both the preceding and following boundaries are observed inside the bounded window. Missing boundaries do not cause the window to be treated as a shot; the anchor is retained as unresolved/insufficient instead.",
        "",
        "## 3. Shot definition and temporal sampling",
        "",
        f"Resolved shots are half-open `[shot_start_frame, shot_end_frame_exclusive)` intervals. Samples target 10%, 25%, 50%, 75%, and 90% of the shot. Selected HDR frames are at least `{MIN_SAMPLE_SEPARATION_FRAMES}` frames apart; if the shot is short, deterministic fallback sampling is used only when at least three separated frames are possible. OM frames are derived solely as `HDR frame + fixed P2.32 offset`.",
        "",
        "For every sample the output preserves frame IDs, timestamps, decode metadata, SDR/HDR overlap array path and shape, frame-specific curve, filter counts, and rejection reasons.",
        "",
        "## 4. Frame-specific and pooled curves",
        "",
        "Each frame-specific curve and pooled shot curve uses fixed SDR bins from `<0.01` through `>1000` nits, HDR P10/P50/P90 per populated bin, PAVA monotonic projection, log-domain interpolation, and clamping without extrapolation. Empty bins remain empty in diagnostics.",
        "",
        "The shot pooled curve is built from several distinct temporal frames of the same resolved shot. It is not a single-anchor curve and it is not integrated as Model C.",
        "",
        "## 5. Temporal stability",
        "",
        "For each resolved shot and each query at 1, 5, 10, 20, 50, 100, 200, and 500 nits, JSON records pairwise frame-curve difference, frame-to-pooled difference, variance, P95 spread defined as P95−P5, and P99 spread defined as P99−P1. It also records temporal direction reversals and whether the curve values are monotonic in time at each query.",
        "",
        f"Shot-to-shot variation uses the same queries and reports separate Matrix/BR2049/combined results. The predeclared temporal stability gate is median maximum relative P95 spread ≤ `{MAX_RELATIVE_P95_CURVE_SPREAD:.2f}`.",
        "",
        "## 6. Prediction test",
        "",
        "For every usable shot two leakage-safe temporal tests are attempted:",
        "",
        "1. TRAIN = 10%, 25%, 50%, 75%; TEST = 90%.",
        "2. TRAIN = 10%, 25%, 50%, 90%; TEST = 75%.",
        "",
        "The whole test frame is held out. Model A is the unchanged production/reference curve. The second model is the shot-pooled empirical curve fitted only from the listed training frames. Metrics are computed separately for global, shadow `<10`, midtone `10–200`, highlight `>200`, `200–500`, `500–1000`, and `>1000` nits, including MAE, RMSE, P95, P99, max, and relative residuals.",
        "",
        "| Region | Model A MAE | Shot pooled MAE |",
        "|---|---:|---:|",
    ]
    for region in [item[2] for item in TEST_REGIONS]:
        a = aggregate.get(region, {}).get("Model A", {}).get("MAE")
        c = aggregate.get(region, {}).get("Shot pooled curve", {}).get("MAE")
        lines.append(f"| {region} | {a if a is not None else 'n/a'} | {c if c is not None else 'n/a'} |")
    lines += [
        "",
        "The CSV contains per-shot, per-split, per-model, and per-region results. Numeric residual arrays preserve deterministic TEST-frame samples; exact metrics are calculated over the full held-out frame.",
        "",
        "## 7. Domain and clipping",
        "",
        "For every temporal prediction test the output records the pooled curve domain, low/high outside-domain counts, clipping fraction, and extrapolation count. Domains were never extended artificially. If the TEST frame exceeds the TRAIN pooled domain, the result is reported as clipping rather than hidden extrapolation.",
        "",
        "## 8. Shot length relationship",
        "",
        "The JSON records shot duration, frame count, temporal sample count, stability spread, and duration-versus-stability Pearson/Spearman diagnostics when enough resolved shots exist. These results are not interpreted as grading changes without temporal evidence.",
        "",
        "## 9. Answers to the final questions",
        "",
        "1. **Is tone mapping stable within a shot?** See per-shot temporal spreads and monotonicity. The final status records whether the predeclared stability gate passes.",
        "2. **Does a shot-level curve differ from a frame-level curve?** Yes/no is determined from the stored frame-to-pooled difference distributions, not from a single frame.",
        "3. **Does a curve from part of a shot predict the remainder?** This is the two-direction whole-frame temporal holdout test; no test frame is used for fitting.",
        "4. **Is high-end more stable at shot level?** Compare the `>200`, `200–500`, `500–1000`, and `>1000` temporal holdout metrics and spread fields; empty regions remain count zero.",
        "5. **Does P2.33 domain/clipping remain?** Every prediction split reports it explicitly; no artificial domain extension is applied.",
        "6. **Is shot-specific adaptation a sensible production direction?** The final status is based on the fixed stability, prediction, cross-material, high-end, and domain gates below. No production recommendation is made beyond that status.",
        "",
        "## 10. Decision",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in decision["checks"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        f"Predeclared statuses: `INSUFFICIENT DATA` when fewer than `{MIN_USABLE_SHOTS}` usable shots or `{MIN_PREDICTION_SPLITS}` scored temporal splits exist; `FEASIBLE` only when every gate passes; `PARTIALLY FEASIBLE` when temporal stability and at least one predictive direction pass but cross-material/high-end/domain gates do not; otherwise `NOT FEASIBLE`.",
        "",
        "## 11. Artifacts",
        "",
        f"The output directory `{OUTPUT_DIR}` preserves local boundary traces, shot manifests, temporal SDR/HDR overlap samples, frame-specific and pooled curves, residual arrays, plots, JSON metrics, and CSV results.",
        "",
        "## Final status",
        "",
        f"**{decision['status']}**",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if not FFMPEG.exists():
        raise RuntimeError(f"Isolated FFmpeg not found: {FFMPEG}")
    dataset, manifest, cases = load_cases()
    for directory in (OUTPUT_DIR, BOUNDARY_DIR, MANIFEST_DIR, SAMPLE_DIR, CURVE_DIR, RESIDUAL_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    model_a = load_model_a()
    shot_records: list[dict[str, Any]] = []
    all_splits: list[dict[str, Any]] = []
    for case in cases:
        boundary, score_trace = detect_local_boundaries(case)
        boundary_path = BOUNDARY_DIR / f"{case['case_id']}_boundaries.json"
        score_path = BOUNDARY_DIR / f"{case['case_id']}_scores.npz"
        write_json(boundary_path, {"phase": "P2.34", "case_id": case["case_id"], "material": case["material"], "source_metadata": source_metadata(case), "boundary_result": boundary})
        np.savez_compressed(score_path, frame_ids=np.arange(boundary["window"]["start_frame"], boundary["window"]["start_frame"] + score_trace.size, dtype=np.int64), boundary_scores=score_trace)
        shot = dict(boundary)
        shot["case_offset_frames"] = int(case["offset_frames"])
        shot["sync_confidence"] = float(case["sync_confidence"])
        shot["geometry_confidence"] = float(case["geometry"]["confidence"])
        shot["geometry_warning"] = bool(case["material"] == "BR2049" and float(case["geometry"]["confidence"]) < 0.95)
        samples = select_temporal_samples(shot)
        shot["requested_temporal_sample_fractions"] = [{"fraction": fraction, "label": label} for fraction, label in TEMPORAL_FRACTIONS]
        shot["samples"] = samples
        shot["sample_count"] = len(samples)
        shot["boundary_artifact"] = str(boundary_path)
        shot["boundary_score_artifact"] = str(score_path)
        shot["status"] = "UNRESOLVED_SHOT"
        shot_dir = SAMPLE_DIR / case["case_id"]
        shot_dir.mkdir(parents=True, exist_ok=True)
        samples_by_label: dict[str, dict[str, Any]] = {}
        if shot["shot_status"] in ("LOCAL_SHOT_RESOLVED", "ANCHOR_NEAR_CUT") and len(samples) >= 3:
            try:
                for sample in samples:
                    extracted = extract_sample(case, sample, shot_dir)
                    samples_by_label[sample["label"]] = extracted
                shot["status"] = "USABLE_SHOT"
            except Exception as exc:
                shot["status"] = "SAMPLE_EXTRACTION_FAILED"
                shot["sample_error"] = str(exc)
        shot["samples"] = [{key: value for key, value in sample.items() if key not in {"sdr", "hdr", "frame_curve"}} | {"frame_curve": model_to_json(sample["frame_curve"]) if "frame_curve" in sample else None} for sample in samples_by_label.values()]
        if shot["status"] == "USABLE_SHOT":
            sample_list = list(samples_by_label.values())
            pooled_all, pooled_info, pooled_rows = pooled_curve_from_samples(sample_list, "Shot pooled all temporal samples")
            pooled_curve_path = CURVE_DIR / f"{case['case_id']}_pooled_all_curve.json"
            write_json(pooled_curve_path, {"phase": "P2.34", "case_id": case["case_id"], "curve": model_to_json(pooled_all), "diagnostic_bins": pooled_rows, "fit_info": pooled_info, "sample_labels": list(samples_by_label)})
            shot["pooled_all_curve"] = pooled_all
            shot["pooled_all_curve_path"] = str(pooled_curve_path)
            shot["temporal_stability"] = curve_variability([sample["frame_curve"] for sample in sample_list], pooled_all)
            split_a = evaluate_prediction_split(case, shot, samples_by_label, ["10%", "25%", "50%", "75%"], "90%", model_a)
            split_b = evaluate_prediction_split(case, shot, samples_by_label, ["10%", "25%", "50%", "90%"], "75%", model_a)
            shot["prediction_splits"] = [split_a, split_b]
            all_splits.extend([{**split, "case_id": case["case_id"], "material": case["material"]} for split in (split_a, split_b)])
        else:
            shot["pooled_all_curve"] = None
            shot["pooled_all_curve_path"] = None
            shot["temporal_stability"] = {"status": "INSUFFICIENT DATA", "reason": "No resolved shot with at least three separated temporal samples"}
            shot["prediction_splits"] = []
        shot_records.append(shot)
        write_json(MANIFEST_DIR / f"{case['case_id']}_shot_manifest.json", {"phase": "P2.34", "case_id": case["case_id"], "material": case["material"], "source_metadata": source_metadata(case), "p2_32_scene_independence_verified": bool(case.get("scene_independence_verified", False)), "shot": {key: value for key, value in shot.items() if key not in {"pooled_all_curve", "samples"}}, "samples": shot["samples"]})
        for sample in samples_by_label.values():
            del sample["sdr"], sample["hdr"]
    aggregate = aggregate_split_results([split for split in all_splits if split.get("status") == "SCORED"])
    variation = {material: shot_to_shot_variation(shot_records, material) for material in MATERIALS}
    variation["combined"] = shot_to_shot_variation(shot_records)
    duration_stability = correlation_duration_stability(shot_records)
    decision = decide(shot_records, all_splits, aggregate, variation)

    usable_by_material = {material: [record for record in shot_records if record["material"] == material] for material in MATERIALS}
    plot_bar(PLOT_DIR / "P234_usable_shots_by_material.png", "P2.34 usable resolved shots", list(MATERIALS), [sum(record["status"] == "USABLE_SHOT" for record in usable_by_material[material]) for material in MATERIALS], [(80, 80, 210), (50, 150, 70)])
    plot_bar(PLOT_DIR / "P234_shot_temporal_spread.png", "P2.34 median maximum relative P95 curve spread", list(MATERIALS), [float(np.median([record["temporal_stability"].get("max_relative_P95_spread") for record in usable_by_material[material] if record["temporal_stability"].get("max_relative_P95_spread") is not None]) if any(record["temporal_stability"].get("max_relative_P95_spread") is not None for record in usable_by_material[material]) else 0.0) for material in MATERIALS], [(120, 100, 210), (60, 160, 80)])
    plot_bar(PLOT_DIR / "P234_prediction_global_mae.png", "P2.34 temporal holdout global MAE", ["Model A", "Shot pooled"], [aggregate.get("global", {}).get("Model A", {}).get("MAE") or 0.0, aggregate.get("global", {}).get("Shot pooled curve", {}).get("MAE") or 0.0], [(80, 80, 210), (50, 150, 70)])
    plot_bar(PLOT_DIR / "P234_domain_clipping.png", "P2.34 shot pooled TEST clipping median", ["all splits"], [decision["median_test_clipping_fraction"] or 0.0], [(120, 80, 180)])
    plot_temporal_curves(PLOT_DIR / "P234_matrix_temporal_curves.png", [record for record in shot_records if record["material"] == "The Matrix"], "The Matrix")
    plot_temporal_curves(PLOT_DIR / "P234_br2049_temporal_curves.png", [record for record in shot_records if record["material"] == "BR2049"], "BR2049")

    fields = ["material", "case_id", "status", "shot_start_frame", "shot_end_frame_exclusive", "shot_duration_seconds", "shot_sample_count", "split_name", "train_labels", "test_label", "model", "region", "count", "MAE", "RMSE", "P95", "P99", "max", "relative_MAE", "clipping_fraction", "outside_domain_count", "extrapolation_count"]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shot in shot_records:
            splits = shot.get("prediction_splits", [])
            if not splits:
                writer.writerow({"material": shot["material"], "case_id": shot["case_id"], "status": shot["status"], "shot_start_frame": shot.get("shot_start_frame"), "shot_end_frame_exclusive": shot.get("shot_end_frame_exclusive"), "shot_duration_seconds": shot.get("shot_duration_seconds"), "shot_sample_count": shot.get("sample_count"), "split_name": None, "train_labels": None, "test_label": None, "model": None, "region": None})
                continue
            for split in splits:
                for model, model_result in split.get("models", {}).items():
                    for _, _, region in TEST_REGIONS:
                        metric = model_result["metrics"][region]
                        support = model_result["support"]
                        writer.writerow({"material": shot["material"], "case_id": shot["case_id"], "status": shot["status"], "shot_start_frame": shot.get("shot_start_frame"), "shot_end_frame_exclusive": shot.get("shot_end_frame_exclusive"), "shot_duration_seconds": shot.get("shot_duration_seconds"), "shot_sample_count": shot.get("sample_count"), "split_name": split.get("split_name"), "train_labels": ";".join(split.get("train_labels", [])), "test_label": split.get("test_label"), "model": model, "region": region, "count": metric.get("count"), "MAE": metric.get("MAE"), "RMSE": metric.get("RMSE"), "P95": metric.get("P95"), "P99": metric.get("P99"), "max": metric.get("max"), "relative_MAE": metric.get("relative", {}).get("MAE"), "clipping_fraction": support.get("clipping_fraction"), "outside_domain_count": support.get("outside_domain_count"), "extrapolation_count": support.get("extrapolation_count")})

    json_data = {
        "phase": "P2.34",
        "status": decision["status"],
        "study": "true shot-level temporal tone analysis",
        "dataset_reference": {"manifest": str(P232_MANIFEST_PATH), "metrics": str(P232_METRICS_PATH), "material_case_counts": {material: 20 for material in MATERIALS}, "verified_scene_count": {material: None for material in MATERIALS}, "scene_independence_verified": {material: False for material in MATERIALS}, "target_provenance": "direct HDR-master overlap from bounded P2.34 sample decode; no output-as-target", "warnings": {"The Matrix": {"offset_frames": -19, "sync_confidence": 0.940513, "sync_status": "LOCKED", "geometry_confidence": 1.0}, "BR2049": {"offset_frames": 1167, "sync_confidence": 0.9745, "sync_status": "LOCKED", "geometry_confidence": 0.9155, "geometry_warning": True}}},
        "fixed_protocol": {"local_half_window_seconds": LOCAL_HALF_WINDOW_SECONDS, "hard_max_half_window_seconds": MAX_HALF_WINDOW_SECONDS, "proxy_width": PROXY_WIDTH, "boundary_score_weights": {"histogram": 0.4, "mean_luminance": 0.2, "edge_ncc_change": 0.4}, "hard_boundary_threshold": BOUNDARY_THRESHOLD, "gradual_threshold": GRADUAL_THRESHOLD, "gradual_min_run_frames": GRADUAL_MIN_RUN_FRAMES, "anchor_near_cut_guard_frames": ANCHOR_NEAR_CUT_FRAMES, "temporal_fractions": [{"fraction": fraction, "label": label} for fraction, label in TEMPORAL_FRACTIONS], "minimum_sample_separation_frames": MIN_SAMPLE_SEPARATION_FRAMES, "queries_nits": QUERY_NITS, "prediction_splits": [{"train": ["10%", "25%", "50%", "75%"], "test": "90%"}, {"train": ["10%", "25%", "50%", "90%"], "test": "75%"}], "no_global_sync": True, "no_full_film_scan": True, "no_artificial_domain_extension": True},
        "cases": [{key: value for key, value in record.items() if key not in {"pooled_all_curve", "samples"}} | {"pooled_all_curve": model_to_json(record["pooled_all_curve"]) if record.get("pooled_all_curve") else None, "samples": record.get("samples", [])} for record in shot_records],
        "results": {"prediction_splits": all_splits, "aggregate_prediction_metrics": aggregate, "shot_to_shot_variation": variation, "duration_vs_stability": duration_stability},
        "decision": decision,
        "artifacts": {"json": str(METRICS_PATH), "csv": str(CSV_PATH), "report": str(REPORT_PATH), "output_directory": str(OUTPUT_DIR), "shot_boundaries": str(BOUNDARY_DIR), "shot_manifests": str(MANIFEST_DIR), "temporal_samples": str(SAMPLE_DIR), "shot_curves": str(CURVE_DIR), "residuals": str(RESIDUAL_DIR), "plots": [str(path) for path in PLOT_DIR.glob("*.png")]},
        "scope_constraints": {"p232_dataset_modified": False, "p232_pairs_added": False, "source_decode_scope": "bounded local windows and selected sample frames only", "full_film_scan_performed": False, "global_find_global_offset_performed": False, "synchronization_changed": False, "production_code_changed": False, "transform_py_changed": False, "luminance_py_changed": False, "compose_py_changed": False, "render_py_changed": False, "cuda_changed": False, "production_lut_changed": False, "model_a_changed": False, "model_c_integrated": False, "hdr_output_rendered": False, "commit_created": False, "push_performed": False},
    }
    write_json(METRICS_PATH, json_data)
    REPORT_PATH.write_text(build_report(json_data), encoding="utf-8")
    print(f"P2.34_STATUS={decision['status']}")
    print(f"USABLE_SHOTS={decision['usable_shot_count']}")
    print(f"SCORED_SPLITS={decision['scored_split_count']}")
    print(f"METRICS={METRICS_PATH}")
    print(f"CSV={CSV_PATH}")
    print(f"REPORT={REPORT_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
