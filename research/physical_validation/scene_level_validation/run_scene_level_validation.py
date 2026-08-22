"""Phase 6 bounded shared-transform validation for frozen Phase-5 MMR-1.

This research-only harness discovers only local HDR boundaries around four fixed
P2.32 anchors, decodes a fixed five-frame temporal sample when resolvable, and
fits one pooled H0 + MMR-1 transform from 10/25/50/75% frames. The 90% frame is
held out in time. It never modifies Phase 5, P2.32, sync, geometry, or production.
"""
from __future__ import annotations

import gc
import hashlib
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

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PROTOCOL_PATH = BASE / "PHASE6_FROZEN_PROTOCOL.json"
OUTPUT = BASE / "results"
METRICS_PATH = OUTPUT / "scene_level_metrics.json"
REPORT_PATH = BASE / "SCENE_LEVEL_VALIDATION_REPORT.md"
P232_METRICS = ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset" / "P232_dataset_metrics.json"
FFMPEG = ROOT / "dev" / "ffmpeg-build" / "install" / "bin" / "ffmpeg.exe"
FPS = 24000.0 / 1001.0
PEAK_NITS = 10000.0
RGB16_MAX = 65535.0


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_frozen_module(protocol: dict[str, Any]) -> Any:
    expected = protocol["frozen_dependency_hashes_sha256"]["mmr1_source"]
    source = ROOT / expected["path"]
    require(sha256(source) == expected["sha256"], "Frozen Phase-5 MMR source hash mismatch")
    for key in ("h0_and_output_helpers", "p232_metrics", "p232_manifest"):
        entry = protocol["frozen_dependency_hashes_sha256"][key]
        require(sha256(ROOT / entry["path"]) == entry["sha256"], f"Frozen dependency hash mismatch: {key}")
    source_parent = str(source.parent)
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)
    spec = importlib.util.spec_from_file_location("phase5_frozen_mmr1", source)
    require(spec is not None and spec.loader is not None, "Could not load frozen MMR module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = protocol["frozen_mmr1_contract"]
    require(tuple(module.LAMBDA_GRID) == tuple(contract["lambda_grid"]), "Frozen lambda grid mismatch")
    require(module.CHROMA_BOUND == contract["chroma_component_bound"], "Frozen chroma bound mismatch")
    require(module.LUMA_FEATURE_SCALE_NITS == contract["luma_feature_scale_nits"], "Frozen luminance feature scale mismatch")
    require((module.SPATIAL_TRAIN, module.SPATIAL_HOLDOUT) == (contract["spatial_train_limit_per_frame"], contract["spatial_holdout_limit_per_frame"]), "Frozen spatial sample limits mismatch")
    require(module.identity_coefficients("MMR1").shape == (6, 2), "Frozen MMR-1 parameterization mismatch")
    return module


def timestamp(frame: int) -> float:
    return float(frame / FPS)


def run_rgb48_frame(source: Path, width: int, height: int, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
    expected = width * height * 3 * 2
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp(frame):.9f}", "-i", str(source), "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb48le", "-vsync", "0", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    raw = completed.stdout
    metadata = {"command": command, "returncode": int(completed.returncode), "stderr": completed.stderr.decode("utf-8", errors="replace"), "stdout_bytes": len(raw), "expected_bytes": expected, "pixel_format": "rgb48le", "frame": int(frame), "timestamp_seconds": timestamp(frame)}
    if completed.returncode != 0 or len(raw) < expected:
        raise RuntimeError(f"RGB48LE extraction failed: {metadata}")
    array = np.frombuffer(raw[:expected], dtype="<u2").reshape(height, width, 3).copy()
    return array, metadata


def decode_gray_window(case: dict[str, Any], half_seconds: float, proxy_width: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    anchor = int(case["hdr_frame"])
    half = int(round(half_seconds * FPS))
    start = max(0, anchor - half)
    requested_end = anchor + half + 1
    width = int(proxy_width)
    height = max(2, int(round(case["hdr_resolution"][1] * width / case["hdr_resolution"][0] / 2.0) * 2))
    count = requested_end - start
    expected = width * height
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp(start):.9f}", "-i", case["hdr_source"], "-map", "0:v:0", "-vf", f"scale={width}:{height},format=gray", "-frames:v", str(count), "-pix_fmt", "gray", "-vsync", "0", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    decoded = min(count, len(completed.stdout) // expected)
    meta = {"command": command, "returncode": int(completed.returncode), "stderr": completed.stderr.decode("utf-8", errors="replace"), "requested_start_frame": start, "requested_end_frame_exclusive": requested_end, "requested_frame_count": count, "decoded_frame_count": decoded, "proxy_shape": [height, width], "full_film_scan": False, "seek_mode": "bounded fast input seek"}
    if completed.returncode != 0 or decoded < 2:
        raise RuntimeError(f"Bounded proxy decode failed: {meta}")
    frames = np.frombuffer(completed.stdout[:decoded * expected], dtype=np.uint8).reshape(decoded, height, width).astype(np.float32) / 255.0
    return frames, np.arange(start, start + decoded, dtype=np.int64), meta


def boundary_score(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float, float]:
    hist_left, _ = np.histogram(left, bins=64, range=(0.0, 1.0))
    hist_right, _ = np.histogram(right, bins=64, range=(0.0, 1.0))
    hist_left = hist_left.astype(np.float64) / max(float(hist_left.sum()), 1.0)
    hist_right = hist_right.astype(np.float64) / max(float(hist_right.sum()), 1.0)
    total = hist_left + hist_right
    valid = total > 0
    hist = float(min(1.0, np.sum((hist_left[valid] - hist_right[valid]) ** 2 / total[valid]) / 2.0)) if np.any(valid) else 0.0
    luminance = float(abs(np.mean(left) - np.mean(right)))
    edge_left = cv2.Canny(np.asarray(left * 255.0, dtype=np.uint8), 32, 96).reshape(-1).astype(np.float64)
    edge_right = cv2.Canny(np.asarray(right * 255.0, dtype=np.uint8), 32, 96).reshape(-1).astype(np.float64)
    edge_left -= np.mean(edge_left)
    edge_right -= np.mean(edge_right)
    denom = float(np.linalg.norm(edge_left) * np.linalg.norm(edge_right))
    ncc = float(np.dot(edge_left, edge_right) / denom) if denom > 1e-12 else 0.0
    edge = max(0.0, 1.0 - ncc)
    return float(0.4 * hist + 0.2 * luminance + 0.4 * edge), hist, luminance, edge


def discover_local_shot(case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    setup = protocol["local_shot_grouping"]
    frames, frame_ids, decode = decode_gray_window(case, setup["local_half_window_seconds"], setup["proxy_width"])
    scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for index in range(frames.shape[0] - 1):
        score, hist, lum, edge = boundary_score(frames[index], frames[index + 1])
        scores.append(score)
        rows.append({"frame_a": int(frame_ids[index]), "frame_b": int(frame_ids[index + 1]), "score": score, "histogram": hist, "luminance": lum, "edge_ncc_change": edge})
    score_array = np.asarray(scores, dtype=np.float64)
    hard, gradual, min_run = setup["score"]["hard_threshold"], setup["score"]["gradual_threshold"], setup["score"]["gradual_min_run_frames"]
    candidates: list[dict[str, Any]] = []
    for index, score in enumerate(score_array):
        prior = score_array[index - 1] if index else 0.0
        following = score_array[index + 1] if index + 1 < score_array.size else 0.0
        if score >= hard and prior < gradual and following < gradual:
            candidates.append({"frame": int(frame_ids[index + 1]), "score": float(score), "type": "hard"})
    index = 0
    while index < score_array.size:
        if score_array[index] < gradual:
            index += 1
            continue
        start = index
        while index < score_array.size and score_array[index] >= gradual:
            index += 1
        if index - start >= min_run:
            peak = start + int(np.argmax(score_array[start:index]))
            candidates.append({"frame": int(frame_ids[peak + 1]), "score": float(score_array[peak]), "type": "gradual", "run_start_frame": int(frame_ids[start]), "run_end_frame": int(frame_ids[index])})
    candidates.sort(key=lambda item: item["frame"])
    boundaries: list[dict[str, Any]] = []
    for candidate in candidates:
        if boundaries and candidate["frame"] - boundaries[-1]["frame"] <= min_run:
            if candidate["score"] > boundaries[-1]["score"]:
                boundaries[-1] = candidate
        else:
            boundaries.append(candidate)
    anchor = int(case["hdr_frame"])
    left = [item for item in boundaries if item["frame"] <= anchor]
    right = [item for item in boundaries if item["frame"] > anchor]
    nearest = min((abs(item["frame"] - anchor) for item in boundaries), default=None)
    near_cut = nearest is not None and nearest <= setup["anchor_near_cut_guard_frames"]
    start = int(left[-1]["frame"]) if left else None
    end = int(right[0]["frame"]) if right else None
    status = "LOCAL_SHOT_RESOLVED" if start is not None and end is not None and end - start >= 3 else "LOCAL_SHOT_UNRESOLVED"
    if start is not None and end is not None and end - start < 3:
        status = "SHOT_TOO_SHORT"
    if near_cut:
        status = "ANCHOR_NEAR_CUT"
    return {"case_id": case["case_id"], "material": case["material"], "anchor_hdr_frame": anchor, "anchor_om_frame": int(case["om_frame"]), "window": {"start_frame": int(frame_ids[0]), "end_frame_exclusive": int(frame_ids[-1] + 1), "half_window_seconds": setup["local_half_window_seconds"]}, "decode": decode, "boundary_detection": setup["score"], "boundaries": boundaries, "score_rows": rows, "shot_status": status, "shot_start_frame": start, "shot_end_frame_exclusive": end, "shot_duration_frames": end - start if start is not None and end is not None else None, "anchor_near_cut": near_cut, "nearest_boundary_distance_frames": nearest, "full_film_scan": False}


def select_samples(discovery: dict[str, Any], protocol: dict[str, Any], offset: int) -> list[dict[str, Any]]:
    if discovery["shot_status"] != "LOCAL_SHOT_RESOLVED":
        return []
    start, end = discovery["shot_start_frame"], discovery["shot_end_frame_exclusive"]
    separation = protocol["temporal_sampling_and_partition"]["minimum_sample_separation_frames"]
    samples: list[dict[str, Any]] = []
    for fraction, label in protocol["temporal_sampling_and_partition"]["fractions"]:
        desired = int(round(start + fraction * max(end - start - 1, 0)))
        candidates = sorted(range(start, end), key=lambda frame: (abs(frame - desired), frame))
        choice = next((frame for frame in candidates if all(abs(frame - item["hdr_frame"]) >= separation for item in samples)), None)
        if choice is None:
            return []
        samples.append({"label": label, "fraction": fraction, "hdr_frame": int(choice), "om_frame": int(choice + offset), "hdr_timestamp_seconds": timestamp(int(choice)), "om_timestamp_seconds": timestamp(int(choice + offset))})
    return samples


def decode_pair_exact(frozen: Any, case: dict[str, Any], hdr_raw: np.ndarray, om_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    require(hdr_raw.dtype == np.uint16 and om_raw.dtype == np.uint16, "Expected RGB48LE uint16 arrays")
    require(tuple(hdr_raw.shape) == tuple(case["hdr_rgb_shape"]) and tuple(om_raw.shape) == tuple(case["om_rgb_shape"]), "Extracted source shape mismatch")
    x1, y1, x2, y2 = map(int, case["geometry"]["overlap"])
    hdr_code = frozen.cv2.resize(hdr_raw.astype(np.float64) / frozen.RGB16_MAX, (x2 - x1, y2 - y1), interpolation=frozen.cv2.INTER_LINEAR)
    target = frozen.pq_eotf_normalized(hdr_code)
    sdr_full = np.maximum(frozen.bt709_to_bt2020(frozen.bt1886_eotf(om_raw.astype(np.float64) / frozen.RGB16_MAX)), 0.0)
    sdr_common = sdr_full[y1:y2, x1:x2]
    require(sdr_common.shape == target.shape and np.isfinite(sdr_full).all() and np.isfinite(target).all(), "Invalid exact Phase-5-compatible decode")
    return sdr_full, sdr_common, target, (x1, y1, x2, y2)


def aggregate_bounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([row["pixel_count"] for row in rows], dtype=np.float64)
    def weighted(key: str) -> float:
        return float(np.average([row["bounds"][key] for row in rows], weights=counts))
    return {"frame_count": len(rows), "pixel_count": int(np.sum(counts)), "chroma_bound_pixel_fraction": weighted("chroma_bound_pixel_fraction"), "inverse_negative_pixel_fraction": weighted("inverse_negative_pixel_fraction"), "inverse_negative_channel_fraction": weighted("inverse_negative_channel_fraction"), "Y_target_max_abs_error_nits": float(max(row["bounds"]["Y_target_max_abs_error_nits"] for row in rows)), "finite": bool(all(row["bounds"]["finite"] for row in rows)), "reimposition_dark_fallback_pixels": int(sum(row["bounds"]["reimposition_dark_fallback_pixels"] for row in rows))}


def pooled_fit_mmr(frozen: Any, training: list[dict[str, Any]]) -> dict[str, Any]:
    """Frozen MMR-1 ridge, with only frame pooling added before the same fit/objective."""
    theta0 = frozen.identity_coefficients("MMR1")
    train_x: list[np.ndarray] = []; train_c: list[np.ndarray] = []; hold_x: list[np.ndarray] = []; hold_c: list[np.ndarray] = []
    frame_splits: list[dict[str, Any]] = []
    for frame in training:
        base, target, target_y = frame["h0_common"], frame["target"], frame["target_y_common"]
        spatial_train, spatial_hold = frozen.spatial_indices(base.shape[:2])
        design, _ = frozen.features(base, target_y, "MMR1")
        chroma = frozen.linear_bt2020_to_ictcp(target * PEAK_NITS)[..., 1:]
        flat_design, flat_chroma = design.reshape(-1, 6), chroma.reshape(-1, 2)
        train_x.append(flat_design[spatial_train]); train_c.append(flat_chroma[spatial_train]); hold_x.append(flat_design[spatial_hold]); hold_c.append(flat_chroma[spatial_hold])
        frame_splits.append({"label": frame["sample"]["label"], "spatial_train_count": len(spatial_train), "spatial_holdout_count": len(spatial_hold)})
    x_train, c_train = np.concatenate(train_x), np.concatenate(train_c)
    x_hold, c_hold = np.concatenate(hold_x), np.concatenate(hold_c)
    candidates: list[dict[str, Any]] = []
    for lam in frozen.LAMBDA_GRID:
        started = time.perf_counter()
        gram = x_train.T @ x_train / len(x_train) + lam * np.eye(x_train.shape[1])
        rhs = x_train.T @ c_train / len(x_train) + lam * theta0
        theta = np.linalg.solve(gram, rhs)
        train_metrics: list[dict[str, Any]] = []; hold_metrics: list[dict[str, Any]] = []; train_bounds: list[dict[str, Any]] = []; hold_bounds: list[dict[str, Any]] = []
        for frame in training:
            base, target, target_y = frame["h0_common"], frame["target"], frame["target_y_common"]
            spatial_train, spatial_hold = frozen.spatial_indices(base.shape[:2])
            output_train, bounds_train = frozen.apply_mmr(base.reshape(-1, 1, 3)[spatial_train], target_y.reshape(-1, 1)[spatial_train], theta, "MMR1")
            output_hold, bounds_hold = frozen.apply_mmr(base.reshape(-1, 1, 3)[spatial_hold], target_y.reshape(-1, 1)[spatial_hold], theta, "MMR1")
            train_metrics.append(frozen.metrics(output_train, target.reshape(-1, 1, 3)[spatial_train])); hold_metrics.append(frozen.metrics(output_hold, target.reshape(-1, 1, 3)[spatial_hold]))
            train_bounds.append({"pixel_count": len(spatial_train), "bounds": bounds_train}); hold_bounds.append({"pixel_count": len(spatial_hold), "bounds": bounds_hold})
        # The pooled metrics below use exactly the frozen metric implementation over all frozen spatial samples.
        pool_train_output, _ = frozen.apply_mmr(np.concatenate([item["h0_common"].reshape(-1, 1, 3)[frozen.spatial_indices(item["h0_common"].shape[:2])[0]] for item in training]), np.concatenate([item["target_y_common"].reshape(-1, 1)[frozen.spatial_indices(item["h0_common"].shape[:2])[0]] for item in training]), theta, "MMR1")
        pool_hold_output, _ = frozen.apply_mmr(np.concatenate([item["h0_common"].reshape(-1, 1, 3)[frozen.spatial_indices(item["h0_common"].shape[:2])[1]] for item in training]), np.concatenate([item["target_y_common"].reshape(-1, 1)[frozen.spatial_indices(item["h0_common"].shape[:2])[1]] for item in training]), theta, "MMR1")
        pool_train_target = np.concatenate([item["target"].reshape(-1, 1, 3)[frozen.spatial_indices(item["h0_common"].shape[:2])[0]] for item in training])
        pool_hold_target = np.concatenate([item["target"].reshape(-1, 1, 3)[frozen.spatial_indices(item["h0_common"].shape[:2])[1]] for item in training])
        train_metric = frozen.metrics(pool_train_output, pool_train_target)
        hold_metric = frozen.metrics(pool_hold_output, pool_hold_target)
        hold_bound = aggregate_bounds(hold_bounds)
        condition = float(np.linalg.cond(gram)); deviation = float(np.linalg.norm(theta - theta0))
        stable = bool(np.isfinite(condition) and condition <= 1e6 and deviation <= 5.0 and hold_bound["chroma_bound_pixel_fraction"] <= 0.005 and hold_bound["inverse_negative_pixel_fraction"] <= 0.005 and hold_bound["finite"])
        candidates.append({"lambda": lam, "coefficients": theta, "regularized_gram_condition_number": condition, "coefficient_identity_deviation_L2": deviation, "coefficient_max_abs": float(np.max(np.abs(theta))), "training_metrics": train_metric, "holdout_metrics": hold_metric, "training_bounds": aggregate_bounds(train_bounds), "holdout_bounds": hold_bound, "holdout_objective": frozen.objective(hold_metric), "stable": stable, "fit_seconds_cpu": time.perf_counter() - started, "per_frame_train_metrics": train_metrics, "per_frame_holdout_metrics": hold_metrics})
    accepted = [item for item in candidates if item["stable"]]
    require(bool(accepted), "All pooled frozen MMR-1 candidates were pathological")
    selected = min(accepted, key=lambda item: item["holdout_objective"])
    return {"model": "MMR1", "feature_names": ["constant", "Y_target_nits/100", "Ct_base", "Cp_base", "(Y_target_nits/100)*Ct_base", "(Y_target_nits/100)*Cp_base"], "parameter_count": 12, "identity_coefficients": theta0, "training_sample_count": len(x_train), "spatial_holdout_sample_count": len(x_hold), "frame_splits": frame_splits, "selected": selected, "candidates": candidates, "regularized_closed_form": "(XᵀX/N + λI)⁻¹(XᵀC/N + λΘ_identity)", "pooling_note": "Only same-shot training-frame sample rows are concatenated. All features, ridge formula, lambda candidates, identity prior, constraints, spatial indices, objective, and application are frozen Phase-5 behavior."}


def write_frame_artifacts(frozen: Any, shot_dir: Path, frame: dict[str, Any], h0_full: np.ndarray, mmr_full: np.ndarray, bounds: dict[str, Any]) -> dict[str, str]:
    label = frame["sample"]["label"].replace("%", "pct")
    x1, y1, x2, y2 = frame["bbox"]
    preview_dir, seam_dir, difference_dir, scientific_dir = (shot_dir / "previews", shot_dir / "seams", shot_dir / "differences", shot_dir / "scientific")
    for directory in (preview_dir, seam_dir, difference_dir, scientific_dir): directory.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros_like(h0_full); canvas[y1:y2, x1:x2] = frame["target"]
    montage = frozen.comparison_montage(frame["sdr_full"], canvas, h0_full, [mmr_full], height=180)
    montage_path = preview_dir / f"{label}_comparison_SDR_HDR_H0_shared_MMR1.png"
    frozen.write_cv_image(montage_path, montage[..., ::-1])
    frozen.write_preview(preview_dir / f"{label}_H0_log.png", h0_full)
    frozen.write_preview(preview_dir / f"{label}_MMR1_shared_log.png", mmr_full)
    difference = frozen.diff_map(mmr_full[y1:y2, x1:x2], frame["target"])
    difference_path = difference_dir / f"{label}_MMR1_shared_luminance_difference.png"
    frozen.write_cv_image(difference_path, difference)
    composite = mmr_full.copy(); composite[y1:y2, x1:x2] = frame["target"]
    strip = frozen.panel([mmr_full[max(0, y1 - 60):y1 + 60], composite[max(0, y1 - 60):y1 + 60], mmr_full[y2 - 60:min(mmr_full.shape[0], y2 + 60)], composite[y2 - 60:min(mmr_full.shape[0], y2 + 60)]], 120)
    seam_path = seam_dir / f"{label}_MMR1_shared_seam.png"
    frozen.write_cv_image(seam_path, strip[..., ::-1])
    scientific: dict[str, str] = {}
    if frame["sample"]["label"] == "90%":
        frozen.write_scientific(scientific_dir / "heldout_90pct_H0_full", h0_full)
        frozen.write_scientific(scientific_dir / "heldout_90pct_MMR1_shared_full", mmr_full)
        scientific = {"heldout_h0_full": str(scientific_dir / "heldout_90pct_H0_full"), "heldout_mmr1_full": str(scientific_dir / "heldout_90pct_MMR1_shared_full")}
    return {"montage": str(montage_path), "difference": str(difference_path), "seam": str(seam_path), **scientific, "bound_finite": str(bounds["finite"])}


def fit_and_evaluate_shot(frozen: Any, case: dict[str, Any], discovery: dict[str, Any], samples: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    shot_dir = OUTPUT / case["case_id"]
    shot_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for sample in samples:
        hdr_raw, hdr_decode = run_rgb48_frame(Path(case["hdr_source"]), *map(int, case["hdr_resolution"]), sample["hdr_frame"])
        om_raw, om_decode = run_rgb48_frame(Path(case["om_source"]), *map(int, case["om_resolution"]), sample["om_frame"])
        sdr_full, sdr_common, target, bbox = decode_pair_exact(frozen, case, hdr_raw, om_raw)
        frames.append({"sample": sample, "sdr_full": sdr_full, "sdr_common": sdr_common, "target": target, "bbox": bbox, "source_provenance": {"hdr_decode": hdr_decode, "om_decode": om_decode, "hdr_rgb_sha256": hashlib.sha256(hdr_raw.tobytes()).hexdigest().upper(), "om_rgb_sha256": hashlib.sha256(om_raw.tobytes()).hexdigest().upper()}})
        del hdr_raw, om_raw
    train_labels = protocol["temporal_sampling_and_partition"]["training_labels"]
    train_frames = [frame for frame in frames if frame["sample"]["label"] in train_labels]
    heldout = [frame for frame in frames if frame["sample"]["label"] == protocol["temporal_sampling_and_partition"]["held_out_temporal_label"]]
    require(len(train_frames) == 4 and len(heldout) == 1, "Protocol frame partition missing")
    started = time.perf_counter()
    h0_model = frozen.ModelECdfRegularized()
    h0_params = h0_model.fit(np.concatenate([frozen.compute_luminance(frame["sdr_common"]).reshape(-1) for frame in train_frames]), np.concatenate([frozen.compute_luminance(frame["target"]).reshape(-1) for frame in train_frames]))
    for frame in frames:
        frame["h0_full"], frame["target_y_full"] = frozen.map_luma_ratio(frame["sdr_full"], h0_params)
        x1, y1, x2, y2 = frame["bbox"]
        frame["h0_common"] = frame["h0_full"][y1:y2, x1:x2]
        frame["target_y_common"] = frame["target_y_full"][y1:y2, x1:x2]
    h0_elapsed = time.perf_counter() - started
    fitted = pooled_fit_mmr(frozen, train_frames)
    theta = fitted["selected"]["coefficients"]
    theta_hash = hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest().upper()
    frame_rows: list[dict[str, Any]] = []
    for frame in frames:
        mmr_full, bounds = frozen.apply_mmr(frame["h0_full"], frame["target_y_full"], theta, "MMR1")
        x1, y1, x2, y2 = frame["bbox"]
        mmr_common = mmr_full[y1:y2, x1:x2]
        h0_metrics = frozen.metrics(frame["h0_common"], frame["target"])
        mmr_metrics = frozen.metrics(mmr_common, frame["target"])
        h0_seam = frozen.seam_metrics(frame["h0_full"], frame["target"], frame["bbox"])
        mmr_seam = frozen.seam_metrics(mmr_full, frame["target"], frame["bbox"])
        improvement = 100.0 * (h0_metrics["deltaE2000"]["mean"] - mmr_metrics["deltaE2000"]["mean"]) / max(h0_metrics["deltaE2000"]["mean"], 1e-12)
        seam_ok = all(mmr_seam["boundaries"][index]["luma_mean_nits"] <= h0_seam["boundaries"][index]["luma_mean_nits"] * 1.10 for index in (0, 1))
        bounded = bool(bounds["chroma_bound_pixel_fraction"] <= 0.005 and bounds["inverse_negative_pixel_fraction"] <= 0.005 and bounds["finite"] and bounds["Y_target_max_abs_error_nits"] <= 1e-5)
        artifacts = write_frame_artifacts(frozen, shot_dir, frame, frame["h0_full"], mmr_full, bounds)
        independent = None
        if frame["sample"]["label"] in protocol["temporal_sampling_and_partition"]["independent_per_frame_fit_diagnostic_labels"]:
            diagnostic = frozen.fit_mmr(frame["h0_common"], frame["target"], frame["target_y_common"], "MMR1")
            independent = {"diagnostic_only": True, "selected_lambda": diagnostic["selected"]["lambda"], "selected_coefficients": diagnostic["selected"]["coefficients"], "coefficient_distance_from_shared_l2": float(np.linalg.norm(diagnostic["selected"]["coefficients"] - theta)), "full_reference_metrics_if_leakage_fit_were_allowed": diagnostic["selected"]["holdout_metrics"], "warning": "This fit sees the held-out temporal frame and is never used for output, scoring, or model selection."}
        frame_rows.append({"label": frame["sample"]["label"], "fraction": frame["sample"]["fraction"], "frame_pair": [frame["sample"]["hdr_frame"], frame["sample"]["om_frame"]], "role": "temporal_holdout" if frame in heldout else "training_frame", "shared_coefficient_sha256": theta_hash, "H0_metrics": h0_metrics, "MMR1_shared_metrics": mmr_metrics, "deltaE2000_improvement_vs_H0_percent": improvement, "H0_seam": h0_seam, "MMR1_shared_seam": mmr_seam, "bounds": bounds, "seam_not_severely_worse": seam_ok, "bounded_and_finite": bounded, "phase5_style_per_frame_temporal_gate": bool(improvement >= 5.0 and seam_ok and bounded), "independent_frame_fit_diagnostic": independent, "artifacts": artifacts, "source_provenance": frame["source_provenance"]})
        del mmr_full, mmr_common
    heldout_row = next(row for row in frame_rows if row["role"] == "temporal_holdout")
    return {"status": "SCORABLE", "case_id": case["case_id"], "material": case["material"], "conditional_geometry": float(case["geometry"]["confidence"]) < 0.95, "geometry_confidence": float(case["geometry"]["confidence"]), "discovery": discovery, "samples": samples, "shared_transform": {"H0_fit_training_labels": train_labels, "H0_parameters": h0_params, "H0_fit_seconds_cpu": h0_elapsed, "MMR1": fitted, "coefficients_sha256": theta_hash, "application": "The exact selected coefficient matrix was applied unchanged to all five frames."}, "frames": frame_rows, "heldout_90pct": heldout_row, "artifacts_root": str(shot_dir)}


def contact_sheet(frozen: Any, cases: list[dict[str, Any]]) -> str | None:
    previews = []
    for case in cases:
        if case.get("status") != "SCORABLE":
            continue
        path = Path(case["heldout_90pct"]["artifacts"]["montage"])
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        if image is not None:
            previews.append(image)
    if not previews:
        return None
    width = max(image.shape[1] for image in previews)
    normalized = [cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for image in previews]
    path = OUTPUT / "phase6_heldout_contact_sheet.png"
    frozen.write_cv_image(path, np.concatenate(normalized, axis=0))
    return str(path)


def report(data: dict[str, Any]) -> str:
    scored = [item for item in data["cases"] if item["status"] == "SCORABLE"]
    matrix = [item for item in scored if item["material"] == "The Matrix"]
    br = [item for item in scored if item["material"] == "BR2049"]
    matrix_pass = bool(matrix) and all(item["heldout_90pct"]["phase5_style_per_frame_temporal_gate"] for item in matrix)
    lines = ["# Phase 6 — Real Scene-Level Validation of Frozen MMR-1", "", f"**Final status:** `{data['final_status']}`", "", "Research-only bounded validation. No MMR-2, alternate model family, production code, P2.32 data, sync offset, geometry, full-film scan, or full-shot render was created.", "", "## Frozen contract and protocol", "", "The immutable protocol is `PHASE6_FROZEN_PROTOCOL.json`. Startup verified SHA-256 hashes for the frozen Phase-5 MMR source, H0/output helpers, and P2.32 inputs; it also asserted the 12-parameter MMR-1 feature set, lambda grid `(1e-4, 1e-3, 1e-2)`, ±0.5 chroma limit, 4096/2048 spatial diagnostic split, identity ridge prior, and final Model-E luminance reimposition.", "", "Local scene evidence is deliberately modest: each fixed anchor used one HDR-only ±5 s, 320-pixel proxy window with P2.34's deterministic histogram/luminance/edge boundary score. A shot is accepted only with a boundary on both sides inside that window. No missing boundary was inferred or widened. The P2.32 temporal labels remain unverified and are not reported as film-wide shot IDs.", "", "## Per-anchor outcomes", "", "| Anchor | Local grouping | Frames | 90% ΔE H0 → shared MMR-1 | ΔE improvement | 90% gate | Geometry role |", "|---|---|---:|---:|---:|---|---|"]
    for item in data["cases"]:
        if item["status"] != "SCORABLE":
            lines.append(f"| {item['case_id']} | {item['status']} | 0 | n/a | n/a | n/a | {'conditional' if item.get('conditional_geometry') else 'primary'} |")
            continue
        held = item["heldout_90pct"]
        h0 = held["H0_metrics"]["deltaE2000"]["mean"]; mmr = held["MMR1_shared_metrics"]["deltaE2000"]["mean"]
        lines.append(f"| {item['case_id']} | LOCAL_SHOT_RESOLVED | 5 | {h0:.6f} → {mmr:.6f} | {held['deltaE2000_improvement_vs_H0_percent']:+.3f}% | {held['phase5_style_per_frame_temporal_gate']} | {'conditional, confidence 0.9155' if item['conditional_geometry'] else 'primary Matrix'} |")
    lines += ["", "## One-transform consistency", "", "For every scorable anchor, one pooled Model-E luminance parameter set was fit only from the 10/25/50/75% frames. One MMR-1 coefficient matrix was then fit from those same training frames using frozen Phase-5 equations and constraints. Every frame record retains the same coefficient SHA-256; the 90% temporal holdout did not contribute to either fit. Independent fitting of the 90% frame is retained solely as a leakage-labelled diagnostic, never as output or model selection.", "", "## Eight final questions", "", f"1. **One transform consistency:** {'Confirmed for every scorable case by identical coefficient hash on all five applications.' if scored else 'Not testable: no anchor satisfied the local-shot requirement.'}", f"2. **HDR reproduction versus H0:** {'The held-out 90% metrics above are the primary evidence; a pass requires ≥5% ΔE2000 improvement, bounded/finite output, and no >10% luminance seam worsening.' if scored else 'No temporal holdout could be scored.'}", f"3. **SDR/DVD→HDR remaster visual improvement:** {'Review the per-frame four-panel artifacts (SDR OM, HDR overlap canvas, H0, shared MMR-1); numerical overlap evidence does not claim HDR ground truth in OM-only pixels.' if scored else 'Not established without resolved Matrix shots.'}", f"4. **OM extension coherence:** {'Each scorable frame stores top/bottom H0/MMR seam metrics and seam strips; only continuity is assessed outside the HDR overlap.' if scored else 'Not established.'}", f"5. **Artifacts:** metrics JSON, per-anchor discovery traces, extraction provenance/hashes, scientific 90% H0/MMR arrays, five temporal montages, luminance-difference maps, seams, and the held-out contact sheet are under `results/`.", f"6. **Multi-frame stability:** {'Measured across all five frames under the exact same transform; per-frame metrics and the diagnostic independent 90% fit quantify transfer rather than per-frame adaptation.' if scored else 'Insufficient data.'}", f"7. **Near-identity BR2049:** {'Reported only as a conditional transformation-fidelity result because geometry confidence is 0.9155 (<0.95); it cannot confirm creative-grade accuracy.' if br else 'Not available.'}", f"8. **Readiness for a controlled full-shot render:** {'Numerically eligible only for bounded follow-up review, never automatically rendered by this phase.' if matrix_pass else 'NOT READY: all three Matrix anchors must resolve locally and pass the held-out gate before a controlled render can be considered.'}", "", "## Synthetic control", "", "The retained synthetic MMR-1 result reuses the frozen Phase-5 known-grade hidden-region diagnostic. It verifies frozen-model mechanics only; it is not evidence that real OM-only regions have HDR ground truth.", "", "## Scope stop", "", "STOP after this validation. The result does not authorize another model, automated production integration, full-film processing, or a full-shot render."]
    return "\n".join(lines) + "\n"


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(protocol["status"] == "PREDECLARED BEFORE EXTRACTION", "Protocol was not predeclared")
    require(FFMPEG.exists(), f"Missing local FFmpeg: {FFMPEG}")
    frozen = load_frozen_module(protocol)
    data = json.loads(P232_METRICS.read_text(encoding="utf-8"))
    require(data.get("phase") == "P2.32" and data.get("status") == "DATASET READY", "P2.32 unavailable")
    records = {record["case_id"]: record for material in data["case_records"].values() for record in material}
    cases: list[dict[str, Any]] = []
    for anchor in protocol["anchors"]:
        case = records[anchor["case_id"]]
        discovery = discover_local_shot(case, protocol)
        discovery_path = OUTPUT / case["case_id"] / "local_shot_discovery.json"
        write_json(discovery_path, discovery)
        samples = select_samples(discovery, protocol, int(case["offset_frames"]))
        if discovery["shot_status"] != "LOCAL_SHOT_RESOLVED":
            cases.append({"status": discovery["shot_status"], "case_id": case["case_id"], "material": case["material"], "conditional_geometry": anchor["conditional_geometry"], "geometry_confidence": anchor["geometry_confidence"], "discovery": discovery, "discovery_artifact": str(discovery_path), "reason": "Bounded local window did not prove both shot boundaries."})
            continue
        if len(samples) != protocol["temporal_sampling_and_partition"]["minimum_samples_required"]:
            cases.append({"status": "INSUFFICIENT_SEPARATED_SAMPLES", "case_id": case["case_id"], "material": case["material"], "conditional_geometry": anchor["conditional_geometry"], "geometry_confidence": anchor["geometry_confidence"], "discovery": discovery, "discovery_artifact": str(discovery_path), "reason": "The fixed five temporal samples could not be separated without protocol fallback."})
            continue
        try:
            result = fit_and_evaluate_shot(frozen, case, discovery, samples, protocol)
            result["discovery_artifact"] = str(discovery_path)
            cases.append(result)
        except Exception as exc:
            cases.append({"status": "SAMPLE_EXTRACTION_FAILED", "case_id": case["case_id"], "material": case["material"], "conditional_geometry": anchor["conditional_geometry"], "geometry_confidence": anchor["geometry_confidence"], "discovery": discovery, "discovery_artifact": str(discovery_path), "reason": str(exc)})
        gc.collect()
    # Existing frozen diagnostic, deliberately not a new model or a real-material transform.
    synthetic = frozen.synthetic_mmr("MMR1")
    sheet = contact_sheet(frozen, cases)
    matrix = [item for item in cases if item["material"] == "The Matrix"]
    all_matrix_scorable = len(matrix) == 3 and all(item["status"] == "SCORABLE" for item in matrix)
    all_matrix_gate = all_matrix_scorable and all(item["heldout_90pct"]["phase5_style_per_frame_temporal_gate"] for item in matrix)
    final_status = "READY_FOR_CONTROLLED_REVIEW_ONLY" if all_matrix_gate else "INSUFFICIENT_DATA_OR_GATE_FAILURE"
    result = {"phase": "Phase 6", "status": "COMPLETE", "final_status": final_status, "scope": protocol["scope"], "protocol": str(PROTOCOL_PATH), "frozen_equivalence": {"mmr1_source_hash": sha256(ROOT / protocol["frozen_dependency_hashes_sha256"]["mmr1_source"]["path"]), "assertions_passed": True, "frozen_model": "MMR1", "parameter_count": 12}, "cases": cases, "synthetic_control": synthetic, "artifacts": {"metrics": str(METRICS_PATH), "report": str(REPORT_PATH), "contact_sheet": sheet, "root": str(OUTPUT)}, "decision": {"all_matrix_primary_cases_scorable": all_matrix_scorable, "all_matrix_heldout_gates_pass": all_matrix_gate, "controlled_full_shot_render_performed": False, "production_modified": False, "stop_after_validation": True}}
    write_json(METRICS_PATH, result)
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    print("SCENE_LEVEL_VALIDATION_STATUS=COMPLETE")
    print(f"FINAL_STATUS={final_status}")
    print(f"METRICS={METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
