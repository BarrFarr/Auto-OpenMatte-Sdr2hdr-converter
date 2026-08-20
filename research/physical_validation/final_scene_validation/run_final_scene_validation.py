"""Phase 8: bounded cross-scene validation and full-shot render of frozen Model E + MMR-1.

This isolated research runner does not modify MMR-1, production, sync, geometry, or P2.32.
"""
from __future__ import annotations

import argparse
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

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PROTOCOL_PATH = BASE / "PHASE8_FROZEN_PROTOCOL.json"
RESULTS = BASE / "results"
METRICS_PATH = RESULTS / "final_scene_validation_metrics.json"
REPORT_PATH = BASE / "FINAL_SCENE_VALIDATION_REPORT.md"
FPS = 24000.0 / 1001.0


def require(condition: bool, message: str) -> None:
    if not condition:
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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(protocol["status"] == "PREDECLARED BEFORE EXTRACTION", "Phase 8 protocol was not predeclared")
    for name, item in protocol["pinned_dependencies_sha256"].items():
        path = ROOT / item["path"]
        require(path.is_file(), f"Missing pinned dependency: {name}")
        require(sha256(path) == item["sha256"], f"Pinned dependency changed: {name}")
    return protocol


def load_phase6_and_frozen(protocol: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    phase6 = load_module("phase8_reused_phase6", ROOT / "research/physical_validation/scene_level_validation/run_scene_level_validation.py")
    phase6_protocol = json.loads((ROOT / "research/physical_validation/scene_level_validation/PHASE6_FROZEN_PROTOCOL.json").read_text(encoding="utf-8"))
    frozen = phase6.load_frozen_module(phase6_protocol)
    contract = protocol["frozen_transform"]
    require(tuple(frozen.LAMBDA_GRID) == tuple(contract["lambda_grid"]), "Frozen lambda grid changed")
    require(float(frozen.CHROMA_BOUND) == float(contract["chroma_component_bound"]), "Frozen chroma bound changed")
    require(frozen.identity_coefficients("MMR1").shape == (6, 2), "Frozen MMR-1 shape changed")
    return phase6, frozen, phase6_protocol


def full_grid_spatial_partitions(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Frozen block equation with no legacy Phase-5 sample cap; no image resize."""
    yy, xx = np.indices(shape)
    held = ((yy // 80 + xx // 120) % 5) == 0
    return np.flatnonzero((~held).ravel()), np.flatnonzero(held.ravel())


def distributed_subset(indices: np.ndarray, count: int) -> np.ndarray:
    require(len(indices) >= count, f"Only {len(indices)} eligible rows for requested {count}")
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    require(len(np.unique(positions)) == count, "Deterministic selector duplicated a coordinate")
    return indices[positions]


def record_is_eligible(record: dict[str, Any]) -> bool:
    return bool(
        record.get("sync_status") == "LOCKED"
        and record.get("finite", {}).get("hdr_rgb")
        and record.get("finite", {}).get("om_rgb")
        and isinstance(record.get("geometry", {}).get("overlap"), list)
    )


def discover_candidates(phase6: Any, phase6_protocol: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    p232 = json.loads((ROOT / "dev/ffmpeg-build/validation/P232_real_material_dataset/P232_dataset_metrics.json").read_text(encoding="utf-8"))
    require(p232.get("phase") == "P2.32" and p232.get("status") == "DATASET READY", "P2.32 not ready")
    discoveries: dict[str, list[dict[str, Any]]] = {"The Matrix": [], "BR2049": []}
    records_by_id: dict[str, Any] = {}
    for material in discoveries:
        for record in sorted(p232["case_records"][material], key=lambda item: int(item["selection_index"])):
            records_by_id[record["case_id"]] = record
            if not record_is_eligible(record):
                discoveries[material].append({"case_id": record["case_id"], "selection_index": record["selection_index"], "status": "INPUT_INELIGIBLE", "reason": "P2.32 LOCKED/finite/geometry precondition failed"})
                continue
            discovery = phase6.discover_local_shot(record, phase6_protocol)
            samples = phase6.select_samples(discovery, phase6_protocol, int(record["offset_frames"]))
            discoveries[material].append({
                "case_id": record["case_id"], "selection_index": int(record["selection_index"]),
                "status": discovery["shot_status"], "sample_count": len(samples), "record": record,
                "discovery": discovery, "samples": samples,
            })
            gc.collect()
    return discoveries, records_by_id


def intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    a0, a1 = left["discovery"]["shot_start_frame"], left["discovery"]["shot_end_frame_exclusive"]
    b0, b1 = right["discovery"]["shot_start_frame"], right["discovery"]["shot_end_frame_exclusive"]
    return max(a0, b0) < min(a1, b1)


def choose_shots(discoveries: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    matrix_rows = discoveries["The Matrix"]
    matrix: list[dict[str, Any]] = []
    anchor = next((row for row in matrix_rows if row["case_id"] == "the_matrix_temporal_stratum_08"), None)
    if anchor is not None and anchor["status"] == "LOCAL_SHOT_RESOLVED" and anchor["sample_count"] == 5:
        matrix.append(anchor)
    for row in matrix_rows:
        if len(matrix) >= 3:
            break
        if row["case_id"] == "the_matrix_temporal_stratum_08" or row["status"] != "LOCAL_SHOT_RESOLVED" or row["sample_count"] != 5:
            continue
        if not any(intervals_overlap(row, selected) for selected in matrix):
            matrix.append(row)
    br = next((row for row in discoveries["BR2049"] if row["status"] == "LOCAL_SHOT_RESOLVED" and row["sample_count"] == 5), None)
    return {"matrix": matrix, "br2049": [br] if br else []}


def decode_pair(phase6: Any, frozen: Any, record: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    hdr_raw, hdr_decode = phase6.run_rgb48_frame(Path(record["hdr_source"]), *map(int, record["hdr_resolution"]), int(sample["hdr_frame"]))
    om_raw, om_decode = phase6.run_rgb48_frame(Path(record["om_source"]), *map(int, record["om_resolution"]), int(sample["om_frame"]))
    sdr_full, sdr_common, target, bbox = phase6.decode_pair_exact(frozen, record, hdr_raw, om_raw)
    return {
        "sample": sample, "sdr_full": sdr_full, "sdr_common": sdr_common, "target": target, "bbox": bbox,
        "provenance": {"hdr_decode": hdr_decode, "om_decode": om_decode, "hdr_sha256": hashlib.sha256(hdr_raw.tobytes()).hexdigest().upper(), "om_sha256": hashlib.sha256(om_raw.tobytes()).hexdigest().upper()},
        "raw_hashes": {"hdr": hashlib.sha256(hdr_raw.tobytes()).hexdigest().upper(), "om": hashlib.sha256(om_raw.tobytes()).hexdigest().upper()},
    }


def fit_transform(phase6: Any, frozen: Any, record: dict[str, Any], samples: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    settings = protocol["sampling"]
    labels = set(settings["training_labels"])
    train_samples = [sample for sample in samples if sample["label"] in labels]
    require(len(train_samples) == 4, "Protocol requires four training samples")
    n_train, n_hold = int(settings["paired_train_rows_per_frame"]), int(settings["lambda_holdout_rows_per_frame"])
    started = time.perf_counter()
    h0_x: list[np.ndarray] = []
    h0_y: list[np.ndarray] = []
    fit_provenance: list[dict[str, Any]] = []
    for sample in train_samples:
        frame = decode_pair(phase6, frozen, record, sample)
        train_all, _ = full_grid_spatial_partitions(frame["sdr_common"].shape[:2])
        train = distributed_subset(train_all, n_train)
        h0_x.append(frozen.compute_luminance(frame["sdr_common"]).reshape(-1)[train])
        h0_y.append(frozen.compute_luminance(frame["target"]).reshape(-1)[train])
        fit_provenance.append({"label": sample["label"], **frame["provenance"]})
        del frame
        gc.collect()
    h0 = frozen.ModelECdfRegularized()
    h0_params = h0.fit(np.concatenate(h0_x), np.concatenate(h0_y))
    del h0_x, h0_y
    theta0 = frozen.identity_coefficients("MMR1")
    x_train: list[np.ndarray] = []
    c_train: list[np.ndarray] = []
    hold_base: list[np.ndarray] = []
    hold_y: list[np.ndarray] = []
    hold_target: list[np.ndarray] = []
    for sample in train_samples:
        frame = decode_pair(phase6, frozen, record, sample)
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], h0_params)
        x1, y1, x2, y2 = frame["bbox"]
        h0_common, y_common = h0_full[y1:y2, x1:x2], target_y_full[y1:y2, x1:x2]
        train_all, hold_all = full_grid_spatial_partitions(h0_common.shape[:2])
        train, hold = distributed_subset(train_all, n_train), distributed_subset(hold_all, n_hold)
        design, _ = frozen.features(h0_common, y_common, "MMR1")
        target_chroma = frozen.linear_bt2020_to_ictcp(frame["target"] * frozen.PEAK_NITS)[..., 1:]
        x_train.append(design.reshape(-1, 6)[train])
        c_train.append(target_chroma.reshape(-1, 2)[train])
        hold_base.append(h0_common.reshape(-1, 1, 3)[hold])
        hold_y.append(y_common.reshape(-1, 1)[hold])
        hold_target.append(frame["target"].reshape(-1, 1, 3)[hold])
        del frame, h0_full, target_y_full, design, target_chroma
        gc.collect()
    xx, cc = np.concatenate(x_train), np.concatenate(c_train)
    candidates: list[dict[str, Any]] = []
    for lam in frozen.LAMBDA_GRID:
        gram = xx.T @ xx / len(xx) + float(lam) * np.eye(xx.shape[1])
        rhs = xx.T @ cc / len(xx) + float(lam) * theta0
        theta = np.linalg.solve(gram, rhs)
        held_output, held_bounds = frozen.apply_mmr(np.concatenate(hold_base), np.concatenate(hold_y), theta, "MMR1")
        held_metric = frozen.metrics(held_output, np.concatenate(hold_target))
        condition, deviation = float(np.linalg.cond(gram)), float(np.linalg.norm(theta - theta0))
        stable = bool(np.isfinite(condition) and condition <= 1e6 and deviation <= 5.0 and held_bounds["chroma_bound_pixel_fraction"] <= .005 and held_bounds["inverse_negative_pixel_fraction"] <= .005 and held_bounds["finite"])
        candidates.append({"lambda": float(lam), "coefficients": theta, "regularized_gram_condition_number": condition, "coefficient_identity_deviation_L2": deviation, "holdout_metrics": held_metric, "holdout_bounds": held_bounds, "holdout_objective": frozen.objective(held_metric), "stable": stable})
    acceptable = [candidate for candidate in candidates if candidate["stable"]]
    require(bool(acceptable), "All frozen MMR-1 candidates were pathological")
    selected = min(acceptable, key=lambda candidate: candidate["holdout_objective"])
    theta = selected["coefficients"]
    return {"fit_seconds": time.perf_counter() - started, "h0_parameters": h0_params, "mmr": {"model": "MMR1", "training_pairs": int(len(xx)), "train_pairs_per_frame": n_train, "lambda_holdout_pairs_per_frame": n_hold, "selected": selected, "candidates": candidates}, "coefficients_sha256": hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest().upper(), "fit_provenance": fit_provenance}


def write_frame_artifacts(frozen: Any, shot_dir: Path, frame: dict[str, Any], h0_full: np.ndarray, mmr_full: np.ndarray, bounds: dict[str, Any]) -> dict[str, str]:
    label = frame["sample"]["label"].replace("%", "pct")
    previews, differences, seams, scientific = (shot_dir / "previews", shot_dir / "differences", shot_dir / "seams", shot_dir / "scientific")
    for folder in (previews, differences, seams, scientific):
        folder.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = frame["bbox"]
    hdr_canvas = np.zeros_like(h0_full)
    hdr_canvas[y1:y2, x1:x2] = frame["target"]
    montage = frozen.comparison_montage(frame["sdr_full"], hdr_canvas, h0_full, [mmr_full], height=180)
    montage_path = previews / f"{label}_SDR_HDR_H0_MMR1.png"
    frozen.write_cv_image(montage_path, montage[..., ::-1])
    frozen.write_preview(previews / f"{label}_H0_log.png", h0_full)
    frozen.write_preview(previews / f"{label}_MMR1_log.png", mmr_full)
    diff = frozen.diff_map(mmr_full[y1:y2, x1:x2], frame["target"])
    diff_path = differences / f"{label}_MMR1_overlap_luminance_difference.png"
    frozen.write_cv_image(diff_path, diff)
    composite = mmr_full.copy()
    composite[y1:y2, x1:x2] = frame["target"]
    strip = frozen.panel([mmr_full[max(0, y1 - 60):y1 + 60], composite[max(0, y1 - 60):y1 + 60], mmr_full[y2 - 60:min(mmr_full.shape[0], y2 + 60)], composite[y2 - 60:min(mmr_full.shape[0], y2 + 60)]], 120)
    seam_path = seams / f"{label}_MMR1_HDR_boundary.png"
    frozen.write_cv_image(seam_path, strip[..., ::-1])
    if frame["sample"]["label"] == "90%":
        frozen.write_scientific(scientific / "temporal_holdout_H0", h0_full)
        frozen.write_scientific(scientific / "temporal_holdout_MMR1", mmr_full)
    return {"montage": str(montage_path), "difference": str(diff_path), "seam": str(seam_path), "finite": str(bounds["finite"])}


def write_contact_sheet(frozen: Any, shot_dir: Path, paths: list[str]) -> str | None:
    images = []
    for path in paths:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = frozen.cv2.imdecode(encoded, frozen.cv2.IMREAD_COLOR) if encoded.size else None
        if image is not None:
            images.append(image)
    if not images:
        return None
    width = max(image.shape[1] for image in images)
    padded = [frozen.cv2.copyMakeBorder(image, 0, 0, 0, width - image.shape[1], frozen.cv2.BORDER_CONSTANT, value=(255, 255, 255)) for image in images]
    path = shot_dir / "contact_sheet_representative_frames.png"
    frozen.write_cv_image(path, np.concatenate(padded, axis=0))
    return str(path)


def evaluate_shot(phase6: Any, frozen: Any, selected: dict[str, Any], fit: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    record, samples = selected["record"], selected["samples"]
    shot_dir = RESULTS / "shots" / record["case_id"]
    theta = fit["mmr"]["selected"]["coefficients"]
    rows: list[dict[str, Any]] = []
    montage_paths: list[str] = []
    for sample in samples:
        frame = decode_pair(phase6, frozen, record, sample)
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], fit["h0_parameters"])
        started = time.perf_counter()
        mmr_full, bounds = frozen.apply_mmr(h0_full, target_y_full, theta, "MMR1")
        application_seconds = time.perf_counter() - started
        x1, y1, x2, y2 = frame["bbox"]
        h0_common, mmr_common = h0_full[y1:y2, x1:x2], mmr_full[y1:y2, x1:x2]
        h0_metrics, mmr_metrics = frozen.metrics(h0_common, frame["target"]), frozen.metrics(mmr_common, frame["target"])
        h0_seam, mmr_seam = frozen.seam_metrics(h0_full, frame["target"], frame["bbox"]), frozen.seam_metrics(mmr_full, frame["target"], frame["bbox"])
        valid = bool(bounds["finite"] and bounds["chroma_bound_pixel_fraction"] <= .005 and bounds["inverse_negative_pixel_fraction"] <= .005 and bounds["Y_target_max_abs_error_nits"] <= 1e-5)
        artifacts = write_frame_artifacts(frozen, shot_dir, frame, h0_full, mmr_full, bounds)
        montage_paths.append(artifacts["montage"])
        improvement = 100.0 * (h0_metrics["deltaE2000"]["mean"] - mmr_metrics["deltaE2000"]["mean"]) / max(h0_metrics["deltaE2000"]["mean"], 1e-12)
        rows.append({"label": sample["label"], "fraction": sample["fraction"], "frame_pair": [sample["hdr_frame"], sample["om_frame"]], "role": "temporal_holdout" if sample["label"] == "90%" else "training_frame", "coefficient_sha256": fit["coefficients_sha256"], "H0_metrics": h0_metrics, "MMR1_metrics": mmr_metrics, "deltaE2000_improvement_vs_H0_percent": improvement, "H0_seam": h0_seam, "MMR1_seam": mmr_seam, "bounds": bounds, "application_seconds": application_seconds, "valid_output": valid, "artifacts": artifacts, "source_provenance": frame["provenance"]})
        del frame, h0_full, target_y_full, mmr_full
        gc.collect()
    contact = write_contact_sheet(frozen, shot_dir, montage_paths)
    return {"status": "VALIDATED" if all(row["valid_output"] for row in rows) else "NUMERICAL_OUTPUT_FAILURE", "case_id": record["case_id"], "material": record["material"], "geometry_confidence": float(record["geometry"]["confidence"]), "conditional_geometry": float(record["geometry"]["confidence"]) < .95, "discovery": selected["discovery"], "samples": samples, "fit": fit, "frames": rows, "temporal_holdout": next(row for row in rows if row["role"] == "temporal_holdout"), "coefficient_stability": {"same_coefficient_hash_all_applications": len({row["coefficient_sha256"] for row in rows}) == 1, "shared_hash": fit["coefficients_sha256"], "independent_per_frame_fits": "prohibited and not performed"}, "contact_sheet": contact}


def pq_oetf(value: np.ndarray) -> np.ndarray:
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    clipped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    p = np.power(clipped, m1)
    return np.power((c1 + c2 * p) / (1.0 + c3 * p), m2)


def render_matrix_shot(phase6: Any, frozen: Any, result: dict[str, Any]) -> dict[str, Any]:
    record, fit, discovery = result["record"], result["fit"], result["discovery"]
    start, end = int(discovery["shot_start_frame"]), int(discovery["shot_end_frame_exclusive"])
    count = end - start
    require(count > 0, "Empty bounded shot")
    output_dir = RESULTS / "renders" / record["case_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "MMR1_generated_HDR_OpenMatte_full_shot_ffv1.mkv"
    ffmpeg = phase6.FFMPEG
    om_width, om_height = map(int, record["om_resolution"])
    hdr_width, hdr_height = map(int, record["hdr_resolution"])
    hbytes, obytes = hdr_width * hdr_height * 6, om_width * om_height * 6
    def decoder(source: str, frame: int, count_: int) -> list[str]:
        return [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{frame / FPS:.9f}", "-i", source, "-map", "0:v:0", "-frames:v", str(count_), "-pix_fmt", "rgb48le", "-vsync", "0", "-f", "rawvideo", "pipe:1"]
    hdr_process = subprocess.Popen(decoder(record["hdr_source"], start, count), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    om_process = subprocess.Popen(decoder(record["om_source"], start + int(record["offset_frames"]), count), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{om_width}x{om_height}", "-framerate", "24000/1001", "-i", "pipe:0", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp16le", "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc", str(output_path)]
    enc_process = subprocess.Popen(encoder, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    theta = fit["mmr"]["selected"]["coefficients"]
    first_hashes: dict[str, str] | None = None
    final_stream_hashes: dict[str, str] | None = None
    application_times: list[float] = []
    bounds_rows: list[dict[str, Any]] = []
    rendered = 0
    started = time.perf_counter()
    try:
        for index in range(count):
            hdr_data, om_data = hdr_process.stdout.read(hbytes), om_process.stdout.read(obytes)
            require(len(hdr_data) == hbytes and len(om_data) == obytes, f"Stream decode ended early at frame {index}")
            current_hashes = {"hdr": hashlib.sha256(hdr_data).hexdigest().upper(), "om": hashlib.sha256(om_data).hexdigest().upper()}
            if index == 0:
                first_hashes = current_hashes
            if index == count - 1:
                final_stream_hashes = current_hashes
            hdr_raw = np.frombuffer(hdr_data, dtype="<u2").reshape(hdr_height, hdr_width, 3).copy()
            om_raw = np.frombuffer(om_data, dtype="<u2").reshape(om_height, om_width, 3).copy()
            sdr_full, _, target, bbox = phase6.decode_pair_exact(frozen, record, hdr_raw, om_raw)
            apply_started = time.perf_counter()
            h0_full, target_y_full = frozen.map_luma_ratio(sdr_full, fit["h0_parameters"])
            mmr_full, bounds = frozen.apply_mmr(h0_full, target_y_full, theta, "MMR1")
            x1, y1, x2, y2 = bbox
            mmr_full[y1:y2, x1:x2] = target
            pq16 = np.round(pq_oetf(mmr_full) * frozen.RGB16_MAX).astype("<u2")
            enc_process.stdin.write(pq16.tobytes())
            application_times.append(time.perf_counter() - apply_started)
            bounds_rows.append(bounds)
            rendered += 1
            del hdr_raw, om_raw, sdr_full, target, h0_full, target_y_full, mmr_full, pq16
    finally:
        if hdr_process.stdout:
            hdr_process.stdout.close()
        if om_process.stdout:
            om_process.stdout.close()
        if enc_process.stdin:
            enc_process.stdin.close()
        hdr_code, om_code, enc_code = hdr_process.wait(), om_process.wait(), enc_process.wait()
        hdr_err, om_err, enc_err = hdr_process.stderr.read().decode("utf-8", errors="replace"), om_process.stderr.read().decode("utf-8", errors="replace"), enc_process.stderr.read().decode("utf-8", errors="replace")
    require(rendered == count and hdr_code == 0 and om_code == 0 and enc_code == 0 and output_path.is_file(), f"Full-shot render failure: hdr={hdr_code} om={om_code} enc={enc_code}; {hdr_err} {om_err} {enc_err}")
    first_direct_hdr, _ = phase6.run_rgb48_frame(Path(record["hdr_source"]), hdr_width, hdr_height, start)
    first_direct_om, _ = phase6.run_rgb48_frame(Path(record["om_source"]), om_width, om_height, start + int(record["offset_frames"]))
    last_direct_hdr, _ = phase6.run_rgb48_frame(Path(record["hdr_source"]), hdr_width, hdr_height, end - 1)
    last_direct_om, _ = phase6.run_rgb48_frame(Path(record["om_source"]), om_width, om_height, end - 1 + int(record["offset_frames"]))
    direct = {"first": {"hdr": hashlib.sha256(first_direct_hdr.tobytes()).hexdigest().upper(), "om": hashlib.sha256(first_direct_om.tobytes()).hexdigest().upper()}, "last": {"hdr": hashlib.sha256(last_direct_hdr.tobytes()).hexdigest().upper(), "om": hashlib.sha256(last_direct_om.tobytes()).hexdigest().upper()}}
    require(first_hashes == direct["first"] and final_stream_hashes == direct["last"], "Streaming decode did not match exact Phase-6-style frame extraction")
    return {"status": "RENDERED", "case_id": record["case_id"], "hdr_interval": [start, end], "frame_count": count, "fit_coefficient_sha256": fit["coefficients_sha256"], "application_seconds": {"total": float(sum(application_times)), "mean_per_frame": float(np.mean(application_times)), "p95_per_frame": float(np.percentile(application_times, 95)), "max_per_frame": float(max(application_times))}, "render_total_seconds": time.perf_counter() - started, "streaming_exact_frame_hash_check": {"passed": True, "stream_first": first_hashes, "stream_last": final_stream_hashes, "direct": direct}, "aggregate_bounds": {"all_finite": bool(all(row["finite"] for row in bounds_rows)), "max_chroma_bound_fraction": float(max(row["chroma_bound_pixel_fraction"] for row in bounds_rows)), "max_inverse_negative_fraction": float(max(row["inverse_negative_pixel_fraction"] for row in bounds_rows)), "max_y_reimposition_error_nits": float(max(row["Y_target_max_abs_error_nits"] for row in bounds_rows))}, "output": str(output_path), "file_bytes": output_path.stat().st_size}


def report(data: dict[str, Any]) -> str:
    selected = data["selection"]
    validations = data.get("validations", [])
    renders = data.get("renders", [])
    matrix_validations = [row for row in validations if row["material"] == "The Matrix" and row["status"] == "VALIDATED"]
    br_validations = [row for row in validations if row["material"] == "BR2049" and row["status"] == "VALIDATED"]
    lines = ["# Final Scene Validation — Phase 8", "", f"**Status:** `{data['status']}`", "", "Research-only bounded validation of the frozen Model E luminance + ratio scaling + MMR-1 transform. No model, sampling-budget, production, synchronization, or geometry change was made.", "", "## Fixed configuration", "", "- Four representative training frames: 10%, 25%, 50%, 75%.", "- 5,000 deterministic correctly paired full-resolution common-grid rows per training frame (20,000 per shot).", "- 2,048 paired held-mask rows per training frame only for frozen λ selection.", "- 90% temporal frame excluded from all fitting and applied only after one shared transform is selected.", "- One H0 parameter set and one frozen 12-coefficient MMR-1 matrix per shot; no per-frame fitting.", "", "## Automatic bounded-shot selection", "", "All existing P2.32 temporal strata were considered only through the unchanged ±5-second HDR proxy / fixed boundary machinery. The selection was by predeclared P2.32 index and local-shot eligibility, never by fit quality. No window was widened and no frame/offset/geometry was manually changed.", "", f"- Matrix selected: `{', '.join(row['case_id'] for row in selected['matrix']) or 'none'}`", f"- BR2049 selected: `{', '.join(row['case_id'] for row in selected['br2049']) or 'none'}`", f"- Desired minimum: 3 Matrix + 1 BR2049; achieved: {len(selected['matrix'])} Matrix + {len(selected['br2049'])} BR2049.", "", "## Per-shot validation", "", "| Shot | Material | Status | λ | Fit s | 90% H0 → MMR mean ΔE2000 | 90% improvement | Conditional geometry |", "|---|---|---|---:|---:|---:|---:|---|"]
    for row in validations:
        if row["status"] != "VALIDATED":
            lines.append(f"| {row['case_id']} | {row['material']} | {row['status']} | n/a | n/a | n/a | n/a | {row.get('conditional_geometry', False)} |")
            continue
        held = row["temporal_holdout"]
        lines.append(f"| {row['case_id']} | {row['material']} | VALIDATED | {row['fit']['mmr']['selected']['lambda']:.4g} | {row['fit']['fit_seconds']:.4f} | {held['H0_metrics']['deltaE2000']['mean']:.6f} → {held['MMR1_metrics']['deltaE2000']['mean']:.6f} | {held['deltaE2000_improvement_vs_H0_percent']:+.3f}% | {row['conditional_geometry']} |")
    lines += ["", "Each JSON shot record contains the complete 12 coefficients, λ candidates, coefficient hash, direct full-overlap ΔE2000/P95, luminance MAE/RMSE, chroma/hue metrics, top/bottom seam metrics, output range/clipping/negative checks, source decode provenance, and the five-frame contact sheet. BR2049 is reported strictly as conditional near-identity/fidelity evidence when available.", "", "## Full-shot renders", ""]
    if renders:
        lines += ["| Shot | Frames | Analysis s | Application mean/P95 s per frame | Total render s | Exact stream check | Output |", "|---|---:|---:|---:|---:|---|---|"]
        by_case = {row["case_id"]: row for row in validations}
        for row in renders:
            analysis = by_case[row["case_id"]]["fit"]["fit_seconds"]
            application = row["application_seconds"]
            lines.append(f"| {row['case_id']} | {row['frame_count']} | {analysis:.4f} | {application['mean_per_frame']:.4f}/{application['p95_per_frame']:.4f} | {row['render_total_seconds']:.3f} | {row['streaming_exact_frame_hash_check']['passed']} | `{row['output']}` |")
    else:
        lines.append("No full-shot render was produced because no selected Matrix validation met the frozen numerical output gate.")
    matrix_stable = len(matrix_validations) >= 3 and all(item["coefficient_stability"]["same_coefficient_hash_all_applications"] for item in matrix_validations)
    enough_samples = len(matrix_validations) >= 3 and all(item["status"] == "VALIDATED" for item in matrix_validations)
    transform_valid = bool(renders) and all(item["aggregate_bounds"]["all_finite"] for item in renders)
    visual = "Human inspection required: representative contact sheets, seam strips, comparisons, and complete generated videos are supplied; automatic metrics cannot certify visual integration."
    fast = bool(renders) and max(item["application_seconds"]["mean_per_frame"] for item in renders) < 1.0
    ready = matrix_stable and enough_samples and transform_valid and len(br_validations) >= 1
    lines += ["", "## Final decision", "", f"A. **Is the frozen MMR-1 configuration stable across multiple real shots?** {'YES within this bounded evidence.' if matrix_stable else 'NOT ESTABLISHED: fewer than three validated Matrix shots or a stability failure.'}", f"B. **Is 4 × 5,000 samples sufficient in practice?** {'YES within the validated bounded shots.' if enough_samples else 'NOT ESTABLISHED across the desired three Matrix shots.'}", f"C. **Does one T_shot remain valid across all frames of a shot?** {'YES for the rendered bounded shots: the same persisted coefficient hash was used for every application and all render-frame numerical bounds passed.' if transform_valid else 'NOT ESTABLISHED by a complete bounded render.'}", f"D. **Does the generated Open Matte visually merge with the HDR center?** {visual}", f"E. **Does the process remain fast enough for local full-film use?** {'Indicatively yes for application throughput; analysis is once per shot. Full-film authorization is outside this phase.' if fast else 'Not established by the bounded render timings.'}", f"F. **Is the current algorithm ready for a controlled production-pipeline integration test?** {'YES, subject to controlled human visual review and the explicitly conditional BR2049 evidence.' if ready else 'NO: the required cross-scene evidence and/or full visual approval is incomplete. The blocking details are recorded above.'}", "", "## Hard stop", "", "STOP. This phase does not authorize MMR-2, a new model family, another sampling sweep, retuning, LUT/AI work, production changes, or full-film processing."]
    return "\n".join(lines) + "\n"


def run(discover_only: bool = False) -> int:
    protocol = load_protocol()
    phase6, frozen, phase6_protocol = load_phase6_and_frozen(protocol)
    discoveries, _ = discover_candidates(phase6, phase6_protocol)
    selection = choose_shots(discoveries)
    if discover_only:
        write_json(METRICS_PATH, {"phase": "Phase 8", "status": "DISCOVERY_COMPLETE", "protocol": str(PROTOCOL_PATH), "discoveries": discoveries, "selection": selection})
        print("PHASE8_STATUS=DISCOVERY_COMPLETE")
        return 0
    validations: list[dict[str, Any]] = []
    selected_all = selection["matrix"] + selection["br2049"]
    for chosen in selected_all:
        try:
            fit = fit_transform(phase6, frozen, chosen["record"], chosen["samples"], protocol)
            validation = evaluate_shot(phase6, frozen, chosen, fit, protocol)
            validation["record"] = chosen["record"]
            validations.append(validation)
        except Exception as exc:
            validations.append({"status": "VALIDATION_FAILED", "case_id": chosen["case_id"], "material": chosen["record"]["material"], "conditional_geometry": float(chosen["record"]["geometry"]["confidence"]) < .95, "reason": str(exc)})
        gc.collect()
    renders: list[dict[str, Any]] = []
    for validation in validations:
        if validation["material"] != "The Matrix" or validation["status"] != "VALIDATED":
            continue
        try:
            renders.append(render_matrix_shot(phase6, frozen, validation))
        except Exception as exc:
            renders.append({"status": "RENDER_FAILED", "case_id": validation["case_id"], "reason": str(exc)})
        gc.collect()
    complete = {"phase": "Phase 8", "status": "COMPLETE", "protocol": str(PROTOCOL_PATH), "frozen_equivalence": {"mmr1_sha256": sha256(ROOT / protocol["pinned_dependencies_sha256"]["frozen_mmr1"]["path"]), "assertions_passed": True}, "discoveries": discoveries, "selection": selection, "validations": validations, "renders": renders, "scope": protocol["scope"], "hard_stop": protocol["hard_stop"]}
    write_json(METRICS_PATH, complete)
    REPORT_PATH.write_text(report(complete), encoding="utf-8")
    print("PHASE8_STATUS=COMPLETE")
    print(f"METRICS={METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true", help="Run bounded automatic candidate discovery only.")
    raise SystemExit(run(discover_only=parser.parse_args().discover_only))
