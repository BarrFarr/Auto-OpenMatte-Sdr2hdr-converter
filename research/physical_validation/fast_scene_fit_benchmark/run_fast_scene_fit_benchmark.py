"""Phase 7 research-only deterministic fast scene-fit benchmark for frozen MMR-1.

This runner never changes the frozen Phase-5 MMR-1 implementation. It re-decodes
only the fixed Matrix 08 pairs, proves the Phase-6 B4 reference can be recreated,
and then reduces only deterministic fitting rows before applying one transform to
all five fixed frames. It does not process a full film or alter production code.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PROTOCOL_PATH = BASE / "PHASE7_FROZEN_PROTOCOL.json"
OUTPUT = BASE / "results"
METRICS_PATH = OUTPUT / "fast_scene_fit_metrics.json"
REPORT_PATH = BASE / "FAST_SCENE_FIT_BENCHMARK_REPORT.md"
FPS = 24000.0 / 1001.0
PEAK_NITS = 10000.0
RGB16_MAX = 65535.0


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


def timestamp(frame: int) -> float:
    return float(frame / FPS)


class ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS_EX layout through the needed fields."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
PSAPI.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ProcessMemoryCounters),
    wintypes.DWORD,
]
PSAPI.GetProcessMemoryInfo.restype = wintypes.BOOL


def working_set_bytes() -> int:
    """Measured current resident working-set memory for this process on Windows."""
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    success = PSAPI.GetProcessMemoryInfo(
        KERNEL32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not success:
        error = ctypes.get_last_error()
        raise RuntimeError(f"GetProcessMemoryInfo failed while measuring RSS: {error}")
    return int(counters.WorkingSetSize)


class PeakRSS:
    """Sample Windows resident working-set memory during one candidate fit."""

    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while self._running:
            self.peak_bytes = max(self.peak_bytes, working_set_bytes())
            time.sleep(self.interval_seconds)

    def __enter__(self) -> "PeakRSS":
        self.peak_bytes = working_set_bytes()
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_bytes = max(self.peak_bytes, working_set_bytes())


def verify_and_load(protocol: dict[str, Any]) -> Any:
    expected = protocol["frozen_dependency_hashes_sha256"]
    for label, entry in expected.items():
        path = ROOT / entry["path"]
        require(path.is_file(), f"Pinned dependency is absent: {label}: {path}")
        require(sha256(path) == entry["sha256"], f"Pinned dependency hash mismatch: {label}")
    source = ROOT / expected["mmr1_source"]["path"]
    source_parent = str(source.parent)
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)
    spec = importlib.util.spec_from_file_location("phase7_frozen_mmr1", source)
    require(spec is not None and spec.loader is not None, "Could not load frozen MMR-1 source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = protocol["frozen_mmr1_contract"]
    require(tuple(module.LAMBDA_GRID) == tuple(contract["lambda_grid"]), "Frozen lambda grid mismatch")
    require(module.CHROMA_BOUND == contract["chroma_component_bound"], "Frozen chroma bound mismatch")
    require(module.LUMA_FEATURE_SCALE_NITS == contract["luma_feature_scale_nits"], "Frozen feature luminance scale mismatch")
    require(module.identity_coefficients("MMR1").shape == (6, 2), "Frozen MMR-1 parameterization mismatch")
    return module


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(protocol["status"] == "PREDECLARED BEFORE EXTRACTION_OR_BENCHMARK", "Phase 7 protocol status mismatch")
    return protocol


def run_rgb48_frame(ffmpeg: Path, source: Path, width: int, height: int, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
    expected = width * height * 3 * 2
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{timestamp(frame):.9f}", "-i", str(source), "-map", "0:v:0", "-frames:v", "1", "-pix_fmt", "rgb48le", "-vsync", "0", "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    raw = completed.stdout
    metadata = {"command": command, "returncode": int(completed.returncode), "stderr": completed.stderr.decode("utf-8", errors="replace"), "stdout_bytes": len(raw), "expected_bytes": expected, "pixel_format": "rgb48le", "frame": int(frame), "timestamp_seconds": timestamp(frame)}
    require(completed.returncode == 0 and len(raw) >= expected, f"RGB48LE extraction failed: {metadata}")
    return np.frombuffer(raw[:expected], dtype="<u2").reshape(height, width, 3).copy(), metadata


def decode_pair_exact(frozen: Any, reference: dict[str, Any], hdr_raw: np.ndarray, om_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    require(hdr_raw.dtype == np.uint16 and om_raw.dtype == np.uint16, "Expected RGB48LE uint16 arrays")
    hdr_shape = (int(reference["hdr_resolution"][1]), int(reference["hdr_resolution"][0]), 3)
    om_shape = (
        int(reference["open_matte_resolution"][1]),
        int(reference["open_matte_resolution"][0]),
        3,
    )
    require(tuple(hdr_raw.shape) == hdr_shape, "Unexpected HDR source shape")
    require(tuple(om_raw.shape) == om_shape, "Unexpected Open-Matte source shape")
    x1, y1, x2, y2 = map(int, reference["overlap_open_matte"])
    hdr_code = frozen.cv2.resize(hdr_raw.astype(np.float64) / RGB16_MAX, (x2 - x1, y2 - y1), interpolation=frozen.cv2.INTER_LINEAR)
    target = frozen.pq_eotf_normalized(hdr_code)
    sdr_full = np.maximum(frozen.bt709_to_bt2020(frozen.bt1886_eotf(om_raw.astype(np.float64) / RGB16_MAX)), 0.0)
    sdr_common = sdr_full[y1:y2, x1:x2]
    require(sdr_common.shape == target.shape and np.isfinite(sdr_full).all() and np.isfinite(target).all(), "Invalid frozen-compatible decode")
    return sdr_full, sdr_common, target, (x1, y1, x2, y2)


def decode_fixed_frames(frozen: Any, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    reference = protocol["reference_case"]
    ffmpeg = ROOT / reference["ffmpeg"]
    require(ffmpeg.is_file(), f"Missing local FFmpeg: {ffmpeg}")
    frames: list[dict[str, Any]] = []
    for sample in reference["sample_pairs"]:
        hdr_raw, hdr_decode = run_rgb48_frame(ffmpeg, Path(reference["source_paths"]["hdr"]), *map(int, reference["hdr_resolution"]), int(sample["hdr_frame"]))
        om_raw, om_decode = run_rgb48_frame(ffmpeg, Path(reference["source_paths"]["open_matte"]), *map(int, reference["open_matte_resolution"]), int(sample["open_matte_frame"]))
        sdr_full, sdr_common, target, bbox = decode_pair_exact(frozen, reference, hdr_raw, om_raw)
        frames.append({"sample": sample, "sdr_full": sdr_full, "sdr_common": sdr_common, "target": target, "bbox": bbox, "source_provenance": {"hdr_decode": hdr_decode, "om_decode": om_decode, "hdr_rgb_sha256": hashlib.sha256(hdr_raw.tobytes()).hexdigest().upper(), "om_rgb_sha256": hashlib.sha256(om_raw.tobytes()).hexdigest().upper()}})
        del hdr_raw, om_raw
        gc.collect()
    return frames


def fixed_indices(shape: tuple[int, int], train_rows: int, hold_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Frozen 80x120 partition, with individual deterministic evenly distributed subsets."""
    yy, xx = np.indices(shape)
    held = ((yy // 80 + xx // 120) % 5) == 0

    def select(mask: np.ndarray, count: int) -> np.ndarray:
        available = np.flatnonzero(mask.ravel())
        require(len(available) >= count, f"Requested {count} deterministic rows but only {len(available)} are eligible")
        positions = np.linspace(0, len(available) - 1, count, dtype=np.int64)
        require(len(np.unique(positions)) == count, "Deterministic sample positions unexpectedly duplicate")
        return available[positions]

    return select(~held, train_rows), select(held, hold_rows)


def aggregate_bounds(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([item["pixel_count"] for item in rows], dtype=np.float64)

    def weighted(key: str) -> float:
        return float(np.average([item["bounds"][key] for item in rows], weights=counts))

    return {"frame_count": len(rows), "pixel_count": int(np.sum(counts)), "chroma_bound_pixel_fraction": weighted("chroma_bound_pixel_fraction"), "inverse_negative_pixel_fraction": weighted("inverse_negative_pixel_fraction"), "inverse_negative_channel_fraction": weighted("inverse_negative_channel_fraction"), "Y_target_max_abs_error_nits": float(max(item["bounds"]["Y_target_max_abs_error_nits"] for item in rows)), "finite": bool(all(item["bounds"]["finite"] for item in rows)), "reimposition_dark_fallback_pixels": int(sum(item["bounds"]["reimposition_dark_fallback_pixels"] for item in rows))}


def fit_pooled_mmr(frozen: Any, prepared: list[dict[str, Any]], train_rows: int, hold_rows: int) -> dict[str, Any]:
    """Frozen MMR ridge/lambda/objective with only declared deterministic row limits."""
    theta0 = frozen.identity_coefficients("MMR1")
    train_x: list[np.ndarray] = []
    train_c: list[np.ndarray] = []
    hold_x: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    for frame in prepared:
        train, hold = fixed_indices(frame["h0_common"].shape[:2], train_rows, hold_rows)
        design, _ = frozen.features(frame["h0_common"], frame["target_y_common"], "MMR1")
        target_chroma = frozen.linear_bt2020_to_ictcp(frame["target"] * PEAK_NITS)[..., 1:]
        flat_x, flat_c = design.reshape(-1, 6), target_chroma.reshape(-1, 2)
        train_x.append(flat_x[train])
        train_c.append(flat_c[train])
        hold_x.append(flat_x[hold])
        frame_rows.append({"label": frame["sample"]["label"], "train_rows": int(len(train)), "holdout_rows": int(len(hold))})
        del design, target_chroma
    x_train, c_train = np.concatenate(train_x), np.concatenate(train_c)
    x_hold = np.concatenate(hold_x)
    candidates: list[dict[str, Any]] = []
    for lam in frozen.LAMBDA_GRID:
        gram = x_train.T @ x_train / len(x_train) + lam * np.eye(x_train.shape[1])
        rhs = x_train.T @ c_train / len(x_train) + lam * theta0
        theta = np.linalg.solve(gram, rhs)
        hold_out, hold_bounds = frozen.apply_mmr(
            np.concatenate([item["h0_common"].reshape(-1, 1, 3)[fixed_indices(item["h0_common"].shape[:2], train_rows, hold_rows)[1]] for item in prepared]),
            np.concatenate([item["target_y_common"].reshape(-1, 1)[fixed_indices(item["h0_common"].shape[:2], train_rows, hold_rows)[1]] for item in prepared]),
            theta,
            "MMR1",
        )
        hold_target = np.concatenate([item["target"].reshape(-1, 1, 3)[fixed_indices(item["h0_common"].shape[:2], train_rows, hold_rows)[1]] for item in prepared])
        metric = frozen.metrics(hold_out, hold_target)
        condition = float(np.linalg.cond(gram))
        deviation = float(np.linalg.norm(theta - theta0))
        stable = bool(np.isfinite(condition) and condition <= 1e6 and deviation <= 5.0 and hold_bounds["chroma_bound_pixel_fraction"] <= 0.005 and hold_bounds["inverse_negative_pixel_fraction"] <= 0.005 and hold_bounds["finite"])
        candidates.append({"lambda": float(lam), "coefficients": theta, "regularized_gram_condition_number": condition, "coefficient_identity_deviation_L2": deviation, "coefficient_max_abs": float(np.max(np.abs(theta))), "holdout_metrics": metric, "holdout_bounds": hold_bounds, "holdout_objective": frozen.objective(metric), "stable": stable})
    accepted = [item for item in candidates if item["stable"]]
    require(bool(accepted), "All reduced-budget frozen MMR candidates were pathological")
    selected = min(accepted, key=lambda item: item["holdout_objective"])
    return {"model": "MMR1", "frame_rows": frame_rows, "training_sample_count": int(len(x_train)), "spatial_holdout_sample_count": int(len(x_hold)), "selected": selected, "candidates": candidates, "regularized_closed_form": "(XᵀX/N + λI)⁻¹(XᵀC/N + λΘ_identity)"}


def h0_fit_and_prepare(frozen: Any, frames: list[dict[str, Any]], train_rows: int, hold_rows: int, full_h0_fit: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for frame in frames:
        if full_h0_fit:
            x_parts.append(frozen.compute_luminance(frame["sdr_common"]).reshape(-1))
            y_parts.append(frozen.compute_luminance(frame["target"]).reshape(-1))
        else:
            train, _ = fixed_indices(frame["sdr_common"].shape[:2], train_rows, hold_rows)
            x_parts.append(frozen.compute_luminance(frame["sdr_common"]).reshape(-1)[train])
            y_parts.append(frozen.compute_luminance(frame["target"]).reshape(-1)[train])
    model = frozen.ModelECdfRegularized()
    params = model.fit(np.concatenate(x_parts), np.concatenate(y_parts))
    prepared: list[dict[str, Any]] = []
    for frame in frames:
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], params)
        x1, y1, x2, y2 = frame["bbox"]
        prepared.append({**frame, "h0_common": h0_full[y1:y2, x1:x2].copy(), "target_y_common": target_y_full[y1:y2, x1:x2].copy()})
        del h0_full, target_y_full
    return params, prepared


def reference_phase6_metrics(protocol: dict[str, Any]) -> dict[str, dict[str, float]]:
    data = json.loads((ROOT / protocol["frozen_dependency_hashes_sha256"]["phase6_metrics"]["path"]).read_text(encoding="utf-8"))
    record = next(item for item in data["cases"] if item.get("case_id") == protocol["reference_case"]["case_id"])
    return {item["label"]: {"mean": float(item["MMR1_shared_metrics"]["deltaE2000"]["mean"]), "p95": float(item["MMR1_shared_metrics"]["deltaE2000"]["p95"])} for item in record["frames"]}


def fit_reference_control(frozen: Any, frames: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[dict[str, Any], list[np.ndarray]]:
    training_labels = set(protocol["reference_case"]["phase6_reference_transform"]["training_labels"])
    training = [frame for frame in frames if frame["sample"]["label"] in training_labels]
    reference = protocol["reference_case"]["phase6_reference_transform"]
    params, prepared = h0_fit_and_prepare(frozen, training, int(reference["reference_fit_rows_per_training_frame"]["train"]), int(reference["reference_fit_rows_per_training_frame"]["holdout"]), full_h0_fit=True)
    fitted = fit_pooled_mmr(frozen, prepared, int(reference["reference_fit_rows_per_training_frame"]["train"]), int(reference["reference_fit_rows_per_training_frame"]["holdout"]))
    theta = fitted["selected"]["coefficients"]
    theta_hash = hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest().upper()
    expected_hash = protocol["reproducibility_control"]["required_coefficient_sha256"]
    expected = np.asarray(reference["coefficients"], dtype=np.float64)
    coefficient_l2 = float(np.linalg.norm(theta - expected))
    require(theta_hash == expected_hash, f"REPRODUCIBILITY_FAILURE: coefficient hash {theta_hash} != {expected_hash}")
    require(coefficient_l2 <= protocol["reproducibility_control"]["maximum_coefficient_l2_difference"], "REPRODUCIBILITY_FAILURE: coefficient L2 tolerance exceeded")
    phase6 = reference_phase6_metrics(protocol)
    reference_outputs: list[np.ndarray] = []
    checks: list[dict[str, Any]] = []
    for frame in frames:
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], params)
        output, bounds = frozen.apply_mmr(h0_full, target_y_full, theta, "MMR1")
        x1, y1, x2, y2 = frame["bbox"]
        common = output[y1:y2, x1:x2].copy()
        metric = frozen.metrics(common, frame["target"])
        label = frame["sample"]["label"]
        mean_difference = abs(metric["deltaE2000"]["mean"] - phase6[label]["mean"])
        p95_difference = abs(metric["deltaE2000"]["p95"] - phase6[label]["p95"])
        require(mean_difference <= protocol["reproducibility_control"]["maximum_mean_deltaE2000_difference_per_tested_frame"], f"REPRODUCIBILITY_FAILURE: {label} mean DeltaE differs from Phase 6")
        require(p95_difference <= protocol["reproducibility_control"]["maximum_p95_deltaE2000_difference_per_tested_frame"], f"REPRODUCIBILITY_FAILURE: {label} P95 DeltaE differs from Phase 6")
        reference_outputs.append(common)
        checks.append({"label": label, "mean_deltaE2000": metric["deltaE2000"]["mean"], "p95_deltaE2000": metric["deltaE2000"]["p95"], "phase6_mean_difference": mean_difference, "phase6_p95_difference": p95_difference, "bounds": bounds})
        del h0_full, target_y_full, output
    return {"h0_parameters": params, "mmr": fitted, "coefficients_sha256": theta_hash, "coefficient_l2_vs_protocol": coefficient_l2, "per_frame": checks}, reference_outputs


def fit_candidate_once(frozen: Any, frames: list[dict[str, Any]], labels: list[str], fit_rows: int, hold_rows: int) -> dict[str, Any]:
    selected = [frame for frame in frames if frame["sample"]["label"] in set(labels)]
    require(len(selected) == len(labels), "Candidate training labels were unavailable")
    params, prepared = h0_fit_and_prepare(frozen, selected, fit_rows, hold_rows, full_h0_fit=False)
    fitted = fit_pooled_mmr(frozen, prepared, fit_rows, hold_rows)
    theta = fitted["selected"]["coefficients"]
    return {"h0_parameters": params, "mmr": fitted, "coefficients_sha256": hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest().upper()}


def evaluate_candidate(frozen: Any, frames: list[dict[str, Any]], candidate: dict[str, Any], reference_outputs: list[np.ndarray], protocol: dict[str, Any]) -> dict[str, Any]:
    theta = candidate["mmr"]["selected"]["coefficients"]
    validity = protocol["equivalence_and_validity"]["required_validity"]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for frame, reference_output in zip(frames, reference_outputs, strict=True):
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], candidate["h0_parameters"])
        output, bounds = frozen.apply_mmr(h0_full, target_y_full, theta, "MMR1")
        x1, y1, x2, y2 = frame["bbox"]
        common = output[y1:y2, x1:x2]
        difference = frozen.metrics(common, reference_output)
        direct = frozen.metrics(common, frame["target"])
        equivalent = bool(difference["deltaE2000"]["mean"] <= protocol["equivalence_and_validity"]["mean_deltaE2000_max"] and difference["deltaE2000"]["p95"] <= protocol["equivalence_and_validity"]["p95_deltaE2000_max"])
        bounded = bool(bounds["finite"] and bounds["chroma_bound_pixel_fraction"] <= 0.005 and bounds["inverse_negative_pixel_fraction"] <= 0.005 and bounds["Y_target_max_abs_error_nits"] <= 1e-5)
        rows.append({"label": frame["sample"]["label"], "frame_pair": [frame["sample"]["hdr_frame"], frame["sample"]["open_matte_frame"]], "reference_difference_metrics": difference, "direct_hdr_metrics": direct, "bounds": bounds, "equivalent_to_reference": equivalent, "valid_output": bounded, "required_validity": validity})
        del h0_full, target_y_full, output, common
    return {"application_and_evaluation_seconds": time.perf_counter() - started, "frames": rows, "passes_all_frames": bool(all(row["equivalent_to_reference"] and row["valid_output"] for row in rows))}


def benchmark_candidate(frozen: Any, frames: list[dict[str, Any]], frame_budget: dict[str, Any], fit_rows: int, protocol: dict[str, Any], reference_outputs: list[np.ndarray]) -> dict[str, Any]:
    hold_rows = min(2048, max(250, round(fit_rows / 2)))
    repetitions: list[dict[str, Any]] = []
    final_candidate: dict[str, Any] | None = None
    for repetition in range(int(protocol["measurement"]["repetitions_per_candidate"])):
        gc.collect()
        started = time.perf_counter()
        with PeakRSS() as peak:
            candidate = fit_candidate_once(frozen, frames, frame_budget["labels"], fit_rows, hold_rows)
        repetitions.append({"repetition": repetition + 1, "fit_seconds": time.perf_counter() - started, "peak_rss_bytes": peak.peak_bytes, "coefficients_sha256": candidate["coefficients_sha256"]})
        final_candidate = candidate
    require(final_candidate is not None, "Candidate did not execute")
    coefficient_hashes = {item["coefficients_sha256"] for item in repetitions}
    require(len(coefficient_hashes) == 1, "Deterministic candidate coefficient hash varied across repetitions")
    evaluation = evaluate_candidate(frozen, frames, final_candidate, reference_outputs, protocol)
    times = np.asarray([item["fit_seconds"] for item in repetitions], dtype=np.float64)
    rss = np.asarray([item["peak_rss_bytes"] for item in repetitions], dtype=np.int64)
    return {"frame_budget": frame_budget["id"], "training_labels": frame_budget["labels"], "production_eligible": "reference_frame_set" not in frame_budget or True, "fit_rows_per_frame": int(fit_rows), "lambda_holdout_rows_per_frame": int(hold_rows), "total_fit_rows": int(len(frame_budget["labels"]) * fit_rows), "fit_repetitions": repetitions, "fit_time_seconds": {"median": float(np.median(times)), "worst": float(np.max(times)), "all": times.tolist()}, "peak_rss_bytes": {"median": int(np.median(rss)), "worst": int(np.max(rss)), "all": rss.tolist()}, "candidate": final_candidate, "evaluation": evaluation}


def curve_image(rows: list[dict[str, Any]], path: Path, title: str, y_key: str, y_label: str) -> str:
    """Minimal deterministic CV2 curve; avoids adding plotting dependencies."""
    width, height, margin = 900, 520, 70
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(image, title, (margin, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    values = np.asarray([float(item[y_key]) for item in rows], dtype=np.float64)
    xs = np.asarray([float(item["total_fit_rows"]) for item in rows], dtype=np.float64)
    x_log = np.log10(np.maximum(xs, 1.0))
    x0, x1 = float(x_log.min()), float(x_log.max())
    y0, y1 = float(values.min()), float(values.max())
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    cv2.rectangle(image, (margin, margin), (width - margin, height - margin), (0, 0, 0), 1)
    points: list[tuple[int, int]] = []
    for x, y in zip(x_log, values, strict=True):
        px = int(round(margin + (x - x0) / (x1 - x0) * (width - 2 * margin)))
        py = int(round(height - margin - (y - y0) / (y1 - y0) * (height - 2 * margin)))
        points.append((px, py))
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (40, 90, 220), 2, cv2.LINE_AA)
    for point, row in zip(points, rows, strict=True):
        cv2.circle(image, point, 4, (20, 40, 180), -1, cv2.LINE_AA)
        cv2.putText(image, f"{row['frame_budget']}/{row['fit_rows_per_frame']}", (point[0] - 35, min(height - 15, point[1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "log10(total fit rows)", (width // 2 - 90, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, y_label, (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    require(bool(ok), f"Could not encode curve: {path}")
    encoded.tofile(str(path))
    return str(path)


def choose_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [row for row in rows if row["production_eligible"] and row["evaluation"]["passes_all_frames"]]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (row["total_fit_rows"], len(row["training_labels"]), row["fit_rows_per_frame"]))


def write_selected_artifacts(frozen: Any, frames: list[dict[str, Any]], candidate: dict[str, Any], reference_outputs: list[np.ndarray]) -> list[dict[str, str]]:
    """Create panels only after a candidate has been selected; no full-shot render."""
    target_dir = OUTPUT / "selected_visuals"
    target_dir.mkdir(parents=True, exist_ok=True)
    theta = candidate["candidate"]["mmr"]["selected"]["coefficients"]
    paths: list[dict[str, str]] = []
    for frame, reference in zip(frames, reference_outputs, strict=True):
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], candidate["candidate"]["h0_parameters"])
        output, _ = frozen.apply_mmr(h0_full, target_y_full, theta, "MMR1")
        x1, y1, x2, y2 = frame["bbox"]
        canvas = np.zeros_like(output); canvas[y1:y2, x1:x2] = frame["target"]
        montage = frozen.comparison_montage(frame["sdr_full"], canvas, h0_full, [reference, output], height=180)
        label = frame["sample"]["label"].replace("%", "pct")
        panel_path = target_dir / f"{label}_reference_and_selected.png"
        frozen.write_cv_image(panel_path, montage[..., ::-1])
        diff_path = target_dir / f"{label}_selected_vs_reference_delta.png"
        frozen.write_cv_image(diff_path, frozen.diff_map(output[y1:y2, x1:x2], reference))
        paths.append({"label": frame["sample"]["label"], "panel": str(panel_path), "difference": str(diff_path)})
        del h0_full, target_y_full, output, canvas
    return paths


def report(data: dict[str, Any]) -> str:
    rows = data["candidates"]
    selected = data.get("recommendation")
    lines = ["# Phase 7 — Fast Scene-Fit Benchmark for Frozen MMR-1", "", f"**Status:** `{data['status']}`", "", "Research-only deterministic Matrix 08 benchmark. Frozen MMR-1 equations, ICtCp representation, Model-E mapping, ridge objective, constraints, and final luminance reimposition were not modified. The 90% frame remained a temporal holdout for B1–B4; B5 is diagnostic-only and cannot support the recommendation.", "", "## Reference reproducibility", "", f"Phase-6 B4 coefficient SHA-256: `{data['reference_control']['coefficients_sha256']}`. The re-decode/re-fit control passed before the reduced-budget sweep.", "", "## Sweep", "", "| Frames | Rows/frame | Total rows | Fit median s | Fit worst s | Peak RSS worst MiB | Worst frame mean ΔE vs ref | Worst frame P95 ΔE vs ref | Equivalent |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        frame_metrics = [item["reference_difference_metrics"]["deltaE2000"] for item in row["evaluation"]["frames"]]
        worst_mean = max(float(metric["mean"]) for metric in frame_metrics)
        worst_p95 = max(float(metric["p95"]) for metric in frame_metrics)
        lines.append(f"| {row['frame_budget']} | {row['fit_rows_per_frame']} | {row['total_fit_rows']} | {row['fit_time_seconds']['median']:.4f} | {row['fit_time_seconds']['worst']:.4f} | {row['peak_rss_bytes']['worst'] / 1024**2:.1f} | {worst_mean:.6f} | {worst_p95:.6f} | {row['evaluation']['passes_all_frames']} |")
    lines += ["", "## Required A–H recommendation", ""]
    if selected is None:
        lines += ["No production-eligible reduced configuration met every predeclared reference-equivalence and validity criterion. Therefore A–H cannot recommend a smaller budget; this phase stops without a production recommendation."]
    else:
        median = selected["fit_time_seconds"]["median"]
        worst = selected["fit_time_seconds"]["worst"]
        frame_metrics = [item["reference_difference_metrics"]["deltaE2000"] for item in selected["evaluation"]["frames"]]
        lines += [f"A. **Representative frames per shot:** `{len(selected['training_labels'])}` ({', '.join(selected['training_labels'])}).", f"B. **Fit rows per representative frame:** `{selected['fit_rows_per_frame']}`.", f"C. **Total fit rows per shot:** `{selected['total_fit_rows']}`.", f"D. **Fitting time:** median `{median:.4f} s`, worst `{worst:.4f} s` (warm-cache H0 + pooled MMR fitting only).", f"E. **Fit-only projection:** 100/500/1000 shots = `{median * 100:.1f}/{median * 500:.1f}/{median * 1000:.1f} s` median and `{worst * 100:.1f}/{worst * 500:.1f}/{worst * 1000:.1f} s` worst. Decode/application are separately recorded and not included.", f"F. **Measured peak RSS:** median `{selected['peak_rss_bytes']['median'] / 1024**2:.1f} MiB`, worst `{selected['peak_rss_bytes']['worst'] / 1024**2:.1f} MiB`; GPU memory was not measured and is not inferred.", f"G. **Expected quality difference versus reference:** worst tested frame mean ΔE2000 `{max(float(item['mean']) for item in frame_metrics):.6f}`, P95 `{max(float(item['p95']) for item in frame_metrics):.6f}`, within predeclared limits 0.05/0.15.", "H. **RTX 3080 / CPU practicality:** the measured fit runs on CPU and requires no GPU-memory claim. It is practical only insofar as the reported fit-only and measured RSS budgets suit the local CPU workflow; full decode/application remains a separate throughput cost."]
    lines += ["", "## Cross-scene sanity", "", "`EXCLUDED_BY_PHASE6_STATUS`: Matrix 03 was `ANCHOR_NEAR_CUT`, Matrix 12 `LOCAL_SHOT_UNRESOLVED`, and BR2049 01 `ANCHOR_NEAR_CUT` with conditional geometry. No new anchor, sync, geometry, or source window was used.", "", "## Stop", "", "STOP after this report. No model modification, production integration, full-shot render, or full-film processing is authorized."]
    return "\n".join(lines) + "\n"


def run(check_only: bool = False) -> int:
    protocol = load_protocol()
    frozen = verify_and_load(protocol)
    if check_only:
        print("PHASE7_CHECK_ONLY=PASS")
        return 0
    decode_started = time.perf_counter()
    frames = decode_fixed_frames(frozen, protocol)
    decode_seconds = time.perf_counter() - decode_started
    reference_control, reference_outputs = fit_reference_control(frozen, frames, protocol)
    eligible_budgets = protocol["benchmark_frame_budgets"]["production_eligible_prefixes"]
    b5 = protocol["benchmark_frame_budgets"]["five_frame_diagnostic"]
    candidates: list[dict[str, Any]] = []
    for frame_budget in [*eligible_budgets, b5]:
        for fit_rows in protocol["benchmark_pixel_budgets"]["fit_rows_per_frame"]:
            row = benchmark_candidate(frozen, frames, frame_budget, int(fit_rows), protocol, reference_outputs)
            row["production_eligible"] = frame_budget["id"] != "B5"
            candidates.append(row)
            print(f"PHASE7_CANDIDATE={frame_budget['id']}/{fit_rows} PASS={row['evaluation']['passes_all_frames']}", flush=True)
    recommendation = choose_recommendation(candidates)
    curves_root = OUTPUT / "curves"
    curves_input = []
    for row in candidates:
        worst_mean = max(float(item["reference_difference_metrics"]["deltaE2000"]["mean"]) for item in row["evaluation"]["frames"])
        curves_input.append({**row, "worst_mean_deltaE": worst_mean, "fit_median_seconds": row["fit_time_seconds"]["median"], "peak_rss_mib": row["peak_rss_bytes"]["worst"] / 1024**2})
    artifacts = {"quality_curve": curve_image(curves_input, curves_root / "quality_vs_total_fit_rows.png", "Worst frame mean DeltaE2000 vs reference", "worst_mean_deltaE", "mean DeltaE2000"), "time_curve": curve_image(curves_input, curves_root / "time_vs_total_fit_rows.png", "Fit median time vs total fit rows", "fit_median_seconds", "seconds"), "rss_curve": curve_image(curves_input, curves_root / "rss_vs_total_fit_rows.png", "Peak RSS vs total fit rows", "peak_rss_mib", "MiB")}
    if recommendation is not None:
        artifacts["selected_visuals"] = write_selected_artifacts(frozen, frames, recommendation, reference_outputs)
    result = {"phase": "Phase 7", "status": "COMPLETE", "scope": protocol["scope"], "protocol": str(PROTOCOL_PATH), "frozen_equivalence": {"mmr1_source_hash": sha256(ROOT / protocol["frozen_dependency_hashes_sha256"]["mmr1_source"]["path"]), "assertions_passed": True, "frozen_model": "MMR1", "parameter_count": 12}, "decode_seconds": decode_seconds, "reference_control": reference_control, "source_frames": [{"label": frame["sample"]["label"], "frame_pair": [frame["sample"]["hdr_frame"], frame["sample"]["open_matte_frame"]], "source_provenance": frame["source_provenance"]} for frame in frames], "candidates": candidates, "recommendation": recommendation, "cross_scene_sanity": {"status": "EXCLUDED_BY_PHASE6_STATUS", "details": protocol["cross_scene_sanity"]}, "artifacts": artifacts}
    write_json(METRICS_PATH, result)
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    print("PHASE7_BENCHMARK_STATUS=COMPLETE")
    print(f"METRICS={METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify pinned dependencies and frozen contract without decoding sources.")
    return run(check_only=parser.parse_args().check_only)


if __name__ == "__main__":
    raise SystemExit(main())
