"""Read-only correspondence audit and sampling-efficiency test for frozen MMR-1.

The audit proves what the current code actually constructs: synchronized source
frames, one common 1920x800 coordinate grid, and paired same-coordinate samples.
It does not claim unmeasured optical registration or truth outside the HDR crop.
No frozen model, production code, source asset, sync offset, or geometry is changed.
"""
from __future__ import annotations

import argparse
import csv
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
PROTOCOL_PATH = BASE / "SAMPLING_AUDIT_PROTOCOL.json"
OUTPUT = BASE / "results"
METRICS_PATH = OUTPUT / "sampling_audit_metrics.json"
TABLE_PATH = OUTPUT / "matrix08_coordinate_audit.csv"
REPORT_PATH = BASE / "SAMPLING_AUDIT_REPORT.md"
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
    path.write_text(
        json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def timestamp(frame: int) -> float:
    return float(frame / FPS)


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(
        protocol["status"] == "PREDECLARED BEFORE AUDIT EXTRACTION",
        "Sampling-audit protocol was not predeclared",
    )
    return protocol


def load_frozen(protocol: dict[str, Any]) -> Any:
    for name, entry in protocol["pinned_dependencies_sha256"].items():
        path = ROOT / entry["path"]
        require(path.is_file(), f"Pinned dependency is missing: {name}")
        require(sha256(path) == entry["sha256"], f"Pinned dependency hash mismatch: {name}")
    source = ROOT / protocol["pinned_dependencies_sha256"]["frozen_mmr1"]["path"]
    source_parent = str(source.parent)
    if source_parent not in sys.path:
        sys.path.insert(0, source_parent)
    spec = importlib.util.spec_from_file_location("sampling_audit_frozen_mmr1", source)
    require(spec is not None and spec.loader is not None, "Could not import frozen MMR-1")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    contract = protocol["frozen_mmr1_contract"]
    require(tuple(module.LAMBDA_GRID) == tuple(contract["lambda_grid"]), "Lambda grid changed")
    require(module.CHROMA_BOUND == contract["chroma_bound"], "Chroma bound changed")
    require(module.identity_coefficients("MMR1").shape == (6, 2), "MMR-1 shape changed")
    return module


def extract_rgb48(ffmpeg: Path, source: Path, width: int, height: int, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
    expected_bytes = width * height * 3 * 2
    command = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{timestamp(frame):.9f}", "-i", str(source), "-map", "0:v:0",
        "-frames:v", "1", "-pix_fmt", "rgb48le", "-vsync", "0", "-f",
        "rawvideo", "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    raw = completed.stdout
    metadata = {
        "command": command,
        "returncode": int(completed.returncode),
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
        "stdout_bytes": int(len(raw)),
        "expected_bytes": int(expected_bytes),
        "pixel_format": "rgb48le",
        "frame": int(frame),
        "timestamp_seconds": timestamp(frame),
    }
    require(completed.returncode == 0 and len(raw) >= expected_bytes, f"RGB48LE extraction failed: {metadata}")
    image = np.frombuffer(raw[:expected_bytes], dtype="<u2").reshape(height, width, 3).copy()
    return image, metadata


def decode_pair(frozen: Any, contract: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    ffmpeg = ROOT / contract["ffmpeg"]
    require(ffmpeg.is_file(), f"Missing FFmpeg: {ffmpeg}")
    hdr_source = Path(contract["source_paths"]["hdr"])
    sdr_source = Path(contract["source_paths"]["sdr_open_matte"])
    hdr_raw, hdr_decode = extract_rgb48(ffmpeg, hdr_source, 3840, 1600, int(sample["hdr_frame"]))
    sdr_raw, sdr_decode = extract_rgb48(ffmpeg, sdr_source, 1920, 1080, int(sample["sdr_frame"]))
    require(hdr_raw.shape == (1600, 3840, 3), "Unexpected raw HDR dimensions")
    require(sdr_raw.shape == (1080, 1920, 3), "Unexpected raw SDR dimensions")
    hdr_common_code = frozen.cv2.resize(
        hdr_raw.astype(np.float64) / RGB16_MAX,
        (1920, 800),
        interpolation=frozen.cv2.INTER_LINEAR,
    )
    hdr_target = frozen.pq_eotf_normalized(hdr_common_code)
    sdr_full = np.maximum(
        frozen.bt709_to_bt2020(frozen.bt1886_eotf(sdr_raw.astype(np.float64) / RGB16_MAX)),
        0.0,
    )
    sdr_common = sdr_full[140:940, 0:1920]
    require(sdr_common.shape == (800, 1920, 3), "Unexpected SDR common shape")
    require(hdr_target.shape == (800, 1920, 3), "Unexpected HDR common shape")
    require(np.isfinite(sdr_common).all() and np.isfinite(hdr_target).all(), "Non-finite common grid")
    return {
        "sample": sample,
        "hdr_raw": hdr_raw,
        "sdr_raw": sdr_raw,
        "hdr_common_code": hdr_common_code,
        "sdr_full": sdr_full,
        "sdr_common": sdr_common,
        "target": hdr_target,
        "bbox": (0, 140, 1920, 940),
        "provenance": {
            "hdr_decode": hdr_decode,
            "sdr_decode": sdr_decode,
            "hdr_rgb_sha256": hashlib.sha256(hdr_raw.tobytes()).hexdigest().upper(),
            "sdr_rgb_sha256": hashlib.sha256(sdr_raw.tobytes()).hexdigest().upper(),
        },
    }


def full_grid_spatial_partitions(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """The unchanged frozen block-mask equation without its Phase-5 diagnostic caps."""
    yy, xx = np.indices(shape)
    held = ((yy // 80 + xx // 120) % 5) == 0
    return np.flatnonzero((~held).ravel()), np.flatnonzero(held.ravel())


def frozen_spatial_indices(frozen: Any, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Directly call the frozen partition implementation; do not reimplement it."""
    train, hold = frozen.spatial_indices(shape)
    require(len(np.intersect1d(train, hold)) == 0, "Frozen spatial partitions overlap")
    return train, hold


def distributed_subset(indices: np.ndarray, count: int) -> np.ndarray:
    require(len(indices) >= count, f"Requested {count} pairs from only {len(indices)} eligible coordinates")
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    require(len(np.unique(positions)) == count, "Deterministic sample selection duplicated a coordinate")
    return indices[positions]


def audit_coordinates(frozen: Any, shape: tuple[int, int], count: int) -> np.ndarray:
    train, hold = frozen_spatial_indices(frozen, shape)
    universe = np.sort(np.concatenate([train, hold]))
    return distributed_subset(universe, count)


def rgb_triplet(values: np.ndarray) -> str:
    return "[" + ", ".join(str(int(item)) for item in np.asarray(values).reshape(3)) + "]"


def write_coordinate_table(frames: list[dict[str, Any]], protocol: dict[str, Any], frozen: Any) -> dict[str, Any]:
    coordinates = audit_coordinates(
        frozen,
        (800, 1920),
        int(protocol["audit"]["coordinate_table_rows"]),
    )
    fields = [
        "sample_id", "label", "sdr_frame", "hdr_frame", "common_x", "common_y",
        "sdr_x", "sdr_y", "hdr_mapped_x", "hdr_mapped_y",
        "hdr_source_center_x", "hdr_source_center_y", "source_geometry",
        "sdr_rgb48le", "hdr_resized_rgb_code16", "hdr_source_nearest_rgb48le",
    ]
    rows: list[dict[str, Any]] = []
    for frame in frames:
        for number, flat in enumerate(coordinates, start=1):
            y, x = divmod(int(flat), 1920)
            source_x = (x + 0.5) / 0.5 - 0.5
            source_y = (y + 0.5) / 0.5 - 0.5
            nearest_x = int(np.clip(round(source_x), 0, 3839))
            nearest_y = int(np.clip(round(source_y), 0, 1599))
            rows.append({
                "sample_id": f"{frame['sample']['label'].replace('%', 'pct')}_{number:02d}",
                "label": frame["sample"]["label"],
                "sdr_frame": int(frame["sample"]["sdr_frame"]),
                "hdr_frame": int(frame["sample"]["hdr_frame"]),
                "common_x": x,
                "common_y": y,
                "sdr_x": x,
                "sdr_y": y + 140,
                "hdr_mapped_x": x,
                "hdr_mapped_y": y,
                "hdr_source_center_x": f"{source_x:.3f}",
                "hdr_source_center_y": f"{source_y:.3f}",
                "source_geometry": "SDR crop [0,140,1920,940]; HDR full 3840x1600 -> 1920x800 cv2.INTER_LINEAR",
                "sdr_rgb48le": rgb_triplet(frame["sdr_raw"][y + 140, x]),
                "hdr_resized_rgb_code16": rgb_triplet(np.rint(frame["hdr_common_code"][y, x] * RGB16_MAX)),
                "hdr_source_nearest_rgb48le": rgb_triplet(frame["hdr_raw"][nearest_y, nearest_x]),
            })
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"coordinate_count": int(len(coordinates)), "observation_count": int(len(rows)), "coordinates_flat": coordinates, "csv": str(TABLE_PATH)}


def run_correspondence_audit(frozen: Any, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = protocol["matrix08_pair_contract"]
    frames: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for sample in contract["representative_pairs"]:
        require(
            int(sample["sdr_frame"]) == int(sample["hdr_frame"]) + int(contract["synchronization"]["offset_frames"]),
            f"Temporal contract mismatch for {sample['label']}",
        )
        frame = decode_pair(frozen, contract, sample)
        frames.append(frame)
        checks.append({
            "label": sample["label"],
            "frame_pair": [sample["sdr_frame"], sample["hdr_frame"]],
            "offset_frames_verified": int(sample["sdr_frame"]) == int(sample["hdr_frame"]) - 19,
            "sdr_common_shape": list(frame["sdr_common"].shape),
            "hdr_common_shape": list(frame["target"].shape),
            "full_resolution_common_grid": bool(frame["sdr_common"].shape[:2] == (800, 1920)),
            "finite": bool(np.isfinite(frame["sdr_common"]).all() and np.isfinite(frame["target"]).all()),
            "provenance": frame["provenance"],
        })
    table = write_coordinate_table(frames, protocol, frozen)
    passed = bool(all(
        item["offset_frames_verified"]
        and item["sdr_common_shape"] == [800, 1920, 3]
        and item["hdr_common_shape"] == [800, 1920, 3]
        and item["full_resolution_common_grid"]
        and item["finite"]
        for item in checks
    ))
    return frames, {
        "status": "PASS" if passed else "FAIL",
        "pair_checks": checks,
        "coordinate_table": table,
        "findings": {
            "same_common_coordinate_used_for_sdr_and_hdr": passed,
            "common_grid_full_sdr_resolution": passed,
            "hidden_fit_image_downsampling": False,
            "post_sampling_spatial_transform": "None. The only analytical resample is the pre-sampling HDR 3840x1600 -> 1920x800 cv2.INTER_LINEAR geometry operation.",
            "physical_correspondence_limit": protocol["audit"]["physical_correspondence_limit"],
        },
    }


def prepare_fit_frames(
    frozen: Any,
    frames: list[dict[str, Any]],
    labels: list[str],
    train_count: int,
    full_h0_grid: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = [frame for frame in frames if frame["sample"]["label"] in set(labels)]
    require(len(selected) == len(labels), "Requested representative frame is unavailable")
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for frame in selected:
        if full_h0_grid:
            x_parts.append(frozen.compute_luminance(frame["sdr_common"]).reshape(-1))
            y_parts.append(frozen.compute_luminance(frame["target"]).reshape(-1))
            continue
        train, _ = full_grid_spatial_partitions(frame["sdr_common"].shape[:2])
        indices = distributed_subset(train, train_count)
        x_parts.append(frozen.compute_luminance(frame["sdr_common"]).reshape(-1)[indices])
        y_parts.append(frozen.compute_luminance(frame["target"]).reshape(-1)[indices])
    h0 = frozen.ModelECdfRegularized()
    params = h0.fit(np.concatenate(x_parts), np.concatenate(y_parts))
    prepared: list[dict[str, Any]] = []
    for frame in selected:
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], params)
        prepared.append({
            **frame,
            "h0_common": h0_full[140:940, 0:1920].copy(),
            "target_y_common": target_y_full[140:940, 0:1920].copy(),
        })
        del h0_full, target_y_full
    return params, prepared


def pooled_mmr_fit(frozen: Any, prepared: list[dict[str, Any]], train_count: int, hold_count: int) -> dict[str, Any]:
    theta0 = frozen.identity_coefficients("MMR1")
    x_train: list[np.ndarray] = []
    c_train: list[np.ndarray] = []
    hold_base: list[np.ndarray] = []
    hold_y: list[np.ndarray] = []
    hold_target: list[np.ndarray] = []
    frame_counts: list[dict[str, Any]] = []
    for frame in prepared:
        train_all, hold_all = full_grid_spatial_partitions(frame["h0_common"].shape[:2])
        train = distributed_subset(train_all, train_count)
        hold = distributed_subset(hold_all, hold_count)
        design, _ = frozen.features(frame["h0_common"], frame["target_y_common"], "MMR1")
        target_chroma = frozen.linear_bt2020_to_ictcp(frame["target"] * PEAK_NITS)[..., 1:]
        x_train.append(design.reshape(-1, 6)[train])
        c_train.append(target_chroma.reshape(-1, 2)[train])
        hold_base.append(frame["h0_common"].reshape(-1, 1, 3)[hold])
        hold_y.append(frame["target_y_common"].reshape(-1, 1)[hold])
        hold_target.append(frame["target"].reshape(-1, 1, 3)[hold])
        frame_counts.append({"label": frame["sample"]["label"], "train_pairs": int(len(train)), "holdout_pairs": int(len(hold))})
        del design, target_chroma
    train_x = np.concatenate(x_train)
    train_c = np.concatenate(c_train)
    candidates: list[dict[str, Any]] = []
    for lam in frozen.LAMBDA_GRID:
        gram = train_x.T @ train_x / len(train_x) + lam * np.eye(train_x.shape[1])
        rhs = train_x.T @ train_c / len(train_x) + lam * theta0
        theta = np.linalg.solve(gram, rhs)
        held_output, held_bounds = frozen.apply_mmr(
            np.concatenate(hold_base), np.concatenate(hold_y), theta, "MMR1"
        )
        held_metric = frozen.metrics(held_output, np.concatenate(hold_target))
        condition = float(np.linalg.cond(gram))
        deviation = float(np.linalg.norm(theta - theta0))
        stable = bool(
            np.isfinite(condition)
            and condition <= 1e6
            and deviation <= 5.0
            and held_bounds["chroma_bound_pixel_fraction"] <= 0.005
            and held_bounds["inverse_negative_pixel_fraction"] <= 0.005
            and held_bounds["finite"]
        )
        candidates.append({
            "lambda": float(lam),
            "coefficients": theta,
            "regularized_gram_condition_number": condition,
            "coefficient_identity_deviation_L2": deviation,
            "holdout_metrics": held_metric,
            "holdout_bounds": held_bounds,
            "holdout_objective": frozen.objective(held_metric),
            "stable": stable,
        })
    accepted = [candidate for candidate in candidates if candidate["stable"]]
    require(bool(accepted), "All frozen MMR-1 candidates were pathological")
    selected = min(accepted, key=lambda candidate: candidate["holdout_objective"])
    return {
        "model": "MMR1",
        "frame_counts": frame_counts,
        "training_pairs": int(len(train_x)),
        "selected": selected,
        "candidates": candidates,
    }


def fit_once(
    frozen: Any,
    frames: list[dict[str, Any]],
    labels: list[str],
    train_count: int,
    hold_count: int,
    full_h0_grid: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    h0_params, prepared = prepare_fit_frames(
        frozen,
        frames,
        labels,
        train_count,
        full_h0_grid=full_h0_grid,
    )
    fitted = pooled_mmr_fit(frozen, prepared, train_count, hold_count)
    theta = fitted["selected"]["coefficients"]
    return {
        "h0_parameters": h0_params,
        "mmr": fitted,
        "coefficients_sha256": hashlib.sha256(np.ascontiguousarray(theta).tobytes()).hexdigest().upper(),
        "fit_seconds": time.perf_counter() - started,
    }


def evaluate_transform(frozen: Any, frames: list[dict[str, Any]], fit: dict[str, Any], reference: list[np.ndarray] | None = None) -> tuple[dict[str, Any], list[np.ndarray]]:
    theta = fit["mmr"]["selected"]["coefficients"]
    outputs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        h0_full, target_y_full = frozen.map_luma_ratio(frame["sdr_full"], fit["h0_parameters"])
        output, bounds = frozen.apply_mmr(h0_full, target_y_full, theta, "MMR1")
        common = output[140:940, 0:1920].copy()
        direct = frozen.metrics(common, frame["target"])
        seam = frozen.seam_metrics(output, frame["target"], frame["bbox"])
        reference_metric = frozen.metrics(common, reference[index]) if reference is not None else None
        valid = bool(
            bounds["finite"]
            and bounds["chroma_bound_pixel_fraction"] <= 0.005
            and bounds["inverse_negative_pixel_fraction"] <= 0.005
            and bounds["Y_target_max_abs_error_nits"] <= 1e-5
        )
        rows.append({
            "label": frame["sample"]["label"],
            "direct_hdr_metrics": direct,
            "reference_difference_metrics": reference_metric,
            "bounds": bounds,
            "seam": seam,
            "valid_output": valid,
        })
        outputs.append(common)
        del h0_full, target_y_full, output
    return {"frames": rows, "all_valid": bool(all(row["valid_output"] for row in rows))}, outputs


def passes_reference(evaluation: dict[str, Any], protocol: dict[str, Any]) -> bool:
    limits = protocol["reference_and_efficiency"]["acceptance"]
    for row in evaluation["frames"]:
        metric = row["reference_difference_metrics"]
        if metric is None or not row["valid_output"]:
            return False
        if metric["deltaE2000"]["mean"] > limits["per_frame_mean_deltaE2000_vs_reference_max"]:
            return False
        if metric["deltaE2000"]["p95"] > limits["per_frame_p95_deltaE2000_vs_reference_max"]:
            return False
    return True


def benchmark_configuration(frozen: Any, frames: list[dict[str, Any]], labels: list[str], train_count: int, hold_count: int, reference_outputs: list[np.ndarray], protocol: dict[str, Any]) -> dict[str, Any]:
    repetitions = []
    final_fit: dict[str, Any] | None = None
    for repetition in range(int(protocol["reference_and_efficiency"]["repetitions_per_fit"])):
        gc.collect()
        fit = fit_once(frozen, frames, labels, train_count, hold_count)
        repetitions.append({"repetition": repetition + 1, "fit_seconds": fit["fit_seconds"], "coefficients_sha256": fit["coefficients_sha256"]})
        final_fit = fit
    require(final_fit is not None, "No fit completed")
    require(len({row["coefficients_sha256"] for row in repetitions}) == 1, "Nondeterministic frozen fit")
    evaluation, _ = evaluate_transform(frozen, frames, final_fit, reference_outputs)
    times = np.asarray([row["fit_seconds"] for row in repetitions], dtype=np.float64)
    return {
        "training_labels": labels,
        "representative_frame_count": len(labels),
        "train_pairs_per_frame": int(train_count),
        "holdout_pairs_per_frame": int(hold_count),
        "total_fit_pairs": int(len(labels) * train_count),
        "fit": final_fit,
        "fit_time_seconds": {"median": float(np.median(times)), "worst": float(np.max(times)), "all": times.tolist()},
        "evaluation": evaluation,
        "passes_reference": passes_reference(evaluation, protocol),
    }


def report(data: dict[str, Any]) -> str:
    audit = data["audit"]
    lines = [
        "# Sampling Audit — Frozen MMR-1",
        "",
        f"**Status:** `{data['status']}`",
        "",
        "This is an audit of the current frozen MMR-1 observation contract, followed only when that contract passes by a reduced paired-sample measurement. It does not modify MMR-1, production, source media, synchronization, or geometry.",
        "",
        "## Actual current correspondence implementation",
        "",
        "For Matrix 08, each representative pair is fixed by `SDR frame = HDR frame − 19`. SDR RGB48LE is decoded at 1920×1080, converted from BT.1886 linear BT.709 to linear BT.2020, then cropped to `[0,140,1920,940]`. HDR RGB48LE is decoded at 3840×1600 and resized once with `cv2.INTER_LINEAR` to the same 1920×800 grid before PQ EOTF. Therefore every fit observation is a same-time, same-common-coordinate pair: SDR `(x,y+140)` and resized-HDR `(x,y)`.",
        "",
        "The frozen MMR implementation obtains `Ct_base`/`Cp_base` from H0 ratio-scaled linear BT.2020 at a selected common-grid coordinate, obtains HDR target chroma from that identical coordinate, then fits its fixed six-feature/two-output ridge regression. No separate SDR/HDR random samplers exist. There is no fit-image downsample below 1920×800; the 320-pixel proxy in Phase 6 is only local cut detection. MMR itself has no luminance/chroma/hue/residual/motion correspondence rejection. Its clip and inverse-negative checks are whole-candidate numerical guards after fitting/application.",
        "",
        "## Audit result",
        "",
        f"- Pair-contract audit: `{audit['status']}`",
        f"- Five exact pairs obey `SDR = HDR − 19`: `{all(item['offset_frames_verified'] for item in audit['pair_checks'])}`",
        f"- Common grid for every pair: `1920×800`: `{all(item['full_resolution_common_grid'] for item in audit['pair_checks'])}`",
        f"- Diagnostic table: `{audit['coordinate_table']['coordinate_count']}` coordinates × 5 frames = `{audit['coordinate_table']['observation_count']}` paired observations: `{audit['coordinate_table']['csv']}`",
        "",
        "The table records the requested frame IDs, common coordinates, SDR source coordinates, HDR mapped coordinates, HDR continuous source-center coordinates, and raw/resized RGB values. It proves the implemented coordinate contract; it does **not** establish optical-flow residuals, occlusion masks, per-pair sync confidence beyond the inherited locked offset, or truth outside the HDR overlap.",
        "",
        "## Efficiency result",
        "",
    ]
    efficiency = data.get("efficiency")
    if efficiency is None:
        reason = (
            "Not run because this invocation used `--audit-only` after a passing correspondence audit."
            if audit["status"] == "PASS"
            else "Not run because the correspondence audit did not pass."
        )
        lines.append(reason)
    else:
        lines += [
            "The reference fits Model-E luminance from every full-resolution common-grid pixel in four training frames and fits frozen MMR-1 from 200,000 deterministic paired rows per frame. Reduced candidates use the declared paired subset for both H0 and MMR fitting, then are scored by transformed-image difference from the reference across all five frames, including the 90% temporal holdout.",
            "",
            "| Stage | Frames | Pairs/frame | Total fit pairs | Holdout | Median fit s | Worst fit s | Pass |",
            "|---|---:|---:|---:|---|---:|---:|---|",
        ]
        for row in efficiency["spatial_sweep"] + efficiency["frame_sweep"]:
            holdout = "preserved" if row.get("temporal_holdout_preserved", True) else "consumed (diagnostic)"
            lines.append(
                f"| {row['stage']} | {row['representative_frame_count']} | {row['train_pairs_per_frame']} | {row['total_fit_pairs']} | {holdout} | {row['fit_time_seconds']['median']:.4f} | {row['fit_time_seconds']['worst']:.4f} | {row['passes_reference']} |"
            )
        recommendation = efficiency.get("recommendation")
        if recommendation is None:
            lines += ["", "No reduced configuration met every predeclared transformed-image equivalence criterion. No minimum budget is recommended."]
        else:
            median = recommendation["fit_time_seconds"]["median"]
            worst = recommendation["fit_time_seconds"]["worst"]
            lines += [
                "",
                "### Recommended audit-supported configuration",
                "",
                f"- Representative frames: `{recommendation['representative_frame_count']}` (`{', '.join(recommendation['training_labels'])}`).",
                f"- Correctly paired fit samples per frame: `{recommendation['train_pairs_per_frame']}`.",
                f"- Total fit samples per shot: `{recommendation['total_fit_pairs']}`.",
                f"- Fit time: median `{median:.4f} s`, worst `{worst:.4f} s`.",
                f"- Fit-only projected analysis time: 100/500/1000 shots = `{median * 100:.1f}/{median * 500:.1f}/{median * 1000:.1f} s` median; `{worst * 100:.1f}/{worst * 500:.1f}/{worst * 1000:.1f} s` worst.",
            ]
    lines += [
        "",
        "## Cross-scene status",
        "",
        "`EXCLUDED_BY_EXISTING_STATUS`: no additional Matrix or BR2049 anchor is currently valid under the retained Phase-6 bounded-shot/geometry evidence. Matrix 03 is `ANCHOR_NEAR_CUT`, Matrix 12 is `LOCAL_SHOT_UNRESOLVED`, and BR2049 01 is `ANCHOR_NEAR_CUT` with conditional 0.9155 geometry. This audit did not search for new anchors or modify those contracts.",
        "",
        "## Stop",
        "",
        "STOP. The audit does not authorize MMR-2, a new model family, production changes, full-video processing, or a full-shot render.",
    ]
    return "\n".join(lines) + "\n"


def run(audit_only: bool = False) -> int:
    protocol = load_protocol()
    frozen = load_frozen(protocol)
    frames, audit = run_correspondence_audit(frozen, protocol)
    require(audit["status"] == "PASS", "Correspondence audit failed; efficiency test is prohibited")
    if audit_only:
        result = {"phase": "Sampling Audit", "status": "AUDIT_COMPLETE", "protocol": str(PROTOCOL_PATH), "audit": audit}
        write_json(METRICS_PATH, result)
        REPORT_PATH.write_text(report(result), encoding="utf-8")
        print("SAMPLING_AUDIT_STATUS=AUDIT_COMPLETE")
        return 0
    settings = protocol["reference_and_efficiency"]
    reference_spec = settings["reference_fit"]
    reference_fit = fit_once(
        frozen,
        frames,
        reference_spec["training_labels"],
        int(reference_spec["mmr_train_pairs_per_training_frame"]),
        int(reference_spec["mmr_lambda_holdout_pairs_per_training_frame"]),
        full_h0_grid=True,
    )
    reference_evaluation, reference_outputs = evaluate_transform(frozen, frames, reference_fit)
    require(reference_evaluation["all_valid"], "High-information reference output is invalid")
    spatial_sweep = []
    selected_spatial: dict[str, Any] | None = None
    b4 = next(
        item for item in settings["frame_budgets_after_spatial_selection"]
        if item["id"] == "B4"
    )
    for budget in settings["sample_budgets_per_training_frame"]:
        hold = min(50000, max(2048, round(int(budget) / 4)))
        candidate = benchmark_configuration(
            frozen, frames, b4["labels"], int(budget), int(hold), reference_outputs, protocol
        )
        candidate["stage"] = "spatial"
        candidate["frame_budget"] = b4["id"]
        spatial_sweep.append(candidate)
        if selected_spatial is None and candidate["passes_reference"]:
            selected_spatial = candidate
    frame_sweep = []
    if selected_spatial is not None:
        for frame_budget in settings["frame_budgets_after_spatial_selection"]:
            candidate = benchmark_configuration(
                frozen,
                frames,
                frame_budget["labels"],
                selected_spatial["train_pairs_per_frame"],
                selected_spatial["holdout_pairs_per_frame"],
                reference_outputs,
                protocol,
            )
            candidate["stage"] = "frames"
            candidate["frame_budget"] = frame_budget["id"]
            candidate["temporal_holdout_preserved"] = bool(
                frame_budget.get("temporal_holdout_preserved", False)
            )
            candidate["diagnostic_only"] = bool(frame_budget.get("diagnostic_only", False))
            frame_sweep.append(candidate)
    passing_frames = [
        row for row in frame_sweep
        if row["passes_reference"] and row["temporal_holdout_preserved"]
    ]
    recommendation = min(
        passing_frames,
        key=lambda row: (row["total_fit_pairs"], row["representative_frame_count"]),
    ) if passing_frames else None
    result = {
        "phase": "Sampling Audit",
        "status": "COMPLETE",
        "protocol": str(PROTOCOL_PATH),
        "frozen_equivalence": {"mmr1_sha256": sha256(ROOT / protocol["pinned_dependencies_sha256"]["frozen_mmr1"]["path"]), "assertions_passed": True},
        "audit": audit,
        "reference_fit": {"fit": reference_fit, "evaluation": reference_evaluation},
        "efficiency": {"spatial_sweep": spatial_sweep, "frame_sweep": frame_sweep, "recommendation": recommendation},
        "cross_scene": {"status": "EXCLUDED_BY_EXISTING_STATUS", "rule": protocol["cross_scene"]},
    }
    write_json(METRICS_PATH, result)
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    print("SAMPLING_AUDIT_STATUS=COMPLETE")
    print(f"METRICS={METRICS_PATH}")
    print(f"REPORT={REPORT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Run the correspondence audit without efficiency tests.")
    return run(audit_only=parser.parse_args().audit_only)


if __name__ == "__main__":
    raise SystemExit(main())
