"""Phase 16 prerequisite — confirm the actual shot boundaries inside Matrix-08.

The P2.34 manifest records [73274, 73460) as one resolved local shot, but its
closing boundary is a *gradual* detection whose run spans frames 73294 to 73460,
that is 166 of the 186 frames. The gradual branch collapses an entire run into a
single peak, so multiple real cuts inside such a run would be merged and reported
as one shot. Phase 16 must not pool multiple cuts into one transform, so the
segmentation is verified here before anything is fitted.

This script decodes nothing. It reads the existing Matrix-08 cache and computes the
unchanged boundary score between consecutive frames:

    0.4 * histogram chi-square + 0.2 * mean luminance delta + 0.4 * edge NCC change

with the documented thresholds: hard 0.30, gradual 0.15, minimum gradual run 5.

It reports every frame-to-frame score, the hard peaks, the gradual runs, and a
thumbnail strip so the segmentation can also be confirmed by eye.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
TOOLS = WORKSPACE / "tools" / "openmatte_hdr"

PROXY_WIDTH = 320
HISTOGRAM_WEIGHT = 0.4
LUMINANCE_WEIGHT = 0.2
EDGE_WEIGHT = 0.4
HARD_THRESHOLD = 0.30
GRADUAL_THRESHOLD = 0.15
GRADUAL_MIN_RUN_FRAMES = 5
MATRIX08 = {
    "hdr": "G:\\Filmy\\The Matrix UHD\\The Matrix.mkv",
    "om": "G:\\Filmy\\IMAX format (open matte)\\The Matrix (1999) [OPEN MATTE] [WEB-DL 1080p 10bit DD5.1 x265].mkv",
    "shot_start": 73274,
    "shot_end": 73460,
    "offset_frames": -19,
    "overlap": [0, 140, 1920, 940],
    "om_size": [1920, 1080],
    "hdr_size": [3840, 1600],
}


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adapter = _load("run_br2049_v04_legacy_p16_confirm", TOOLS / "run_br2049_v04_legacy.py")
v1 = adapter.v1
om = adapter.om


def matrix08_profile(directory: Path) -> Path:
    start, end = MATRIX08["shot_start"], MATRIX08["shot_end"]
    profile = {
        "schema": "openmatte-hdr-input-profile/v1",
        "name": "matrix08_boundary_confirmation",
        "description": "Matrix-08 bounded interval, used to confirm actual shot boundaries before Phase 16.",
        "hdr": MATRIX08["hdr"],
        "om": MATRIX08["om"],
        "ffmpeg": "dev/ffmpeg-build/install/bin/ffmpeg.exe",
        "shot_start": start,
        "shot_end": end,
        "expected_frame_count": end - start,
        "om_start_frame": start + MATRIX08["offset_frames"],
        "offset_frames": MATRIX08["offset_frames"],
        "fps": "24000/1001",
        "hdr_size": MATRIX08["hdr_size"],
        "om_size": MATRIX08["om_size"],
        "overlap": MATRIX08["overlap"],
        "seam_band": 48,
        "feather": 24,
        "cache_root": "tools/openmatte_hdr/cache",
        "legacy_sync": {
            "equation": "om_frame = hdr_frame - 19",
            "frame_offset": MATRIX08["offset_frames"],
            "offset_seconds_at_24000_1001": -0.7924583333333333,
            "status": "LOCKED",
            "frame_locked": True,
            "confidence": 0.940513,
            "historical_time_seconds": 3060.0152916666666,
            "frame_rounding": "inherited from the verified Matrix-08 anchor",
        },
        "legacy_geometry": {
            "scale": [0.5, 0.5],
            "offset": [0, 140],
            "overlap": MATRIX08["overlap"],
            "confidence": 1.0,
            "status": "preserved geometry contract",
        },
        "limitations": ["Boundary confirmation only."],
    }
    target = directory / "matrix08_boundary_confirmation.json"
    target.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


PQ_M1, PQ_M2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
PQ_C1, PQ_C2, PQ_C3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0


def pq_oetf(nits: np.ndarray) -> np.ndarray:
    y = np.clip(nits, 0.0, 10000.0) / 10000.0
    powered = np.power(y, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * powered) / (1.0 + PQ_C3 * powered), PQ_M2)


def proxy_gray(frame: np.ndarray, width: int) -> np.ndarray:
    """Reconstruct the documented proxy: the PQ-encoded master signal as gray in [0, 1].

    The original scan ran ffmpeg `scale=...,format=gray` on the PQ-encoded HDR master.
    The cached frame is PQ-decoded normalized linear light, so the PQ OETF must be
    re-applied. Absolute PQ encoding is used; no per-frame normalization, because
    per-frame normalization would make the histogram and mean-luminance terms move
    with content and manufacture boundaries that do not exist.
    """
    height = max(2, int(round(frame.shape[0] * width / frame.shape[1] / 2.0) * 2))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    luma = 0.2627 * small[..., 0] + 0.6780 * small[..., 1] + 0.0593 * small[..., 2]
    return np.clip(pq_oetf(luma * 10000.0), 0.0, 1.0).astype(np.float32)


def boundary_score(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    hist_left, _ = np.histogram(left, bins=64, range=(0.0, 1.0))
    hist_right, _ = np.histogram(right, bins=64, range=(0.0, 1.0))
    hist_left = hist_left.astype(np.float64) / max(float(hist_left.sum()), 1.0)
    hist_right = hist_right.astype(np.float64) / max(float(hist_right.sum()), 1.0)
    total = hist_left + hist_right
    valid = total > 0
    histogram = (
        float(min(1.0, np.sum((hist_left[valid] - hist_right[valid]) ** 2 / total[valid]) / 2.0))
        if np.any(valid)
        else 0.0
    )
    luminance = float(abs(np.mean(left) - np.mean(right)))
    edge_left = cv2.Canny(np.asarray(left * 255.0, dtype=np.uint8), 32, 96).reshape(-1).astype(np.float64)
    edge_right = cv2.Canny(np.asarray(right * 255.0, dtype=np.uint8), 32, 96).reshape(-1).astype(np.float64)
    edge_left -= np.mean(edge_left)
    edge_right -= np.mean(edge_right)
    denominator = float(np.linalg.norm(edge_left) * np.linalg.norm(edge_right))
    ncc = float(np.dot(edge_left, edge_right) / denominator) if denominator > 1e-12 else 0.0
    edge = max(0.0, 1.0 - ncc)
    return {
        "score": HISTOGRAM_WEIGHT * histogram + LUMINANCE_WEIGHT * luminance + EDGE_WEIGHT * edge,
        "histogram": histogram,
        "luminance": luminance,
        "edge_ncc_change": edge,
    }


ROBUST_WINDOW = 12
ROBUST_K = 6.0
ROBUST_LOCAL_MAX_RADIUS = 2
ROBUST_MIN_SEGMENT_FRAMES = 8


def robust_cuts(
    scores: np.ndarray, histogram: np.ndarray, first_frame: int, last_frame_exclusive: int
) -> dict[str, Any]:
    """Motion-robust cut detection: prominence against a local median, not an absolute threshold.

    In sustained fast action the absolute score stays high for hundreds of frames, so a
    fixed threshold cannot separate cuts from motion. A cut is instead a sharp local
    outlier relative to its own neighbourhood, which is what this measures.

    Predeclared: window +/-12 frames, robust z threshold 6.0 using median and MAD,
    the candidate must be a local maximum within +/-2 frames, and segments shorter
    than 8 frames are marked unresolved rather than pooled.
    """
    count = scores.size
    rows: list[dict[str, Any]] = []
    for index in range(count):
        low = max(0, index - ROBUST_WINDOW)
        high = min(count, index + ROBUST_WINDOW + 1)
        neighbourhood = np.concatenate((scores[low:index], scores[index + 1 : high]))
        median = float(np.median(neighbourhood))
        mad = float(np.median(np.abs(neighbourhood - median)))
        scale = 1.4826 * mad
        z = (float(scores[index]) - median) / scale if scale > 1e-9 else 0.0
        window_low = max(0, index - ROBUST_LOCAL_MAX_RADIUS)
        window_high = min(count, index + ROBUST_LOCAL_MAX_RADIUS + 1)
        is_local_max = bool(scores[index] >= scores[window_low:window_high].max())
        histogram_low = max(0, index - ROBUST_WINDOW)
        histogram_high = min(count, index + ROBUST_WINDOW + 1)
        histogram_median = float(
            np.median(np.concatenate((histogram[histogram_low:index], histogram[index + 1 : histogram_high])))
        )
        rows.append(
            {
                "frame": first_frame + index + 1,
                "score": float(scores[index]),
                "local_median": median,
                "robust_z": z,
                "is_local_max": is_local_max,
                "histogram": float(histogram[index]),
                "histogram_local_median": histogram_median,
                "histogram_is_peak": bool(histogram[index] > histogram_median),
                "accepted": bool(z >= ROBUST_K and is_local_max and histogram[index] > histogram_median),
            }
        )
    accepted = [row for row in rows if row["accepted"]]
    boundaries = sorted({first_frame} | {row["frame"] for row in accepted} | {last_frame_exclusive})
    segments: list[dict[str, Any]] = []
    for position in range(len(boundaries) - 1):
        start, end = boundaries[position], boundaries[position + 1]
        frames = end - start
        segments.append(
            {
                "index": position,
                "start_frame": start,
                "end_frame_exclusive": end,
                "frames": frames,
                "status": "RESOLVED" if frames >= ROBUST_MIN_SEGMENT_FRAMES else "UNRESOLVED_TOO_SHORT",
            }
        )
    return {
        "parameters": {
            "window_frames": ROBUST_WINDOW,
            "robust_z_threshold": ROBUST_K,
            "local_max_radius": ROBUST_LOCAL_MAX_RADIUS,
            "minimum_segment_frames": ROBUST_MIN_SEGMENT_FRAMES,
            "declared_before_inspection": True,
        },
        "candidates": rows,
        "accepted_cuts": accepted,
        "cut_frames": [row["frame"] for row in accepted],
        "segments": segments,
        "resolved_segments": [entry for entry in segments if entry["status"] == "RESOLVED"],
        "unresolved_segments": [entry for entry in segments if entry["status"] != "RESOLVED"],
    }


def boundary_pair_strip(path: Path, pairs: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    """One row per detected cut: the frame before and the frame after."""
    rows = []
    for frame, before, after in pairs:
        tiles = []
        for label, image in ((f"f{frame - 1} before", before), (f"f{frame} after", after)):
            tile = np.round(om.preview(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)) * 255.0).astype(
                np.uint8
            )
            cv2.rectangle(tile, (0, 0), (480, 20), (0, 0, 0), -1)
            cv2.putText(tile, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    if not rows:
        rows = [np.zeros((270, 960, 3), dtype=np.uint8)]
    om.write_png(path, np.concatenate(rows, axis=0))


def segment(scores: np.ndarray, first_frame: int) -> dict[str, Any]:
    hard: list[dict[str, Any]] = []
    for index, score in enumerate(scores):
        prior = scores[index - 1] if index else 0.0
        following = scores[index + 1] if index + 1 < scores.size else 0.0
        if score >= HARD_THRESHOLD and prior < GRADUAL_THRESHOLD and following < GRADUAL_THRESHOLD:
            hard.append({"frame": first_frame + index + 1, "score": float(score), "type": "hard_isolated"})
    above = scores >= GRADUAL_THRESHOLD
    runs: list[dict[str, Any]] = []
    index = 0
    while index < scores.size:
        if not above[index]:
            index += 1
            continue
        start = index
        while index < scores.size and above[index]:
            index += 1
        length = index - start
        window = scores[start:index]
        peaks = [
            {"frame": first_frame + start + offset + 1, "score": float(window[offset])}
            for offset in range(window.size)
            if window[offset] >= HARD_THRESHOLD
        ]
        runs.append(
            {
                "run_start_frame": first_frame + start,
                "run_end_frame_exclusive": first_frame + index,
                "length_frames": int(length),
                "qualifies_as_gradual": bool(length >= GRADUAL_MIN_RUN_FRAMES),
                "peak_frame": first_frame + start + int(np.argmax(window)) + 1,
                "peak_score": float(window.max()),
                "frames_above_hard_threshold": len(peaks),
                "hard_peaks_inside_run": peaks,
            }
        )
    return {"hard_isolated": hard, "gradual_runs": runs}


def score_plot(path: Path, scores: np.ndarray, first_frame: int) -> None:
    height, width = 420, max(960, scores.size * 5)
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    top, bottom, left, right = 40, height - 50, 60, width - 20
    ceiling = max(0.6, float(scores.max()) * 1.1)

    def to_y(value: float) -> int:
        return int(bottom - (value / ceiling) * (bottom - top))

    for value, colour, label in (
        (HARD_THRESHOLD, (90, 90, 255), "hard 0.30"),
        (GRADUAL_THRESHOLD, (90, 200, 255), "gradual 0.15"),
    ):
        y = to_y(value)
        cv2.line(canvas, (left, y), (right, y), colour, 1)
        cv2.putText(canvas, label, (right - 110, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    points = [
        (int(left + (index / max(scores.size - 1, 1)) * (right - left)), to_y(float(value)))
        for index, value in enumerate(scores)
    ]
    for start, end in zip(points, points[1:]):
        cv2.line(canvas, start, end, (120, 255, 120), 1, cv2.LINE_AA)
    for index, value in enumerate(scores):
        if value >= HARD_THRESHOLD:
            x, y = points[index]
            cv2.circle(canvas, (x, y), 3, (90, 90, 255), -1)
            cv2.putText(
                canvas, str(first_frame + index + 1), (x - 22, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (200, 200, 255), 1, cv2.LINE_AA,
            )
    cv2.putText(
        canvas, "Matrix-08 frame-to-frame boundary score", (left, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"frames {first_frame}..{first_frame + scores.size}", (left, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
    )
    om.write_png(path, canvas)


def thumbnail_strip(path: Path, frames: list[tuple[int, np.ndarray]], columns: int = 8) -> None:
    tiles = []
    for number, image in frames:
        tile = np.round(om.preview(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)) * 255.0).astype(
            np.uint8
        )
        cv2.rectangle(tile, (0, 0), (320, 18), (0, 0, 0), -1)
        cv2.putText(tile, f"f{number}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[index : index + columns], axis=1) for index in range(0, len(tiles), columns)]
    om.write_png(path, np.concatenate(rows, axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(prog="confirm-shot-boundaries")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--thumbnail-stride", type=int, default=8)
    args = parser.parse_args()

    profile_path = matrix08_profile(args.output_dir)
    profile = adapter.load_profile(profile_path)
    config = adapter.build_config(profile, args.output_dir)
    v1.CACHE_ROOT = adapter.workspace_path(profile["cache_root"])
    count = int(profile["expected_frame_count"])
    cache = v1.cache_for(config, count)
    om.require(
        cache.valid(),
        f"Matrix-08 cache is required and must not be rebuilt here: {cache.directory}",
    )
    sdr_cache, hdr_cache, manifest = cache.open()
    first_frame = int(profile["shot_start"])
    print(f"Matrix-08 boundary confirmation: {count} frames from {first_frame}, cache reused")

    proxies: list[np.ndarray] = []
    thumbnails: list[tuple[int, np.ndarray]] = []
    for index in range(count):
        frame = np.asarray(hdr_cache[index])
        proxies.append(proxy_gray(frame, PROXY_WIDTH))
        if index % args.thumbnail_stride == 0:
            thumbnails.append((first_frame + index, frame.copy()))
    rows = [boundary_score(proxies[index], proxies[index + 1]) for index in range(count - 1)]
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    histograms = np.asarray([row["histogram"] for row in rows], dtype=np.float64)
    segmentation = segment(scores, first_frame)
    robust = robust_cuts(scores, histograms, first_frame, int(profile["shot_end"]))

    hard_frames = sorted(
        {entry["frame"] for entry in segmentation["hard_isolated"]}
        | {peak["frame"] for run in segmentation["gradual_runs"] for peak in run["hard_peaks_inside_run"]}
    )
    boundaries = [first_frame] + hard_frames + [int(profile["shot_end"])]
    boundaries = sorted(set(boundaries))
    segments = [
        {
            "index": position,
            "start_frame": boundaries[position],
            "end_frame_exclusive": boundaries[position + 1],
            "frames": boundaries[position + 1] - boundaries[position],
        }
        for position in range(len(boundaries) - 1)
    ]

    maps_dir = args.output_dir / "boundary_confirmation"
    maps_dir.mkdir(parents=True, exist_ok=True)
    score_plot(maps_dir / "boundary_scores.png", scores, first_frame)
    thumbnail_strip(maps_dir / "thumbnail_strip.png", thumbnails)
    pairs = [
        (
            frame,
            np.asarray(hdr_cache[frame - 1 - first_frame]).copy(),
            np.asarray(hdr_cache[frame - first_frame]).copy(),
        )
        for frame in robust["cut_frames"]
        if 0 < frame - first_frame < count
    ]
    boundary_pair_strip(maps_dir / "detected_cut_pairs.png", pairs)
    segment_thumbnails = [
        (
            entry["start_frame"],
            np.asarray(hdr_cache[entry["start_frame"] - first_frame]).copy(),
        )
        for entry in robust["segments"]
        if 0 <= entry["start_frame"] - first_frame < count
    ]
    thumbnail_strip(maps_dir / "segment_first_frames.png", segment_thumbnails, columns=6)
    # The original lossless dense sheet exceeded the chat client’s 5 MiB image limit.
    # Retain every frame, but encode a bounded diagnostic representation and produce
    # focused full-resolution review sheets for all materially ambiguous transitions.
    chat_max_bytes = 5 * 1024 * 1024

    def review_tile(number: int, image: np.ndarray, width: int, height: int) -> np.ndarray:
        tile = np.round(om.preview(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)) * 255.0).astype(
            np.uint8
        )
        label_height = 18 if height >= 90 else 16
        font_scale = 0.42 if height >= 90 else 0.32
        cv2.rectangle(tile, (0, 0), (width, label_height), (0, 0, 0), -1)
        cv2.putText(
            tile, str(number), (3, label_height - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return tile

    def review_sheet(frame_numbers: list[int], width: int, height: int, columns: int) -> np.ndarray:
        tiles = [
            review_tile(number, np.asarray(hdr_cache[number - first_frame]), width, height)
            for number in frame_numbers
        ]
        while len(tiles) % columns:
            tiles.append(np.zeros_like(tiles[0]))
        return np.concatenate(
            [np.concatenate(tiles[index : index + columns], axis=1) for index in range(0, len(tiles), columns)],
            axis=0,
        )

    def write_bounded_image(path: Path, image: np.ndarray) -> dict[str, Any]:
        """Encode a deterministic review artifact below the chat attachment cap."""
        import hashlib

        suffix = path.suffix.lower()
        for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            if scale == 1.0:
                candidate = image
            else:
                candidate = cv2.resize(
                    image,
                    (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            parameter_sets = (
                ([cv2.IMWRITE_PNG_COMPRESSION, 9],)
                if suffix == ".png"
                else tuple([cv2.IMWRITE_JPEG_QUALITY, quality] for quality in (92, 88, 84, 80, 76, 72))
            )
            encoded_input = (
                np.ascontiguousarray(candidate[..., ::-1])
                if candidate.ndim == 3 and candidate.shape[2] == 3
                else candidate
            )
            for parameters in parameter_sets:
                ok, encoded = cv2.imencode(suffix, encoded_input, parameters)
                if ok and encoded.size <= chat_max_bytes:
                    path.write_bytes(encoded.tobytes())
                    data = path.read_bytes()
                    return {
                        "path": str(path),
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "dimensions": [int(candidate.shape[1]), int(candidate.shape[0])],
                        "chat_safe": True,
                    }
        raise RuntimeError(f"Could not encode a chat-safe image: {path}")

    dense_path = maps_dir / "all_frames_dense_sheet.png"
    dense_info = write_bounded_image(
        dense_path,
        review_sheet(list(range(first_frame, first_frame + count)), width=160, height=67, columns=14),
    )
    dense_info.update({"role": "all-frame chronological review", "source_frame_interval": [first_frame, first_frame + count]})

    # These windows are fixed review evidence, not detector-selected fit input. They
    # cover the observed hard cut, the ambiguous action transition, and both frames
    # around the candidate late cut; all frames remain available in the dense sheet.
    review_windows = [
        ("review_73289_73301.jpg", 73289, 73302, "early transition review"),
        ("review_73322_73340.jpg", 73322, 73341, "mid-action continuity review"),
        ("review_73428_73440.jpg", 73428, 73441, "high-score late-action review"),
        ("review_73437_73448.jpg", 73437, 73449, "late framing-transition review"),
    ]
    review_infos: list[dict[str, Any]] = []
    for filename, start, end, role in review_windows:
        info = write_bounded_image(
            maps_dir / filename,
            review_sheet(list(range(start, end)), width=320, height=135, columns=4),
        )
        info.update({"role": role, "source_frame_interval": [start, end]})
        review_infos.append(info)

    def describe_artifact(path: Path, role: str, source_frame_interval: list[int] | None = None) -> dict[str, Any]:
        import hashlib

        data = path.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return {
            "path": str(path),
            "role": role,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "dimensions": [int(decoded.shape[1]), int(decoded.shape[0])],
            "chat_safe": len(data) <= chat_max_bytes,
            **({"source_frame_interval": source_frame_interval} if source_frame_interval is not None else {}),
        }

    artifact_manifest = [
        describe_artifact(maps_dir / "boundary_scores.png", "score timeline", [first_frame, first_frame + count]),
        describe_artifact(maps_dir / "thumbnail_strip.png", "stride-8 chronological overview", [first_frame, first_frame + count]),
        describe_artifact(maps_dir / "detected_cut_pairs.png", "robust-detector pair review", [first_frame, first_frame + count]),
        describe_artifact(maps_dir / "segment_first_frames.png", "detector-only segment-start overview", [first_frame, first_frame + count]),
        dense_info,
        *review_infos,
    ]

    payload = {
        "schema": "openmatte-hdr-phase16-boundary-confirmation/v2",
        "protocol": str(HERE / "PHASE16_FROZEN_PROTOCOL.json"),
        "interval": [first_frame, int(profile["shot_end"])],
        "frames": count,
        "verification_status": "PENDING_REVIEWED_SEGMENTATION",
        "proxy": {
            "width": PROXY_WIDTH,
            "source": "cached HDR frames, resized, BT.2020 luma, then absolute PQ OETF encoding",
            "note": "no decoding was performed; the cached HDR frame is the full HDR master frame at half resolution; no per-frame normalization was used",
        },
        "scoring": {
            "histogram_weight": HISTOGRAM_WEIGHT,
            "mean_luminance_weight": LUMINANCE_WEIGHT,
            "edge_ncc_change_weight": EDGE_WEIGHT,
            "hard_threshold": HARD_THRESHOLD,
            "gradual_threshold": GRADUAL_THRESHOLD,
            "gradual_min_run_frames": GRADUAL_MIN_RUN_FRAMES,
        },
        "statistics": {
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_mean": float(scores.mean()),
            "score_p95": float(np.percentile(scores, 95)),
            "frames_above_hard_threshold": int(np.sum(scores >= HARD_THRESHOLD)),
            "frames_above_gradual_threshold": int(np.sum(scores >= GRADUAL_THRESHOLD)),
        },
        "legacy_absolute_threshold_detector": {
            "segmentation": segmentation,
            "threshold_exceedance_frames": hard_frames,
            "implied_segments": segments,
            "verdict": "unusable in this material: the absolute threshold is exceeded by motion for most of the interval",
        },
        "robust_prominence_detector": {
            **robust,
            "interpretation": "DETECTOR_ONLY: no accepted robust cut is not proof that the interval contains one actual shot",
        },
        "p2_34_manifest_claim": {
            "shot_status": "LOCAL_SHOT_RESOLVED",
            "shot_start_frame": 73274,
            "shot_end_frame_exclusive": 73460,
            "shot_duration_frames": 186,
            "closing_boundary_type": "gradual",
            "closing_boundary_run": [73294, 73460],
            "concern": "the closing gradual run spans 166 of 186 frames, and the gradual branch collapses a whole run into one peak, so interior cuts could have been merged",
        },
        "per_frame_scores": [
            {"frame_a": first_frame + index, "frame_b": first_frame + index + 1, **rows[index]}
            for index in range(len(rows))
        ],
        "artifacts": [entry["path"] for entry in artifact_manifest],
        "artifact_manifest": artifact_manifest,
        "source_frame_hashes": manifest["source_frame_hashes"],
    }
    target = args.output_dir / "boundary_confirmation.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"score min {scores.min():.4f}, mean {scores.mean():.4f}, p95 {np.percentile(scores, 95):.4f}, "
        f"max {scores.max():.4f}"
    )
    print(
        f"frames above gradual 0.15: {int(np.sum(scores >= GRADUAL_THRESHOLD))}; "
        f"above hard 0.30: {int(np.sum(scores >= HARD_THRESHOLD))}"
    )
    print(
        f"legacy absolute-threshold detector: {len(hard_frames)} threshold exceedances, {len(segments)} implied segments "
        f"-> unusable in this material"
    )
    print(
        f"\nrobust prominence detector (window +/-{ROBUST_WINDOW}, z >= {ROBUST_K}, local max +/-"
        f"{ROBUST_LOCAL_MAX_RADIUS}):"
    )
    print(f"  detected cuts: {robust['cut_frames'] if robust['cut_frames'] else 'none'}")
    for row in robust["accepted_cuts"]:
        print(
            f"    frame {row['frame']}: score {row['score']:.4f} vs local median {row['local_median']:.4f}, "
            f"robust z {row['robust_z']:.2f}"
        )
    print(f"  segments ({len(robust['segments'])}):")
    for entry in robust["segments"]:
        print(
            f"    {entry['index']}: [{entry['start_frame']}, {entry['end_frame_exclusive']}) = "
            f"{entry['frames']} frames  {entry['status']}"
        )
    print(
        f"  resolved {len(robust['resolved_segments'])}, unresolved {len(robust['unresolved_segments'])}"
    )
    print(f"chat-safe dense sheet: {dense_info['bytes']} bytes")
    print(f"\n{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
