#!/usr/bin/env python3
"""Compare boundary-gain estimation band widths on one shot.

The v0.1 pipeline is reused unchanged: identical decode, identical base/detail
math, identical composite. The only variable is how many HDR rows next to each
seam are used to estimate the gain.

Speed comes from decoding and computing the shared base/detail terms once for
all variants, and from encoding every review output on the NVIDIA encoder.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("openmatte_hdr", HERE / "openmatte_hdr.py")
om = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = om
SPEC.loader.exec_module(om)

BANDS = (24, 48, 96, 192)
OUT_DIR = HERE / "out" / "seam_band_compare"
MATRIX = {
    "hdr": "G:\\Filmy\\The Matrix UHD\\The Matrix.mkv",
    "om": "G:\\Filmy\\IMAX format (open matte)\\The Matrix (1999) [OPEN MATTE] [WEB-DL 1080p 10bit DD5.1 x265].mkv",
    "shot_start": 73274,
    "shot_end": 73460,
    "offset": -19,
}


def nvenc_command(config: om.Config, path: Path, width: int, height: int) -> list[str]:
    return [str(config.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{width}x{height}", "-framerate", config.fps_text, "-i", "pipe:0", "-an", "-vf", om.SETPARAMS, "-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p5", "-cq", "16", "-pix_fmt", "p010le", *om.HDR_TAGS, str(path)]


def band_profiles(config: om.Config, hdr_y: np.ndarray, base_y: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray]:
    """Exactly the v0.1 seam measurement, but on precomputed luminance and a given band."""
    _, y1, _, y2 = config.overlap
    top = np.median(np.log2((hdr_y[:band] + om.EPS) / (base_y[y1:y1 + band] + om.EPS)), axis=0)
    bottom = np.median(np.log2((hdr_y[-band:] + om.EPS) / (base_y[y2 - band:y2] + om.EPS)), axis=0)
    return top.astype(np.float64), bottom.astype(np.float64)


def finalize_gain(config: om.Config, top_rows: list[np.ndarray], bottom_rows: list[np.ndarray]) -> dict[str, Any]:
    """The unchanged v0.1 aggregation: median over frames, smooth in x, clamp to the shot median."""
    top_median = np.median(np.stack(top_rows), axis=0)
    bottom_median = np.median(np.stack(bottom_rows), axis=0)
    shot_stops = float(np.median(np.concatenate((top_median, bottom_median))))
    limit = config.gain_clamp_stops
    top_smooth = om.smooth_profile(top_median, config.gain_smooth_sigma)
    bottom_smooth = om.smooth_profile(bottom_median, config.gain_smooth_sigma)
    top_profile = np.clip(top_smooth, shot_stops - limit, shot_stops + limit)
    bottom_profile = np.clip(bottom_smooth, shot_stops - limit, shot_stops + limit)
    return {
        "shot_gain_stops": shot_stops,
        "top_profile_stops": top_profile,
        "bottom_profile_stops": bottom_profile,
        "top_profile_range_stops": [float(top_profile.min()), float(top_profile.max())],
        "bottom_profile_range_stops": [float(bottom_profile.min()), float(bottom_profile.max())],
        "top_profile_std_stops": float(top_profile.std()),
        "bottom_profile_std_stops": float(bottom_profile.std()),
        "clamped_fraction": float(np.mean(np.abs(np.concatenate((top_smooth, bottom_smooth)) - shot_stops) > limit)),
        "per_frame_measurement_spread_stops": float(np.percentile(np.stack(top_rows), 95) - np.percentile(np.stack(top_rows), 5)),
    }


def label_overlay(width: int, height: int, labels: list[str]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    half_w, half_h = width // 2, height // 2
    for index, text in enumerate(labels):
        x = (index % 2) * half_w + 12
        y = (index // 2) * half_h + 34
        cv2.putText(mask, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    return mask


def measure(config: om.Config, count: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, tuple[list[np.ndarray], list[np.ndarray]]] = {band: ([], []) for band in BANDS}
    started = time.perf_counter()
    with om.pair_stream(config, count) as frames:
        for hdr_common, sdr, _ in frames:
            hdr_y = om.compute_luminance(hdr_common)
            base_y = om.compute_luminance(om.base_layer(config, sdr))
            for band in BANDS:
                top, bottom = band_profiles(config, hdr_y, base_y, band)
                rows[band][0].append(top)
                rows[band][1].append(bottom)
    result = {band: finalize_gain(config, rows[band][0], rows[band][1]) for band in BANDS}
    print(f"measurement pass for {len(BANDS)} bands: {time.perf_counter() - started:.1f} s")
    return result


def render(config: om.Config, gains: dict[int, dict[str, Any]], count: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, y1, _, y2 = config.overlap
    om_width, om_height = config.om_size
    fields = {band: om.gain_field(config, gains[band]) for band in BANDS}
    encoders = {band: subprocess.Popen(nvenc_command(config, OUT_DIR / f"band_{band:03d}px_hdr10.mkv", om_width, om_height), stdin=subprocess.PIPE, stderr=subprocess.PIPE) for band in BANDS}
    grid_path = OUT_DIR / "grid_2x2_hdr10.mkv"
    grid_encoder = subprocess.Popen(nvenc_command(config, grid_path, om_width, om_height), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    overlay = label_overlay(om_width, om_height, [f"{band} px" for band in BANDS])
    extension = np.r_[0:y1, y2:om_height]
    sample_index = int(round((count - 1) * 0.5))
    samples: dict[str, np.ndarray] = {}
    stats: dict[int, list[dict[str, float]]] = {band: [] for band in BANDS}
    diff_rows: dict[int, list[dict[str, float]]] = {band: [] for band in BANDS}
    rendered = 0
    started = time.perf_counter()
    try:
        with om.pair_stream(config, count) as frames:
            for index, (hdr_common, sdr, _) in enumerate(frames):
                base = om.base_layer(config, sdr)
                low, high = config.detail_ratio_clip
                ratio_raw = sdr / np.maximum(base, om.EPS)
                ratio = np.clip(ratio_raw, low, high)
                detail = np.power(ratio, config.detail_strength)
                shared = {
                    "detail_ratio_clip_low_fraction": float(np.mean(ratio_raw < low)),
                    "detail_ratio_clip_high_fraction": float(np.mean(ratio_raw > high)),
                }
                quadrants: list[np.ndarray] = []
                baseline_luma: np.ndarray | None = None
                for band in BANDS:
                    predicted_raw = (fields[band][..., None] * base) * detail
                    predicted = np.clip(predicted_raw, 0.0, 1.0).astype(np.float32)
                    output = om.composite(config, predicted, hdr_common)
                    pq16 = om.to_pq16(output)
                    encoders[band].stdin.write(pq16.tobytes())
                    quadrants.append(cv2.resize(pq16, (om_width // 2, om_height // 2), interpolation=cv2.INTER_AREA))
                    luma = om.compute_luminance(output[extension]) * om.PEAK_NITS
                    if band == BANDS[0]:
                        baseline_luma = luma
                        difference = {"mean_abs_nits_vs_baseline": 0.0, "P95_abs_nits_vs_baseline": 0.0}
                    else:
                        delta = np.abs(luma - baseline_luma)
                        difference = {"mean_abs_nits_vs_baseline": float(delta.mean()), "P95_abs_nits_vs_baseline": float(np.percentile(delta, 95))}
                    diff_rows[band].append(difference)
                    stats[band].append({
                        **shared, **om.seam_continuity(config, output),
                        "output_clip_high_fraction": float(np.mean(predicted_raw > 1.0)),
                        "extension_mean_nits": float(luma.mean()),
                        "extension_P95_nits": float(np.percentile(luma, 95)),
                    })
                    if index == sample_index:
                        samples[f"band_{band}"] = output.copy()
                grid = np.concatenate((np.concatenate(quadrants[:2], axis=1), np.concatenate(quadrants[2:], axis=1)), axis=0)
                grid[overlay > 0] = 60000
                grid_encoder.stdin.write(grid.tobytes())
                if index == sample_index:
                    reference = np.zeros_like(hdr_common, shape=(om_height, om_width, 3))
                    reference[y1:y2] = hdr_common
                    samples["reference"] = reference
                rendered += 1
    finally:
        processes = list(encoders.values()) + [grid_encoder]
        for process in processes:
            if process.stdin:
                process.stdin.close()
        codes = [process.wait() for process in processes]
        errors = [process.stderr.read().decode("utf-8", errors="replace") for process in processes]
        om.require(all(code == 0 for code in codes), f"NVENC encoding failed: {codes}; {errors}")
    om.require(rendered == count, f"Render incomplete: {rendered}/{count}")
    print(f"render pass for {len(BANDS)} bands + grid: {time.perf_counter() - started:.1f} s ({(time.perf_counter() - started) / count:.3f} s/frame)")
    summary: dict[str, Any] = {}
    for band in BANDS:
        frame_stats = stats[band]
        extension_means = np.array([row["extension_mean_nits"] for row in frame_stats])
        summary[str(band)] = {
            "seam_top_mean_nits": float(np.mean([row["top_seam_mean_nits"] for row in frame_stats])),
            "seam_top_P95_nits": float(np.mean([row["top_seam_P95_nits"] for row in frame_stats])),
            "seam_bottom_mean_nits": float(np.mean([row["bottom_seam_mean_nits"] for row in frame_stats])),
            "seam_bottom_P95_nits": float(np.mean([row["bottom_seam_P95_nits"] for row in frame_stats])),
            "extension_mean_nits": float(extension_means.mean()),
            "extension_temporal_std_nits": float(extension_means.std()),
            "extension_P95_nits": float(np.mean([row["extension_P95_nits"] for row in frame_stats])),
            "output_clip_high_fraction_max": float(max(row["output_clip_high_fraction"] for row in frame_stats)),
            "mean_abs_nits_vs_24px": float(np.mean([row["mean_abs_nits_vs_baseline"] for row in diff_rows[band]])),
            "P95_abs_nits_vs_24px": float(np.mean([row["P95_abs_nits_vs_baseline"] for row in diff_rows[band]])),
            "review_video": str((OUT_DIR / f"band_{band:03d}px_hdr10.mkv").relative_to(HERE)),
        }
    contact_sheet(config, samples, sample_index)
    return {"per_band": summary, "grid_video": str(grid_path.relative_to(HERE)), "grid_layout": [f"{band} px" for band in BANDS], "sample_frame_index": sample_index, "rendered_frames": rendered}


def contact_sheet(config: om.Config, samples: dict[str, np.ndarray], sample_index: int) -> None:
    _, y1, _, y2 = config.overlap
    crop_x = 720

    def tile(image: np.ndarray, label: str) -> np.ndarray:
        view = cv2.resize(np.round(om.preview(image) * 255.0).astype(np.uint8), (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(view, (0, 0), (480, 24), (0, 0, 0), -1)
        cv2.putText(view, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def crop(image: np.ndarray, seam: int, label: str) -> np.ndarray:
        top = max(0, min(image.shape[0] - 270, seam - 135))
        patch = np.round(om.preview(image[top:top + 270, crop_x:crop_x + 480]) * 255.0).astype(np.uint8)
        cv2.rectangle(patch, (0, 0), (480, 22), (0, 0, 0), -1)
        cv2.putText(patch, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return patch

    rows = [np.concatenate((tile(samples["reference"], "HDR reference"), crop(samples["reference"], y1, "reference top seam 1:1"), crop(samples["reference"], y2, "reference bottom seam 1:1")), axis=1)]
    for band in BANDS:
        image = samples[f"band_{band}"]
        rows.append(np.concatenate((tile(image, f"{band} px band"), crop(image, y1, f"{band} px top seam 1:1"), crop(image, y2, f"{band} px bottom seam 1:1")), axis=1))
    om.write_png(OUT_DIR / "seam_band_contact_sheet.png", np.concatenate(rows, axis=0))


def report(gains: dict[int, dict[str, Any]], result: dict[str, Any]) -> str:
    lines = [
        "# Boundary-gain band width comparison - Matrix-08", "",
        f"One shot, {result['rendered_frames']} frames. The v0.1 decode, base/detail transform, composite and feather are unchanged; only the number of HDR rows used to estimate the gain differs. Decode and the shared base/detail terms were computed once for all four variants, and every review file was encoded on the NVIDIA encoder.", "",
        "## Gain estimation", "| Band | Shot gain (stops) | Top profile range | Top profile std | Bottom profile range | Clamped |", "|---|---:|---|---:|---|---:|",
    ]
    for band in BANDS:
        gain = gains[band]
        top = gain["top_profile_range_stops"]
        bottom = gain["bottom_profile_range_stops"]
        lines.append(f"| {band} px | {gain['shot_gain_stops']:+.4f} | {top[0]:+.4f} .. {top[1]:+.4f} | {gain['top_profile_std_stops']:.4f} | {bottom[0]:+.4f} .. {bottom[1]:+.4f} | {gain['clamped_fraction'] * 100:.2f}% |")
    lines += ["", "## Rendered result", "| Band | Top seam mean | Bottom seam mean | Extension mean | Temporal std | vs 24 px mean | vs 24 px P95 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for band in BANDS:
        row = result["per_band"][str(band)]
        lines.append(f"| {band} px | {row['seam_top_mean_nits']:.3f} nits | {row['seam_bottom_mean_nits']:.3f} nits | {row['extension_mean_nits']:.3f} nits | {row['extension_temporal_std_nits']:.3f} nits | {row['mean_abs_nits_vs_24px']:.4f} nits | {row['P95_abs_nits_vs_24px']:.4f} nits |")
    lines += [
        "", "## Review artifacts",
        f"- 2x2 grid, layout {result['grid_layout']}: `{result['grid_video']}`",
        *[f"- {band} px: `{result['per_band'][str(band)]['review_video']}`" for band in BANDS],
        "- `out/seam_band_compare/seam_band_contact_sheet.png`", "",
        "All review files are HDR10 (PQ, BT.2020). The extension regions have no HDR ground truth, so the seam and difference values are diagnostics and the decision is visual.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    argv = [
        "--hdr", MATRIX["hdr"], "--om", MATRIX["om"],
        "--output", str(OUT_DIR / "unused.mkv"),
        "--shot-start", str(MATRIX["shot_start"]), "--shot-end", str(MATRIX["shot_end"]),
        "--offset-frames", str(MATRIX["offset"]),
    ]
    if len(sys.argv) > 1:
        argv += ["--max-frames", sys.argv[1]]
    config = om.build_config(argv)
    count = config.frame_count
    print(f"seam-band comparison: {BANDS} px on HDR [{config.shot_start},{config.shot_end}) -> {count} frames")
    started = time.perf_counter()
    gains = measure(config, count)
    for band in BANDS:
        gain = gains[band]
        print(f"  {band:3d} px: shot {gain['shot_gain_stops']:+.4f} stops, top {gain['top_profile_range_stops'][0]:+.4f}..{gain['top_profile_range_stops'][1]:+.4f}, std {gain['top_profile_std_stops']:.4f}")
    result = render(config, gains, count)
    payload = {
        "shot": {"hdr_interval": [config.shot_start, config.shot_end], "frames": count, "offset_frames": config.offset_frames, "overlap": list(config.overlap)},
        "unchanged_from_v01": {"base_sigma_px": config.base_sigma, "detail_strength": config.detail_strength, "detail_ratio_clip": list(config.detail_ratio_clip), "gain_smooth_sigma_px": config.gain_smooth_sigma, "gain_clamp_stops": config.gain_clamp_stops, "feather_px": config.feather},
        "bands_px": list(BANDS),
        "gain": {str(band): {key: value for key, value in gains[band].items() if not key.endswith("_profile_stops")} for band in BANDS},
        "result": result,
        "total_seconds": time.perf_counter() - started,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "comparison.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT_DIR / "COMPARISON.md").write_text(report(gains, result), encoding="utf-8")
    print(f"total {payload['total_seconds']:.1f} s")
    print(OUT_DIR / "COMPARISON.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
