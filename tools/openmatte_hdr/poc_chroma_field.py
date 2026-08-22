#!/usr/bin/env python3
"""PoC: low-frequency chroma field for Open Matte extensions.

The v0.1 luminance path is unchanged (base sigma 16, detail^0.85, boundary gain
measured on a 48 px band). On top of it, only the *heavily smoothed* chroma of
the extension is corrected by a similarity transform in the ICtCp chroma plane:
one scale (saturation) and one rotation (hue) per location.

The field is deliberately low order: a quadratic in x per seam, blended
vertically towards the shot-level value. That is 12 real numbers of chroma
correction for the whole scene. Chroma detail and intensity are untouched, and
the HDR centre is copied, never regenerated.

No temporal state, no AI, no LUT, no MMR.
"""
from __future__ import annotations

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

from reshaping_research.utils.color_spaces import BT2020_TO_LMS, ICTCP_TO_LMS, LMS_TO_BT2020, LMS_TO_ICTCP  # noqa: E402
from reshaping_research.utils.transfer_functions import pq_eotf, pq_oetf  # noqa: E402

SEAM_BAND = 48
CHROMA_SIGMA = 16.0
POLYNOMIAL_DEGREE = 2
SCALE_CLAMP = (0.50, 2.00)
ROTATION_CLAMP_DEG = 30.0
# The chroma field is one least-squares fit for the whole scene, so the accumulators do not
# need every frame; the diagnostics need even fewer.
MEASURE_STRIDE = 3
DIAGNOSTIC_STRIDE = 6
OUT_DIR = HERE / "out" / "chroma_field_poc"
MATRIX = {
    "hdr": "G:\\Filmy\\The Matrix UHD\\The Matrix.mkv",
    "om": "G:\\Filmy\\IMAX format (open matte)\\The Matrix (1999) [OPEN MATTE] [WEB-DL 1080p 10bit DD5.1 x265].mkv",
    "shot_start": 73274,
    "shot_end": 73460,
    "offset": -19,
}
REGIONS = (("wall_wood", 40), ("dark_clothing", 300), ("skin_head", 780), ("rice_paper_screens", 1400))

M_RGB_LMS = BT2020_TO_LMS.astype(np.float32)
M_LMS_ICTCP = LMS_TO_ICTCP.astype(np.float32)
M_ICTCP_LMS = ICTCP_TO_LMS.astype(np.float32)
M_LMS_RGB = LMS_TO_BT2020.astype(np.float32)


def to_ictcp(rgb_nits: np.ndarray) -> np.ndarray:
    lms = np.maximum(np.einsum("ij,...j->...i", M_RGB_LMS, rgb_nits), 0.0)
    return np.einsum("ij,...j->...i", M_LMS_ICTCP, pq_oetf(lms).astype(np.float32))


def from_ictcp(ictcp: np.ndarray) -> np.ndarray:
    lms = pq_eotf(np.einsum("ij,...j->...i", M_ICTCP_LMS, ictcp)).astype(np.float32)
    return np.einsum("ij,...j->...i", M_LMS_RGB, lms)


def smooth_chroma(chroma: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(chroma, (0, 0), sigmaX=CHROMA_SIGMA, sigmaY=CHROMA_SIGMA, borderType=cv2.BORDER_REFLECT_101)


def clamp_z(z: np.ndarray) -> tuple[np.ndarray, float]:
    scale = np.abs(z)
    angle = np.angle(z)
    limit = np.deg2rad(ROTATION_CLAMP_DEG)
    clamped = np.mean((scale < SCALE_CLAMP[0]) | (scale > SCALE_CLAMP[1]) | (np.abs(angle) > limit))
    scale = np.clip(scale, *SCALE_CLAMP)
    angle = np.clip(angle, -limit, limit)
    return scale * np.exp(1j * angle), float(clamped)


def predict(config: om.Config, sdr: np.ndarray, field: np.ndarray) -> np.ndarray:
    """The unchanged v0.1 luminance prediction."""
    base = om.base_layer(config, sdr)
    low, high = config.detail_ratio_clip
    ratio = np.clip(sdr / np.maximum(base, om.EPS), low, high)
    return np.clip((field[..., None] * base) * np.power(ratio, config.detail_strength), 0.0, 1.0).astype(np.float32)


def measure_chroma(config: om.Config, field: np.ndarray, count: int) -> dict[str, Any]:
    """One complex least-squares fit per seam over the entire shot."""
    _, y1, _, y2 = config.overlap
    width = config.om_size[0]
    accumulators = {"top": [np.zeros(width, np.complex128), np.zeros(width, np.float64)], "bottom": [np.zeros(width, np.complex128), np.zeros(width, np.float64)]}
    before: list[float] = []
    used = 0
    started = time.perf_counter()
    with om.pair_stream(config, count) as frames:
        for index, (hdr_common, sdr, _) in enumerate(frames):
            if index % MEASURE_STRIDE:
                continue
            used += 1
            predicted = predict(config, sdr, field)
            pred_chroma = smooth_chroma(to_ictcp(predicted * om.PEAK_NITS)[..., 1:])
            hdr_chroma = smooth_chroma(to_ictcp(hdr_common * om.PEAK_NITS)[..., 1:])
            bands = {
                "top": (pred_chroma[y1:y1 + SEAM_BAND], hdr_chroma[:SEAM_BAND]),
                "bottom": (pred_chroma[y2 - SEAM_BAND:y2], hdr_chroma[-SEAM_BAND:]),
            }
            for name, (pred_band, hdr_band) in bands.items():
                p = pred_band[..., 0].astype(np.float64) + 1j * pred_band[..., 1].astype(np.float64)
                h = hdr_band[..., 0].astype(np.float64) + 1j * hdr_band[..., 1].astype(np.float64)
                accumulators[name][0] += np.sum(np.conj(p) * h, axis=0)
                accumulators[name][1] += np.sum(np.abs(p) ** 2, axis=0)
                before.append(float(np.mean(np.abs(h - p))))
    coefficients: dict[str, list[complex]] = {}
    per_column: dict[str, np.ndarray] = {}
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    total_num = sum(accumulators[name][0].sum() for name in accumulators)
    total_den = sum(accumulators[name][1].sum() for name in accumulators)
    z_shot = complex(total_num / max(total_den, 1e-30))
    clamped: dict[str, float] = {}
    for name, (num, den) in accumulators.items():
        weights = np.sqrt(np.maximum(den, 0.0))
        z_column = num / np.maximum(den, 1e-30)
        real = np.polyfit(x, z_column.real, POLYNOMIAL_DEGREE, w=weights)
        imag = np.polyfit(x, z_column.imag, POLYNOMIAL_DEGREE, w=weights)
        fitted = np.polyval(real, x) + 1j * np.polyval(imag, x)
        limited, clamp_fraction = clamp_z(fitted)
        coefficients[name] = [complex(r, i) for r, i in zip(real, imag)]
        per_column[name] = limited
        clamped[name] = clamp_fraction
    z_shot_limited, _ = clamp_z(np.asarray([z_shot]))
    return {
        "coefficients": coefficients,
        "per_column": per_column,
        "z_shot": complex(z_shot_limited[0]),
        "clamped_fraction": clamped,
        "band_chroma_error_before": float(np.mean(before)),
        "measurement_seconds": time.perf_counter() - started,
        "frames_used": used,
        "measure_stride": MEASURE_STRIDE,
        "parameter_count": 2 * 2 * (POLYNOMIAL_DEGREE + 1),
    }


def chroma_field(config: om.Config, chroma: dict[str, Any]) -> np.ndarray:
    """Quadratic in x at each seam, blended vertically towards the shot value."""
    _, y1, _, y2 = config.overlap
    width, height = config.om_size
    top = chroma["per_column"]["top"][None, :]
    bottom = chroma["per_column"]["bottom"][None, :]
    shot = chroma["z_shot"]
    field = np.empty((height, width), dtype=np.complex128)
    if y1 > 0:
        weight = ((y1 - np.arange(y1, dtype=np.float64)) / float(y1))[:, None]
        field[:y1] = (1.0 - weight) * top + weight * shot
    rows = y2 - y1
    position = (np.arange(rows, dtype=np.float64) / float(max(rows - 1, 1)))[:, None]
    field[y1:y2] = (1.0 - position) * top + position * bottom
    if y2 < height:
        weight = ((np.arange(y2, height, dtype=np.float64) - (y2 - 1)) / float(height - y2))[:, None]
        field[y2:] = (1.0 - weight) * bottom + weight * shot
    return field


def apply_chroma(predicted: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns the corrected image, its corrected smoothed chroma, and the negative fraction."""
    ictcp = to_ictcp(predicted * om.PEAK_NITS)
    chroma = ictcp[..., 1:]
    base = smooth_chroma(chroma)
    detail = chroma - base
    ct, cp = base[..., 0], base[..., 1]
    real = field.real.astype(np.float32)
    imag = field.imag.astype(np.float32)
    corrected_base = np.stack((ct * real - cp * imag, ct * imag + cp * real), axis=-1)
    ictcp[..., 1:] = corrected_base + detail
    result = from_ictcp(ictcp) / om.PEAK_NITS
    negative = float(np.mean(result < -1e-4))
    return np.clip(result, 0.0, 1.0).astype(np.float32), corrected_base, negative


def seam_chroma_metrics(output: np.ndarray, config: om.Config) -> dict[str, float]:
    _, y1, _, y2 = config.overlap
    result: dict[str, float] = {}
    for label, outside, inside in (("top", y1 - 1, y1), ("bottom", y2 - 1, y2)):
        rows = to_ictcp(output[[outside, inside]] * om.PEAK_NITS)
        chroma = np.hypot(rows[..., 1], rows[..., 2])
        hue = np.degrees(np.arctan2(rows[..., 2], rows[..., 1]))
        difference = np.abs(((hue[0] - hue[1] + 180.0) % 360.0) - 180.0)
        valid = (chroma[0] > 1e-4) & (chroma[1] > 1e-4)
        result[f"{label}_chroma_step"] = float(np.mean(np.abs(chroma[0] - chroma[1])))
        result[f"{label}_hue_step_degrees"] = float(np.mean(difference[valid])) if valid.any() else 0.0
    return result


def nvenc(config: om.Config, path: Path, width: int, height: int) -> subprocess.Popen[bytes]:
    command = [str(config.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{width}x{height}", "-framerate", config.fps_text, "-i", "pipe:0", "-an", "-vf", om.SETPARAMS, "-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p5", "-cq", "16", "-pix_fmt", "p010le", *om.HDR_TAGS, str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def region_sheet(path: Path, seam: int, samples: dict[str, np.ndarray], height: int) -> None:
    top = max(0, min(height - 270, seam - 135))
    rows = []
    for name, x in REGIONS:
        tiles = []
        for variant in ("luminance_only", "luminance_plus_chroma"):
            patch = np.round(om.preview(samples[variant][top:top + 270, x:x + 480]) * 255.0).astype(np.uint8)
            cv2.rectangle(patch, (0, 0), (480, 22), (0, 0, 0), -1)
            cv2.putText(patch, f"{name} / {variant}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(patch, (0, seam - top), (480, seam - top), (0, 0, 0), 1)
            tiles.append(patch)
        rows.append(np.concatenate(tiles, axis=1))
    om.write_png(path, np.concatenate(rows, axis=0))


def render(config: om.Config, gain: np.ndarray, chroma: dict[str, Any], count: int) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    om_width, om_height = config.om_size
    field = chroma_field(config, chroma)
    encoders = {
        "luminance_only": nvenc(config, OUT_DIR / "v01_luminance_only_hdr10.mkv", om_width, om_height),
        "luminance_plus_chroma": nvenc(config, OUT_DIR / "v02_luminance_plus_chroma_hdr10.mkv", om_width, om_height),
    }
    side_by_side = nvenc(config, OUT_DIR / "side_by_side_hdr10.mkv", om_width * 2, om_height)
    sample_index = int(round((count - 1) * 0.5))
    _, y1, _, y2 = config.overlap
    samples: dict[str, np.ndarray] = {}
    stats: list[dict[str, float]] = []
    residual_after: list[float] = []
    roundtrip: dict[str, float] = {}
    rendered = 0
    started = time.perf_counter()
    try:
        with om.pair_stream(config, count) as frames:
            for index, (hdr_common, sdr, _) in enumerate(frames):
                predicted = predict(config, sdr, gain)
                corrected, corrected_chroma, negative = apply_chroma(predicted, field)
                outputs = {"luminance_only": om.composite(config, predicted, hdr_common), "luminance_plus_chroma": om.composite(config, corrected, hdr_common)}
                pq = {}
                for name, image in outputs.items():
                    pq[name] = om.to_pq16(image)
                    encoders[name].stdin.write(pq[name].tobytes())
                side_by_side.stdin.write(np.concatenate((pq["luminance_only"], pq["luminance_plus_chroma"]), axis=1).tobytes())
                row = {f"gain_{key}": value for key, value in seam_chroma_metrics(outputs["luminance_only"], config).items()}
                row.update({f"chroma_{key}": value for key, value in seam_chroma_metrics(outputs["luminance_plus_chroma"], config).items()})
                row["negative_fraction_after_chroma"] = negative
                stats.append(row)
                if index % DIAGNOSTIC_STRIDE == 0:
                    hdr_chroma = smooth_chroma(to_ictcp(hdr_common * om.PEAK_NITS)[..., 1:])
                    residual_after.append(float(np.mean(np.abs(hdr_chroma[:SEAM_BAND] - corrected_chroma[y1:y1 + SEAM_BAND]))))
                    residual_after.append(float(np.mean(np.abs(hdr_chroma[-SEAM_BAND:] - corrected_chroma[y2 - SEAM_BAND:y2]))))
                if index == sample_index:
                    samples = {name: image.copy() for name, image in outputs.items()}
                    identity, _, _ = apply_chroma(predicted, np.ones_like(field))
                    roundtrip = {"identity_roundtrip_max_abs_nits": float(np.max(np.abs(identity - predicted)) * om.PEAK_NITS), "identity_roundtrip_mean_abs_nits": float(np.mean(np.abs(identity - predicted)) * om.PEAK_NITS)}
                rendered += 1
    finally:
        processes = list(encoders.values()) + [side_by_side]
        for process in processes:
            if process.stdin:
                process.stdin.close()
        codes = [process.wait() for process in processes]
        errors = [process.stderr.read().decode("utf-8", errors="replace") for process in processes]
        om.require(all(code == 0 for code in codes), f"NVENC failed: {codes}; {errors}")
    om.require(rendered == count, f"Render incomplete: {rendered}/{count}")
    region_sheet(OUT_DIR / "top_seam_regions.png", y1, samples, om_height)
    region_sheet(OUT_DIR / "bottom_seam_regions.png", y2, samples, om_height)
    full = [np.concatenate([np.round(om.preview(cv2.resize(samples[name], (960, 540), interpolation=cv2.INTER_AREA)) * 255.0).astype(np.uint8) for name in ("luminance_only", "luminance_plus_chroma")], axis=1)]
    om.write_png(OUT_DIR / "full_frame_pair.png", np.concatenate(full, axis=0))
    aggregate = {key: float(np.mean([row[key] for row in stats])) for key in stats[0]}
    aggregate["band_chroma_error_after"] = float(np.mean(residual_after))
    aggregate.update(roundtrip)
    return {"rendered_frames": rendered, "render_seconds": time.perf_counter() - started, "seam_and_range": aggregate}


def report(chroma: dict[str, Any], result: dict[str, Any], gain_info: dict[str, Any]) -> str:
    metrics = result["seam_and_range"]
    improvement = 100.0 * (1.0 - metrics["band_chroma_error_after"] / max(chroma["band_chroma_error_before"], 1e-12))
    lines = [
        "# PoC - low-frequency chroma field", "",
        f"Matrix-08, {result['rendered_frames']} frames. The v0.1 luminance path is unchanged: base sigma 16, detail^0.85, boundary gain on a {SEAM_BAND} px band ({gain_info['shot_gain_stops']:+.4f} stops shot level). The HDR centre is copied, not regenerated.", "",
        "## The correction",
        f"Only the ICtCp chroma smoothed at sigma {CHROMA_SIGMA:.0f} is corrected, by one complex factor per location: magnitude is saturation, angle is hue. Intensity and chroma detail are untouched. The field is a degree-{POLYNOMIAL_DEGREE} polynomial in x at each seam, blended vertically to the shot value, so the entire scene correction is **{chroma['parameter_count']} real numbers**.", "",
        "| Seam | Scale at x=0 | Scale at x=mid | Scale at x=max | Hue at x=0 | Hue at x=mid | Hue at x=max | Clamped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("top", "bottom"):
        column = chroma["per_column"][name]
        picks = [column[0], column[len(column) // 2], column[-1]]
        scales = " | ".join(f"{abs(value):.4f}" for value in picks)
        hues = " | ".join(f"{np.degrees(np.angle(value)):+.3f}°" for value in picks)
        lines.append(f"| {name} | {scales} | {hues} | {chroma['clamped_fraction'][name] * 100:.2f}% |")
    lines += [
        "", f"Shot-level factor: scale `{abs(chroma['z_shot']):.4f}`, hue `{np.degrees(np.angle(chroma['z_shot'])):+.3f}°`.", "",
        "## Numeric diagnostics", "| Quantity | Luminance only | Luminance + chroma field |", "|---|---:|---:|",
        f"| Top seam chroma step | {metrics['gain_top_chroma_step']:.6f} | {metrics['chroma_top_chroma_step']:.6f} |",
        f"| Bottom seam chroma step | {metrics['gain_bottom_chroma_step']:.6f} | {metrics['chroma_bottom_chroma_step']:.6f} |",
        f"| Top seam hue step | {metrics['gain_top_hue_step_degrees']:.3f}° | {metrics['chroma_top_hue_step_degrees']:.3f}° |",
        f"| Bottom seam hue step | {metrics['gain_bottom_hue_step_degrees']:.3f}° | {metrics['chroma_bottom_hue_step_degrees']:.3f}° |",
        "",
        f"Smoothed-chroma disagreement inside the seam bands: `{chroma['band_chroma_error_before']:.6f}` before, `{metrics['band_chroma_error_after']:.6f}` after ({improvement:+.1f}%).",
        f"ICtCp round-trip at identity: mean `{metrics['identity_roundtrip_mean_abs_nits']:.6f}` nits, max `{metrics['identity_roundtrip_max_abs_nits']:.4f}` nits, so the colour path itself is effectively transparent.",
        f"Negative RGB fraction after correction: `{metrics['negative_fraction_after_chroma']:.8f}`.", "",
        "## Visual review, which is the actual criterion",
        "- `out/chroma_field_poc/side_by_side_hdr10.mkv` - left luminance only, right with the chroma field",
        "- `out/chroma_field_poc/v01_luminance_only_hdr10.mkv`, `out/chroma_field_poc/v02_luminance_plus_chroma_hdr10.mkv`",
        "- `out/chroma_field_poc/top_seam_regions.png`, `out/chroma_field_poc/bottom_seam_regions.png` - 1:1 crops across the seam for " + ", ".join(name for name, _ in REGIONS),
        "- `out/chroma_field_poc/full_frame_pair.png`", "",
        "Judge rice paper and walls for tint, skin for plausibility, dark clothing for colour cast, and saturated elements for over- or under-saturation. The extension has no HDR ground truth, so these numbers cannot decide acceptance.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    argv = [
        "--hdr", MATRIX["hdr"], "--om", MATRIX["om"], "--output", str(OUT_DIR / "unused.mkv"),
        "--shot-start", str(MATRIX["shot_start"]), "--shot-end", str(MATRIX["shot_end"]),
        "--offset-frames", str(MATRIX["offset"]), "--seam-band", str(SEAM_BAND),
    ]
    if len(sys.argv) > 1:
        argv += ["--max-frames", sys.argv[1]]
    config = om.build_config(argv)
    count = config.frame_count
    print(f"chroma-field PoC: {count} frames, seam band {SEAM_BAND} px")
    started = time.perf_counter()
    gain_info = om.measure_gain(config, count)
    gain = om.gain_field(config, gain_info)
    print(f"luminance gain: {gain_info['shot_gain_stops']:+.4f} stops (unchanged v0.1 path)")
    chroma = measure_chroma(config, gain, count)
    for name in ("top", "bottom"):
        column = chroma["per_column"][name]
        print(f"  {name}: scale {abs(column).min():.4f}..{abs(column).max():.4f}, hue {np.degrees(np.angle(column)).min():+.3f}..{np.degrees(np.angle(column)).max():+.3f} deg, clamped {chroma['clamped_fraction'][name] * 100:.2f}%")
    result = render(config, gain, chroma, count)
    payload = {
        "shot": {"hdr_interval": [config.shot_start, config.shot_start + count], "frames": count},
        "luminance_path_unchanged": {key: value for key, value in gain_info.items() if not key.endswith("_profile_stops")},
        "chroma_field": {
            "space": "ICtCp chroma plane, similarity transform (scale = saturation, angle = hue)",
            "smoothing_sigma_px": CHROMA_SIGMA, "polynomial_degree": POLYNOMIAL_DEGREE,
            "parameter_count": chroma["parameter_count"],
            "coefficients_highest_order_first": {name: [[value.real, value.imag] for value in values] for name, values in chroma["coefficients"].items()},
            "z_shot": [chroma["z_shot"].real, chroma["z_shot"].imag],
            "scale_clamp": list(SCALE_CLAMP), "rotation_clamp_degrees": ROTATION_CLAMP_DEG,
            "clamped_fraction": chroma["clamped_fraction"],
            "band_chroma_error_before": chroma["band_chroma_error_before"],
        },
        "result": result,
        "guarantees": {"temporal_state": False, "ai": False, "lut": False, "mmr": False, "hdr_centre_regenerated": False, "intensity_modified_by_chroma_step": False},
        "total_seconds": time.perf_counter() - started,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "chroma_field.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT_DIR / "CHROMA_FIELD_POC.md").write_text(report(chroma, result, gain_info), encoding="utf-8")
    print(f"total {payload['total_seconds']:.1f} s")
    print(OUT_DIR / "CHROMA_FIELD_POC.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
