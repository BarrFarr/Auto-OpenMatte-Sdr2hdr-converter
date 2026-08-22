#!/usr/bin/env python3
"""GPU-accelerated chroma-field PoC. Same model as poc_chroma_field.py, much faster.

Decode happens once into a cache; the gain pass, the chroma pass and the render
all read from it and run their math on the GPU. Encoder pipes are written from
worker threads.
"""
from __future__ import annotations

import argparse
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
ROOT = HERE.parents[1]
if str(ROOT / "research" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "research" / "src"))


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fastcore = _load("fastcore")
om = _load("openmatte_hdr")

SEAM_BAND = 48
CHROMA_SIGMA = 16.0
POLYNOMIAL_DEGREE = 2
SCALE_CLAMP = (0.50, 2.00)
ROTATION_CLAMP_DEG = 30.0
OUT_DIR = HERE / "out" / "chroma_field_fast"
CACHE_ROOT = HERE / "cache"
MATRIX = {
    "hdr": "G:\\Filmy\\The Matrix UHD\\The Matrix.mkv",
    "om": "G:\\Filmy\\IMAX format (open matte)\\The Matrix (1999) [OPEN MATTE] [WEB-DL 1080p 10bit DD5.1 x265].mkv",
    "shot_start": 73274,
    "shot_end": 73460,
    "offset": -19,
}
REGIONS = (("wall_wood", 40), ("dark_clothing", 300), ("skin_head", 780), ("rice_paper_screens", 1400))


def nvenc(config: om.Config, path: Path, width: int, height: int) -> subprocess.Popen[bytes]:
    command = [str(config.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "0", "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{width}x{height}", "-framerate", config.fps_text, "-i", "pipe:0", "-an", "-vf", om.SETPARAMS, "-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p5", "-cq", "16", "-pix_fmt", "p010le", *om.HDR_TAGS, str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def cache_for(config: om.Config, count: int) -> fastcore.FrameCache:
    key = {"hdr": str(config.hdr_source), "om": str(config.om_source), "start": config.shot_start, "count": count, "offset": config.offset_frames, "overlap": list(config.overlap), "decode": "v0.1 rgb48le/BT.1886/PQ float32"}
    directory = CACHE_ROOT / f"{config.shot_start}_{count}"
    _, y1, _, y2 = config.overlap
    return fastcore.FrameCache(directory, count, (config.om_size[1], config.om_size[0]), (y2 - y1, config.om_size[0]), key)


def predict(be: fastcore.Backend, config: om.Config, sdr: Any, base: Any, gain: Any) -> Any:
    low, high = config.detail_ratio_clip
    ratio = be.xp.clip(sdr / be.xp.maximum(base, fastcore.EPS), low, high)
    return be.xp.clip((gain[..., None] * base) * be.xp.power(ratio, config.detail_strength), 0.0, 1.0)


def composite(be: fastcore.Backend, config: om.Config, predicted: Any, hdr: Any) -> Any:
    _, y1, _, y2 = config.overlap
    feather = config.feather
    out = predicted.copy()
    out[y1:y2] = hdr
    if feather > 0:
        alpha = be.xp.linspace(0.0, 1.0, feather, dtype=be.xp.float32)[:, None, None]
        out[y1:y1 + feather] = (1.0 - alpha) * predicted[y1:y1 + feather] + alpha * hdr[:feather]
        out[y2 - feather:y2] = alpha * predicted[y2 - feather:y2] + (1.0 - alpha) * hdr[-feather:]
    return out


def measure_gain(be: fastcore.Backend, config: om.Config, sdr_cache: np.memmap, hdr_cache: np.memmap, stride: int) -> dict[str, Any]:
    _, y1, _, y2 = config.overlap
    tops: list[np.ndarray] = []
    bottoms: list[np.ndarray] = []
    started = time.perf_counter()
    for index in range(0, len(sdr_cache), stride):
        sdr = be.asarray(sdr_cache[index])
        hdr_y = be.luminance(be.asarray(hdr_cache[index]))
        base_y = be.luminance(be.blur(sdr, config.base_sigma))
        top = be.xp.median(be.xp.log2((hdr_y[:SEAM_BAND] + fastcore.EPS) / (base_y[y1:y1 + SEAM_BAND] + fastcore.EPS)), axis=0)
        bottom = be.xp.median(be.xp.log2((hdr_y[-SEAM_BAND:] + fastcore.EPS) / (base_y[y2 - SEAM_BAND:y2] + fastcore.EPS)), axis=0)
        tops.append(be.tohost(top).astype(np.float64))
        bottoms.append(be.tohost(bottom).astype(np.float64))
    top_median = np.median(np.stack(tops), axis=0)
    bottom_median = np.median(np.stack(bottoms), axis=0)
    shot = float(np.median(np.concatenate((top_median, bottom_median))))
    limit = config.gain_clamp_stops
    top_profile = np.clip(om.smooth_profile(top_median, config.gain_smooth_sigma), shot - limit, shot + limit)
    bottom_profile = np.clip(om.smooth_profile(bottom_median, config.gain_smooth_sigma), shot - limit, shot + limit)
    return {"shot_gain_stops": shot, "top_profile_stops": top_profile, "bottom_profile_stops": bottom_profile, "frames_used": len(tops), "seconds": time.perf_counter() - started, "top_profile_range_stops": [float(top_profile.min()), float(top_profile.max())], "bottom_profile_range_stops": [float(bottom_profile.min()), float(bottom_profile.max())]}


def clamp_z(z: np.ndarray) -> tuple[np.ndarray, float]:
    scale, angle = np.abs(z), np.angle(z)
    limit = np.deg2rad(ROTATION_CLAMP_DEG)
    clamped = float(np.mean((scale < SCALE_CLAMP[0]) | (scale > SCALE_CLAMP[1]) | (np.abs(angle) > limit)))
    return np.clip(scale, *SCALE_CLAMP) * np.exp(1j * np.clip(angle, -limit, limit)), clamped


def measure_chroma(be: fastcore.Backend, config: om.Config, sdr_cache: np.memmap, hdr_cache: np.memmap, gain: Any, stride: int) -> dict[str, Any]:
    _, y1, _, y2 = config.overlap
    width = config.om_size[0]
    totals = {name: {"num_re": np.zeros(width), "num_im": np.zeros(width), "den": np.zeros(width)} for name in ("top", "bottom")}
    before: list[float] = []
    used = 0
    started = time.perf_counter()
    for index in range(0, len(sdr_cache), stride):
        sdr = be.asarray(sdr_cache[index])
        hdr = be.asarray(hdr_cache[index])
        predicted = predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
        pred_chroma = be.blur(be.to_ictcp(predicted * fastcore.PEAK_NITS)[..., 1:], CHROMA_SIGMA)
        hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], CHROMA_SIGMA)
        bands = {"top": (pred_chroma[y1:y1 + SEAM_BAND], hdr_chroma[:SEAM_BAND]), "bottom": (pred_chroma[y2 - SEAM_BAND:y2], hdr_chroma[-SEAM_BAND:])}
        for name, (p, h) in bands.items():
            p_re, p_im, h_re, h_im = p[..., 0], p[..., 1], h[..., 0], h[..., 1]
            totals[name]["num_re"] += be.tohost(be.xp.sum(p_re * h_re + p_im * h_im, axis=0)).astype(np.float64)
            totals[name]["num_im"] += be.tohost(be.xp.sum(p_re * h_im - p_im * h_re, axis=0)).astype(np.float64)
            totals[name]["den"] += be.tohost(be.xp.sum(p_re * p_re + p_im * p_im, axis=0)).astype(np.float64)
            before.append(float(be.tohost(be.xp.mean(be.xp.abs(h - p)))))
        used += 1
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    coefficients: dict[str, list[complex]] = {}
    per_column: dict[str, np.ndarray] = {}
    clamped: dict[str, float] = {}
    num_total = sum(totals[name]["num_re"].sum() + 1j * totals[name]["num_im"].sum() for name in totals)
    den_total = sum(totals[name]["den"].sum() for name in totals)
    z_shot, _ = clamp_z(np.asarray([num_total / max(den_total, 1e-30)]))
    for name, entry in totals.items():
        den = np.maximum(entry["den"], 1e-30)
        z_column = (entry["num_re"] + 1j * entry["num_im"]) / den
        weights = np.sqrt(den)
        real = np.polyfit(x, z_column.real, POLYNOMIAL_DEGREE, w=weights)
        imag = np.polyfit(x, z_column.imag, POLYNOMIAL_DEGREE, w=weights)
        limited, fraction = clamp_z(np.polyval(real, x) + 1j * np.polyval(imag, x))
        coefficients[name] = [complex(r, i) for r, i in zip(real, imag)]
        per_column[name] = limited
        clamped[name] = fraction
    return {"coefficients": coefficients, "per_column": per_column, "z_shot": complex(z_shot[0]), "clamped_fraction": clamped, "band_chroma_error_before": float(np.mean(before)), "frames_used": used, "seconds": time.perf_counter() - started, "parameter_count": 2 * 2 * (POLYNOMIAL_DEGREE + 1)}


def chroma_field(config: om.Config, chroma: dict[str, Any]) -> np.ndarray:
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


def apply_chroma(be: fastcore.Backend, predicted: Any, real: Any, imag: Any) -> tuple[Any, Any]:
    ictcp = be.to_ictcp(predicted * fastcore.PEAK_NITS)
    chroma = ictcp[..., 1:]
    base = be.blur(chroma, CHROMA_SIGMA)
    detail = chroma - base
    ct, cp = base[..., 0], base[..., 1]
    corrected = be.xp.stack((ct * real - cp * imag, ct * imag + cp * real), axis=-1)
    ictcp[..., 1:] = corrected + detail
    return be.xp.clip(be.from_ictcp(ictcp) / fastcore.PEAK_NITS, 0.0, 1.0), corrected


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


def render(be: fastcore.Backend, config: om.Config, sdr_cache: np.memmap, hdr_cache: np.memmap, gain: Any, field: np.ndarray) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = config.om_size
    _, y1, _, y2 = config.overlap
    real = be.asarray(field.real.astype(np.float32))
    imag = be.asarray(field.imag.astype(np.float32))
    pipes = [nvenc(config, OUT_DIR / "v01_luminance_only_hdr10.mkv", width, height), nvenc(config, OUT_DIR / "v02_luminance_plus_chroma_hdr10.mkv", width, height), nvenc(config, OUT_DIR / "side_by_side_hdr10.mkv", width * 2, height)]
    writer = fastcore.ParallelWriter(pipes)
    count = len(sdr_cache)
    sample_index = int(round((count - 1) * 0.5))
    samples: dict[str, np.ndarray] = {}
    seam_rows: list[dict[str, float]] = []
    residual: list[float] = []
    started = time.perf_counter()
    try:
        for index in range(count):
            sdr = be.asarray(sdr_cache[index])
            hdr = be.asarray(hdr_cache[index])
            predicted = predict(be, config, sdr, be.blur(sdr, config.base_sigma), gain)
            corrected, corrected_chroma = apply_chroma(be, predicted, real, imag)
            out_lum = composite(be, config, predicted, hdr)
            out_chroma = composite(be, config, corrected, hdr)
            pq_lum, pq_chroma = be.to_pq16(out_lum), be.to_pq16(out_chroma)
            writer.submit([pq_lum.tobytes(), pq_chroma.tobytes(), np.concatenate((pq_lum, pq_chroma), axis=1).tobytes()])
            if index % 6 == 0:
                hdr_chroma = be.blur(be.to_ictcp(hdr * fastcore.PEAK_NITS)[..., 1:], CHROMA_SIGMA)
                residual.append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_chroma[:SEAM_BAND] - corrected_chroma[y1:y1 + SEAM_BAND])))))
                residual.append(float(be.tohost(be.xp.mean(be.xp.abs(hdr_chroma[-SEAM_BAND:] - corrected_chroma[y2 - SEAM_BAND:y2])))))
                row = {}
                for label, image in (("gain", out_lum), ("chroma", out_chroma)):
                    for side, outside, inside in (("top", y1 - 1, y1), ("bottom", y2 - 1, y2)):
                        pair = be.to_ictcp(image[[outside, inside]] * fastcore.PEAK_NITS)
                        magnitude = be.xp.hypot(pair[..., 1], pair[..., 2])
                        hue = be.xp.degrees(be.xp.arctan2(pair[..., 2], pair[..., 1]))
                        step = be.xp.abs(((hue[0] - hue[1] + 180.0) % 360.0) - 180.0)
                        row[f"{label}_{side}_chroma_step"] = float(be.tohost(be.xp.mean(be.xp.abs(magnitude[0] - magnitude[1]))))
                        row[f"{label}_{side}_hue_step_degrees"] = float(be.tohost(be.xp.mean(step)))
                seam_rows.append(row)
            if index == sample_index:
                samples = {"luminance_only": be.tohost(out_lum), "luminance_plus_chroma": be.tohost(out_chroma)}
    finally:
        writer.close()
        for pipe in pipes:
            if pipe.stdin:
                pipe.stdin.close()
        codes = [pipe.wait() for pipe in pipes]
        errors = [pipe.stderr.read().decode("utf-8", errors="replace") for pipe in pipes]
        om.require(all(code == 0 for code in codes), f"NVENC failed: {codes}; {errors}")
    region_sheet(OUT_DIR / "top_seam_regions.png", y1, samples, height)
    region_sheet(OUT_DIR / "bottom_seam_regions.png", y2, samples, height)
    aggregate = {key: float(np.mean([row[key] for row in seam_rows])) for key in seam_rows[0]}
    aggregate["band_chroma_error_after"] = float(np.mean(residual))
    return {"seconds": time.perf_counter() - started, "frames": count, "seam": aggregate}


def main() -> int:
    parser = argparse.ArgumentParser(prog="fast-chroma-field")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=3, help="frame stride for the two measurement passes")
    parser.add_argument("--cpu", action="store_true", help="force the NumPy backend")
    args = parser.parse_args()
    argv = ["--hdr", MATRIX["hdr"], "--om", MATRIX["om"], "--output", str(OUT_DIR / "unused.mkv"), "--shot-start", str(MATRIX["shot_start"]), "--shot-end", str(MATRIX["shot_end"]), "--offset-frames", str(MATRIX["offset"]), "--seam-band", str(SEAM_BAND)]
    if args.max_frames:
        argv += ["--max-frames", str(args.max_frames)]
    config = om.build_config(argv)
    count = config.frame_count
    be = fastcore.Backend(use_gpu=not args.cpu and fastcore.gpu_available())
    cache = cache_for(config, count)
    print(f"fast chroma field: {count} frames, backend {be.name}, cache {cache.gigabytes:.2f} GB")
    total = time.perf_counter()
    if cache.valid():
        print("cache: reusing decoded frames")
        decode_seconds = 0.0
    else:
        started = time.perf_counter()
        cache.build(lambda: om.pair_stream(config, count))
        decode_seconds = time.perf_counter() - started
        print(f"cache: decoded once in {decode_seconds:.1f} s")
    sdr_cache, hdr_cache, manifest = cache.open()
    gain_info = measure_gain(be, config, sdr_cache, hdr_cache, args.stride)
    gain = be.asarray(om.gain_field(config, gain_info))
    print(f"gain pass {gain_info['seconds']:.1f} s: {gain_info['shot_gain_stops']:+.4f} stops")
    chroma = measure_chroma(be, config, sdr_cache, hdr_cache, gain, args.stride)
    for name in ("top", "bottom"):
        column = chroma["per_column"][name]
        print(f"  {name}: scale {abs(column).min():.4f}..{abs(column).max():.4f}, hue {np.degrees(np.angle(column)).min():+.3f}..{np.degrees(np.angle(column)).max():+.3f} deg, clamped {chroma['clamped_fraction'][name] * 100:.2f}%")
    print(f"chroma pass {chroma['seconds']:.1f} s")
    result = render(be, config, sdr_cache, hdr_cache, gain, chroma_field(config, chroma))
    elapsed = time.perf_counter() - total
    print(f"render pass {result['seconds']:.1f} s ({result['seconds'] / result['frames']:.3f} s/frame)")
    print(f"total {elapsed:.1f} s")
    payload = {
        "backend": be.name, "frames": count, "stride": args.stride,
        "timing_seconds": {"decode_cache": decode_seconds, "gain_pass": gain_info["seconds"], "chroma_pass": chroma["seconds"], "render_pass": result["seconds"], "total": elapsed},
        "cache": {"gigabytes": cache.gigabytes, "reused": decode_seconds == 0.0, "source_frame_hashes": manifest["source_frame_hashes"]},
        "gain": {key: value for key, value in gain_info.items() if not key.endswith("_profile_stops")},
        "chroma_field": {"parameter_count": chroma["parameter_count"], "z_shot": [chroma["z_shot"].real, chroma["z_shot"].imag], "clamped_fraction": chroma["clamped_fraction"], "band_chroma_error_before": chroma["band_chroma_error_before"], "coefficients_highest_order_first": {name: [[v.real, v.imag] for v in values] for name, values in chroma["coefficients"].items()}},
        "result": result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "fast_chroma_field.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_DIR / "fast_chroma_field.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
