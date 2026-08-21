#!/usr/bin/env python3
"""openmatte-hdr v0.1 - bounded Open Matte HDR extension renderer.

The HDR master is treated as a boundary reference, not as a prediction target.
Only the Open Matte rows outside the HDR overlap are generated. The centre is
copied from the existing HDR master except for two explicit feather bands.

The gain is measured against the HDR rows adjacent to each seam, smoothed
across x, clamped, and fixed for the whole shot by a median over a first
measurement pass. There is no MMR, no LUT, no AI, no per-pixel temporal
filtering, and no change to synchronisation or geometry.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_SRC = ROOT / "research" / "src"
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

from reshaping_research.utils.color_spaces import bt709_to_bt2020, compute_luminance  # noqa: E402
from reshaping_research.utils.transfer_functions import bt1886_eotf, pq_eotf, pq_oetf  # noqa: E402

RGB16_MAX = 65535.0
PEAK_NITS = 10000.0
EPS = 1e-6
DEFAULT_FFMPEG = ROOT / "dev" / "ffmpeg-build" / "install" / "bin" / "ffmpeg.exe"


class ToolError(RuntimeError):
    """Raised for contract, decode, or encode failures."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ToolError(message)


@dataclass(frozen=True)
class Config:
    hdr_source: Path
    om_source: Path
    output: Path
    shot_start: int
    shot_end: int
    offset_frames: int
    overlap: tuple[int, int, int, int]
    hdr_size: tuple[int, int]
    om_size: tuple[int, int]
    fps: float
    fps_text: str
    base_sigma: float
    detail_strength: float
    detail_ratio_clip: tuple[float, float]
    seam_band: int
    gain_smooth_sigma: float
    gain_clamp_stops: float
    feather: int
    ffmpeg: Path
    comparison: Path | None
    qc_json: Path | None
    contact_sheet: Path | None
    max_frames: int | None
    review_hdr10: bool
    review_cq: int

    @property
    def frame_count(self) -> int:
        planned = self.shot_end - self.shot_start
        return min(planned, self.max_frames) if self.max_frames else planned


def parse_fps(text: str) -> float:
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def build_config(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(prog="openmatte-hdr", description="Render Open Matte HDR extensions from an existing HDR master used as a boundary reference.")
    parser.add_argument("--hdr", required=True, type=Path, help="HDR master video path")
    parser.add_argument("--om", required=True, type=Path, help="Open Matte SDR video path")
    parser.add_argument("--output", required=True, type=Path, help="Generated Open Matte HDR output (FFV1 Matroska)")
    parser.add_argument("--shot-start", required=True, type=int, help="First HDR frame of the shot")
    parser.add_argument("--shot-end", required=True, type=int, help="HDR frame after the last frame of the shot")
    parser.add_argument("--offset-frames", required=True, type=int, help="Open Matte frame offset relative to HDR frames")
    parser.add_argument("--overlap", nargs=4, type=int, default=[0, 140, 1920, 940], metavar=("X1", "Y1", "X2", "Y2"), help="HDR overlap rectangle in Open Matte coordinates")
    parser.add_argument("--hdr-size", nargs=2, type=int, default=[3840, 1600], metavar=("W", "H"), help="HDR frame size")
    parser.add_argument("--om-size", nargs=2, type=int, default=[1920, 1080], metavar=("W", "H"), help="Open Matte frame size")
    parser.add_argument("--fps", default="24000/1001", help="Frame rate as a fraction or decimal")
    parser.add_argument("--base-sigma", type=float, default=16.0, help="Gaussian sigma of the base layer in pixels")
    parser.add_argument("--detail-strength", type=float, default=0.85, help="Exponent applied to the detail ratio")
    parser.add_argument("--detail-ratio-clip", nargs=2, type=float, default=[0.25, 4.0], metavar=("LOW", "HIGH"), help="Bounds of the detail ratio")
    parser.add_argument("--seam-band", type=int, default=24, help="HDR rows adjacent to each seam used to measure gain")
    parser.add_argument("--gain-smooth-sigma", type=float, default=128.0, help="Horizontal smoothing sigma of the gain profile in pixels")
    parser.add_argument("--gain-clamp-stops", type=float, default=1.5, help="Maximum deviation of the gain profile from the shot median, in stops")
    parser.add_argument("--feather", type=int, default=24, help="Feather rows inside the HDR centre at each seam")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG, help="FFmpeg executable")
    parser.add_argument("--comparison", type=Path, default=None, help="Optional side-by-side reference/output video")
    parser.add_argument("--qc-json", type=Path, default=None, help="Optional QC metrics JSON path")
    parser.add_argument("--contact-sheet", type=Path, default=None, help="Optional representative-frame contact sheet PNG")
    parser.add_argument("--max-frames", type=int, default=None, help="Render only the first N frames of the shot")
    parser.add_argument("--review-hdr10", action="store_true", help="Also write HDR10 HEVC copies for playback review")
    parser.add_argument("--review-cq", type=int, default=16, help="Quality level of the HDR10 review encode (lower is better)")
    args = parser.parse_args(argv)
    x1, y1, x2, y2 = (int(value) for value in args.overlap)
    om_width, om_height = (int(value) for value in args.om_size)
    require(args.shot_end > args.shot_start, "--shot-end must be greater than --shot-start")
    require(0 <= x1 < x2 <= om_width and 0 <= y1 < y2 <= om_height, "Overlap rectangle is outside the Open Matte frame")
    require(y1 > 0 or y2 < om_height, "Overlap covers the full frame height, so there is no extension to generate")
    require(args.seam_band >= 1 and args.seam_band <= (y2 - y1) // 2, "--seam-band must fit inside half of the overlap height")
    require(args.feather >= 0 and args.feather <= (y2 - y1) // 2, "--feather must fit inside half of the overlap height")
    require(args.base_sigma > 0.0, "--base-sigma must be positive")
    require(args.detail_ratio_clip[0] > 0.0 and args.detail_ratio_clip[1] > args.detail_ratio_clip[0], "Invalid --detail-ratio-clip")
    require(args.gain_clamp_stops > 0.0, "--gain-clamp-stops must be positive")
    require(args.max_frames is None or args.max_frames > 0, "--max-frames must be positive")
    require(args.hdr.is_file(), f"HDR source not found: {args.hdr}")
    require(args.om.is_file(), f"Open Matte source not found: {args.om}")
    require(args.ffmpeg.is_file(), f"FFmpeg not found: {args.ffmpeg}")
    return Config(
        hdr_source=args.hdr, om_source=args.om, output=args.output,
        shot_start=int(args.shot_start), shot_end=int(args.shot_end), offset_frames=int(args.offset_frames),
        overlap=(x1, y1, x2, y2), hdr_size=(int(args.hdr_size[0]), int(args.hdr_size[1])), om_size=(om_width, om_height),
        fps=parse_fps(args.fps), fps_text=str(args.fps), base_sigma=float(args.base_sigma),
        detail_strength=float(args.detail_strength), detail_ratio_clip=(float(args.detail_ratio_clip[0]), float(args.detail_ratio_clip[1])),
        seam_band=int(args.seam_band), gain_smooth_sigma=float(args.gain_smooth_sigma), gain_clamp_stops=float(args.gain_clamp_stops),
        feather=int(args.feather), ffmpeg=args.ffmpeg, comparison=args.comparison, qc_json=args.qc_json,
        contact_sheet=args.contact_sheet, max_frames=args.max_frames,
        review_hdr10=bool(args.review_hdr10), review_cq=int(args.review_cq),
    )


def decoder_command(config: Config, source: Path, start_frame: int, count: int) -> list[str]:
    return [str(config.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{start_frame / config.fps:.9f}", "-i", str(source), "-map", "0:v:0", "-frames:v", str(count), "-pix_fmt", "rgb48le", "-vsync", "0", "-f", "rawvideo", "pipe:1"]


HDR_TAGS = ("-colorspace", "bt2020nc", "-color_trc", "smpte2084", "-color_primaries", "bt2020", "-color_range", "tv")
# Output options alone are not enough: FFmpeg keeps the unknown transfer/primaries of the
# raw input frames, so the frame properties are stamped explicitly before conversion.
SETPARAMS = "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc"


def encoder_command(config: Config, path: Path, width: int, height: int) -> list[str]:
    return [str(config.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{width}x{height}", "-framerate", config.fps_text, "-i", "pipe:0", "-an", "-vf", SETPARAMS, "-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv444p16le", *HDR_TAGS, str(path)]


def review_command(config: Config, source: Path, destination: Path) -> list[str]:
    return [str(config.ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source), "-an", "-vf", SETPARAMS, "-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p5", "-cq", str(config.review_cq), "-pix_fmt", "p010le", *HDR_TAGS, str(destination)]


def verify_hdr_tags(config: Config, path: Path) -> dict[str, str]:
    command = [str(config.ffmpeg.with_name("ffprobe.exe")), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_space,color_transfer,color_primaries,color_range", "-of", "json", str(path)]
    completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
    require(completed.returncode == 0, f"Cannot probe {path}: {completed.stderr.decode(errors='replace')}")
    stream = json.loads(completed.stdout.decode("utf-8"))["streams"][0]
    require(stream.get("color_transfer") == "smpte2084" and stream.get("color_primaries") == "bt2020", f"{path.name} is not tagged as PQ/BT.2020: {stream}")
    return {key: str(value) for key, value in stream.items()}


def write_review(config: Config, source: Path) -> dict[str, Any]:
    destination = source.with_name(f"{source.stem}_hdr10.mkv")
    completed = subprocess.run(review_command(config, source, destination), capture_output=True, timeout=7200, check=False)
    require(completed.returncode == 0 and destination.is_file(), f"HDR10 review encode failed: {completed.stderr.decode(errors='replace')}")
    return {"path": str(destination), "bytes": destination.stat().st_size, "tags": verify_hdr_tags(config, destination)}


def decode_hdr(config: Config, raw: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = config.overlap
    code = cv2.resize(raw.astype(np.float64) / RGB16_MAX, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return (pq_eotf(code) / PEAK_NITS).astype(np.float32)


def decode_sdr(raw: np.ndarray) -> np.ndarray:
    return np.maximum(bt709_to_bt2020(bt1886_eotf(raw.astype(np.float64) / RGB16_MAX)), 0.0).astype(np.float32)


@contextlib.contextmanager
def pair_stream(config: Config, count: int) -> Iterator[Iterator[tuple[np.ndarray, np.ndarray, dict[str, str]]]]:
    hdr_width, hdr_height = config.hdr_size
    om_width, om_height = config.om_size
    hdr_bytes, om_bytes = hdr_width * hdr_height * 6, om_width * om_height * 6
    hdr_process = subprocess.Popen(decoder_command(config, config.hdr_source, config.shot_start, count), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    om_process = subprocess.Popen(decoder_command(config, config.om_source, config.shot_start + config.offset_frames, count), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def frames() -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, str]]]:
        for index in range(count):
            hdr_data = hdr_process.stdout.read(hdr_bytes)
            om_data = om_process.stdout.read(om_bytes)
            require(len(hdr_data) == hdr_bytes and len(om_data) == om_bytes, f"Source stream ended early at frame index {index}")
            hdr_raw = np.frombuffer(hdr_data, dtype="<u2").reshape(hdr_height, hdr_width, 3)
            om_raw = np.frombuffer(om_data, dtype="<u2").reshape(om_height, om_width, 3)
            hashes = {"hdr": hashlib.sha256(hdr_data).hexdigest().upper(), "om": hashlib.sha256(om_data).hexdigest().upper()}
            yield decode_hdr(config, hdr_raw), decode_sdr(om_raw), hashes

    try:
        yield frames()
    finally:
        for process in (hdr_process, om_process):
            if process.stdout:
                process.stdout.close()
        codes = [process.wait() for process in (hdr_process, om_process)]
        errors = [process.stderr.read().decode("utf-8", errors="replace") for process in (hdr_process, om_process)]
        require(codes == [0, 0], f"Source decode failed: codes={codes}; {errors}")


def base_layer(config: Config, sdr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(sdr, (0, 0), sigmaX=config.base_sigma, sigmaY=config.base_sigma, borderType=cv2.BORDER_REFLECT_101)


def seam_profiles(config: Config, hdr_common: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median log2 luminance gain per column in the HDR bands adjacent to each seam."""
    _, y1, _, y2 = config.overlap
    band = config.seam_band
    hdr_y = compute_luminance(hdr_common)
    base_y = compute_luminance(base)
    top = np.median(np.log2((hdr_y[:band] + EPS) / (base_y[y1:y1 + band] + EPS)), axis=0)
    bottom = np.median(np.log2((hdr_y[-band:] + EPS) / (base_y[y2 - band:y2] + EPS)), axis=0)
    return top.astype(np.float64), bottom.astype(np.float64)


def smooth_profile(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values.astype(np.float64)
    radius = int(math.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def measure_gain(config: Config, count: int) -> dict[str, Any]:
    """First pass: one fixed gain profile per seam for the whole shot."""
    top_rows: list[np.ndarray] = []
    bottom_rows: list[np.ndarray] = []
    started = time.perf_counter()
    with pair_stream(config, count) as frames:
        for hdr_common, sdr, _ in frames:
            top, bottom = seam_profiles(config, hdr_common, base_layer(config, sdr))
            require(np.isfinite(top).all() and np.isfinite(bottom).all(), "Nonfinite seam gain measurement")
            top_rows.append(top)
            bottom_rows.append(bottom)
    require(len(top_rows) == count, "Measurement pass did not cover every frame")
    top_median = np.median(np.stack(top_rows), axis=0)
    bottom_median = np.median(np.stack(bottom_rows), axis=0)
    shot_stops = float(np.median(np.concatenate((top_median, bottom_median))))
    limit = config.gain_clamp_stops
    top_profile = np.clip(smooth_profile(top_median, config.gain_smooth_sigma), shot_stops - limit, shot_stops + limit)
    bottom_profile = np.clip(smooth_profile(bottom_median, config.gain_smooth_sigma), shot_stops - limit, shot_stops + limit)
    return {
        "shot_gain_stops": shot_stops,
        "shot_gain_linear": float(np.exp2(shot_stops)),
        "top_profile_stops": top_profile,
        "bottom_profile_stops": bottom_profile,
        "top_profile_range_stops": [float(top_profile.min()), float(top_profile.max())],
        "bottom_profile_range_stops": [float(bottom_profile.min()), float(bottom_profile.max())],
        "per_frame_top_median_spread_stops": float(np.percentile(np.stack(top_rows), 95) - np.percentile(np.stack(top_rows), 5)),
        "clamped_fraction": float(np.mean(np.abs(np.concatenate((smooth_profile(top_median, config.gain_smooth_sigma), smooth_profile(bottom_median, config.gain_smooth_sigma))) - shot_stops) > limit)),
        "measurement_seconds": time.perf_counter() - started,
        "method": "median over frames of per-column median log2(Y_HDR/Y_base) in the seam bands; smoothed in x, clamped to the shot median; constant for the shot",
    }


def gain_field(config: Config, gain: dict[str, Any]) -> np.ndarray:
    """Boundary gain at each seam, decaying to the shot gain away from the HDR data."""
    _, y1, _, y2 = config.overlap
    _, height = config.om_size
    top = gain["top_profile_stops"][None, :]
    bottom = gain["bottom_profile_stops"][None, :]
    shot = gain["shot_gain_stops"]
    stops = np.empty((height, config.om_size[0]), dtype=np.float64)
    if y1 > 0:
        distance = (y1 - np.arange(y1, dtype=np.float64))[:, None]
        weight = distance / float(y1)
        stops[:y1] = (1.0 - weight) * top + weight * shot
    centre_rows = y2 - y1
    if centre_rows > 1:
        position = (np.arange(centre_rows, dtype=np.float64) / float(centre_rows - 1))[:, None]
        stops[y1:y2] = (1.0 - position) * top + position * bottom
    else:
        stops[y1:y2] = top
    if y2 < height:
        distance = (np.arange(y2, height, dtype=np.float64) - (y2 - 1))[:, None]
        weight = distance / float(height - y2)
        stops[y2:] = (1.0 - weight) * bottom + weight * shot
    field = np.exp2(stops).astype(np.float32)
    require(np.isfinite(field).all() and field.min() > 0.0, "Invalid gain field")
    return field


def render_frame(config: Config, sdr: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    base = base_layer(config, sdr)
    low, high = config.detail_ratio_clip
    ratio_raw = sdr / np.maximum(base, EPS)
    ratio = np.clip(ratio_raw, low, high)
    predicted_raw = field[..., None] * base * np.power(ratio, config.detail_strength)
    predicted = np.clip(predicted_raw, 0.0, 1.0).astype(np.float32)
    stats = {
        "detail_ratio_clip_low_fraction": float(np.mean(ratio_raw < low)),
        "detail_ratio_clip_high_fraction": float(np.mean(ratio_raw > high)),
        "output_clip_high_fraction": float(np.mean(predicted_raw > 1.0)),
        "output_max_before_clip": float(predicted_raw.max()),
    }
    return predicted, stats


def composite(config: Config, predicted: np.ndarray, hdr_common: np.ndarray) -> np.ndarray:
    _, y1, _, y2 = config.overlap
    feather = config.feather
    output = predicted.copy()
    output[y1:y2] = hdr_common
    if feather > 0:
        alpha = np.linspace(0.0, 1.0, feather, dtype=np.float32)[:, None, None]
        output[y1:y1 + feather] = (1.0 - alpha) * predicted[y1:y1 + feather] + alpha * hdr_common[:feather]
        output[y2 - feather:y2] = alpha * predicted[y2 - feather:y2] + (1.0 - alpha) * hdr_common[-feather:]
    require(np.isfinite(output).all(), "Composite contains nonfinite values")
    return output


def seam_continuity(config: Config, output: np.ndarray) -> dict[str, float]:
    _, y1, _, y2 = config.overlap
    _, height = config.om_size
    result: dict[str, float] = {}
    for label, outside, inside in (("top", y1 - 1, y1), ("bottom", y2 - 1, y2)):
        if outside < 0 or inside >= height:
            continue
        delta = np.abs(compute_luminance(output[outside]) - compute_luminance(output[inside])) * PEAK_NITS
        result[f"{label}_seam_mean_nits"] = float(np.mean(delta))
        result[f"{label}_seam_P95_nits"] = float(np.percentile(delta, 95))
    return result


def to_pq16(image: np.ndarray) -> np.ndarray:
    return np.round(pq_oetf(np.clip(image, 0.0, 1.0) * PEAK_NITS) * RGB16_MAX).astype("<u2")


def preview(image: np.ndarray) -> np.ndarray:
    nits = np.maximum(image.astype(np.float64) * PEAK_NITS, 0.0)
    return np.power(np.clip(np.log1p(nits) / math.log1p(1000.0), 0.0, 1.0), 1.0 / 2.2)


def contact_tile(image: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(np.round(preview(image) * 255.0).astype(np.uint8), (480, 270), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (480, 25), (0, 0, 0), -1)
    cv2.putText(tile, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def write_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", rgb[..., ::-1])
    require(bool(ok), f"Cannot encode {path}")
    encoded.tofile(str(path))


def render(config: Config, gain: dict[str, Any], count: int) -> dict[str, Any]:
    field = gain_field(config, gain)
    om_width, om_height = config.om_size
    _, y1, _, y2 = config.overlap
    config.output.parent.mkdir(parents=True, exist_ok=True)
    main_encoder = subprocess.Popen(encoder_command(config, config.output, om_width, om_height), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    comparison_encoder = None
    if config.comparison:
        config.comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison_encoder = subprocess.Popen(encoder_command(config, config.comparison, om_width * 2, om_height), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    sample_positions = {int(round((count - 1) * fraction)): label for fraction, label in ((0.10, "10%"), (0.50, "50%"), (0.90, "90%"))}
    samples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    frame_stats: list[dict[str, float]] = []
    first_hashes: dict[str, str] | None = None
    last_hashes: dict[str, str] | None = None
    rendered = 0
    started = time.perf_counter()
    try:
        with pair_stream(config, count) as frames:
            for index, (hdr_common, sdr, hashes) in enumerate(frames):
                if index == 0:
                    first_hashes = hashes
                last_hashes = hashes
                predicted, stats = render_frame(config, sdr, field)
                output = composite(config, predicted, hdr_common)
                main_encoder.stdin.write(to_pq16(output).tobytes())
                if comparison_encoder or (config.contact_sheet and index in sample_positions):
                    reference = np.zeros_like(output)
                    reference[y1:y2] = hdr_common
                    if comparison_encoder:
                        comparison_encoder.stdin.write(np.concatenate((to_pq16(reference), to_pq16(output)), axis=1).tobytes())
                    if config.contact_sheet and index in sample_positions:
                        samples.append((sample_positions[index], sdr.copy(), predicted.copy(), output.copy(), reference))
                frame_stats.append({**stats, **seam_continuity(config, output)})
                rendered += 1
    finally:
        encoders = [process for process in (main_encoder, comparison_encoder) if process]
        for process in encoders:
            if process.stdin:
                process.stdin.close()
        codes = [process.wait() for process in encoders]
        errors = [process.stderr.read().decode("utf-8", errors="replace") for process in encoders]
        require(all(code == 0 for code in codes), f"Encoding failed: codes={codes}; {errors}")
    require(rendered == count and config.output.is_file(), f"Render incomplete: {rendered}/{count} frames")
    if config.comparison:
        require(config.comparison.is_file(), "Comparison video was not written")
    if config.contact_sheet:
        require(len(samples) >= 1, "No representative frames were captured")
        rows = [np.concatenate((contact_tile(reference, f"{label}: HDR reference"), contact_tile(sdr, f"{label}: decoded SDR Open Matte"), contact_tile(predicted, f"{label}: generated extension"), contact_tile(output, f"{label}: HDR-centre composite")), axis=1) for label, sdr, predicted, output, reference in samples]
        write_png(config.contact_sheet, np.concatenate(rows, axis=0))
    aggregate = {key: float(np.mean([row[key] for row in frame_stats])) for key in frame_stats[0] if key.endswith("_nits")}
    aggregate.update({f"{key}_max": float(max(row[key] for row in frame_stats)) for key in ("detail_ratio_clip_low_fraction", "detail_ratio_clip_high_fraction", "output_clip_high_fraction", "output_max_before_clip")})
    outputs: dict[str, Any] = {
        "generated": str(config.output), "generated_bytes": config.output.stat().st_size,
        "generated_tags": verify_hdr_tags(config, config.output),
        "comparison": str(config.comparison) if config.comparison else None,
        "contact_sheet": str(config.contact_sheet) if config.contact_sheet else None,
    }
    if config.comparison:
        outputs["comparison_tags"] = verify_hdr_tags(config, config.comparison)
    if config.review_hdr10:
        outputs["generated_hdr10"] = write_review(config, config.output)
        if config.comparison:
            outputs["comparison_hdr10"] = write_review(config, config.comparison)
    return {
        "rendered_frames": rendered,
        "render_seconds": time.perf_counter() - started,
        "seconds_per_frame": (time.perf_counter() - started) / max(rendered, 1),
        "source_frame_hashes": {"first": first_hashes, "last": last_hashes},
        "diagnostics": aggregate,
        "outputs": outputs,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        config = build_config(argv)
        count = config.frame_count
        print(f"openmatte-hdr v0.1: shot HDR [{config.shot_start},{config.shot_start + count}) -> {count} frames")
        gain = measure_gain(config, count)
        print(f"boundary gain: shot {gain['shot_gain_stops']:+.4f} stops; top {gain['top_profile_range_stops'][0]:+.4f}..{gain['top_profile_range_stops'][1]:+.4f}; bottom {gain['bottom_profile_range_stops'][0]:+.4f}..{gain['bottom_profile_range_stops'][1]:+.4f}")
        result = render(config, gain, count)
        report = {
            "tool": "openmatte-hdr v0.1",
            "configuration": {
                "hdr_source": str(config.hdr_source), "om_source": str(config.om_source),
                "hdr_interval": [config.shot_start, config.shot_start + count],
                "om_interval": [config.shot_start + config.offset_frames, config.shot_start + config.offset_frames + count],
                "offset_frames": config.offset_frames, "overlap": list(config.overlap), "fps": config.fps_text,
                "base_sigma_px": config.base_sigma, "detail_strength": config.detail_strength,
                "detail_ratio_clip": list(config.detail_ratio_clip), "seam_band_px": config.seam_band,
                "gain_smooth_sigma_px": config.gain_smooth_sigma, "gain_clamp_stops": config.gain_clamp_stops,
                "feather_px": config.feather,
            },
            "gain": {key: value for key, value in gain.items() if not key.endswith("_profile_stops")},
            "result": result,
            "guarantees": {"mmr": False, "lut": False, "ai": False, "per_pixel_temporal_filtering": False, "hdr_centre_regenerated": False, "geometry_or_sync_modified": False},
            "acceptance": "Extension regions have no HDR ground truth. Seam and range values are diagnostics; the decision criterion is human review of the generated video.",
        }
        if config.qc_json:
            config.qc_json.parent.mkdir(parents=True, exist_ok=True)
            config.qc_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seam = result["diagnostics"]
        print(f"seam continuity mean: top {seam.get('top_seam_mean_nits', float('nan')):.3f} nits, bottom {seam.get('bottom_seam_mean_nits', float('nan')):.3f} nits")
        print(f"clipping fraction max: {seam['output_clip_high_fraction_max']:.6f}")
        print(f"rendered {result['rendered_frames']} frames in {result['render_seconds']:.1f} s ({result['seconds_per_frame']:.3f} s/frame)")
        print(f"colour tags: {result['outputs']['generated_tags']}")
        print(f"output: {config.output}")
        for key in ("generated_hdr10", "comparison_hdr10"):
            if key in result["outputs"]:
                print(f"{key}: {result['outputs'][key]['path']}")
        return 0
    except ToolError as error:
        print(f"openmatte-hdr error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
