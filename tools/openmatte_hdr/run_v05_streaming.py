"""Run the bounded v05 renderer against a fully specified v04 input profile.

The profile describes synchronization and geometry.  The v04 transform modules
remain the quality reference; this runner only changes frame transport and
execution scheduling.  Use ``--cpu`` explicitly when CUDA processing is not
available.  The runner refuses cache/out/temp-like output locations.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import v05_streaming as v05

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_PROFILE = HERE / "profiles" / "br2049_legacy_t2714_30f.json"
DEFAULT_OUTPUT = WORKSPACE / "benchmarks" / "v05_streaming.mkv"

REQUIRED_PROFILE_FIELDS = {
    "name",
    "hdr",
    "om",
    "ffmpeg",
    "shot_start",
    "shot_end",
    "expected_frame_count",
    "om_start_frame",
    "offset_frames",
    "fps",
    "hdr_size",
    "om_size",
    "overlap",
    "seam_band",
    "feather",
    "legacy_sync",
    "legacy_geometry",
}


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(REQUIRED_PROFILE_FIELDS.difference(profile))
    if missing:
        raise ValueError(f"Profile {path} is missing fields: {', '.join(missing)}")
    if profile.get("schema") != "openmatte-hdr-input-profile/v1":
        raise ValueError(f"Unsupported profile schema: {profile.get('schema')!r}")
    if int(profile["shot_end"]) - int(profile["shot_start"]) != int(profile["expected_frame_count"]):
        raise ValueError("Profile frame interval does not equal expected_frame_count")
    if int(profile["shot_start"]) + int(profile["offset_frames"]) != int(profile["om_start_frame"]):
        raise ValueError("Profile om_start_frame does not match shot_start + offset_frames")
    return profile


def build_config(profile: dict[str, Any], output: Path, max_frames: int | None) -> Any:
    argv = [
        "--hdr", str(workspace_path(profile["hdr"])),
        "--om", str(workspace_path(profile["om"])),
        "--output", str(output),
        "--shot-start", str(profile["shot_start"]),
        "--shot-end", str(profile["shot_end"]),
        "--offset-frames", str(profile["offset_frames"]),
        "--overlap", *(str(value) for value in profile["overlap"]),
        "--hdr-size", *(str(value) for value in profile["hdr_size"]),
        "--om-size", *(str(value) for value in profile["om_size"]),
        "--fps", str(profile["fps"]),
        "--seam-band", str(profile["seam_band"]),
        "--feather", str(profile["feather"]),
    ]
    config = v05.om.build_config(argv)
    output_size = profile.get("output_size")
    if output_size is not None:
        if len(output_size) != 2:
            raise ValueError("Profile output_size must contain [width, height]")
        output_width, output_height = (int(value) for value in output_size)
        if output_width <= 0 or output_height <= 0 or output_width % 2 or output_height % 2:
            raise ValueError("Profile output_size must be positive and even")
        config = dataclasses.replace(config, output_size=(output_width, output_height))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive")
        if max_frames > int(profile["expected_frame_count"]):
            raise ValueError("--max-frames cannot exceed the profile shot length")
        config = dataclasses.replace(config, max_frames=int(max_frames))
    if config.frame_count <= 0:
        raise ValueError("Profile has no frames to render")
    return config


def parse_bytes(value: str) -> int:
    text = value.strip().lower()
    multipliers = {"k": 1024, "kb": 1024, "m": 1024**2, "mb": 1024**2, "g": 1024**3, "gb": 1024**3}
    for suffix, multiplier in multipliers.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def model_summary(fit: dict[str, Any], rendered_model: dict[str, Any]) -> dict[str, Any]:
    gain = fit["gain"]
    spatial = fit["spatial"]
    return {
        "training_indices": fit["training_indices"],
        "holdout_indices": fit["holdout_indices"],
        "gain": {
            key: value
            for key, value in gain.items()
            if key not in {"top_profile_stops", "bottom_profile_stops"}
        },
        "gain_profiles": {
            "top_profile_stops": gain["top_profile_stops"],
            "bottom_profile_stops": gain["bottom_profile_stops"],
        },
        "spatial_field": {
            "parameter_count": spatial["parameter_count"],
            "frames_used": spatial["frames_used"],
            "z_shot": spatial["z_shot"],
            "clamped_fraction": spatial["clamped_fraction"],
            "band_chroma_error_before": spatial["band_chroma_error_before"],
            "coefficients": spatial["coefficients"],
        },
        "intensity_controls": fit["controls"],
        "selection": fit["selection"],
        "default_model": fit["default_model"],
        "review_model": fit["review_model"],
        "rendered_model": rendered_model,
        "fit_seconds": fit["fit_seconds"],
        "fit_scale": float(fit.get("fit_scale", fit.get("fit_profile", {}).get("fit_scale_requested", 1.0))),
        "fit_profile": fit.get("fit_profile", {}),
        "quality_contract": {
            "seam_band_px": int(v05.v1.SEAM_BAND),
            "base_sigma_px": float(fit["config"].base_sigma),
            "chroma_sigma_px": float(v05.v4.INTENSITY_SIGMA),
            "detail_strength": float(fit["config"].detail_strength),
            "detail_ratio_clip": list(fit["config"].detail_ratio_clip),
            "field_degree": int(v05.v1.POLYNOMIAL_DEGREE),
            "hdr_centre_regenerated": False,
            "chroma_detail_modified": False,
            "geometry_or_sync_modified": False,
        },
    }


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    profile_path = workspace_path(args.profile)
    profile = load_profile(profile_path)
    output = workspace_path(args.output)
    report = workspace_path(args.report) if args.report else output.with_suffix(".json")
    config = build_config(profile, output, args.max_frames)
    fit_scale = float(getattr(args, "fit_scale", v05.DEFAULT_FIT_SCALE))
    v43_overlap = bool(getattr(args, "v43_overlap", False))
    cuda = v05.cuda_diagnostics()
    use_gpu = not bool(args.cpu)
    if v43_overlap and not use_gpu:
        raise v05.V05Error("--v43-overlap requires CUDA processing; do not combine it with --cpu")
    if use_gpu and not cuda.get("available"):
        raise v05.V05Error("CUDA processing is unavailable; rerun with explicit --cpu for the CPU fallback")
    if use_gpu and not cuda.get("runtime_ready"):
        raise v05.V05Error(f"CUDA runtime probe failed: {cuda.get('runtime_error')}; rerun with explicit --cpu for the CPU fallback")
    minimum_slots = 3 if v43_overlap else 2
    counts = v05.resolve_counts(
        config,
        use_gpu,
        args.ram_buffer_count,
        args.vram_buffer_count,
        args.gpu_streams,
        minimum_slots=minimum_slots,
    )
    counts["minimum_slots"] = minimum_slots
    counts["cpu_workers"] = max(1, int(args.cpu_workers))
    startup = v05.startup_diagnostics(config, counts, args.pinned_memory_limit, use_gpu)
    if args.dry_run:
        print(json.dumps(v05.json_safe({"profile": profile_path, "startup": startup}), indent=2, sort_keys=True))
        return {"status": "DRY_RUN", "startup": startup, "counts": counts}

    backend = v05.fastcore.Backend(use_gpu=use_gpu)
    total_started = time.perf_counter()
    source = v05.FrameSource(config, config.frame_count)
    print(f"v05: {config.frame_count} frames, mode={'cuda' if use_gpu else 'cpu-explicit'}, ring={counts['ram_buffer_count']}, vram={counts['vram_buffer_count']}, streams={counts['gpu_streams']}")
    print(f"HDR [{config.shot_start},{config.shot_start + config.frame_count}), OM [{config.shot_start + config.offset_frames},{config.shot_start + config.offset_frames + config.frame_count})")
    fit = v05.fit_shot(source, config, backend, args.fit_stride, fit_scale)
    fit["config"] = config
    rendered_model = fit["default_model"] if args.render_model == "default" else fit["review_model"]
    fit["rendered_model"] = rendered_model
    if use_gpu:
        try:
            import cupy

            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    context = {
        "output": output,
        "report": report,
        "gain_field": fit["gain_field"],
        "spatial_field": fit["spatial_field"],
        "rendered_model": rendered_model,
        "use_gpu": use_gpu,
        "pipeline_mode": "v43_overlap" if v43_overlap else "v42",
        "v43_overlap": v43_overlap,
    }
    render = v05.render_streaming(config, context, counts, args.pinned_memory_limit, args.diagnostic_stride)
    elapsed = time.perf_counter() - total_started
    payload = {
        "schema": "openmatte-hdr-v05-streaming-render/v1",
        "status": "PASS",
        "renderer": "v05",
        "profile_path": profile_path,
        "profile": profile,
        "configuration": {
            "hdr_source": config.hdr_source,
            "om_source": config.om_source,
            "output": output,
            "report": report,
            "hdr_interval": [config.shot_start, config.shot_start + config.frame_count],
            "om_interval": [config.shot_start + config.offset_frames, config.shot_start + config.offset_frames + config.frame_count],
            "offset_frames": config.offset_frames,
            "overlap": config.overlap,
            "fps": config.fps_text,
            "max_frames": args.max_frames,
            "fit_stride": args.fit_stride,
            "fit_scale": fit_scale,
            "render_model": args.render_model,
            "pipeline_mode": context["pipeline_mode"],
            "v43_overlap": v43_overlap,
            "minimum_frame_slots": minimum_slots,
        },
        "startup": startup,
        "model": model_summary(fit, rendered_model),
        "render": render,
        "overlap_contract": render["overlap_contract"],
        "timing_seconds": {"fit": fit["fit_seconds"], "stream_render": render["wall_seconds"], "total": elapsed},
        "guarantees": {
            "disk_frame_cache": False,
            "intermediate_frame_files": False,
            "bounded_host_ring": True,
            "bounded_device_ring": bool(use_gpu),
            "backpressure": True,
            "ordered_encode": True,
            "per_frame_global_cuda_synchronization": False,
            "v04_colour_transform_reused": True,
            "hdr_centre_regenerated": False,
            "chroma_detail_modified": False,
            "geometry_or_sync_modified": False,
        },
        "limitations": {
            "fit_gain_exact_storage": "preallocated on the GPU/device, size is recorded in model.gain.fit_storage_bytes",
            "gpu_numeric_parity": "not claimed bit-identical to CPU; use --cpu for the v04 CPU reference",
            "full_material": "not authorized by this command; run Test A and Test B gates first",
        },
    }
    v05.write_json(report, payload)
    print(f"fit {fit['fit_seconds']:.1f}s; render {render['wall_seconds']:.1f}s ({render['fps']:.3f} fps); total {elapsed:.1f}s")
    print(f"output: {output}")
    print(f"report: {report}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-v05-streaming", description="Bounded v05 renderer using the frozen v04 quality path")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the selected profile to a positive prefix; cannot exceed profile length")
    parser.add_argument("--ram-buffer-count", type=int, default=None)
    parser.add_argument("--vram-buffer-count", type=int, default=None)
    parser.add_argument("--gpu-streams", type=int, default=2)
    parser.add_argument("--pinned-memory-limit", type=parse_bytes, default=256 * 1024**2, metavar="BYTES")
    parser.add_argument("--fit-stride", type=int, default=1)
    parser.add_argument("--fit-scale", type=float, default=v05.DEFAULT_FIT_SCALE, help="Scale the bounded V4 fit grid only (default: 0.5); renderer geometry remains full resolution")
    parser.add_argument("--diagnostic-stride", type=int, default=6)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Explicitly select the CPU fallback")
    parser.add_argument("--v43-overlap", action="store_true", help="Enable V4.3 bounded GPU/CPU/NVENC output-completion overlap (requires CUDA)")
    parser.add_argument("--render-model", choices=("default", "review"), default="default")
    parser.add_argument("--dry-run", action="store_true", help="Print startup diagnostics and sizing only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (
            args.fit_stride <= 0
            or args.diagnostic_stride <= 0
            or args.gpu_streams <= 0
            or args.cpu_workers <= 0
            or not math.isfinite(args.fit_scale)
            or not (0.0 < args.fit_scale <= 1.0)
        ):
            raise v05.V05Error("stride, diagnostic stride, stream count and worker count must be positive; fit-scale must be finite and in (0, 1]")
        run_job(args)
        return 0
    except (v05.V05Error, ValueError, OSError, RuntimeError) as exc:
        print(f"v05 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
