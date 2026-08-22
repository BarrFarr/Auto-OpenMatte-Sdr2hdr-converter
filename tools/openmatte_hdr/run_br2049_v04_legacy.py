#!/usr/bin/env python3
"""Run the current v04 chroma-field pipeline on a legacy-synchronized BR2049 sample.

This is a narrow adapter, not a second colour model. It loads the generic v04
implementation unchanged, supplies every source-dependent setting from a JSON
profile, isolates its cache/output, and records inherited legacy-sync caveats.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_PROFILE = HERE / "profiles" / "br2049_legacy_t2714_30f.json"
DEFAULT_OUTPUT = HERE / "out" / "br2049_v04_legacy_t2714_30f"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v4 = _load("fast_chroma_field_intensity_br2049", HERE / "fast_chroma_field_intensity.py")
v1 = v4.v1
v2 = v4.v2
fastcore = v4.fastcore
om = v4.om


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
    "cache_root",
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
    if len(profile["hdr_size"]) != 2 or len(profile["om_size"]) != 2 or len(profile["overlap"]) != 4:
        raise ValueError("Profile geometry dimensions are malformed")
    return profile


def build_config(profile: dict[str, Any], output_dir: Path) -> Any:
    argv = [
        "--hdr", str(workspace_path(profile["hdr"])),
        "--om", str(workspace_path(profile["om"])),
        "--output", str(output_dir / "unused.mkv"),
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
    config = om.build_config(argv)
    om.require(config.frame_count == int(profile["expected_frame_count"]), "Unexpected profile frame count")
    om.require(config.shot_start + config.offset_frames == int(profile["om_start_frame"]), "Unexpected profile OM frame mapping")
    return config


def report(profile: dict[str, Any], profile_path: Path, payload: dict[str, Any]) -> str:
    sync = profile["legacy_sync"]
    geometry = profile["legacy_geometry"]
    preamble = [
        "# BR2049 v04 legacy-sync sample", "",
        f"Profile: `{profile['name']}` (`{profile_path}`).", "",
        "## Inherited contract", "",
        f"- Mapping: `{sync['equation']}`; status `{sync['status']}`, confidence `{sync['confidence']:.4f}`.",
        f"- HDR frames `[{profile['shot_start']}, {profile['shot_end']})`; OM frames `[{profile['om_start_frame']}, {profile['om_start_frame'] + profile['expected_frame_count']})`; `{profile['expected_frame_count']}` frames.",
        f"- Geometry: overlap `{profile['overlap']}`, confidence `{geometry['confidence']:.4f}` ({geometry['status']}).",
        f"- This is a legacy regression/review sample at `{sync['historical_time_seconds']:.3f}s`, not a newly validated BR2049 scene/anchor.",
        f"- Feather is `{profile['feather']}` px from current v04, not the distinct 4 px old P2.27.2 renderer setting.", "",
    ]
    body = (
        v4.markdown_report(payload)
        .replace("# PoC v3 — bounded intensity-conditioned chroma residual", "## v04 bounded chroma-field result", 1)
        .replace(
            "The extensions have no HDR ground truth. Visual HDR review remains decisive: inspect skin, rice paper, walls, dark clothes, and saturated elements for a more natural merge without intensity-dependent colour artifacts.",
            "The extensions have no HDR ground truth. Visual HDR review remains decisive: inspect the atmospheric bright areas, dark costume fabric, blue-gray backgrounds, and any saturated lights for a more natural merge without intensity-dependent colour artifacts.",
        )
    )
    return "\n".join(preamble) + body


def main() -> int:
    parser = argparse.ArgumentParser(prog="run-br2049-v04-legacy")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="Fully specified BR2049 legacy input profile")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Isolated output directory")
    parser.add_argument("--cache-root", type=Path, default=None, help="Override profile cache root")
    parser.add_argument("--cpu", action="store_true", help="Force the multicore CPU fallback instead of CuPy")
    parser.add_argument("--render-model", choices=("default", "review"), default="default", help="Render the held-out-accepted default or visual-only candidate")
    args = parser.parse_args()

    profile_path = workspace_path(args.profile)
    profile = load_profile(profile_path)
    config = build_config(profile, args.output_dir)
    cache_root = workspace_path(args.cache_root) if args.cache_root else workspace_path(profile["cache_root"])
    v1.CACHE_ROOT = cache_root
    if profile.get("review_regions"):
        v1.REGIONS = tuple((str(name), int(x)) for name, x in profile["review_regions"])

    count = config.frame_count
    training_indices, holdout_indices = v2.sampled_split(count, 1)
    backend = fastcore.Backend(use_gpu=not args.cpu and fastcore.gpu_available())
    cache = v1.cache_for(config, count)
    print(f"BR2049 v04 legacy sample: {count} frames, backend {backend.name}")
    print(f"sync: {profile['legacy_sync']['equation']} (confidence {profile['legacy_sync']['confidence']:.4f}); geometry confidence {profile['legacy_geometry']['confidence']:.4f} CONDITIONAL")
    print(f"HDR [{config.shot_start}, {config.shot_start + count}), OM [{config.shot_start + config.offset_frames}, {config.shot_start + config.offset_frames + count}), cache {cache.gigabytes:.2f} GB")
    total = time.perf_counter()
    if cache.valid():
        decode_seconds = 0.0
        print("cache: reusing decoded frames")
    else:
        started = time.perf_counter()
        cache.build(lambda: om.pair_stream(config, count))
        decode_seconds = time.perf_counter() - started
        print(f"cache: decoded once in {decode_seconds:.1f} s")
    sdr_cache, hdr_cache, manifest = cache.open()
    gain = v1.measure_gain(backend, config, sdr_cache, hdr_cache, stride=1)
    gain_field = backend.asarray(om.gain_field(config, gain))
    spatial = v2.measure_spatial_field(backend, config, sdr_cache, hdr_cache, gain_field, training_indices)
    spatial_field = v1.chroma_field(config, spatial)
    controls = v4.fit_intensity_controls(backend, config, sdr_cache, hdr_cache, gain_field, spatial_field, training_indices)
    selection = v4.evaluate_candidates(backend, config, sdr_cache, hdr_cache, gain_field, spatial_field, controls, holdout_indices)
    models = v4.candidate_models(controls)
    default_model = next(model for model in models if model["name"] == selection["default_model"]["name"])
    review_model = next(model for model in models if model["name"] == selection["review_candidate"]["name"])
    rendered_model = default_model if args.render_model == "default" else review_model
    print(f"gain {gain['seconds']:.1f} s; spatial {spatial['seconds']:.1f} s; intensity fit {controls['seconds']:.1f} s; held-out gate {selection['seconds']:.1f} s")
    print(f"default={default_model['name']}; review={review_model['name']}; rendered={rendered_model['name']}")
    result = v4.render(backend, config, sdr_cache, hdr_cache, gain_field, spatial_field, rendered_model, args.output_dir)
    elapsed = time.perf_counter() - total
    print(f"render {result['seconds']:.1f} s ({result['seconds'] / count:.3f} s/frame); total {elapsed:.1f} s")

    payload = {
        "schema": "openmatte-hdr-br2049-legacy-v04-sample/v1",
        "input_profile_path": str(profile_path),
        "input_profile": profile,
        "profile_warning": "Legacy sync is locked, but inherited geometry confidence is 0.9155 and this 30-frame interval is not a validated BR2049 shot.",
        "backend": backend.name,
        "frames": count,
        "split": {"fold_modulo": v2.FOLD_MODULO, "holdout_remainder": v2.HOLDOUT_REMAINDER, "training_indices": training_indices, "holdout_indices": holdout_indices},
        "cache": {"path": str(cache.directory), "gigabytes": cache.gigabytes, "reused": decode_seconds == 0.0, "source_frame_hashes": manifest["source_frame_hashes"]},
        "timing_seconds": {"decode_cache": decode_seconds, "gain_pass": gain["seconds"], "spatial_fit": spatial["seconds"], "intensity_fit": controls["seconds"], "heldout_gate": selection["seconds"], "render": result["seconds"], "total": elapsed},
        "gain": {key: value for key, value in gain.items() if not key.endswith("_profile_stops")},
        "spatial_field": {
            "parameter_count": spatial["parameter_count"],
            "frames_used": spatial["frames_used"],
            "z_shot": [spatial["z_shot"].real, spatial["z_shot"].imag],
            "clamped_fraction": spatial["clamped_fraction"],
            "band_chroma_error_before": spatial["band_chroma_error_before"],
            "coefficients_highest_order_first": {name: [[value.real, value.imag] for value in values] for name, values in spatial["coefficients"].items()},
        },
        "intensity_controls": controls,
        "selection": selection,
        "default_model": default_model,
        "review_model": review_model,
        "rendered_model": rendered_model,
        "result": result,
        "guarantees": {"temporal_state": False, "ai": False, "lut": False, "mmr": False, "hdr_centre_regenerated": False, "intensity_modified_by_chroma_step": False, "chroma_detail_modified": False, "new_spatial_parameters": 0},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "br2049_v04_legacy.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "BR2049_V04_LEGACY_SAMPLE.md").write_text(report(profile, profile_path, payload), encoding="utf-8")
    print(args.output_dir / "BR2049_V04_LEGACY_SAMPLE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
