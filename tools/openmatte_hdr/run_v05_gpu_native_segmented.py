"""Safe segmented/resumable wrapper for the native V5 Phase 2 pipeline.

This module deliberately keeps the existing native renderer unchanged.  It runs
one full-shot fit, then invokes ``render_native_phase2`` once per missing segment
with a fresh native source and encoder.  A segment is only checkpointed after
its encoder trailer has been written, the temporary file has been probed, and
``.tmp`` has been atomically renamed to ``.mkv``.

The final output is packet-concatenated with FFmpeg ``-c copy`` and then passed
through the existing metadata-only HDR10 remux helper.  No frame cache is used.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import run_v05_gpu_native as phase2
import run_v05_streaming as base
import v05_gpu_native as native
import v05_streaming as v05
from cuda_backend import (
    CudaComputeBackend,
    CudaDecoderBackend,
    make_cuda_encoder_factory,
    make_cuda_frame_factory,
    make_cuda_resample_backend,
)

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
DEFAULT_PROFILE = HERE / "profiles" / "br2049_phase2_t2714_50f.json"
DEFAULT_OUTPUT = WORKSPACE / "benchmarks" / "v05_gpu_native_segmented" / "FINAL.mkv"
DEFAULT_MANIFEST = DEFAULT_OUTPUT.parent / "render_manifest.json"
DEFAULT_SEGMENT_DIR = DEFAULT_OUTPUT.parent / "segments"
MANIFEST_SCHEMA = "openmatte-hdr-v05-segmented-manifest/v1"
REPORT_SCHEMA = "openmatte-hdr-v05-segmented-report/v1"


@dataclass(frozen=True)
class SegmentPlan:
    index: int
    start_offset: int
    frame_count: int

    @property
    def first_frame_offset(self) -> int:
        return self.start_offset

    @property
    def last_frame_offset(self) -> int:
        return self.start_offset + self.frame_count - 1


def _validate_path(path: Path, *, must_not_exist: bool = False) -> None:
    phase2._validate_output(path, must_not_exist=must_not_exist)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint using flush/fsync followed by an atomic replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    encoded = json.dumps(
        v05.json_safe(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        v05.json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _profile_identifier(profile: dict[str, Any]) -> str:
    return _identifier(profile)


def _hdr_metadata_identifier(
    source_hdr10: dict[str, Any],
    source_colors: dict[str, str],
) -> str:
    return _identifier({"hdr10": source_hdr10, "color_signaling": source_colors})


def _renderer_build_identifier(bridge: Path) -> str:
    """Identify the Python/native implementation used by every segment."""

    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(phase2.__file__).resolve(),
        Path(native.__file__).resolve(),
        Path(bridge).resolve(),
    ):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(_sha256_file(path).encode("ascii"))
        else:
            digest.update(b"missing")
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _fit_model_identifier(
    fit: dict[str, Any],
    rendered_model: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    """Hash the actual fitted arrays/model, excluding timing telemetry."""

    digest = hashlib.sha256()
    for name in ("gain_field", "spatial_field"):
        array = np.ascontiguousarray(np.asarray(fit[name]))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    digest.update(_canonical_bytes({"rendered_model": rendered_model, "profile": profile}))
    return "sha256:" + digest.hexdigest()


def _relative_filename(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _segment_path(segment_dir: Path, plan: SegmentPlan) -> Path:
    return segment_dir / f"segment_{plan.index:06d}.mkv"


def _segment_tmp_path(segment_dir: Path, plan: SegmentPlan) -> Path:
    return segment_dir / f"segment_{plan.index:06d}.tmp"


def _make_plans(total_frames: int, segment_frame_count: int) -> list[SegmentPlan]:
    if total_frames <= 0:
        raise v05.V05Error("segmented output requires at least one frame")
    if segment_frame_count <= 0:
        raise v05.V05Error("segment frame count must be positive")
    plans: list[SegmentPlan] = []
    start = 0
    index = 0
    while start < total_frames:
        count = min(segment_frame_count, total_frames - start)
        plans.append(SegmentPlan(index, start, count))
        start += count
        index += 1
    return plans


def _fps_fraction(config: Any) -> tuple[int, int]:
    try:
        value = Fraction(str(config.fps_text))
    except (ValueError, ZeroDivisionError):
        value = Fraction(float(config.fps)).limit_denominator(1_000_000)
    return int(value.numerator), int(value.denominator)


def _segment_frame_count(config: Any, duration_seconds: float) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise v05.V05Error("--segment-duration must be a finite positive number")
    fps_num, fps_den = _fps_fraction(config)
    return max(1, int(math.floor(duration_seconds * fps_num / fps_den + 0.5)))


def _time_base_seconds(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(Fraction(str(value)))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _stream_signature(stream: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "codec_name",
        "profile",
        "level",
        "width",
        "height",
        "pix_fmt",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "sample_aspect_ratio",
        "time_base",
        "r_frame_rate",
        "avg_frame_rate",
        "codec_tag_string",
        "codec_tag",
    )
    return {field: stream.get(field) for field in fields}


def _probe_segment(
    helper: Any,
    ffprobe: Path,
    path: Path,
    expected_count: int,
    fps: float,
) -> dict[str, Any]:
    """Validate one closed MKV before it can enter the manifest."""

    if not path.is_file() or path.stat().st_size <= 0:
        raise v05.V05Error(f"segment is missing or empty: {path}")
    probe = helper._ffprobe_json(
        ffprobe,
        ["-count_frames", "-show_streams", "-show_format", "-of", "json"],
        path,
    )
    stream = helper._first_stream(probe)
    frames = helper._output_frames(ffprobe, path)
    output_count = helper._int_or_none(stream.get("nb_read_frames"))
    timestamps = [
        helper._float_or_none(frame.get("best_effort_timestamp_time"))
        for frame in frames
    ]
    all_timestamps = all(value is not None for value in timestamps)
    monotonic = all(
        timestamps[index] < timestamps[index + 1]
        for index in range(len(timestamps) - 1)
        if timestamps[index] is not None and timestamps[index + 1] is not None
    )
    time_base = _time_base_seconds(stream.get("time_base"))
    expected_step = 1.0 / max(float(fps), 1e-12)
    tolerance = max(0.0025, (time_base or 0.0) * 3.0)
    deltas = [
        float(timestamps[index + 1] - timestamps[index])
        for index in range(len(timestamps) - 1)
        if timestamps[index] is not None and timestamps[index + 1] is not None
    ]
    contiguous = all(
        delta > 0.0 and abs(delta - expected_step) <= tolerance
        for delta in deltas
    )
    stream_contract = {
        "codec_name_hevc": stream.get("codec_name") == "hevc",
        "main10": "10" in str(stream.get("profile", "")) or "10" in str(stream.get("pix_fmt", "")),
        "resolution_3840x2160": stream.get("width") == 3840 and stream.get("height") == 2160,
        "pix_fmt_yuv420p10le": stream.get("pix_fmt") == "yuv420p10le",
        "bt2020nc": stream.get("color_space") == "bt2020nc",
        "smpte2084": stream.get("color_transfer") == "smpte2084",
        "bt2020": stream.get("color_primaries") == "bt2020",
        "tv_range": stream.get("color_range") == "tv",
        "sar_1_1": stream.get("sample_aspect_ratio") == "1:1",
    }
    contract = {
        **stream_contract,
        "frame_count_expected": output_count == expected_count and len(frames) == expected_count,
        "timestamps_present": all_timestamps,
        "timestamps_strictly_monotonic": monotonic,
        "timestamps_contiguous": contiguous,
    }
    if not all(contract.values()):
        raise v05.V05Error(
            f"segment validation failed for {path.name}: "
            f"{[name for name, passed in contract.items() if not passed]}"
        )
    return {
        "probe": probe,
        "stream": helper._metadata_snapshot(stream),
        "stream_signature": _stream_signature(stream),
        "time_base": stream.get("time_base"),
        "frame_count": len(frames),
        "first_pts": frames[0].get("best_effort_timestamp") if frames else None,
        "last_pts": frames[-1].get("best_effort_timestamp") if frames else None,
        "pts_start_seconds": timestamps[0] if timestamps else None,
        "pts_end_seconds": timestamps[-1] if timestamps else None,
        "frame_timestamps": frames,
        "contract": contract,
        "pass": True,
    }


def _segment_record(
    plan: SegmentPlan,
    config: Any,
    probe: dict[str, Any],
    filename: str,
    renderer_build_identifier: str,
    fitting_model_identifier: str | None,
    hdr_metadata_identifier: str,
    render_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fps_num, fps_den = _fps_fraction(config)
    first_frame = int(config.shot_start) + plan.first_frame_offset
    last_frame = int(config.shot_start) + plan.last_frame_offset
    record: dict[str, Any] = {
        "segment_index": plan.index,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "frame_count": plan.frame_count,
        "pts_start": probe["first_pts"],
        "pts_end": probe["last_pts"],
        "pts_start_seconds": probe["pts_start_seconds"],
        "pts_end_seconds": probe["pts_end_seconds"],
        "absolute_pts_start_seconds": first_frame * fps_den / fps_num,
        "absolute_pts_end_seconds": last_frame * fps_den / fps_num,
        "time_base": probe["time_base"],
        "filename": filename,
        "status": "complete",
        "renderer_build_identifier": renderer_build_identifier,
        "fitting_model_identifier": fitting_model_identifier,
        "hdr_metadata_identifier": hdr_metadata_identifier,
        "stream_signature": probe["stream_signature"],
        "validation": {
            "contract": probe["contract"],
            "pass": probe["pass"],
        },
    }
    if render_summary is not None:
        record["render"] = render_summary
    return record


def _compact_render_summary(render: dict[str, Any]) -> dict[str, Any]:
    return {
        "frames": int(render.get("frames", 0)),
        "wall_seconds": float(render.get("wall_seconds", 0.0)),
        "pipeline_seconds": float(render.get("pipeline_seconds", 0.0)),
        "fps": float(render.get("fps", 0.0)),
        "quality": render.get("quality", {}),
        "output_boundary": render.get("output_boundary", {}),
        "overlap_contract": render.get("overlap_contract", {}),
        "extension_contract": render.get("extension_contract", {}),
    }


def _new_manifest(
    manifest_path: Path,
    profile_path: Path,
    profile: dict[str, Any],
    config: Any,
    output: Path,
    segment_dir: Path,
    plans: list[SegmentPlan],
    segment_duration: float,
    renderer_build_identifier: str,
    hdr_metadata_identifier: str,
) -> dict[str, Any]:
    total_frames = sum(plan.frame_count for plan in plans)
    segment_frames = max(plan.frame_count for plan in plans)
    hdr_start = int(config.shot_start)
    om_start = int(config.shot_start + config.offset_frames)
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "in_progress",
        "manifest": _relative_filename(manifest_path, manifest_path.parent),
        "created_by": "run_v05_gpu_native_segmented",
        "profile_path": _relative_filename(profile_path, WORKSPACE),
        "profile_name": profile.get("name"),
        "configuration": {
            "output": _relative_filename(output, manifest_path.parent),
            "segment_directory": _relative_filename(segment_dir, manifest_path.parent),
            "total_frames": total_frames,
            "segment_duration_seconds": float(segment_duration),
            "segment_frame_count": segment_frames,
            "segment_count": len(plans),
            "fps": config.fps_text,
            "fps_float": float(config.fps),
            "hdr_interval": [hdr_start, hdr_start + total_frames],
            "om_interval": [om_start, om_start + total_frames],
            "offset_frames": int(config.offset_frames),
            "hdr_size": list(config.hdr_size),
            "om_size": list(config.om_size),
            "overlap": list(config.overlap),
            "stream_signature": None,
        },
        "identifiers": {
            "profile": _profile_identifier(profile),
            "renderer_build": renderer_build_identifier,
            "fitting_model": None,
            "hdr_metadata": hdr_metadata_identifier,
        },
        "segments": [],
        "extension_contract": None,
        "final": None,
    }


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise v05.V05Error(f"cannot read checkpoint manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise v05.V05Error(f"unsupported or corrupt segmented manifest: {path}")
    return payload


def _validate_manifest_contract(
    manifest: dict[str, Any],
    profile: dict[str, Any],
    config: Any,
    output: Path,
    segment_dir: Path,
    plans: list[SegmentPlan],
    segment_duration: float,
    renderer_build_identifier: str,
    hdr_metadata_identifier: str,
    manifest_path: Path,
) -> None:
    configuration = manifest.get("configuration")
    identifiers = manifest.get("identifiers")
    if not isinstance(configuration, dict) or not isinstance(identifiers, dict):
        raise v05.V05Error("segmented manifest has no configuration/identifiers contract")
    expected = {
        "total_frames": sum(plan.frame_count for plan in plans),
        "segment_duration_seconds": float(segment_duration),
        "segment_count": len(plans),
        "fps": config.fps_text,
        "offset_frames": int(config.offset_frames),
        "segment_directory": _relative_filename(segment_dir, manifest_path.parent),
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise v05.V05Error(
                f"manifest contract mismatch for {key}: "
                f"existing={configuration.get(key)!r}, requested={value!r}"
            )
    if identifiers.get("profile") != _profile_identifier(profile):
        raise v05.V05Error("manifest profile identifier does not match the requested profile")
    if identifiers.get("renderer_build") != renderer_build_identifier:
        raise v05.V05Error(
            "manifest renderer/build identifier does not match current code or bridge"
        )
    if identifiers.get("hdr_metadata") != hdr_metadata_identifier:
        raise v05.V05Error("manifest HDR metadata identifier does not match the source")
    if Path(str(configuration.get("output", ""))).name != output.name:
        raise v05.V05Error("manifest final output name does not match the requested output")


def _quarantine(path: Path, reason: str) -> Path | None:
    """Preserve an orphan/corrupt generated artifact instead of overwriting it."""

    if not path.exists():
        return None
    safe_reason = "".join(character if character.isalnum() else "_" for character in reason)
    destination = path.with_name(
        f".{path.name}.{safe_reason}.{os.getpid()}.{time.time_ns()}"
    )
    os.replace(path, destination)
    return destination


def _existing_records(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = manifest.get("segments", [])
    if not isinstance(records, list):
        raise v05.V05Error("segmented manifest segments field is not a list")
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise v05.V05Error("segmented manifest contains a non-object segment record")
        index = int(record.get("segment_index", -1))
        if index in result:
            raise v05.V05Error(f"segmented manifest contains duplicate segment {index}")
        result[index] = record
    return result


def _reconcile_segments(
    manifest: dict[str, Any],
    manifest_path: Path,
    segment_dir: Path,
    plans: list[SegmentPlan],
    config: Any,
    helper: Any,
    ffprobe: Path,
    renderer_build_identifier: str,
    hdr_metadata_identifier: str,
) -> list[dict[str, Any]]:
    """Recover only a contiguous, valid prefix; everything else is rerendered."""

    existing = _existing_records(manifest)
    stream_signature = manifest.get("configuration", {}).get("stream_signature")
    prefix: list[dict[str, Any]] = []
    first_missing_index = len(plans)
    for plan in plans:
        final_path = _segment_path(segment_dir, plan)
        temporary_path = _segment_tmp_path(segment_dir, plan)
        if temporary_path.exists():
            temporary_path.unlink()
        record = existing.get(plan.index)
        if record is not None and record.get("status") != "complete":
            record = None
        if (
            record is not None
            and record.get("filename")
            != _relative_filename(final_path, manifest_path.parent)
        ):
            record = None
        if not final_path.exists():
            first_missing_index = plan.index
            break
        try:
            probe = _probe_segment(helper, ffprobe, final_path, plan.frame_count, config.fps)
            expected_signature = stream_signature
            if expected_signature is not None and probe["stream_signature"] != expected_signature:
                raise v05.V05Error("segment stream signature differs from the checkpoint")
            if stream_signature is None:
                stream_signature = probe["stream_signature"]
        except BaseException as exc:
            _quarantine(final_path, "invalid_segment")
            first_missing_index = plan.index
            print(f"segment {plan.index} will be rerendered: {exc}", file=sys.stderr, flush=True)
            break
        recovered = _segment_record(
            plan,
            config,
            probe,
            _relative_filename(final_path, manifest_path.parent),
            renderer_build_identifier,
            manifest.get("identifiers", {}).get("fitting_model"),
            hdr_metadata_identifier,
            record.get("render") if record is not None else {"recovered": True},
        )
        if record is not None:
            for key in ("render", "fitting_model_identifier"):
                if key in record:
                    recovered[key] = record[key]
        prefix.append(recovered)
    for plan in plans[first_missing_index:]:
        final_path = _segment_path(segment_dir, plan)
        temporary_path = _segment_tmp_path(segment_dir, plan)
        if temporary_path.exists():
            temporary_path.unlink()
        if final_path.exists():
            _quarantine(final_path, "noncontiguous_segment")
    manifest["segments"] = prefix
    manifest.setdefault("configuration", {})["stream_signature"] = stream_signature
    manifest["status"] = "in_progress"
    manifest["final"] = None
    return prefix


def _update_manifest_identifiers(
    manifest: dict[str, Any],
    fitting_model_identifier: str,
    renderer_build_identifier: str,
    hdr_metadata_identifier: str,
) -> None:
    identifiers = manifest.setdefault("identifiers", {})
    old_fitting = identifiers.get("fitting_model")
    if old_fitting not in (None, fitting_model_identifier):
        raise v05.V05Error("manifest fitting/model identifier does not match the new fit")
    identifiers["fitting_model"] = fitting_model_identifier
    identifiers["renderer_build"] = renderer_build_identifier
    identifiers["hdr_metadata"] = hdr_metadata_identifier
    for record in manifest.get("segments", []):
        record["fitting_model_identifier"] = fitting_model_identifier
        record["renderer_build_identifier"] = renderer_build_identifier
        record["hdr_metadata_identifier"] = hdr_metadata_identifier


def _validate_final(
    helper: Any,
    ffprobe: Path,
    output: Path,
    source_hdr10: dict[str, Any],
    source_colors: dict[str, str],
    extension_contract: dict[str, Any] | None,
    expected_count: int,
    fps: float,
) -> dict[str, Any]:
    final_output = phase2._validate_phase2_output(
        helper,
        ffprobe,
        output,
        source_hdr10,
        source_colors,
        extension_contract or {"extended_open_matte_pass": True},
        expected_count=expected_count,
    )
    structural = _probe_segment(helper, ffprobe, output, expected_count, fps)
    contract = dict(final_output["contract"])
    quality_pass = bool(contract.get("extended_open_matte", False))
    storage_contract = {
        key: value for key, value in contract.items() if key != "extended_open_matte"
    }
    storage_pass = bool(all(storage_contract.values()) and structural["pass"])
    return {
        "output": final_output,
        "structural": structural,
        "contract": contract,
        "storage_contract": storage_contract,
        "storage_pass": storage_pass,
        "quality_pass": quality_pass,
        "pass": bool(storage_pass),
    }


def _run_concat(
    ffmpeg: Path,
    segment_paths: list[Path],
    list_path: Path,
    output: Path,
) -> tuple[list[str], str]:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with list_path.open("w", encoding="utf-8", newline="\n") as handle:
            for path in segment_paths:
                escaped = path.resolve().as_posix().replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
            handle.flush()
            os.fsync(handle.fileno())
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=str(WORKSPACE),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
            check=False,
        )
        log = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0 or not output.is_file():
            raise v05.V05Error(
                f"packet concat failed ({completed.returncode}): {log[-4000:]}"
            )
        return command, log
    finally:
        if list_path.exists():
            list_path.unlink()


def _render_missing_segments(
    args: argparse.Namespace,
    profile: dict[str, Any],
    profile_path: Path,
    full_config: Any,
    plans: list[SegmentPlan],
    manifest: dict[str, Any],
    manifest_path: Path,
    segment_dir: Path,
    helper: Any,
    ffprobe: Path,
    backend: Any,
    fit: dict[str, Any],
    rendered_model: dict[str, Any],
    renderer_build_identifier: str,
    fitting_model_identifier: str,
    hdr_metadata_identifier: str,
    bridge: Path,
) -> list[dict[str, Any]]:
    records_by_index = {
        int(record["segment_index"]): record
        for record in manifest.get("segments", [])
    }
    for plan in plans:
        if plan.index in records_by_index:
            continue
        final_path = _segment_path(segment_dir, plan)
        temporary_path = _segment_tmp_path(segment_dir, plan)
        if temporary_path.exists():
            temporary_path.unlink()
        if final_path.exists():
            _quarantine(final_path, "rerender_segment")
        segment_config = dataclasses.replace(
            full_config,
            shot_start=int(full_config.shot_start) + plan.start_offset,
            shot_end=int(full_config.shot_start) + plan.start_offset + plan.frame_count,
            max_frames=plan.frame_count,
            output=temporary_path,
        )
        print(
            f"segment {plan.index + 1}/{len(plans)}: "
            f"HDR [{segment_config.shot_start},{segment_config.shot_end}), "
            f"OM [{segment_config.shot_start + segment_config.offset_frames},"
            f"{segment_config.shot_start + segment_config.offset_frames + plan.frame_count}), "
            f"frames={plan.frame_count}",
            flush=True,
        )
        source = CudaDecoderBackend(native.NativeFrameSource(
            segment_config,
            plan.frame_count,
            bridge_path=bridge,
            ring_size=int(args.ring_size),
            device_index=int(args.device_index),
            hdr_decoder=args.hdr_decoder,
            om_decoder=args.om_decoder,
        ))
        render = native.render_native_phase2(
            segment_config,
            source,
            backend,
            fit,
            temporary_path,
            rendered_model,
            diagnostic_stride=int(args.diagnostic_stride),
            resample_backend=(
                make_cuda_resample_backend()
                if getattr(segment_config, "output_size", None) is not None
                and tuple(int(value) for value in segment_config.output_size)
                != tuple(int(value) for value in segment_config.om_size)
                else None
            ),
            encoder_backend_factory=make_cuda_encoder_factory(
                bridge_path=bridge,
                ffmpeg_bin=segment_config.ffmpeg.parent,
                device_index=int(args.device_index),
                qp=int(args.encoder_qp),
            ),
            frame_factory=make_cuda_frame_factory(int(args.device_index)),
        )
        if int(render.get("frames", -1)) != plan.frame_count:
            raise v05.V05Error(
                f"segment {plan.index} rendered {render.get('frames')} of {plan.frame_count} frames"
            )
        # render_native_phase2 returns only after the encoder trailer and decoder
        # cleanup.  Probe the temporary file before making it visible as .mkv.
        _probe_segment(helper, ffprobe, temporary_path, plan.frame_count, full_config.fps)
        os.replace(temporary_path, final_path)
        try:
            final_probe = _probe_segment(
                helper, ffprobe, final_path, plan.frame_count, full_config.fps
            )
        except BaseException:
            _quarantine(final_path, "post_rename_validation")
            raise
        stream_signature = manifest["configuration"].get("stream_signature")
        if stream_signature is not None and final_probe["stream_signature"] != stream_signature:
            _quarantine(final_path, "stream_signature_mismatch")
            raise v05.V05Error(
                f"segment {plan.index} stream configuration differs from prior segments"
            )
        if stream_signature is None:
            manifest["configuration"]["stream_signature"] = final_probe["stream_signature"]
        record = _segment_record(
            plan,
            full_config,
            final_probe,
            _relative_filename(final_path, manifest_path.parent),
            renderer_build_identifier,
            fitting_model_identifier,
            hdr_metadata_identifier,
            _compact_render_summary(render),
        )
        records_by_index[plan.index] = record
        manifest["segments"] = [records_by_index[index] for index in sorted(records_by_index)]
        manifest["extension_contract"] = render.get("extension_contract", {})
        manifest["status"] = "in_progress"
        manifest["final"] = None
        _atomic_write_json(manifest_path, manifest)
        print(f"segment complete: {final_path}", flush=True)
    return [records_by_index[plan.index] for plan in plans]


def _run_segmented(args: argparse.Namespace) -> dict[str, Any]:
    if not math.isclose(float(args.fit_scale), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise v05.V05Error("V5 segmented Phase 2 keeps the frozen V4 fit_scale at exactly 0.5")
    profile_path = base.workspace_path(args.profile)
    profile = base.load_profile(profile_path)
    output = base.workspace_path(args.output)
    report = base.workspace_path(args.report) if args.report else output.with_suffix(".json")
    manifest_path = (
        base.workspace_path(args.manifest)
        if args.manifest
        else output.parent / "render_manifest.json"
    )
    segment_dir = (
        base.workspace_path(args.segment_dir)
        if args.segment_dir
        else output.parent / "segments"
    )
    bridge = base.workspace_path(args.bridge)
    ffprobe = base.workspace_path(args.ffprobe)
    remuxer = base.workspace_path(args.metadata_remuxer)
    for path in (output, report, manifest_path, segment_dir):
        _validate_path(path)
    for path, label in (
        (ffprobe, "ffprobe"),
        (remuxer, "HDR10 metadata remuxer"),
    ):
        if not path.is_file():
            raise v05.V05Error(f"{label} is missing: {path}")
    requested_count = (
        int(args.max_frames)
        if args.max_frames is not None
        else int(profile["expected_frame_count"])
    )
    if requested_count <= 0 or requested_count > int(profile["expected_frame_count"]):
        raise v05.V05Error("--max-frames must be between 1 and the profile expected_frame_count")
    full_config = base.build_config(profile, output, requested_count)
    full_config = dataclasses.replace(
        full_config,
        shot_end=int(full_config.shot_start) + requested_count,
        max_frames=requested_count,
        output=output,
    )
    segment_frame_count = _segment_frame_count(full_config, float(args.segment_duration))
    plans = _make_plans(requested_count, segment_frame_count)
    segment_dir.mkdir(parents=True, exist_ok=True)
    helper, source_probe, source_hdr10, source_colors = phase2._source_metadata(
        ffprobe,
        full_config.hdr_source,
    )
    hdr_metadata_identifier = _hdr_metadata_identifier(source_hdr10, source_colors)
    renderer_build_identifier = _renderer_build_identifier(bridge)
    profile_identifier = _profile_identifier(profile)
    del source_probe, profile_identifier

    manifest = _load_manifest(manifest_path)
    if manifest is None:
        if output.exists():
            raise v05.V05Error(
                f"final output exists without a manifest; refusing to overwrite: {output}"
            )
        manifest = _new_manifest(
            manifest_path,
            profile_path,
            profile,
            full_config,
            output,
            segment_dir,
            plans,
            float(args.segment_duration),
            renderer_build_identifier,
            hdr_metadata_identifier,
        )
        _atomic_write_json(manifest_path, manifest)
    else:
        _validate_manifest_contract(
            manifest,
            profile,
            full_config,
            output,
            segment_dir,
            plans,
            float(args.segment_duration),
            renderer_build_identifier,
            hdr_metadata_identifier,
            manifest_path,
        )

    reconciled = _reconcile_segments(
        manifest,
        manifest_path,
        segment_dir,
        plans,
        full_config,
        helper,
        ffprobe,
        renderer_build_identifier,
        hdr_metadata_identifier,
    )
    _atomic_write_json(manifest_path, manifest)

    identifiers = manifest["identifiers"]
    all_segments_complete = len(reconciled) == len(plans)
    needs_fit = not all_segments_complete or identifiers.get("fitting_model") is None
    fit: dict[str, Any] | None = None
    rendered_model: dict[str, Any] | None = None
    fit_seconds = 0.0
    cuda: dict[str, Any] | None = None
    memory_monitor: Any | None = None
    hardware_monitor: Any | None = None
    backend: Any | None = None
    if needs_fit:
        cuda = v05.cuda_diagnostics()
        if not cuda.get("available") or not cuda.get("runtime_ready"):
            raise v05.V05Error(f"CUDA runtime is unavailable: {cuda.get('runtime_error')}")
        if not bridge.is_file():
            raise v05.V05Error(f"V5 native bridge DLL is missing: {bridge}")
        backend = CudaComputeBackend(v05.fastcore.Backend(use_gpu=True))
        native.diagnostic_marker(
            "INIT_OK",
            phase="phase2_segmented",
            frame_count=requested_count,
            segment_frame_count=segment_frame_count,
            segment_count=len(plans),
            device_index=int(args.device_index),
            hdr_decoder=args.hdr_decoder,
            om_decoder=args.om_decoder,
            cuda_device=cuda.get("device_id"),
            cuda_device_name=cuda.get("device_name"),
            bridge=str(bridge),
        )
        memory_monitor = v05.MemoryMonitor(True, 0)
        hardware_monitor = v05.HardwareMonitor(True)
        memory_monitor.start()
        hardware_monitor.start()
        fit_started = time.perf_counter()
        source = CudaDecoderBackend(native.NativeFrameSource(
            full_config,
            requested_count,
            bridge_path=bridge,
            ring_size=int(args.ring_size),
            device_index=int(args.device_index),
            hdr_decoder=args.hdr_decoder,
            om_decoder=args.om_decoder,
        ))
        try:
            print(
                f"v05-native segmented: fit once over {requested_count} frames; "
                f"segment_frames={segment_frame_count}, segments={len(plans)}",
                flush=True,
            )
            fitter = native.NativeV4GpuShotFitter(backend, fit_stride=int(args.fit_stride))
            fit = fitter.fit_shot(source, full_config, float(args.fit_scale))
            fit["config"] = full_config
            rendered_model = (
                fit["default_model"]
                if args.render_model == "default"
                else fit["review_model"]
            )
            fit["rendered_model"] = rendered_model
            fit_profile = fit.get("fit_profile", {})
            fit_profile["native_source"] = source.metadata
            fit_profile["cpu_rgb_transport"] = False
            fit_profile["full_frame_d2h"] = 0
            fit_profile["per_frame_d2h"] = 0
            fitting_model_identifier = _fit_model_identifier(fit, rendered_model, profile)
            _update_manifest_identifiers(
                manifest,
                fitting_model_identifier,
                renderer_build_identifier,
                hdr_metadata_identifier,
            )
            fit_seconds = time.perf_counter() - fit_started
            manifest["fit"] = {
                "fit_seconds": fit_seconds,
                "fit_frame_count": requested_count,
                "render_model": args.render_model,
                "fitting_model_identifier": fitting_model_identifier,
            }
            _atomic_write_json(manifest_path, manifest)
            try:
                del fitter
                import cupy

                cupy.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        finally:
            # fit_shot's iterators own and close their decoders.  This is only
            # a defensive cleanup for an exception before the first iterator.
            try:
                source.close()
            except Exception:
                pass
        reconciled = _render_missing_segments(
            args,
            profile,
            profile_path,
            full_config,
            plans,
            manifest,
            manifest_path,
            segment_dir,
            helper,
            ffprobe,
            backend,
            fit,
            rendered_model,
            renderer_build_identifier,
            fitting_model_identifier,
            hdr_metadata_identifier,
            bridge,
        )
        if memory_monitor is not None:
            memory_monitor.stop()
        if hardware_monitor is not None:
            hardware_monitor.stop()
    else:
        fitting_model_identifier = str(identifiers["fitting_model"])

    if len(reconciled) != len(plans):
        raise v05.V05Error(
            f"checkpoint contains {len(reconciled)} of {len(plans)} complete segments"
        )
    if manifest.get("configuration", {}).get("stream_signature") is None:
        raise v05.V05Error("complete segmented output has no stream signature checkpoint")
    _atomic_write_json(manifest_path, manifest)

    final_validation: dict[str, Any] | None = None
    if output.exists():
        try:
            final_validation = _validate_final(
                helper,
                ffprobe,
                output,
                source_hdr10,
                source_colors,
                manifest.get("extension_contract"),
                requested_count,
                full_config.fps,
            )
        except BaseException as exc:
            _quarantine(output, "invalid_final_output")
            print(f"existing final output will be rebuilt: {exc}", file=sys.stderr, flush=True)
    concat_command: list[str] | None = None
    concat_log = ""
    metadata_command: list[str] | None = None
    metadata_log = ""
    if final_validation is None:
        concat_tmp = output.with_name(f".{output.stem}.concat_pre_hdr10{output.suffix}")
        final_tmp = output.with_name(f".{output.stem}.hdr10{output.suffix}")
        for stale in (concat_tmp, final_tmp):
            if stale.exists():
                stale.unlink()
        list_path = output.with_name(f".{output.stem}.concat.txt")
        segment_paths = [_segment_path(segment_dir, plan) for plan in plans]
        concat_command, concat_log = _run_concat(
            full_config.ffmpeg, segment_paths, list_path, concat_tmp
        )
        try:
            metadata_command, metadata_log = phase2._remux_hdr10(
                helper,
                source_hdr10,
                concat_tmp,
                final_tmp,
                remuxer,
                full_config.ffmpeg,
            )
            os.replace(final_tmp, output)
        finally:
            if concat_tmp.exists():
                concat_tmp.unlink()
            if final_tmp.exists():
                final_tmp.unlink()
        final_validation = _validate_final(
            helper,
            ffprobe,
            output,
            source_hdr10,
            source_colors,
            manifest.get("extension_contract"),
            requested_count,
            full_config.fps,
        )

    previous_final = manifest.get("final") if isinstance(manifest.get("final"), dict) else {}
    manifest_validation = {
        "pass": bool(final_validation["pass"]),
        "storage_pass": bool(final_validation["storage_pass"]),
        "quality_pass": bool(final_validation["quality_pass"]),
        "contract": final_validation["contract"],
        "structural": {
            "frame_count": final_validation["structural"]["frame_count"],
            "pts_start_seconds": final_validation["structural"]["pts_start_seconds"],
            "pts_end_seconds": final_validation["structural"]["pts_end_seconds"],
            "time_base": final_validation["structural"]["time_base"],
            "contract": final_validation["structural"]["contract"],
        },
    }
    final_entry = {
        "filename": _relative_filename(output, manifest_path.parent),
        "status": "complete" if final_validation["pass"] else "invalid",
        "frame_count": requested_count,
        "concat": (
            {
                "command": concat_command,
                "mode": "FFmpeg concat demuxer with -c copy; no re-encode",
                "log": concat_log,
            }
            if concat_command is not None
            else previous_final.get("concat", {})
        ),
        "hdr10_remux": (
            {
                "command": metadata_command,
                "mode": "existing metadata-only HDR10 Matroska remux; encoded packets copied",
                "log": metadata_log,
            }
            if metadata_command is not None
            else previous_final.get("hdr10_remux", {})
        ),
        "validation": final_validation,
    }
    manifest["final"] = {
        key: value for key, value in final_entry.items() if key != "validation"
    }
    manifest["final"]["validation"] = manifest_validation
    manifest["status"] = "complete" if final_validation["pass"] else "in_progress"
    _atomic_write_json(manifest_path, manifest)

    memory = memory_monitor.result() if memory_monitor is not None else None
    hardware = hardware_monitor.result() if hardware_monitor is not None else None
    render_summaries = [record.get("render", {}) for record in manifest["segments"]]
    total_render_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in render_summaries)
    storage_pass = bool(final_validation["storage_pass"])
    quality_pass = bool(final_validation["quality_pass"])
    payload = {
        "schema": REPORT_SCHEMA,
        "status": "PASS" if storage_pass else "FAIL",
        "renderer": "v05-gpu-native-phase2-segmented",
        "profile_path": profile_path,
        "profile": profile,
        "configuration": {
            "output": output,
            "report": report,
            "manifest": manifest_path,
            "segment_directory": segment_dir,
            "max_frames": requested_count,
            "segment_duration_seconds": float(args.segment_duration),
            "segment_frame_count": segment_frame_count,
            "segment_count": len(plans),
            "fit_stride": int(args.fit_stride),
            "fit_scale": float(args.fit_scale),
            "render_model": args.render_model,
            "ring_size_per_source": int(args.ring_size),
            "device_index": int(args.device_index),
            "hdr_decoder": args.hdr_decoder,
            "om_decoder": args.om_decoder,
            "fps": full_config.fps_text,
            "hdr_interval": [full_config.shot_start, full_config.shot_start + requested_count],
            "om_interval": [
                full_config.shot_start + full_config.offset_frames,
                full_config.shot_start + full_config.offset_frames + requested_count,
            ],
        },
        "identifiers": manifest["identifiers"],
        "fit": manifest.get("fit"),
        "segments": manifest["segments"],
        "final": final_entry,
        "validation": {
            "segmented_output_pass": storage_pass,
            "quality_pass": quality_pass,
            "output": final_validation,
            "no_frame_cache": True,
            "no_duplicate_or_missing_frames": bool(
                all(
                    record["frame_count"]
                    == record["last_frame"] - record["first_frame"] + 1
                    for record in manifest["segments"]
                )
                and sum(record["frame_count"] for record in manifest["segments"]) == requested_count
            ),
            "fit_once": True,
            "resume_completed_segments": True,
        },
        "timing_seconds": {
            "fit": fit_seconds,
            "segment_render_sum": total_render_seconds,
        },
        "telemetry": {"memory": memory, "hardware": hardware},
        "guarantees": {
            "frozen_v4_fit": True,
            "frozen_v4_render_equations": True,
            "nvdec_cuda_frames": True,
            "zero_copy_nvenc": True,
            "decoded_frame_d2h_bytes": 0,
            "render_output_d2h_bytes": 0,
            "h2d_decoded_video_bytes": 0,
            "disk_frame_cache": False,
            "atomic_segment_close": True,
            "manifest_atomic_checkpoint": True,
            "concat_without_reencode": True,
            "hdr10_metadata_only_final_remux": True,
        },
        "limitations": {
            "quality_gate": (
                "quality_pass is reported separately; segmented storage PASS does not "
                "override the existing Open Matte extension quality contract"
            ),
            "segment_boundary": (
                "each segment starts a fresh NVENC session because the native ABI has "
                "no append API; final concat is packet-level -c copy"
            ),
        },
    }
    _atomic_write_json(report, payload)
    print(f"segmented output: {output}", flush=True)
    print(f"manifest: {manifest_path}", flush=True)
    print(f"report: {report}", flush=True)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "segments": len(manifest["segments"]),
                "frame_count": requested_count,
                "storage_pass": storage_pass,
                "quality_pass": quality_pass,
                "hdr10": final_validation["contract"].get("hdr10_metadata_matches_source"),
                "concat_without_reencode": True,
                "resume_complete_segments": len(reconciled),
            },
            indent=2,
        ),
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-v05-gpu-native-segmented",
        description="V5 native Phase 2 segmented output with atomic checkpoints and resume",
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--segment-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--bridge", type=Path, default=phase2.DEFAULT_BRIDGE)
    parser.add_argument("--ffprobe", type=Path, default=phase2.DEFAULT_FFPROBE)
    parser.add_argument("--metadata-remuxer", type=Path, default=phase2.DEFAULT_METADATA_REMUXER)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--segment-duration", type=float, default=60.0)
    parser.add_argument("--fit-stride", type=int, default=1)
    parser.add_argument("--fit-scale", type=float, default=0.5)
    parser.add_argument("--diagnostic-stride", type=int, default=6)
    parser.add_argument("--ring-size", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--hdr-decoder", default="hevc_cuvid")
    parser.add_argument("--om-decoder", default="h264_cuvid")
    parser.add_argument("--encoder-qp", type=int, default=18)
    parser.add_argument("--render-model", choices=("default", "review"), default="default")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if (
            args.fit_stride <= 0
            or args.diagnostic_stride <= 0
            or args.ring_size < 2
            or args.device_index < 0
            or not math.isfinite(args.fit_scale)
            or not (0 < args.fit_scale <= 1)
            or args.encoder_qp < 0
            or args.encoder_qp > 51
            or not math.isfinite(args.segment_duration)
            or args.segment_duration <= 0
        ):
            raise v05.V05Error("invalid segmented runner configuration")
        _run_segmented(args)
        return 0
    except BaseException as exc:
        print(f"v05-native-segmented error: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        report = (
            base.workspace_path(args.report)
            if args.report
            else base.workspace_path(args.output).with_suffix(".json")
        )
        try:
            _atomic_write_json(
                report,
                {
                    "schema": REPORT_SCHEMA,
                    "status": "FAIL",
                    "exception": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)
                        ),
                    },
                },
            )
        except BaseException:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
