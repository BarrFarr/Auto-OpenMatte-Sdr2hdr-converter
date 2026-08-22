"""Run the native V5 pipeline.

The default mode preserves the original Phase 1 TEST 10 runner.  The explicit
``--phase2-zero-copy`` mode uses the same native NVDEC source and frozen V4/V5
math, then converts the composited CUDA RGB result to GPU P010 and sends it to
the integrated FFmpeg/NVENC bridge without decoded-frame or render-output D2H.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
DEFAULT_PROFILE = HERE / "profiles" / "br2049_legacy_t2714_30f.json"
DEFAULT_OUTPUT = WORKSPACE / "benchmarks" / "v05_gpu_native_test10" / "v5_native_phase1_test10.mkv"
DEFAULT_REPORT = DEFAULT_OUTPUT.with_suffix(".json")
DEFAULT_PHASE2_PROFILE = HERE / "profiles" / "br2049_phase2_t2714_50f.json"
DEFAULT_PHASE2_OUTPUT = WORKSPACE / "benchmarks" / "v05_gpu_native_phase2_test50" / "v5_native_phase2_test50.mkv"
DEFAULT_PHASE2_REPORT = DEFAULT_PHASE2_OUTPUT.with_suffix(".json")
DEFAULT_BRIDGE = WORKSPACE / "dev" / "v05-native" / "v5_gpu_bridge.dll"
DEFAULT_FFPROBE = WORKSPACE / "dev" / "ffmpeg-build" / "install" / "bin" / "ffprobe.exe"
DEFAULT_METADATA_REMUXER = WORKSPACE / "dev" / "ffmpeg-build" / "api-probe" / "inject_hdr10_mkv_metadata.exe"
V43_BASELINE_FPS = 0.774236581954846
PHASE1_TOTAL_FPS_BASELINE = 3.0178
TRANSPORT_ONLY_FPS_BASELINE = 17.4
TEST_FRAME_COUNT = 10
PHASE2_FRAME_COUNT = 50


def _validate_output(path: Path, *, must_not_exist: bool = False) -> None:
    lowered = {part.lower() for part in path.resolve().parts}
    banned = {part.lower() for part in v05.BANNED_OUTPUT_PARTS}
    if lowered.intersection(banned):
        raise v05.V05Error(f"Refusing output in cache/temp-like path: {path}")
    if must_not_exist and path.exists():
        raise v05.V05Error(f"Refusing to overwrite existing Phase 2 artifact: {path}")


def _load_metadata_helper() -> Any:
    path = WORKSPACE / "dev" / "ffmpeg-build" / "api-probe" / "run_zero_copy_nvdec_nvenc.py"
    spec = importlib.util.spec_from_file_location("v05_hdr10_metadata_helper", path)
    if spec is None or spec.loader is None:
        raise v05.V05Error(f"cannot load HDR10 metadata helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_metadata(ffprobe: Path, source: Path) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, str]]:
    helper = _load_metadata_helper()
    probe = helper._ffprobe_json(
        ffprobe,
        ["-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json"],
        source,
    )
    stream = helper._first_stream(probe)
    hdr10 = helper._source_hdr10_side_data(ffprobe, source)
    fields = {
        field: stream.get(field)
        for field in ("color_range", "color_space", "color_transfer", "color_primaries")
    }
    missing = [field for field, value in fields.items() if not value]
    if missing:
        raise v05.V05Error(f"source color signaling is incomplete; missing: {missing}")
    return helper, probe, hdr10, fields


def _remux_hdr10(
    helper: Any,
    source_hdr10: dict[str, Any],
    pre_output: Path,
    output: Path,
    remuxer: Path,
    ffmpeg_bin: Path,
) -> tuple[list[str], str]:
    command = [
        str(remuxer),
        os.path.relpath(pre_output.resolve(), WORKSPACE),
        os.path.relpath(output.resolve(), WORKSPACE),
        *helper._hdr10_cli_values(source_hdr10),
    ]
    environment = os.environ.copy()
    environment["PATH"] = str(ffmpeg_bin.parent) + os.pathsep + environment.get("PATH", "")
    completed = subprocess.run(
        command,
        cwd=str(WORKSPACE),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0 or not output.is_file():
        raise v05.V05Error(
            f"HDR10 metadata-only remux failed ({completed.returncode}): {log[-4000:]}"
        )
    return command, log


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _write_diagnostic_failure_report(args: argparse.Namespace, error: BaseException) -> None:
    events = native.diagnostic_events()
    if not any(event.get("marker") == "FIRST_FAILURE" for event in events):
        native.diagnostic_exception("runner", error)
        events = native.diagnostic_events()
    first_failure = next(
        (event for event in events if event.get("marker") == "FIRST_FAILURE"),
        None,
    )
    report = base.workspace_path(args.report) if args.report else None
    if report is None:
        return
    payload = {
        "schema": "openmatte-hdr-v05-gpu-native-diagnostic/v1",
        "status": "FAIL",
        "first_failure": first_failure,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        },
        "events": events,
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key in {
                "phase2_zero_copy", "diagnostic_test10", "profile", "max_frames",
                "output", "pre_metadata_output", "report", "fit_stride", "fit_scale",
                "diagnostic_stride", "ring_size", "device_index", "hdr_decoder",
                "om_decoder", "encoder_qp", "render_model",
            }
        },
        "artifacts": {
            "output": args.output,
            "pre_metadata_output": args.pre_metadata_output,
            "report": report,
        },
    }
    try:
        v05.write_json(report, payload)
        print(f"diagnostic failure report: {report}", file=sys.stderr, flush=True)
    except BaseException as report_error:
        print(
            "diagnostic failure report write failed: "
            f"{type(report_error).__name__}: {report_error}",
            file=sys.stderr,
            flush=True,
        )


def _validate_phase2_output(
    helper: Any,
    ffprobe: Path,
    output: Path,
    source_hdr10: dict[str, Any],
    source_colors: dict[str, str],
    extension_contract: dict[str, Any],
    expected_count: int = PHASE2_FRAME_COUNT,
) -> dict[str, Any]:
    probe = helper._ffprobe_json(
        ffprobe,
        ["-count_frames", "-show_streams", "-show_format", "-of", "json"],
        output,
    )
    stream = helper._first_stream(probe)
    frames = helper._output_frames(ffprobe, output)
    output_count = helper._int_or_none(stream.get("nb_read_frames"))
    hdr_comparison = helper._compare_hdr10_metadata(source_hdr10, stream)
    color_match = all(stream.get(field) == source_colors[field] for field in source_colors)
    contract = {
        "codec_name_hevc": stream.get("codec_name") == "hevc",
        "main10": "10" in str(stream.get("profile", "")) or "10" in str(stream.get("pix_fmt", "")),
        "resolution_3840x2160": stream.get("width") == 3840 and stream.get("height") == 2160,
        "pix_fmt_yuv420p10le": stream.get("pix_fmt") == "yuv420p10le",
        "bt2020nc": stream.get("color_space") == "bt2020nc",
        "smpte2084": stream.get("color_transfer") == "smpte2084",
        "bt2020": stream.get("color_primaries") == "bt2020",
        "tv_range": stream.get("color_range") == "tv",
        "color_signaling_matches_source": color_match,
        "sar_1_1": stream.get("sample_aspect_ratio") == "1:1",
        "frame_count_expected": output_count == expected_count and len(frames) == expected_count,
        "hdr10_metadata_matches_source": bool(hdr_comparison["all_match"]),
        "extended_open_matte": bool(extension_contract.get("extended_open_matte_pass")),
    }
    timestamps = [helper._float_or_none(frame.get("best_effort_timestamp_time")) for frame in frames]
    contract["timestamps_monotonic"] = all(
        timestamps[index] <= timestamps[index + 1]
        for index in range(len(timestamps) - 1)
        if timestamps[index] is not None and timestamps[index + 1] is not None
    ) and all(value is not None for value in timestamps)
    return {
        "probe": probe,
        "stream": helper._metadata_snapshot(stream),
        "frame_timestamps": frames,
        "hdr10_comparison": hdr_comparison,
        "contract": contract,
        "pass": all(contract.values()),
    }


def _run_phase1(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.max_frames) != TEST_FRAME_COUNT:
        raise v05.V05Error(
            f"V5 phase-1 is safety-gated to exactly TEST {TEST_FRAME_COUNT}; do not run a different frame count"
        )
    if not math.isclose(float(args.fit_scale), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise v05.V05Error("V5 phase-1 keeps the frozen V4 fit_scale at exactly 0.5")
    profile_path = base.workspace_path(args.profile)
    profile = base.load_profile(profile_path)
    output = base.workspace_path(args.output)
    report = base.workspace_path(args.report) if args.report else output.with_suffix(".json")
    _validate_output(output)
    _validate_output(report)
    config = base.build_config(profile, output, TEST_FRAME_COUNT)
    config = dataclasses.replace(config, max_frames=TEST_FRAME_COUNT)
    cuda = v05.cuda_diagnostics()
    if not cuda.get("available") or not cuda.get("runtime_ready"):
        raise v05.V05Error(f"CUDA runtime is unavailable: {cuda.get('runtime_error')}")
    bridge = base.workspace_path(args.bridge)
    if not bridge.is_file():
        raise v05.V05Error(f"V5 native bridge DLL is missing: {bridge}")
    backend = CudaComputeBackend(v05.fastcore.Backend(use_gpu=True))
    source = CudaDecoderBackend(native.NativeFrameSource(
        config, TEST_FRAME_COUNT, bridge_path=bridge, ring_size=int(args.ring_size),
        device_index=int(args.device_index), hdr_decoder=args.hdr_decoder, om_decoder=args.om_decoder,
    ))
    total_started = time.perf_counter()
    print(f"v05-native: TEST {TEST_FRAME_COUNT}, fit_scale=0.5, HDR decoder={args.hdr_decoder}, OM decoder={args.om_decoder}, ring={args.ring_size}")
    print(f"HDR [{config.shot_start},{config.shot_start + TEST_FRAME_COUNT}), OM [{config.shot_start + config.offset_frames},{config.shot_start + config.offset_frames + TEST_FRAME_COUNT})")
    fitter = native.NativeV4GpuShotFitter(backend, fit_stride=int(args.fit_stride))
    fit = fitter.fit_shot(source, config, float(args.fit_scale))
    fit["config"] = config
    rendered_model = fit["default_model"] if args.render_model == "default" else fit["review_model"]
    fit["rendered_model"] = rendered_model
    fit_profile = fit.get("fit_profile", {})
    fit_profile["native_source"] = source.metadata
    fit_profile["cpu_rgb_transport"] = False
    fit_profile["full_frame_d2h"] = 0
    fit_profile["per_frame_d2h"] = 0
    try:
        import cupy
        del fitter
        cupy.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    render = native.render_native(config, source, backend, fit, output, rendered_model, diagnostic_stride=int(args.diagnostic_stride))
    elapsed = time.perf_counter() - total_started
    render_fps = float(render["fps"])
    payload = {
        "schema": "openmatte-hdr-v05-gpu-native-phase1/v1", "status": "PASS", "renderer": "v05-gpu-native-phase1",
        "profile_path": profile_path, "profile": profile,
        "configuration": {
            "hdr_source": config.hdr_source, "om_source": config.om_source, "output": output, "report": report,
            "hdr_interval": [config.shot_start, config.shot_start + TEST_FRAME_COUNT],
            "om_interval": [config.shot_start + config.offset_frames, config.shot_start + config.offset_frames + TEST_FRAME_COUNT],
            "offset_frames": config.offset_frames, "overlap": config.overlap, "fps": config.fps_text,
            "max_frames": TEST_FRAME_COUNT, "fit_stride": args.fit_stride, "fit_scale": float(args.fit_scale),
            "render_model": args.render_model, "ring_size_per_source": int(args.ring_size), "device_index": int(args.device_index),
            "hdr_decoder": args.hdr_decoder, "om_decoder": args.om_decoder,
        },
        "startup": {"cuda": cuda, "bridge": bridge, "native_source": source.metadata, "backend_capabilities": backend.capabilities.as_dict()},
        "model": base.model_summary(fit, rendered_model), "render": render,
        "timing_seconds": {"fit": fit["fit_seconds"], "stream_render": render["wall_seconds"], "total": elapsed},
        "comparison": {"v43_baseline_fps": V43_BASELINE_FPS, "v5_phase1_render_fps": render_fps, "v5_phase1_total_fps": TEST_FRAME_COUNT / max(elapsed, 1e-9), "render_speedup_vs_v43": render_fps / V43_BASELINE_FPS, "baseline_scope": "V4.3 TEST 30; V5 scope is TEST 10 only"},
        "guarantees": {"nvdec_cuda_frames": True, "hdr_uses_hevc_cuvid": args.hdr_decoder == "hevc_cuvid", "open_matte_uses_h264_cuvid": args.om_decoder == "h264_cuvid", "gpu_working_rgb": True, "decoded_rgb_host_transport": False, "disk_frame_cache": False, "frozen_v4_fit": True, "frozen_v4_render_equations": True, "zero_copy_nvenc": False, "temporary_output_d2h": True, "test_frame_gate": TEST_FRAME_COUNT},
        "limitations": {"phase": "native NVDEC and GPU working RGB only", "output_boundary": "final PQ16 is copied to host for existing FFmpeg/NVENC stdin wrapper", "cuda_fused_render": "not implemented; V4 equations are reused on device", "cuda_graphs": "not implemented", "quality_parity": "native YUV sampling must be compared against V4 reference before production claim"},
    }
    v05.write_json(report, payload)
    print(f"fit {fit['fit_seconds']:.1f}s; render {render['wall_seconds']:.1f}s ({render_fps:.3f} fps); total {elapsed:.1f}s")
    print(f"output: {output}")
    print(f"report: {report}")
    return payload


def _run_phase2(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic_test10 = bool(getattr(args, "diagnostic_test10", False))
    frame_count = TEST_FRAME_COUNT if diagnostic_test10 else PHASE2_FRAME_COUNT
    if int(args.max_frames) != frame_count:
        mode = "diagnostic Phase 2" if diagnostic_test10 else "V5 Phase 2"
        raise v05.V05Error(f"{mode} is fixed to exactly TEST {frame_count}")
    if not math.isclose(float(args.fit_scale), 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise v05.V05Error("V5 Phase 2 requires the frozen V4 fit_scale=0.5")
    profile_path = base.workspace_path(args.profile)
    profile = base.load_profile(profile_path)
    if (int(profile["shot_start"]), int(profile["shot_end"]), int(profile["om_start_frame"])) != (65071, 65121, 66238):
        raise v05.V05Error("Phase 2 profile must be HDR [65071,65121) and OM [66238,66288)")
    output = base.workspace_path(args.output)
    report = base.workspace_path(args.report) if args.report else output.with_suffix(".json")
    pre_output = base.workspace_path(args.pre_metadata_output) if args.pre_metadata_output else output.with_name(output.stem + "_pre_hdr10" + output.suffix)
    for artifact in (output, report, pre_output):
        _validate_output(artifact, must_not_exist=True)
    config = base.build_config(profile, pre_output, frame_count)
    config = dataclasses.replace(config, max_frames=frame_count)
    cuda = v05.cuda_diagnostics()
    if not cuda.get("available") or not cuda.get("runtime_ready"):
        raise v05.V05Error(f"CUDA runtime is unavailable: {cuda.get('runtime_error')}")
    bridge = base.workspace_path(args.bridge)
    ffprobe = base.workspace_path(args.ffprobe)
    remuxer = base.workspace_path(args.metadata_remuxer)
    for path, label in ((bridge, "V5 native bridge"), (ffprobe, "ffprobe"), (remuxer, "HDR10 metadata remuxer")):
        if not path.is_file():
            raise v05.V05Error(f"{label} is missing: {path}")
    helper, source_probe, source_hdr10, source_colors = _source_metadata(ffprobe, config.hdr_source)
    backend = CudaComputeBackend(v05.fastcore.Backend(use_gpu=True))
    source = CudaDecoderBackend(native.NativeFrameSource(config, frame_count, bridge_path=bridge, ring_size=int(args.ring_size), device_index=int(args.device_index), hdr_decoder=args.hdr_decoder, om_decoder=args.om_decoder))
    memory_monitor = v05.MemoryMonitor(True, 0)
    hardware_monitor = v05.HardwareMonitor(True)
    total_started = time.perf_counter()
    memory_monitor.start()
    hardware_monitor.start()
    native.diagnostic_marker(
        "INIT_OK",
        phase="phase2_diagnostic" if diagnostic_test10 else "phase2",
        frame_count=frame_count,
        device_index=int(args.device_index),
        hdr_decoder=args.hdr_decoder,
        om_decoder=args.om_decoder,
        cuda_device=cuda.get("device_id"),
        cuda_device_name=cuda.get("device_name"),
        bridge=str(bridge),
    )
    metadata_command: list[str] | None = None
    metadata_log = ""
    fit: dict[str, Any] | None = None
    render: dict[str, Any] | None = None
    try:
        print(f"v05-native Phase 2: TEST {frame_count}, fit_scale=0.5, HDR decoder={args.hdr_decoder}, OM decoder={args.om_decoder}, ring={args.ring_size}", flush=True)
        print(f"HDR [{config.shot_start},{config.shot_start + frame_count}), OM [{config.shot_start + config.offset_frames},{config.shot_start + config.offset_frames + frame_count})", flush=True)
        fitter = native.NativeV4GpuShotFitter(backend, fit_stride=int(args.fit_stride))
        fit = fitter.fit_shot(source, config, float(args.fit_scale))
        fit["config"] = config
        rendered_model = fit["default_model"] if args.render_model == "default" else fit["review_model"]
        fit["rendered_model"] = rendered_model
        fit_profile = fit.get("fit_profile", {})
        fit_profile["native_source"] = source.metadata
        fit_profile["cpu_rgb_transport"] = False
        fit_profile["full_frame_d2h"] = 0
        fit_profile["per_frame_d2h"] = 0
        native.diagnostic_marker(
            "FIT_OK",
            fit_seconds=float(fit.get("fit_seconds", 0.0)),
            rendered_model=str(rendered_model.get("name", "unknown")),
            gain_field_shape=[int(value) for value in fit["gain_field"].shape],
            spatial_field_shape=[int(value) for value in fit["spatial_field"].shape],
            frame_count=frame_count,
        )
        try:
            import cupy
            del fitter
            cupy.get_default_memory_pool().free_all_blocks()
        except BaseException as cleanup_error:
            native.diagnostic_exception("fit_memory_cleanup", cleanup_error)
            raise
        render = native.render_native_phase2(
            config, source, backend, fit, pre_output, rendered_model,
            diagnostic_stride=int(args.diagnostic_stride),
            resample_backend=(
                make_cuda_resample_backend()
                if getattr(config, "output_size", None) is not None
                and tuple(int(value) for value in config.output_size)
                != tuple(int(value) for value in config.om_size)
                else None
            ),
            encoder_backend_factory=make_cuda_encoder_factory(
                bridge_path=bridge,
                ffmpeg_bin=config.ffmpeg.parent,
                device_index=int(args.device_index),
                qp=int(args.encoder_qp),
            ),
            frame_factory=make_cuda_frame_factory(int(args.device_index)),
        )
        metadata_started = time.perf_counter()
        metadata_command, metadata_log = _remux_hdr10(helper, source_hdr10, pre_output, output, remuxer, config.ffmpeg)
        metadata_seconds = time.perf_counter() - metadata_started
    finally:
        memory_monitor.stop()
        hardware_monitor.stop()
    if fit is None or render is None:
        raise v05.V05Error("Phase 2 did not produce fit/render results")
    elapsed = time.perf_counter() - total_started
    validation = _validate_phase2_output(
        helper,
        ffprobe,
        output,
        source_hdr10,
        source_colors,
        render["extension_contract"],
        expected_count=frame_count,
    )
    quality_pass = bool(render["quality"].get("diagnostic_frames", 0) > 0 and _finite_tree(render["quality"]))
    parity = {"pass": False, "status": "NOT_RUN", "reason": "No independent 50-frame frozen-V4 final-output reference was generated in the physical run"}
    memory = memory_monitor.result()
    hardware = hardware_monitor.result()
    averages = hardware.get("averages", {})
    peaks = hardware.get("peaks", {})
    output_contract = validation["contract"]
    pipeline_pass = bool(validation["pass"] and render["output_boundary"]["zero_copy_nvenc"] and render["output_boundary"]["d2h_bytes"] == 0 and render["output_boundary"]["h2d_bytes"] == 0 and render["frames"] == frame_count)
    payload = {
        "schema": "openmatte-hdr-v05-gpu-native-phase2-diagnostic/v1" if diagnostic_test10 else "openmatte-hdr-v05-gpu-native-phase2/v1",
        "status": "PASS" if pipeline_pass else "FAIL",
        "renderer": "v05-gpu-native-phase2",
        "diagnostic_test10": diagnostic_test10,
        "profile_path": profile_path,
        "profile": profile,
        "source": {"stream": helper._metadata_snapshot(helper._first_stream(source_probe)), "color_signaling": source_colors, "hdr10": source_hdr10},
        "configuration": {
            "hdr_source": config.hdr_source, "om_source": config.om_source, "output": output, "pre_metadata_output": pre_output, "report": report,
            "hdr_interval": [config.shot_start, config.shot_start + frame_count],
            "om_interval": [config.shot_start + config.offset_frames, config.shot_start + config.offset_frames + frame_count],
            "hdr_start_pts_seconds": float(config.shot_start * 1001 / 24000),
            "om_start_pts_seconds": float((config.shot_start + config.offset_frames) * 1001 / 24000),
            "offset_frames": config.offset_frames, "overlap": config.overlap, "fps": config.fps_text,
            "max_frames": frame_count, "fit_stride": args.fit_stride, "fit_scale": float(args.fit_scale),
            "render_model": args.render_model, "ring_size_per_source": int(args.ring_size), "device_index": int(args.device_index),
            "hdr_decoder": args.hdr_decoder, "om_decoder": args.om_decoder,
            "seek": "PTS-based native av_seek_frame backward seek before timestamp drop; no frame-index select",
        },
        "startup": {"cuda": cuda, "bridge": bridge, "native_source": source.metadata, "backend_capabilities": backend.capabilities.as_dict()},
        "model": base.model_summary(fit, rendered_model),
        "render": render,
        "timing_seconds": {"fit": fit["fit_seconds"], "render": render["wall_seconds"], "pipeline": render["pipeline_seconds"], "metadata_remux": metadata_seconds, "total": elapsed},
        "metrics": {
            "fit_time_seconds": fit["fit_seconds"], "render_time_seconds": render["wall_seconds"], "total_time_seconds": elapsed,
            "whole_v5_fps": frame_count / max(elapsed, 1e-9), "render_fps": render["fps"],
            "peak_ram_bytes": memory.get("peak", {}).get("rss_bytes"), "peak_vram_bytes": memory.get("peak", {}).get("vram_used_bytes"),
            "nvdec_utilization_percent": averages.get("nvdec_utilization_percent"), "cuda_gpu_utilization_percent": averages.get("gpu_utilization_percent"), "nvenc_utilization_percent": averages.get("nvenc_utilization_percent"),
            "nvdec_utilization_peak_percent": peaks.get("nvdec_utilization_percent"), "cuda_gpu_utilization_peak_percent": peaks.get("gpu_utilization_percent"), "nvenc_utilization_peak_percent": peaks.get("nvenc_utilization_percent"),
            "d2h_bytes": render["output_boundary"]["d2h_bytes"], "h2d_bytes": render["output_boundary"]["h2d_bytes"], "frame_count": render["frames"], "output_resolution": [3840, 2160],
        },
        "telemetry": {"memory": memory, "hardware": hardware},
        "validation": {"output": validation, "extended_open_matte": render["extension_contract"], "quality": {"pass": quality_pass, "metrics": render["quality"]}, "parity": parity},
        "comparison": {"phase1_total_fps_10_frames": PHASE1_TOTAL_FPS_BASELINE, "transport_only_fps": TRANSPORT_ONLY_FPS_BASELINE, "phase2_total_fps": frame_count / max(elapsed, 1e-9), "phase2_vs_phase1_total_ratio": (frame_count / max(elapsed, 1e-9)) / PHASE1_TOTAL_FPS_BASELINE, "phase2_vs_transport_ratio": (frame_count / max(elapsed, 1e-9)) / TRANSPORT_ONLY_FPS_BASELINE},
        "metadata": {"command": metadata_command, "log": metadata_log, "scope": "metadata-only Matroska remux; encoded packets copied without pixel decode/transfer"},
        "guarantees": {"nvdec_cuda_frames": True, "hdr_uses_hevc_cuvid": args.hdr_decoder == "hevc_cuvid", "open_matte_uses_h264_cuvid": args.om_decoder == "h264_cuvid", "gpu_working_rgb": True, "gpu_p010": True, "decoded_frame_d2h_bytes": 0, "render_output_d2h_bytes": 0, "h2d_decoded_video_bytes": 0, "zero_copy_nvenc": True, "frozen_v4_fit": True, "frozen_v4_render_equations": True, "test_frame_gate": frame_count},
        "limitations": {"quality_parity": parity["reason"], "telemetry": "nvidia-smi is sampled at bounded 0.75s intervals; short GPU bursts may be missed", "encoded_bitstream": "NVENC bitstream retrieval to Matroska remains host-side and is excluded from decoded/render D2H counters"},
    }
    v05.write_json(report, payload)
    print(f"fit {fit['fit_seconds']:.3f}s; render {render['wall_seconds']:.3f}s ({render['fps']:.3f} fps); total {elapsed:.3f}s")
    print(f"output: {output}")
    print(f"report: {report}")
    print(json.dumps({"status": payload["status"], "frame_count": render["frames"], "zero_copy": render["output_boundary"]["zero_copy_nvenc"], "d2h_bytes": render["output_boundary"]["d2h_bytes"], "h2d_bytes": render["output_boundary"]["h2d_bytes"], "hdr10": output_contract["hdr10_metadata_matches_source"], "extended_open_matte": output_contract["extended_open_matte"], "quality_pass": quality_pass, "parity": parity["status"]}, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-v05-gpu-native", description="V5 native Phase 1 TEST 10 or explicit Phase 2 GPU-P010/NVENC test")
    parser.add_argument("--phase2-zero-copy", action="store_true", help="Run the integrated native V5 Phase 2 GPU-P010/NVENC pipeline")
    parser.add_argument("--diagnostic-test10", action="store_true", help="Run Phase 2 GPU-P010/NVENC with exactly 10 frames for first-failure diagnosis")
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--pre-metadata-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--metadata-remuxer", type=Path, default=DEFAULT_METADATA_REMUXER)
    parser.add_argument("--max-frames", type=int, default=None)
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
    phase2 = bool(args.phase2_zero_copy)
    if args.profile is None:
        args.profile = DEFAULT_PHASE2_PROFILE if phase2 else DEFAULT_PROFILE
    if args.output is None:
        args.output = DEFAULT_PHASE2_OUTPUT if phase2 else DEFAULT_OUTPUT
    expected = TEST_FRAME_COUNT if phase2 and bool(args.diagnostic_test10) else (PHASE2_FRAME_COUNT if phase2 else TEST_FRAME_COUNT)
    if args.report is None:
        args.report = DEFAULT_PHASE2_REPORT if phase2 else DEFAULT_REPORT
    if args.max_frames is None:
        args.max_frames = expected
    try:
        if args.fit_stride <= 0 or args.diagnostic_stride <= 0 or args.ring_size < 2 or args.device_index < 0 or not math.isfinite(args.fit_scale) or not (0 < args.fit_scale <= 1) or args.encoder_qp < 0 or args.encoder_qp > 51:
            raise v05.V05Error("invalid stride/ring/device/fit-scale/encoder-qp configuration")
        if phase2:
            _run_phase2(args)
        else:
            _run_phase1(args)
        return 0
    except BaseException as exc:
        if not any(event.get("marker") == "FIRST_FAILURE" for event in native.diagnostic_events()):
            native.diagnostic_exception("runner", exc)
        print(f"v05-native error: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        _write_diagnostic_failure_report(args, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
