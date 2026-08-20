"""Bounded native NVDEC -> CUDA frame -> NVENC feasibility probe.

This experiment intentionally stays outside the production package and does
not import CuPy or the P2.20 transform backend. It runs at most ten frames of
the real BR2049 HDR HEVC source through three FFmpeg-only paths:

1. hevc_cuvid -> hevc_nvenc
2. hevc_cuvid -> scale_cuda -> hevc_nvenc
3. hevc_cuvid -> CUDA format/range conversion -> p010 CUDA frame -> NVENC

The probe treats FFmpeg's CUDA frames as opaque device frames. It records
FFmpeg's benchmark_all stage counters, process wall time, verbose frame-format
messages, ffprobe metadata, and optional nvidia-smi memory snapshots. A CUDA
filter stage is not independently timed by FFmpeg 7.1.1's benchmark output; the
report therefore keeps that field explicit and only reports direct-vs-filter
wall-time deltas as comparative observations.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    Path(
        r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
    )
    / "Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"
)
DEFAULT_OUTPUT_DIR = ROOT / "P2.23_outputs"
DEFAULT_FRAMES = 10
GPU_INDEX = 0

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "variant_1_nvdec_nvenc",
        "label": "NVDEC -> CUDA frame -> NVENC",
        "filter": None,
        "output_name": "P2.23_variant_1_nvdec_nvenc_10f.mkv",
        "expected_width": 3840,
        "expected_height": 1600,
        "operation": "none; decoder CUDA frame is passed directly to NVENC",
    },
    {
        "id": "variant_2_nvdec_scale_cuda_nvenc",
        "label": "NVDEC -> scale_cuda -> NVENC",
        "filter": "scale_cuda=w=1920:h=800:format=p010le",
        "output_name": "P2.23_variant_2_nvdec_scale_cuda_nvenc_10f.mkv",
        "expected_width": 1920,
        "expected_height": 800,
        "operation": "CUDA resize to 1920x800 and p010le CUDA frame",
    },
    {
        "id": "variant_3_nvdec_colorspace_cuda_p010_nvenc",
        "label": "NVDEC -> CUDA colorspace/range -> p010 CUDA -> NVENC",
        "filter": (
            "scale_cuda=format=nv12,"
            "colorspace_cuda=range=tv,"
            "scale_cuda=format=p010le"
        ),
        "output_name": "P2.23_variant_3_nvdec_colorspace_cuda_p010_nvenc_10f.mkv",
        "expected_width": 3840,
        "expected_height": 1600,
        "operation": (
            "CUDA p010->nv12, colorspace_cuda range=tv, then nv12->p010; "
            "this build does not accept p010 directly in colorspace_cuda"
        ),
    },
)

BENCHMARK_RE = re.compile(
    r"bench:\s+(?P<user>\d+)\s+user\s+"
    r"(?P<sys>\d+)\s+sys\s+(?P<real>\d+)\s+real\s+"
    r"(?P<stage>\S+)"
)


def command_text(command: list[str]) -> str:
    """Render a Windows command line without executing a shell."""
    return subprocess.list2cmdline(command)


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} is not available on PATH")
    return path


def run_command(
    command: list[str],
    *,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded command and capture combined stdout/stderr."""
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def ffmpeg_capabilities(ffmpeg: str) -> dict[str, str]:
    """Capture the requested local FFmpeg capability outputs."""
    requests = {
        "version": [ffmpeg, "-version"],
        "hwaccels": [ffmpeg, "-hwaccels"],
        "cuvid_decoders": [ffmpeg, "-decoders"],
        "nvenc_encoders": [ffmpeg, "-encoders"],
        "cuda_filters": [ffmpeg, "-filters"],
        "scale_cuda_help": [ffmpeg, "-h", "filter=scale_cuda"],
        "colorspace_cuda_help": [ffmpeg, "-h", "filter=colorspace_cuda"],
        "hwupload_cuda_help": [ffmpeg, "-h", "filter=hwupload_cuda"],
        "hwdownload_help": [ffmpeg, "-h", "filter=hwdownload"],
        "h264_cuvid_help": [ffmpeg, "-h", "decoder=h264_cuvid"],
        "hevc_cuvid_help": [ffmpeg, "-h", "decoder=hevc_cuvid"],
        "hevc_nvenc_help": [ffmpeg, "-h", "encoder=hevc_nvenc"],
    }
    captured: dict[str, str] = {}
    for key, command in requests.items():
        result = run_command(command, timeout=120.0)
        captured[key] = result.stdout
    return captured


def ffprobe_json(ffprobe: str, path: Path) -> dict[str, Any]:
    """Read the first video stream and counted frames from an output."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,profile,pix_fmt,color_range,color_primaries,"
            "color_transfer,color_space,width,height,nb_read_frames"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command, timeout=120.0)
    if result.returncode != 0:
        return {
            "returncode": result.returncode,
            "error": result.stdout[-4000:],
        }
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "returncode": result.returncode,
            "error": f"Invalid ffprobe JSON: {exc}",
            "raw": result.stdout[-4000:],
        }
    parsed["returncode"] = result.returncode
    return parsed


def nvidia_snapshot() -> dict[str, Any]:
    """Capture coarse GPU identity and memory counters when nvidia-smi exists."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {"available": False, "reason": "nvidia-smi not available on PATH"}
    command = [
        smi,
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(command, timeout=30.0)
    if result.returncode != 0:
        return {
            "available": False,
            "returncode": result.returncode,
            "error": result.stdout[-1000:],
        }
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {
        "available": True,
        "returncode": result.returncode,
        "raw": rows,
    }


def input_probe(ffprobe: str, path: Path) -> dict[str, Any]:
    """Probe the source without decoding the full movie."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=index,codec_name,profile,width,height,pix_fmt,color_range,"
            "color_primaries,color_transfer,color_space,r_frame_rate,avg_frame_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command, timeout=120.0)
    if result.returncode != 0:
        raise RuntimeError(f"Input ffprobe failed:\n{result.stdout[-4000:]}")
    return json.loads(result.stdout)


def build_command(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    *,
    frames: int,
    filter_graph: str | None,
) -> list[str]:
    """Build one native device-frame FFmpeg command."""
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "verbose",
        "-nostats",
        "-nostdin",
        "-y",
        "-benchmark_all",
        "-hwaccel",
        "cuda",
        "-hwaccel_device",
        str(GPU_INDEX),
        "-hwaccel_output_format",
        "cuda",
        "-c:v",
        "hevc_cuvid",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frames),
    ]
    if filter_graph is not None:
        command.extend(["-vf", filter_graph])
    command.extend(
        [
            "-an",
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            "18",
            # Keep the AVFrame hardware format. p010le is the CUDA frame's
            # software format where scale_cuda selects it; using p010le here
            # would request a software download/format negotiation.
            "-pix_fmt",
            "cuda",
            "-color_range",
            "tv",
            "-color_primaries",
            "bt2020",
            "-color_trc",
            "smpte2084",
            "-colorspace",
            "bt2020nc",
            "-f",
            "matroska",
            str(output_path),
        ]
    )
    return command


def parse_benchmark(log: str) -> dict[str, Any]:
    """Aggregate FFmpeg -benchmark_all real-time counters by task."""
    stages: dict[str, list[int]] = {}
    for match in BENCHMARK_RE.finditer(log):
        stage = match.group("stage")
        stages.setdefault(stage, []).append(int(match.group("real")))
    return {
        stage: {
            "calls": len(values),
            "real_us_sum": sum(values),
            "real_ms_sum": sum(values) / 1000.0,
            "real_ms_samples": [value / 1000.0 for value in values],
        }
        for stage, values in stages.items()
    }


def classify_zero_copy(command: list[str], log: str) -> dict[str, Any]:
    """Classify whether the observed FFmpeg graph stays on CUDA frames."""
    command_lower = command_text(command).lower()
    log_lower = log.lower()
    hwdownload = "hwdownload" in command_lower or "hwdownload" in log_lower
    hwupload = "hwupload_cuda" in command_lower or "hwupload_cuda" in log_lower
    rawvideo = "-f rawvideo" in command_lower
    decoder_cuda = bool(
        re.search(r"formats:\s*original:\s*cuda\s*\|\s*hw:\s*cuda", log_lower)
    )
    graph_cuda = bool(re.search(r"pixfmt:cuda", log_lower))
    pass_value = (
        decoder_cuda
        and graph_cuda
        and not hwdownload
        and not hwupload
        and not rawvideo
    )
    return {
        "classification": "ZERO-COPY PASS" if pass_value else "NOT ZERO-COPY",
        "decoder_reports_cuda_frames": decoder_cuda,
        "filter_graph_reports_cuda_input": graph_cuda,
        "hwdownload_present": hwdownload,
        "hwupload_cuda_present": hwupload,
        "cpu_rawvideo_buffer_boundary": rawvideo,
        "reason": (
            "NVDEC reports CUDA frames, the graph input is cuda, and no "
            "hwdownload/hwupload/rawvideo boundary is present."
            if pass_value
            else "At least one CUDA continuity condition was not observed."
        ),
    }


def run_variant(
    ffmpeg: str,
    ffprobe: str,
    input_path: Path,
    output_dir: Path,
    variant: dict[str, Any],
    *,
    frames: int,
) -> dict[str, Any]:
    """Run one bounded variant and return all observations."""
    output_path = output_dir / str(variant["output_name"])
    output_path.unlink(missing_ok=True)
    command = build_command(
        ffmpeg,
        input_path,
        output_path,
        frames=frames,
        filter_graph=variant["filter"],
    )
    before_memory = nvidia_snapshot()
    started = time.perf_counter()
    try:
        result = run_command(command, timeout=600.0)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        result = subprocess.CompletedProcess(
            command,
            -9,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
        )
        timed_out = True
    total_wall_ms = (time.perf_counter() - started) * 1000.0
    after_memory = nvidia_snapshot()
    log = result.stdout
    probe = ffprobe_json(ffprobe, output_path) if output_path.exists() else {}
    stream = (probe.get("streams") or [{}])[0]
    benchmark = parse_benchmark(log)
    zero_copy = classify_zero_copy(command, log)
    return {
        "id": variant["id"],
        "label": variant["label"],
        "operation": variant["operation"],
        "filter_graph": variant["filter"],
        "input_path": str(input_path),
        "output_path": str(output_path),
        "frames_requested": frames,
        "returncode": result.returncode,
        "timed_out": timed_out,
        "output_exists": output_path.exists(),
        "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "total_wall_ms": total_wall_ms,
        "benchmark_all": benchmark,
        "stage_timing_interpretation": {
            "decode_nvdec_ms": benchmark.get("decode_video", {}).get("real_ms_sum"),
            "nvenc_encode_ms": benchmark.get("encode_video", {}).get("real_ms_sum"),
            "nvenc_flush_ms": benchmark.get("flush_video", {}).get("real_ms_sum"),
            "cuda_processing_ms": None,
            "cuda_processing_note": (
                "FFmpeg -benchmark_all exposes decoder/encoder tasks but not a "
                "separate CUDA filter task. Compare total wall time against "
                "variant 1 only; do not treat the delta as an isolated kernel time."
            ),
            "total_process_wall_ms": total_wall_ms,
        },
        "zero_copy": zero_copy,
        "gpu_memory_before": before_memory,
        "gpu_memory_after": after_memory,
        "ffprobe": probe,
        "output_stream": stream,
        "expected_dimensions": {
            "width": variant["expected_width"],
            "height": variant["expected_height"],
        },
        "log_markers": {
            "contains_hwdownload": "hwdownload" in log.lower(),
            "contains_hwupload_cuda": "hwupload_cuda" in log.lower(),
            "contains_cpu_rgb": "rgb" in log.lower(),
            "cuda_frame_markers": [
                line.strip()
                for line in log.splitlines()
                if "Formats: Original: cuda" in line or "pixfmt:cuda" in line
            ],
        },
        "log_tail": log[-6000:],
        "command": command_text(command),
    }


def validate_result(result: dict[str, Any], frames: int) -> list[str]:
    """Return explicit validation failures for one output."""
    failures: list[str] = []
    if result["returncode"] != 0:
        failures.append(f"returncode={result['returncode']}")
    if result["timed_out"]:
        failures.append("timed out")
    if not result["output_exists"] or result["output_bytes"] <= 0:
        failures.append("missing/empty output")
    if result["zero_copy"]["classification"] != "ZERO-COPY PASS":
        failures.append("zero-copy classification failed")
    stream = result["output_stream"]
    required = {
        "codec_name": "hevc",
        "profile": "Main 10",
        "pix_fmt": "yuv420p10le",
        "color_range": "tv",
        "color_primaries": "bt2020",
        "color_transfer": "smpte2084",
        "color_space": "bt2020nc",
    }
    for key, expected in required.items():
        if stream.get(key) != expected:
            failures.append(f"ffprobe {key}={stream.get(key)!r}, expected {expected!r}")
    expected_dimensions = result["expected_dimensions"]
    for key in ("width", "height"):
        expected = expected_dimensions[key]
        if stream.get(key) != expected:
            failures.append(
                f"ffprobe {key}={stream.get(key)!r}, expected {expected!r}"
            )
    try:
        output_frames = int(stream.get("nb_read_frames"))
    except (TypeError, ValueError):
        output_frames = None
    if output_frames != frames:
        failures.append(f"ffprobe frames={output_frames}, expected {frames}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--keep-log-tail",
        action="store_true",
        help="Keep per-variant log tails in JSON; useful for debugging failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.frames <= 10:
        raise SystemExit("--frames must be between 1 and 10")
    input_path = args.input.resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = executable("ffmpeg")
    ffprobe = executable("ffprobe")

    source = input_probe(ffprobe, input_path)
    capabilities = ffmpeg_capabilities(ffmpeg)
    results: dict[str, Any] = {
        "experiment": {
            "name": "P2.23 native CUDA video pipeline feasibility",
            "frames_requested": args.frames,
            "frames_limit": 10,
            "full_film_rendered": False,
            "cupy_used": False,
            "custom_transform_used": False,
            "production_code_modified": False,
            "gpu_index": GPU_INDEX,
            "architecture_target": (
                "NVDEC -> CUDA surface/frame -> CUDA operation -> p010 CUDA frame -> NVENC"
            ),
        },
        "environment": {
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
            "source": str(input_path),
            "source_probe": source,
            "capabilities": capabilities,
            "nvidia_smi_before_all_variants": nvidia_snapshot(),
        },
        "variants": {},
    }

    for variant in VARIANTS:
        print(f"Running {variant['id']} ({args.frames} frames)...", flush=True)
        variant_result = run_variant(
            ffmpeg,
            ffprobe,
            input_path,
            output_dir,
            variant,
            frames=args.frames,
        )
        if not args.keep_log_tail:
            variant_result.pop("log_tail", None)
        failures = validate_result(variant_result, args.frames)
        variant_result["validation_failures"] = failures
        variant_result["status"] = "PASS" if not failures else "FAIL"
        results["variants"][variant["id"]] = variant_result
        print(
            f"  status={variant_result['status']} "
            f"returncode={variant_result['returncode']} "
            f"wall_ms={variant_result['total_wall_ms']:.1f} "
            f"zero_copy={variant_result['zero_copy']['classification']}",
            flush=True,
        )

    direct = results["variants"]["variant_1_nvdec_nvenc"]
    for variant_id, variant_result in results["variants"].items():
        variant_result["comparison_to_direct"] = {
            "wall_delta_ms": (
                variant_result["total_wall_ms"] - direct["total_wall_ms"]
            ),
            "wall_ratio": (
                variant_result["total_wall_ms"] / direct["total_wall_ms"]
                if direct["total_wall_ms"] > 0
                else None
            ),
            "note": (
                "Comparative process wall time only; not an isolated CUDA kernel timing."
            ),
        }

    statuses = [variant["status"] for variant in results["variants"].values()]
    zero_copy_passes = [
        variant["zero_copy"]["classification"] == "ZERO-COPY PASS"
        for variant in results["variants"].values()
    ]
    if all(status == "PASS" for status in statuses) and all(zero_copy_passes):
        overall = "PASS"
    elif any(
        status == "PASS" and zero_copy
        for status, zero_copy in zip(statuses, zero_copy_passes)
    ):
        overall = "PARTIAL"
    else:
        overall = "FAIL"
    results["decision"] = {
        "overall": overall,
        "native_nvdec_cuda_nvenc_available": bool(
            results["variants"]["variant_1_nvdec_nvenc"]["status"] == "PASS"
            and results["variants"]["variant_1_nvdec_nvenc"]["zero_copy"][
                "classification"
            ]
            == "ZERO-COPY PASS"
        ),
        "all_variants_zero_copy": all(zero_copy_passes),
        "next_step": (
            "Design a native AVHWFramesContext/CUDA-frame custom filter or C++/CUDA "
            "extension; do not modify production until ownership and synchronization "
            "are specified."
        ),
    }
    results["environment"]["nvidia_smi_after_all_variants"] = nvidia_snapshot()
    results_path = ROOT / "P2.23_native_cuda_probe_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(results["decision"], indent=2), flush=True)
    print(f"Results: {results_path}", flush=True)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
