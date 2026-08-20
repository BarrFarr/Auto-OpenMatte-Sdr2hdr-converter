"""P2.22 bounded GPU-transform/NVENC streaming probe.

This is an isolated experiment. It does not modify the production renderer or
make the GPU backend the default. It processes at most ten decoded frames.

The measured B path is intentionally explicit:

    FFmpeg decode -> host RGB48 -> host float64 ROI -> CuPy float64 transform
    -> CuPy asnumpy -> host RGB48 pipe -> FFmpeg RGB conversion/NVENC

The current Python/CuPy API has no way to attach a CuPy allocation to an
FFmpeg AVFrame/AVHWFramesContext. The C section therefore probes FFmpeg's
host-upload CUDA path separately, but never labels that path as zero-copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, BinaryIO

import numpy as np

from auto_openmatte.core.models import GeometryModel, ShotTransform
from auto_openmatte.core.transfer_functions import pq_eotf
from auto_openmatte.pipeline.compose import composite_extend
from auto_openmatte.processing.transform_backend import (
    BackendUnavailableError,
    CPUTransformBackend,
    GPUTransformBackend,
)

ROOT = Path(__file__).resolve().parent
CURVE_FILE = ROOT / "luminance_curve_br2049.json"
DEFAULT_OUTPUT_DIR = ROOT / "P2.22_outputs"
DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
DEFAULT_ROI_Y = 1600
DEFAULT_ROI_HEIGHT = 560
DEFAULT_ROI_X = 0
DEFAULT_ROI_WIDTH = 3840
DEFAULT_FRAMES = 10
FPS = "24000/1001"
PEAK_NITS = 10000.0
LUM_WEIGHTS = np.array([0.2627, 0.6780, 0.0593], dtype=np.float64)
JACKET_Y = (300, 450)
JACKET_X = (1500, 2500)


def stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    ordered = [float(value) for value in values]
    med = float(median(ordered))
    return {
        "count": len(ordered),
        "median_ms": med,
        "mean_ms": float(mean(ordered)),
        "min_ms": float(min(ordered)),
        "max_ms": float(max(ordered)),
        "p10_ms": float(np.percentile(ordered, 10)),
        "p90_ms": float(np.percentile(ordered, 90)),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "fps_from_median": float(1000.0 / med) if med > 0.0 else 0.0,
        "samples_ms": ordered,
    }


def run_checked(
    command: list[str],
    *,
    input_data: bytes | None = None,
    timeout: float = 180.0,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        command,
        input=input_data,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg is not available on PATH")
    return path


def ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise RuntimeError("ffprobe is not available on PATH")
    return path


def make_transform() -> ShotTransform:
    curve_data = json.loads(CURVE_FILE.read_text(encoding="utf-8"))
    return ShotTransform(shot_id=2220, luminance_curve=curve_data["curve"])


def generate_input(
    path: Path,
    *,
    width: int,
    height: int,
    frames: int,
) -> dict[str, Any]:
    """Create a bounded lossless synthetic source for the decode stage."""
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={FPS}",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    started = time.perf_counter()
    result = run_checked(command, timeout=300.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(
            "Could not create bounded probe input: "
            f"{result.stderr.decode(errors='replace')[-1000:]}"
        )
    return {
        "path": str(path),
        "command": command_text(command),
        "frames": frames,
        "width": width,
        "height": height,
        "generation_wall_ms": elapsed_ms,
        "bytes": path.stat().st_size,
    }


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def start_decoder(path: Path, frames: int) -> subprocess.Popen[bytes]:
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frames),
        "-pix_fmt",
        "rgb48le",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def start_encoder(
    path: Path,
    *,
    width: int,
    height: int,
    frames: int,
) -> tuple[subprocess.Popen[bytes], list[str]]:
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{width}x{height}",
        "-r",
        FPS,
        "-i",
        "pipe:0",
        "-frames:v",
        str(frames),
        "-vf",
        (
            "format=p010le,"
            "setparams=color_primaries=bt2020:"
            "color_trc=smpte2084:colorspace=bt2020nc:range=tv"
        ),
        "-an",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        "p5",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-pix_fmt",
        "p010le",
        "-color_range",
        "tv",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return process, command


def finish_process(process: subprocess.Popen[bytes]) -> tuple[int, str]:
    if process.stdin is not None and not process.stdin.closed:
        process.stdin.close()
    try:
        process.wait(timeout=180.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        return -9, "process timeout"
    stderr = b""
    if process.stderr is not None:
        stderr = process.stderr.read()
        process.stderr.close()
    return process.returncode, stderr.decode(errors="replace")[-2000:]


def frame_to_float(raw: bytes, width: int, height: int) -> np.ndarray:
    expected = width * height * 3 * 2
    if len(raw) != expected:
        raise RuntimeError(f"Expected {expected} decoded bytes, got {len(raw)}")
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3)
    return u16.astype(np.float64) / 65535.0


def pack_rgb48(frame: np.ndarray) -> bytes:
    values = np.rint(np.clip(frame, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return values.tobytes()


def luminance_nits(signal: np.ndarray) -> np.ndarray:
    signal_y = np.sum(signal * LUM_WEIGHTS, axis=-1)
    return pq_eotf(signal_y)


def cuda_stage(
    cupy: Any,
    action: Any,
) -> tuple[float, float, Any]:
    """Return CUDA event time, host elapsed/wait time, and action result."""
    host_started = time.perf_counter()
    start = cupy.cuda.Event()
    end = cupy.cuda.Event()
    start.record()
    result = action()
    end.record()
    end.synchronize()
    host_ms = (time.perf_counter() - host_started) * 1000.0
    event_ms = float(cupy.cuda.get_elapsed_time(start, end))
    return event_ms, host_ms, result


def prepare_gpu(
    transform: ShotTransform,
    *,
    roi_shape: tuple[int, int, int],
) -> tuple[GPUTransformBackend, Any, dict[str, Any]]:
    backend = GPUTransformBackend()
    workspace = backend.prepare_shot(transform, peak_nits=PEAK_NITS)
    workspace.metadata["transform"] = transform
    memory_before = backend.memory_report(workspace)
    backend._get_buffers(workspace, roi_shape)
    workspace.cupy.cuda.Stream.null.synchronize()
    memory_after = backend.memory_report(workspace)
    return backend, workspace, {
        "memory_before_buffers": memory_before,
        "memory_after_buffers": memory_after,
        "workspace_allocations": workspace.metadata["workspace_allocations"],
    }


def run_variant(
    label: str,
    input_path: Path,
    output_path: Path,
    *,
    frames: int,
    width: int,
    height: int,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    transform: ShotTransform,
    cpu_backend: CPUTransformBackend,
    cpu_workspace: Any,
    gpu_backend: GPUTransformBackend | None = None,
    gpu_workspace: Any = None,
) -> dict[str, Any]:
    """Run one bounded streaming variant and return detailed timings."""
    decode_bytes = width * height * 3 * 2
    decoder = start_decoder(input_path, frames)
    encoder, encoder_command = start_encoder(
        output_path,
        width=roi_width,
        height=roi_height,
        frames=frames,
    )
    stages: dict[str, list[float]] = {
        "decode_read": [],
        "host_rgb16_to_float": [],
        "roi_preparation": [],
        "cpu_transform": [],
        "gpu_upload_event": [],
        "gpu_upload_sync": [],
        "gpu_transform_event": [],
        "gpu_transform_sync": [],
        "gpu_download_event": [],
        "gpu_download_sync": [],
        "rgb_pack": [],
        "cpu_to_nvenc_pipe_write": [],
        "frame_wall": [],
    }
    first_packed: bytes | None = None
    frames_done = 0
    loop_started = time.perf_counter()
    decoder_error = ""
    encoder_write_error = ""

    try:
        for index in range(frames):
            frame_started = time.perf_counter()
            read_started = time.perf_counter()
            raw = read_exact(decoder.stdout, decode_bytes)  # type: ignore[arg-type]
            stages["decode_read"].append(
                (time.perf_counter() - read_started) * 1000.0
            )
            if len(raw) != decode_bytes:
                break

            conversion_started = time.perf_counter()
            full_frame = frame_to_float(raw, width, height)
            stages["host_rgb16_to_float"].append(
                (time.perf_counter() - conversion_started) * 1000.0
            )
            roi_started = time.perf_counter()
            roi_frame = full_frame[
                roi_y : roi_y + roi_height,
                roi_x : roi_x + roi_width,
                :,
            ]
            stages["roi_preparation"].append(
                (time.perf_counter() - roi_started) * 1000.0
            )

            if label == "A_cpu_transform":
                transform_started = time.perf_counter()
                output = cpu_backend.transform_roi(
                    roi_frame,
                    transform,
                    workspace=cpu_workspace,
                    peak_nits=PEAK_NITS,
                )
                stages["cpu_transform"].append(
                    (time.perf_counter() - transform_started) * 1000.0
                )
            elif label == "B_gpu_transform":
                if gpu_backend is None or gpu_workspace is None:
                    raise RuntimeError("GPU state is unavailable for variant B")
                cupy = gpu_workspace.cupy
                sync_started = time.perf_counter()
                cupy.cuda.Stream.null.synchronize()
                stages["gpu_upload_sync"].append(
                    (time.perf_counter() - sync_started) * 1000.0
                )
                upload_event, upload_host, buffers = cuda_stage(
                    cupy,
                    lambda: gpu_backend._upload_to_device(
                        roi_frame, gpu_workspace
                    ),
                )
                stages["gpu_upload_event"].append(upload_event)
                stages["gpu_upload_sync"].append(upload_host)
                transform_event, transform_host, _ = cuda_stage(
                    cupy,
                    lambda: gpu_backend._transform_device(
                        gpu_workspace, buffers
                    ),
                )
                stages["gpu_transform_event"].append(transform_event)
                stages["gpu_transform_sync"].append(transform_host)
                download_event, download_host, output = cuda_stage(
                    cupy,
                    lambda: gpu_backend._download_to_host(
                        gpu_workspace, buffers
                    ),
                )
                stages["gpu_download_event"].append(download_event)
                stages["gpu_download_sync"].append(download_host)
                cupy.cuda.Stream.null.synchronize()
            else:
                raise ValueError(f"Unsupported variant {label}")

            pack_started = time.perf_counter()
            packed = pack_rgb48(output)
            stages["rgb_pack"].append(
                (time.perf_counter() - pack_started) * 1000.0
            )
            if first_packed is None:
                first_packed = packed
            write_started = time.perf_counter()
            try:
                if encoder.stdin is None:
                    raise RuntimeError("NVENC stdin is unavailable")
                encoder.stdin.write(packed)
                encoder.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                encoder_write_error = str(exc)
                break
            stages["cpu_to_nvenc_pipe_write"].append(
                (time.perf_counter() - write_started) * 1000.0
            )
            stages["frame_wall"].append(
                (time.perf_counter() - frame_started) * 1000.0
            )
            frames_done += 1
            del raw, full_frame, output, packed
    finally:
        if decoder.stdout is not None:
            decoder.stdout.close()
        decoder_returncode, decoder_error = finish_process(decoder)

    finalize_started = time.perf_counter()
    encoder_returncode, encoder_error = finish_process(encoder)
    encoder_finalize_ms = (time.perf_counter() - finalize_started) * 1000.0
    loop_ms = (time.perf_counter() - loop_started) * 1000.0
    pipeline_ms = loop_ms + encoder_finalize_ms

    stage_stats = {name: stats(values) for name, values in stages.items()}
    result: dict[str, Any] = {
        "label": label,
        "frames_requested": frames,
        "frames_completed": frames_done,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "encoder_command": command_text(encoder_command),
        "decoder_returncode": decoder_returncode,
        "decoder_error": decoder_error,
        "encoder_returncode": encoder_returncode,
        "encoder_error": encoder_error,
        "encoder_write_error": encoder_write_error,
        "stages_ms": stage_stats,
        "encoder_finalize_wait_ms": encoder_finalize_ms,
        "loop_wall_ms": loop_ms,
        "total_pipeline_ms": pipeline_ms,
        "total_per_frame_ms": pipeline_ms / frames_done if frames_done else None,
        "pipeline_fps": 1000.0 * frames_done / pipeline_ms
        if pipeline_ms > 0.0
        else 0.0,
        "copy_audit": {
            "decoded_full_frame_bytes": decode_bytes,
            "host_rgb16_to_float64_copy": True,
            "roi_view_copy_bytes": 0,
            "rgb48_output_bytes_per_frame": roi_width * roi_height * 3 * 2,
            "gpu_to_cpu_output_bytes_per_frame": (
                roi_width * roi_height * 3 * 8
                if label == "B_gpu_transform"
                else 0
            ),
            "implicit_ffmpeg_encoder_input_upload": True,
            "direct_cupy_to_avframe": False,
        },
        "rgb_to_yuv_conversion": {
            "status": "not_separately_observable_at_current_pipe_boundary",
            "included_in": [
                "cpu_to_nvenc_pipe_write",
                "encoder_finalize_wait_ms",
            ],
            "note": (
                "The raw RGB48 pipe enters FFmpeg; its RGB-to-YUV conversion "
                "is internal to the FFmpeg filter/encoder graph."
            ),
        },
        "first_packed_frame_available": first_packed is not None,
    }
    if label == "B_gpu_transform" and gpu_workspace is not None:
        result["memory_after_variant"] = gpu_backend.memory_report(gpu_workspace)
        result["workspace_allocations_after_variant"] = gpu_workspace.metadata[
            "workspace_allocations"
        ]
    return result


def run_correctness(
    input_path: Path,
    *,
    frames: int,
    width: int,
    height: int,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    transform: ShotTransform,
    cpu_backend: CPUTransformBackend,
    cpu_workspace: Any,
    gpu_backend: GPUTransformBackend,
    gpu_workspace: Any,
) -> dict[str, Any]:
    """Compare ten CPU/GPU frames and run bounded artifact checks."""
    decoder = start_decoder(input_path, frames)
    expected_bytes = width * height * 3 * 2
    rgb_max = 0.0
    rgb_sum = 0.0
    rgb_count = 0
    lum_max = 0.0
    lum_sum = 0.0
    lum_count = 0
    jacket_values: list[np.ndarray] = []
    frames_done = 0

    try:
        for _ in range(frames):
            raw = read_exact(decoder.stdout, expected_bytes)  # type: ignore[arg-type]
            if len(raw) != expected_bytes:
                break
            full_frame = frame_to_float(raw, width, height)
            roi = full_frame[
                roi_y : roi_y + roi_height,
                roi_x : roi_x + roi_width,
                :,
            ]
            cpu_output = cpu_backend.transform_roi(
                roi,
                transform,
                workspace=cpu_workspace,
                peak_nits=PEAK_NITS,
            )
            gpu_output = gpu_backend.transform_roi(
                roi,
                transform,
                workspace=gpu_workspace,
                peak_nits=PEAK_NITS,
            )
            rgb_diff = np.abs(cpu_output - gpu_output)
            cpu_lum = luminance_nits(cpu_output)
            gpu_lum = luminance_nits(gpu_output)
            lum_diff = np.abs(cpu_lum - gpu_lum)
            rgb_max = max(rgb_max, float(np.max(rgb_diff)))
            rgb_sum += float(np.sum(rgb_diff))
            rgb_count += int(rgb_diff.size)
            lum_max = max(lum_max, float(np.max(lum_diff)))
            lum_sum += float(np.sum(lum_diff))
            lum_count += int(lum_diff.size)
            y1, y2 = JACKET_Y
            x1, x2 = JACKET_X
            jacket_values.append(gpu_lum[y1:y2, x1:x2].ravel())
            frames_done += 1
    finally:
        if decoder.stdout is not None:
            decoder.stdout.close()
        decoder_returncode, decoder_error = finish_process(decoder)

    hdr = np.full((4, 64, 3), 0.73, dtype=np.float64)
    om = np.linspace(0.01, 0.8, 8 * 64 * 3, dtype=np.float64).reshape(
        8, 64, 3
    )
    geometry = GeometryModel(overlap_bbox=[0.0, 2.0, 64.0, 6.0])
    extension_mask = np.zeros((8, 64), dtype=np.float64)
    cpu_composite = composite_extend(
        hdr, om, transform, geometry, extension_mask
    )
    gpu_composite = composite_extend(
        hdr,
        om,
        transform,
        geometry,
        extension_mask,
        backend=gpu_backend,
        backend_workspace=gpu_workspace,
    )
    overlap_cpu_exact = bool(np.array_equal(cpu_composite[2:6], hdr))
    overlap_gpu_exact = bool(np.array_equal(gpu_composite[2:6], hdr))
    overlap_diff = float(np.max(np.abs(cpu_composite - gpu_composite)))

    dark_probe = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1e-8, 0.0, 0.0]],
            [[0.0, 1e-7, 0.0]],
            [[1e-6, 1e-6, 1e-6]],
        ],
        dtype=np.float64,
    )
    dark_output = cpu_backend.transform_roi(
        dark_probe,
        transform,
        workspace=cpu_workspace,
        peak_nits=PEAK_NITS,
    )
    dark_nits = luminance_nits(dark_output)
    jacket_all = np.concatenate(jacket_values) if jacket_values else np.array([])
    jacket_rounded = np.round(jacket_all, 3) if jacket_all.size else jacket_all
    if jacket_rounded.size:
        _, jacket_counts = np.unique(jacket_rounded, return_counts=True)
        plateau_fraction = float(np.max(jacket_counts) / jacket_rounded.size)
    else:
        plateau_fraction = None

    return {
        "frames_compared": frames_done,
        "decoder_returncode": decoder_returncode,
        "decoder_error": decoder_error,
        "max_rgb_difference": rgb_max,
        "mean_rgb_difference": rgb_sum / rgb_count if rgb_count else None,
        "max_luminance_difference_nits": lum_max,
        "mean_luminance_difference_nits": (
            lum_sum / lum_count if lum_count else None
        ),
        "acceptance": {
            "rgb_limit": 1e-5,
            "luminance_limit_nits": 0.01,
            "pass": rgb_max <= 1e-5 and lum_max <= 0.01,
        },
        "hdr_overlap_invariant": {
            "cpu_hdr_region_unchanged": overlap_cpu_exact,
            "gpu_hdr_region_unchanged": overlap_gpu_exact,
            "cpu_gpu_max_difference": overlap_diff,
            "pass": overlap_cpu_exact and overlap_gpu_exact and overlap_diff <= 1e-5,
        },
        "jacket_roi": {
            "source_coordinates": {
                "rows": [DEFAULT_ROI_Y + JACKET_Y[0], DEFAULT_ROI_Y + JACKET_Y[1]],
                "columns": list(JACKET_X),
            },
            "roi_coordinates": {
                "rows": list(JACKET_Y),
                "columns": list(JACKET_X),
            },
            "frames_aggregated": frames_done,
            "max_luminance_nits": (
                float(np.max(jacket_all)) if jacket_all.size else None
            ),
            "p50_luminance_nits": (
                float(np.percentile(jacket_all, 50)) if jacket_all.size else None
            ),
            "pixels_over_400_nits": (
                int(np.sum(jacket_all > 400.0)) if jacket_all.size else 0
            ),
            "plateau_fraction_rounded_0_001": plateau_fraction,
            "note": "Synthetic testsrc2 input; not a real jacket source audit.",
        },
        "white_dot_probe": {
            "inputs": [0.0, 1e-8, 1e-7, 1e-6],
            "max_luminance_nits": float(np.max(dark_nits)),
            "pass": bool(np.max(dark_nits) < 1.0),
        },
    }


def ffprobe_output(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    command = [
        ffprobe_path(),
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-count_frames",
        "-select_streams",
        "v:0",
        str(path),
    ]
    result = run_checked(command, timeout=60.0)
    if result.returncode != 0:
        return {
            "exists": True,
            "path": str(path),
            "returncode": result.returncode,
            "error": result.stderr.decode(errors="replace")[-2000:],
        }
    parsed = json.loads(result.stdout.decode(encoding="utf-8"))
    stream = parsed.get("streams", [{}])[0]
    wanted = (
        "codec_name",
        "profile",
        "pix_fmt",
        "color_range",
        "color_primaries",
        "color_transfer",
        "color_space",
        "width",
        "height",
        "nb_frames",
        "nb_read_frames",
    )
    return {
        "exists": True,
        "path": str(path),
        "returncode": result.returncode,
        "file_bytes": path.stat().st_size,
        "stream": {key: stream.get(key) for key in wanted},
        "format": parsed.get("format", {}),
    }


def probe_rgb_to_yuv(
    sample: bytes,
    *,
    width: int,
    height: int,
    frames: int,
) -> dict[str, Any]:
    """Measure bounded software RGB48-to-P010 conversion without encoding."""
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{width}x{height}",
        "-r",
        FPS,
        "-i",
        "pipe:0",
        "-frames:v",
        str(frames),
        "-vf",
        (
            "format=p010le,"
            "setparams=color_primaries=bt2020:"
            "color_trc=smpte2084:colorspace=bt2020nc:range=tv"
        ),
        "-f",
        "null",
        "-",
    ]
    started = time.perf_counter()
    result = run_checked(command, input_data=sample * frames, timeout=180.0)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "command": command_text(command),
        "frames": frames,
        "wall_ms": wall_ms,
        "per_frame_ms": wall_ms / frames if frames else None,
        "stderr": result.stderr.decode(errors="replace")[-2000:],
        "note": (
            "This is a separate bounded software conversion probe. The A/B "
            "main pipeline performs the same conversion inside its FFmpeg "
            "encoder graph, so this value is not added to total timing."
        ),
    }


def probe_cuda_ffmpeg_path(
    sample: bytes,
    *,
    width: int,
    height: int,
    output_path: Path,
) -> dict[str, Any]:
    """Probe FFmpeg host-RGB -> hwupload_cuda -> NVENC, not CuPy zero-copy."""
    command = [
        ffmpeg_path(),
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{width}x{height}",
        "-r",
        FPS,
        "-i",
        "pipe:0",
        "-frames:v",
        "1",
        "-vf",
        "format=gbrp16le,hwupload_cuda,scale_cuda=format=p010le",
        "-c:v",
        "hevc_nvenc",
        "-pix_fmt",
        "cuda",
        "-bsf:v",
        (
            "hevc_metadata=video_full_range_flag=0:"
            "colour_primaries=9:transfer_characteristics=16:"
            "matrix_coefficients=9"
        ),
        "-color_range",
        "tv",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        str(output_path),
    ]
    started = time.perf_counter()
    result = run_checked(command, input_data=sample, timeout=120.0)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "host_upload_cuda_path_succeeded"
        if result.returncode == 0
        else "host_upload_cuda_path_failed",
        "direct_cupy_to_avframe": False,
        "returncode": result.returncode,
        "command": command_text(command),
        "wall_ms": wall_ms,
        "stderr": result.stderr.decode(errors="replace")[-2000:],
        "ffprobe": ffprobe_output(output_path),
        "explanation": (
            "This proves only FFmpeg system-memory upload to CUDA/NVENC. "
            "The input still originates from a host pipe and is not a CuPy "
            "device pointer."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--keep-input", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.frames > 10:
        raise SystemExit("--frames must be between 1 and 10")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "P2.22_probe_input_10f.mkv"
    transform = make_transform()
    roi_shape = (DEFAULT_ROI_HEIGHT, DEFAULT_ROI_WIDTH, 3)
    cpu_backend = CPUTransformBackend()
    cpu_workspace = cpu_backend.prepare_shot(transform, peak_nits=PEAK_NITS)

    input_info = generate_input(
        input_path,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        frames=args.frames,
    )
    try:
        gpu_backend, gpu_workspace, gpu_init = prepare_gpu(
            transform,
            roi_shape=roi_shape,
        )
    except (BackendUnavailableError, RuntimeError) as exc:
        gpu_backend = None
        gpu_workspace = None
        gpu_init = {"status": "unavailable", "error": str(exc)}

    results: dict[str, Any] = {
        "experiment": {
            "name": "P2.22 GPU -> NVENC streaming prototype",
            "frames_requested": args.frames,
            "frames_limit": 10,
            "full_film_rendered": False,
            "dtype": "float64",
            "width": DEFAULT_WIDTH,
            "height": DEFAULT_HEIGHT,
            "roi": {
                "x": DEFAULT_ROI_X,
                "y": DEFAULT_ROI_Y,
                "width": DEFAULT_ROI_WIDTH,
                "height": DEFAULT_ROI_HEIGHT,
            },
            "fps": FPS,
            "transform_math_unchanged": True,
            "production_default_changed": False,
        },
        "environment": {
            "ffmpeg": ffmpeg_path(),
            "ffprobe": ffprobe_path(),
            "cupy_available": gpu_backend is not None,
        },
        "input": input_info,
        "gpu_initialization": gpu_init,
        "variants": {},
        "correctness": None,
        "variant_C": {
            "status": "not_direct_until_probe",
            "reason": (
                "No CuPy-to-AVFrame/AVHWFramesContext pointer interop exists "
                "in the current Python API or repository."
            ),
        },
    }

    output_a = output_dir / "P2.22_variant_A_cpu_nvenc_10f.mkv"
    output_b = output_dir / "P2.22_variant_B_gpu_download_nvenc_10f.mkv"
    variant_a = run_variant(
        "A_cpu_transform",
        input_path,
        output_a,
        frames=args.frames,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        roi_x=DEFAULT_ROI_X,
        roi_y=DEFAULT_ROI_Y,
        roi_width=DEFAULT_ROI_WIDTH,
        roi_height=DEFAULT_ROI_HEIGHT,
        transform=transform,
        cpu_backend=cpu_backend,
        cpu_workspace=cpu_workspace,
    )
    variant_a["ffprobe"] = ffprobe_output(output_a)
    results["variants"]["A"] = variant_a

    variant_b: dict[str, Any]
    if gpu_backend is None or gpu_workspace is None:
        variant_b = {
            "label": "B_gpu_transform",
            "status": "unavailable",
            "error": gpu_init.get("error", "GPU is unavailable"),
        }
    else:
        variant_b = run_variant(
            "B_gpu_transform",
            input_path,
            output_b,
            frames=args.frames,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            roi_x=DEFAULT_ROI_X,
            roi_y=DEFAULT_ROI_Y,
            roi_width=DEFAULT_ROI_WIDTH,
            roi_height=DEFAULT_ROI_HEIGHT,
            transform=transform,
            cpu_backend=cpu_backend,
            cpu_workspace=cpu_workspace,
            gpu_backend=gpu_backend,
            gpu_workspace=gpu_workspace,
        )
        variant_b["ffprobe"] = ffprobe_output(output_b)
    results["variants"]["B"] = variant_b

    if gpu_backend is not None and gpu_workspace is not None:
        results["correctness"] = run_correctness(
            input_path,
            frames=args.frames,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            roi_x=DEFAULT_ROI_X,
            roi_y=DEFAULT_ROI_Y,
            roi_width=DEFAULT_ROI_WIDTH,
            roi_height=DEFAULT_ROI_HEIGHT,
            transform=transform,
            cpu_backend=cpu_backend,
            cpu_workspace=cpu_workspace,
            gpu_backend=gpu_backend,
            gpu_workspace=gpu_workspace,
        )

    sample = None
    if variant_a.get("first_packed_frame_available"):
        # Recreate one bounded sample rather than retaining ten frames.
        decoder = start_decoder(input_path, 1)
        raw = read_exact(
            decoder.stdout, DEFAULT_WIDTH * DEFAULT_HEIGHT * 6  # type: ignore[arg-type]
        )
        if decoder.stdout is not None:
            decoder.stdout.close()
        finish_process(decoder)
        if len(raw) == DEFAULT_WIDTH * DEFAULT_HEIGHT * 6:
            sample_frame = frame_to_float(raw, DEFAULT_WIDTH, DEFAULT_HEIGHT)[
                DEFAULT_ROI_Y : DEFAULT_ROI_Y + DEFAULT_ROI_HEIGHT,
                DEFAULT_ROI_X : DEFAULT_ROI_X + DEFAULT_ROI_WIDTH,
                :,
            ]
            sample = pack_rgb48(
                cpu_backend.transform_roi(
                    sample_frame,
                    transform,
                    workspace=cpu_workspace,
                    peak_nits=PEAK_NITS,
                )
            )

    if sample is not None:
        rgb_yuv_probe = probe_rgb_to_yuv(
            sample,
            width=DEFAULT_ROI_WIDTH,
            height=DEFAULT_ROI_HEIGHT,
            frames=args.frames,
        )
        for key in ("A", "B"):
            if "rgb_to_yuv_conversion" in results["variants"].get(key, {}):
                results["variants"][key]["rgb_to_yuv_conversion"] = {
                    "status": "separate_bounded_probe",
                    "probe": rgb_yuv_probe,
                    "included_in_main_pipeline_total": False,
                }
        c_output = output_dir / "P2.22_variant_C_host_hwupload_probe.mkv"
        results["variant_C"] = probe_cuda_ffmpeg_path(
            sample,
            width=DEFAULT_ROI_WIDTH,
            height=DEFAULT_ROI_HEIGHT,
            output_path=c_output,
        )

    results["outputs"] = {
        "A": ffprobe_output(output_a),
        "B": ffprobe_output(output_b),
        "C": results["variant_C"].get("ffprobe"),
    }
    results_path = ROOT / "P2.22_gpu_nvenc_probe_results.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not args.keep_input:
        input_path.unlink(missing_ok=True)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
