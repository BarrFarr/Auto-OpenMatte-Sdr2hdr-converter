"""Generate 30-second test output for BR2049 using FEAT-007 luminance mapping.

This uses the existing composition pipeline to render HDR center + transformed
OM extension for the 45:00-45:30 segment. Due to the current pipeline lacking
a full FFmpeg encode loop, we implement a minimal frame-by-frame render here.

Output: 3840x2160 HDR PQ 10-bit, 30 seconds.
"""
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

import numpy as np

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, ShotTransform, SyncModel, SyncStatus
from auto_openmatte.processing.luminance import estimate_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.pipeline.compose import composite_extend

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_DIR = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output")
OUTPUT_PATH = OUTPUT_DIR / "BR2049_FEAT007_45m_30s.mkv"


def main():
    print("=" * 70)
    print("30-SECOND HDR OUTPUT GENERATION — Blade Runner 2049 45:00–45:30")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Inspect sources
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)

    fps = hdr_source.selected_stream.fps
    hdr_w = hdr_source.selected_stream.width
    hdr_h = hdr_source.selected_stream.height
    om_w = om_source.selected_stream.width
    om_h = om_source.selected_stream.height

    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)

    start_seconds = 2700.0  # 45:00
    end_seconds = 2730.0    # 45:30
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))
    n_frames = end_frame - start_frame

    print(f"  HDR: {hdr_w}x{hdr_h}")
    print(f"  OM:  {om_w}x{om_h}")
    print(f"  Output: {om_w}x{om_h} (full OM resolution)")
    print(f"  Frames: {n_frames} ({n_frames/fps:.1f}s)")
    print(f"  Sync offset: {sync.frame_offset}")

    # Step 1: Estimate luminance curve from this segment
    print("\n--- Step 1: Estimating luminance curve ---")
    shot = Shot(shot_id=0, hdr_start_frame=start_frame, hdr_end_frame=end_frame,
                om_start_frame=start_frame+1167, om_end_frame=end_frame+1167,
                duration_frames=n_frames)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256)

    samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                       config=config, n_samples=30, proxy_width=960)
    train, _ = train_validation_split(samples, train_ratio=0.8, seed=42)
    curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=config)
    print(f"  Curve points: {len(curve)}, samples: {samples.n_valid_pairs:,}")

    # Build ShotTransform
    transform = ShotTransform(
        shot_id=0,
        luminance_curve=curve,
        exposure=1.0,
        contrast=1.0,
        saturation=1.0,
        confidence=0.97,
    )

    # Generate extension mask
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)
    print(f"  Extension mask: {ext_mask.shape}, HDR region rows 280-1880")

    # Step 2: Render via FFmpeg pipe
    # Decode both sources simultaneously, composite frame-by-frame, encode output
    print("\n--- Step 2: Rendering (frame-by-frame FFmpeg pipe) ---")
    print(f"  This will process {n_frames} frames at {om_w}x{om_h}...")

    hdr_stream_idx = hdr_source.selected_stream.index
    om_stream_idx = om_source.selected_stream.index

    # HDR decode command (output as float RGB via rgb48le)
    hdr_cmd = [
        "ffmpeg", "-v", "quiet", "-nostdin",
        "-ss", f"{start_seconds:.6f}",
        "-i", str(HDR_PATH),
        "-map", f"0:v:{hdr_stream_idx}",
        "-frames:v", str(n_frames),
        "-pix_fmt", "rgb48le",
        "-f", "rawvideo",
        "pipe:1",
    ]

    # OM decode command
    om_start_seconds = (start_frame + 1167) / fps
    om_cmd = [
        "ffmpeg", "-v", "quiet", "-nostdin",
        "-ss", f"{om_start_seconds:.6f}",
        "-i", str(OM_PATH),
        "-map", f"0:v:{om_stream_idx}",
        "-frames:v", str(n_frames),
        "-pix_fmt", "rgb48le",
        "-f", "rawvideo",
        "pipe:1",
    ]

    # Encode command (HDR PQ output)
    enc_cmd = [
        "ffmpeg", "-v", "quiet", "-nostdin", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb48le",
        "-s", f"{om_w}x{om_h}",
        "-r", "24000/1001",
        "-i", "pipe:0",
        "-c:v", "libx265",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-x265-params", "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50):max-cll=457,179",
        str(OUTPUT_PATH),
    ]

    t0 = time.perf_counter()

    # Start decode processes
    hdr_proc = subprocess.Popen(hdr_cmd, stdout=subprocess.PIPE)
    om_proc = subprocess.Popen(om_cmd, stdout=subprocess.PIPE)
    enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    hdr_frame_bytes = hdr_w * hdr_h * 3 * 2  # rgb48le = 6 bytes/pixel
    om_frame_bytes = om_w * om_h * 3 * 2

    frames_written = 0
    try:
        for frame_idx in range(n_frames):
            # Read HDR frame
            hdr_raw = hdr_proc.stdout.read(hdr_frame_bytes)
            om_raw = om_proc.stdout.read(om_frame_bytes)

            if len(hdr_raw) < hdr_frame_bytes or len(om_raw) < om_frame_bytes:
                print(f"  Frame {frame_idx}: decode ended early")
                break

            # Convert to float [0,1]
            hdr_arr = np.frombuffer(hdr_raw, dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
            om_arr = np.frombuffer(om_raw, dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0

            # Composite
            output = composite_extend(
                hdr_arr, om_arr, transform, geom, ext_mask,
                sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=10000.0,
            )

            # Convert back to uint16 for encoding
            output_u16 = (np.clip(output, 0.0, 1.0) * 65535.0).astype(np.uint16)
            enc_proc.stdin.write(output_u16.tobytes())
            frames_written += 1

            if frame_idx % 50 == 0:
                elapsed = time.perf_counter() - t0
                fps_actual = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  Frame {frame_idx}/{n_frames} ({fps_actual:.1f} fps)")

    except Exception as e:
        print(f"  ERROR at frame {frames_written}: {e}")
    finally:
        hdr_proc.stdout.close()
        om_proc.stdout.close()
        enc_proc.stdin.close()
        hdr_proc.wait()
        om_proc.wait()
        enc_proc.wait()

    t1 = time.perf_counter()
    print(f"\n  Frames written: {frames_written}/{n_frames}")
    print(f"  Render time: {t1-t0:.1f}s ({frames_written/(t1-t0):.1f} fps)")
    print(f"  Output: {OUTPUT_PATH}")

    # Verify output
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / (1024*1024)
        print(f"  File size: {size_mb:.1f} MB")
        print("\n--- FFprobe verification ---")
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", "-select_streams", "v",
             str(OUTPUT_PATH)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            stream = info.get("streams", [{}])[0]
            fmt = info.get("format", {})
            print(f"  Resolution:  {stream.get('width')}x{stream.get('height')}")
            print(f"  Duration:    {fmt.get('duration', 'N/A')}s")
            print(f"  FPS:         {stream.get('r_frame_rate')}")
            print(f"  Codec:       {stream.get('codec_name')}")
            print(f"  Pix fmt:     {stream.get('pix_fmt')}")
            print(f"  Color range: {stream.get('color_range')}")
            print(f"  Primaries:   {stream.get('color_primaries')}")
            print(f"  Transfer:    {stream.get('color_transfer')}")
            print(f"  Matrix:      {stream.get('color_space')}")
            print(f"  Bit depth:   {stream.get('bits_per_raw_sample', 'N/A')}")
    else:
        print("  ERROR: Output file not created!")

    print("\n" + "=" * 70)
    print("RENDER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
