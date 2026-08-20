"""P2.10.1: Quick NVENC test — 10 frames, streaming pipe, no precompute."""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, ShotTransform, SyncModel, SyncStatus
from auto_openmatte.processing.luminance import estimate_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance
from auto_openmatte.pipeline.compose import composite_extend

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_DIR = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output")
PEAK = 10000.0
N_FRAMES = 10


def main():
    print("=" * 70)
    print(f"P2.10.1: QUICK NVENC TEST ({N_FRAMES} frames, streaming pipe)")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps
    hdr_w, hdr_h = 3840, 1600
    om_w, om_h = 3840, 2160

    # Reuse curve from previous runs (quick fit from one segment)
    print("\n--- Quick curve fit (1 segment, 10 frames) ---")
    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=10, luminance_bins=256,
                         low_percentile=1.0, high_percentile=99.9)
    sf = int(round(2700.0 * fps))
    ef = int(round(2730.0 * fps))
    shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
    s = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                 config=config, n_samples=10, proxy_width=960)
    curve = estimate_luminance_curve(s.sdr_luminance, s.hdr_luminance, config=config)
    print(f"  Curve: {len(curve)} pts from {s.n_valid_pairs:,} samples")

    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.99)
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)

    # --- SINGLE FRAME BENCHMARK ---
    print(f"\n--- Single frame transform benchmark ---")
    start_s = 2710.0
    om_start_s = start_s + 1167 / fps

    # Extract 1 HDR + 1 OM frame
    hdr_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{start_s:.6f}",
               "-i", str(HDR_PATH), "-frames:v", "1", "-pix_fmt", "rgb48le",
               "-f", "rawvideo", "pipe:1"]
    om_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{om_start_s:.6f}",
              "-i", str(OM_PATH), "-frames:v", "1", "-pix_fmt", "rgb48le",
              "-f", "rawvideo", "pipe:1"]

    r_hdr = subprocess.run(hdr_cmd, capture_output=True, timeout=30)
    r_om = subprocess.run(om_cmd, capture_output=True, timeout=30)

    hdr_arr = np.frombuffer(r_hdr.stdout[:hdr_w*hdr_h*6], dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
    om_arr = np.frombuffer(r_om.stdout[:om_w*om_h*6], dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0

    t0 = time.perf_counter()
    output = composite_extend(hdr_arr, om_arr, transform, geom, ext_mask,
                              sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
    t1 = time.perf_counter()
    frame_time = t1 - t0
    print(f"  Single frame transform: {frame_time:.2f}s")
    print(f"  Estimated transform FPS: {1.0/frame_time:.2f}")
    print(f"  Frame size: {om_w}x{om_h} = {om_w*om_h*6/(1024*1024):.1f} MB raw")

    # --- NVENC STREAMING TEST (10 frames piped directly) ---
    print(f"\n--- NVENC streaming test ({N_FRAMES} frames) ---")
    nvenc_output = OUTPUT_DIR / "BR2049_nvenc_10f.mkv"

    nvenc_cmd = [
        "ffmpeg", "-v", "warning", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{om_w}x{om_h}", "-r", "24000/1001",
        "-i", "pipe:0",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        "-pix_fmt", "p010le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc", "-color_range", "tv",
        str(nvenc_output),
    ]

    # Decode sources streaming
    hdr_dec = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{start_s:.6f}",
               "-i", str(HDR_PATH), "-frames:v", str(N_FRAMES),
               "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    om_dec = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{om_start_s:.6f}",
              "-i", str(OM_PATH), "-frames:v", str(N_FRAMES),
              "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]

    hdr_proc = subprocess.Popen(hdr_dec, stdout=subprocess.PIPE)
    om_proc = subprocess.Popen(om_dec, stdout=subprocess.PIPE)
    enc_proc = subprocess.Popen(nvenc_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    hdr_fb = hdr_w * hdr_h * 6
    om_fb = om_w * om_h * 6
    t_total_start = time.perf_counter()
    transform_times = []
    frames_done = 0

    for i in range(N_FRAMES):
        hdr_raw = hdr_proc.stdout.read(hdr_fb)
        om_raw = om_proc.stdout.read(om_fb)
        if len(hdr_raw) < hdr_fb or len(om_raw) < om_fb:
            break

        h = np.frombuffer(hdr_raw, dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
        o = np.frombuffer(om_raw, dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0

        tt0 = time.perf_counter()
        out = composite_extend(h, o, transform, geom, ext_mask,
                               sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
        tt1 = time.perf_counter()
        transform_times.append(tt1 - tt0)

        out_u16 = (np.clip(out, 0.0, 1.0) * 65535.0).astype(np.uint16)
        enc_proc.stdin.write(out_u16.tobytes())
        frames_done += 1

    enc_proc.stdin.close()
    hdr_proc.stdout.close()
    om_proc.stdout.close()
    stdout, stderr = enc_proc.communicate()
    hdr_proc.wait()
    om_proc.wait()
    t_total_end = time.perf_counter()

    total_time = t_total_end - t_total_start
    avg_transform = np.mean(transform_times)
    enc_returncode = enc_proc.returncode

    print(f"  Frames rendered: {frames_done}")
    print(f"  NVENC return code: {enc_returncode}")
    print(f"  Total pipeline time: {total_time:.2f}s")
    print(f"  Avg transform/frame: {avg_transform:.3f}s ({1/avg_transform:.1f} fps)")
    print(f"  Total transform: {sum(transform_times):.2f}s")
    print(f"  Encoding overhead: {total_time - sum(transform_times):.2f}s")
    print(f"  Effective pipeline FPS: {frames_done/total_time:.2f}")

    if stderr:
        stderr_text = stderr.decode(errors="replace").strip()
        if stderr_text:
            # Filter: only show actual errors, not info messages
            lines = stderr_text.split("\n")
            errors = [l for l in lines if "error" in l.lower() or "fail" in l.lower()]
            info = [l for l in lines if "error" not in l.lower() and "fail" not in l.lower()]
            if errors:
                print(f"  ERRORS: {errors}")
            if info:
                print(f"  Info messages: {len(info)} lines (encoder status)")

    # Verify output
    if nvenc_output.exists():
        size = nvenc_output.stat().st_size
        print(f"\n  Output: {nvenc_output}")
        print(f"  Size: {size / 1024:.1f} KB")

        # FFprobe
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
             "-select_streams", "v", str(nvenc_output)],
            capture_output=True, text=True, timeout=10
        )
        if probe.returncode == 0:
            import json
            info = json.loads(probe.stdout)
            s = info.get("streams", [{}])[0]
            print(f"  Codec: {s.get('codec_name')} ({s.get('profile')})")
            print(f"  Pix fmt: {s.get('pix_fmt')}")
            print(f"  Primaries: {s.get('color_primaries')}")
            print(f"  Transfer: {s.get('color_transfer')}")
            print(f"  Matrix: {s.get('color_space')}")
            print(f"  Range: {s.get('color_range')}")
    else:
        print(f"\n  OUTPUT NOT CREATED!")

    print(f"\n{'='*70}")
    print("P2.10.1 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
