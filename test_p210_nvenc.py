"""P2.10: NVENC test renderer — x265 vs NVENC comparison.

Renders identical 5s output with both encoders, compares speed and output.
"""
import json
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
from auto_openmatte.core.transfer_functions import pq_eotf
from auto_openmatte.processing.luminance import estimate_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.pipeline.compose import composite_extend

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_DIR = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output")
OUTPUT_X265 = OUTPUT_DIR / "BR2049_P28_Log_5s_x265.mkv"
OUTPUT_NVENC = OUTPUT_DIR / "BR2049_P28_Log_5s_nvenc.mkv"
PEAK = 10000.0


def render_with_encoder(frames_data, encoder_cmd, output_path, om_w, om_h, n_frames, fps_str):
    """Encode pre-computed frames with specified encoder."""
    cmd = [
        "ffmpeg", "-v", "warning", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{om_w}x{om_h}", "-r", fps_str,
        "-i", "pipe:0",
    ] + encoder_cmd + [str(output_path)]

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate(input=frames_data)
    t1 = time.perf_counter()

    return {
        "time": t1 - t0,
        "returncode": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


def ffprobe_info(path):
    """Get stream and format info."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", "-select_streams", "v", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def main():
    print("=" * 70)
    print("P2.10: NVENC TEST RENDERER")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps
    hdr_w, hdr_h = 3840, 1600
    om_w, om_h = 3840, 2160

    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256,
                         low_percentile=1.0, high_percentile=99.9)

    # Prepare curve
    print("\n--- Preparing luminance curve ---")
    segments = [(2700.0, 2730.0), (4500.0, 4530.0), (2220.0, 2250.0)]
    all_sdr, all_hdr = [], []
    for start, end in segments:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        s = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                     config=config, n_samples=30, proxy_width=960)
        all_sdr.append(s.sdr_luminance)
        all_hdr.append(s.hdr_luminance)
    sdr_all = np.concatenate(all_sdr)
    hdr_all = np.concatenate(all_hdr)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(sdr_all))
    curve = estimate_luminance_curve(sdr_all[idx[:int(len(sdr_all)*0.8)]], hdr_all[idx[:int(len(sdr_all)*0.8)]], config=config)
    print(f"  Curve: {len(curve)} points")

    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.99)
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)

    # Pre-compute all frames (transform only, no encoding)
    start_seconds = 2710.0
    n_frames = int(round(5.0 * fps))
    om_start = start_seconds + 1167 / fps

    print(f"\n--- Pre-computing {n_frames} frames (transform only) ---")
    hdr_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{start_seconds:.6f}",
               "-i", str(HDR_PATH), "-frames:v", str(n_frames),
               "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    om_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{om_start:.6f}",
              "-i", str(OM_PATH), "-frames:v", str(n_frames),
              "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]

    t_decode_start = time.perf_counter()
    hdr_proc = subprocess.Popen(hdr_cmd, stdout=subprocess.PIPE)
    om_proc = subprocess.Popen(om_cmd, stdout=subprocess.PIPE)

    hdr_fb = hdr_w * hdr_h * 6
    om_fb = om_w * om_h * 6
    all_output_bytes = bytearray()

    for i in range(n_frames):
        hdr_raw = hdr_proc.stdout.read(hdr_fb)
        om_raw = om_proc.stdout.read(om_fb)
        if len(hdr_raw) < hdr_fb or len(om_raw) < om_fb:
            break
        hdr_arr = np.frombuffer(hdr_raw, dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
        om_arr = np.frombuffer(om_raw, dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0
        output = composite_extend(hdr_arr, om_arr, transform, geom, ext_mask,
                                  sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
        out_u16 = (np.clip(output, 0.0, 1.0) * 65535.0).astype(np.uint16)
        all_output_bytes.extend(out_u16.tobytes())

    hdr_proc.stdout.close()
    om_proc.stdout.close()
    hdr_proc.wait()
    om_proc.wait()
    t_decode_end = time.perf_counter()
    actual_frames = len(all_output_bytes) // (om_w * om_h * 6)
    frames_bytes = bytes(all_output_bytes)

    print(f"  Frames computed: {actual_frames}")
    print(f"  Transform time: {t_decode_end - t_decode_start:.1f}s "
          f"({actual_frames / (t_decode_end - t_decode_start):.1f} fps)")
    print(f"  Raw data: {len(frames_bytes) / (1024*1024):.0f} MB")

    # Encoder commands
    x265_cmd = [
        "-c:v", "libx265", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        "-x265-params",
        "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
        "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50):"
        "max-cll=457,179",
    ]

    nvenc_cmd = [
        "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        "-pix_fmt", "p010le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        "-color_range", "tv",
    ]

    fps_str = "24000/1001"

    # Encode with x265
    print(f"\n--- Encoding with x265 ---")
    r_x265 = render_with_encoder(frames_bytes, x265_cmd, OUTPUT_X265, om_w, om_h, actual_frames, fps_str)
    print(f"  Return code: {r_x265['returncode']}")
    print(f"  Encode time: {r_x265['time']:.1f}s ({actual_frames/r_x265['time']:.1f} fps)")
    if r_x265['returncode'] != 0:
        print(f"  STDERR: {r_x265['stderr'][:500]}")

    # Encode with NVENC
    print(f"\n--- Encoding with NVENC ---")
    r_nvenc = render_with_encoder(frames_bytes, nvenc_cmd, OUTPUT_NVENC, om_w, om_h, actual_frames, fps_str)
    print(f"  Return code: {r_nvenc['returncode']}")
    print(f"  Encode time: {r_nvenc['time']:.1f}s ({actual_frames/r_nvenc['time']:.1f} fps)")
    if r_nvenc['returncode'] != 0:
        print(f"  STDERR: {r_nvenc['stderr'][:500]}")

    # FFprobe both
    print(f"\n--- FFprobe comparison ---")
    for label, path in [("x265", OUTPUT_X265), ("NVENC", OUTPUT_NVENC)]:
        if not path.exists():
            print(f"  {label}: FILE NOT FOUND")
            continue
        info = ffprobe_info(path)
        if not info:
            print(f"  {label}: ffprobe failed")
            continue
        s = info.get("streams", [{}])[0]
        f = info.get("format", {})
        size_mb = path.stat().st_size / (1024*1024)
        print(f"\n  {label}:")
        print(f"    Codec:       {s.get('codec_name')} ({s.get('profile')})")
        print(f"    Pix fmt:     {s.get('pix_fmt')}")
        print(f"    Primaries:   {s.get('color_primaries')}")
        print(f"    Transfer:    {s.get('color_transfer')}")
        print(f"    Matrix:      {s.get('color_space')}")
        print(f"    Range:       {s.get('color_range')}")
        print(f"    Size:        {size_mb:.1f} MB")
        print(f"    Duration:    {f.get('duration', 'N/A')}s")
        print(f"    Bitrate:     {int(f.get('bit_rate', 0))//1000} kbps")

    # Quick image comparison: extract frame at 3.5s from both
    print(f"\n--- Image comparison (frame at 3.5s) ---")
    for label, path in [("x265", OUTPUT_X265), ("NVENC", OUTPUT_NVENC)]:
        if not path.exists():
            continue
        cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", "3.5",
               "-i", str(path), "-frames:v", "1", "-pix_fmt", "rgb48le",
               "-f", "rawvideo", "pipe:1"]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        if r.returncode != 0 or len(r.stdout) < om_w * om_h * 6:
            print(f"  {label}: frame extraction failed")
            continue
        frame = np.frombuffer(r.stdout[:om_w*om_h*6], dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0

        # Boundary
        y279 = float(np.mean(pq_eotf(0.2627*frame[279,:,0]+0.6780*frame[279,:,1]+0.0593*frame[279,:,2])))
        y280 = float(np.mean(pq_eotf(0.2627*frame[280,:,0]+0.6780*frame[280,:,1]+0.0593*frame[280,:,2])))
        y1879 = float(np.mean(pq_eotf(0.2627*frame[1879,:,0]+0.6780*frame[1879,:,1]+0.0593*frame[1879,:,2])))
        y1880 = float(np.mean(pq_eotf(0.2627*frame[1880,:,0]+0.6780*frame[1880,:,1]+0.0593*frame[1880,:,2])))

        # Jacket ROI bright pixels
        roi = frame[1900:2050, 1500:2500, :]
        roi_y = pq_eotf(0.2627*roi[...,0]+0.6780*roi[...,1]+0.0593*roi[...,2])
        n_above_100 = int(np.sum(roi_y > 100))
        n_above_200 = int(np.sum(roi_y > 200))
        max_roi = float(np.max(roi_y))

        print(f"\n  {label}:")
        print(f"    Top boundary:  |{y280:.2f} - {y279:.2f}| = {abs(y280-y279):.3f} nits")
        print(f"    Bot boundary:  |{y1880:.2f} - {y1879:.2f}| = {abs(y1880-y1879):.3f} nits")
        print(f"    Jacket ROI: max={max_roi:.1f} nits, >100={n_above_100}, >200={n_above_200}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    t_transform = t_decode_end - t_decode_start
    x265_time = r_x265['time'] if r_x265['returncode'] == 0 else -1
    nvenc_time = r_nvenc['time'] if r_nvenc['returncode'] == 0 else -1
    speedup = x265_time / nvenc_time if nvenc_time > 0 and x265_time > 0 else 0

    print(f"\n  Transform time:  {t_transform:.1f}s ({actual_frames/t_transform:.1f} fps)")
    print(f"  x265 encode:     {x265_time:.1f}s ({actual_frames/x265_time:.1f} fps)" if x265_time > 0 else "  x265: FAILED")
    print(f"  NVENC encode:    {nvenc_time:.1f}s ({actual_frames/nvenc_time:.1f} fps)" if nvenc_time > 0 else "  NVENC: FAILED")
    print(f"  Speedup:         {speedup:.1f}x")
    print(f"  Total x265:      {t_transform + x265_time:.1f}s")
    print(f"  Total NVENC:     {t_transform + nvenc_time:.1f}s")

    print(f"\n{'='*70}")
    print("P2.10 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
