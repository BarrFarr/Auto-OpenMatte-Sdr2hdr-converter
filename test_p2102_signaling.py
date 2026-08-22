"""P2.10.2: NVENC with correct HDR10 signaling — 10 frame test."""
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
from auto_openmatte.processing.luminance import estimate_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance
from auto_openmatte.pipeline.compose import composite_extend

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output\BR2049_nvenc_hdr10_10f.mkv")
PEAK = 10000.0
N_FRAMES = 10


def main():
    print("=" * 70)
    print("P2.10.2: NVENC HDR10 SIGNALING TEST (10 frames)")
    print("=" * 70)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps
    hdr_w, hdr_h = 3840, 1600
    om_w, om_h = 3840, 2160

    # Quick curve
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
    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.99)
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)

    # NVENC command with correct HDR10 signaling via -vf setparams
    # Key: format=p010le + setparams for VUI color metadata
    nvenc_cmd = [
        "ffmpeg", "-v", "warning", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{om_w}x{om_h}", "-r", "24000/1001",
        "-i", "pipe:0",
        "-vf", "format=p010le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        str(OUTPUT),
    ]

    print(f"\n  NVENC command (key part):")
    print(f"    -vf format=p010le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv")
    print(f"    -c:v hevc_nvenc -preset p5 -rc vbr -cq 18")

    # Decode + transform + encode streaming
    start_s = 2710.0
    om_start_s = start_s + 1167 / fps

    hdr_dec = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{start_s:.6f}",
               "-i", str(HDR_PATH), "-frames:v", str(N_FRAMES),
               "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    om_dec = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{om_start_s:.6f}",
              "-i", str(OM_PATH), "-frames:v", str(N_FRAMES),
              "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]

    print(f"\n--- Rendering {N_FRAMES} frames ---")
    hdr_proc = subprocess.Popen(hdr_dec, stdout=subprocess.PIPE)
    om_proc = subprocess.Popen(om_dec, stdout=subprocess.PIPE)
    enc_proc = subprocess.Popen(nvenc_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    hdr_fb = hdr_w * hdr_h * 6
    om_fb = om_w * om_h * 6
    t0 = time.perf_counter()
    frames_done = 0

    for i in range(N_FRAMES):
        hdr_raw = hdr_proc.stdout.read(hdr_fb)
        om_raw = om_proc.stdout.read(om_fb)
        if len(hdr_raw) < hdr_fb or len(om_raw) < om_fb:
            break
        h = np.frombuffer(hdr_raw, dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
        o = np.frombuffer(om_raw, dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0
        out = composite_extend(h, o, transform, geom, ext_mask,
                               sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
        out_u16 = (np.clip(out, 0.0, 1.0) * 65535.0).astype(np.uint16)
        enc_proc.stdin.write(out_u16.tobytes())
        frames_done += 1

    enc_proc.stdin.close()
    hdr_proc.stdout.close()
    om_proc.stdout.close()
    _, stderr = enc_proc.communicate()
    hdr_proc.wait()
    om_proc.wait()
    t1 = time.perf_counter()

    print(f"  Frames: {frames_done}")
    print(f"  Return code: {enc_proc.returncode}")
    print(f"  Time: {t1-t0:.1f}s")
    if stderr:
        err_text = stderr.decode(errors="replace").strip()
        if err_text:
            print(f"  Encoder messages: {err_text[:200]}")

    # FFprobe
    print(f"\n--- FFprobe result ---")
    if OUTPUT.exists():
        print(f"  File: {OUTPUT}")
        print(f"  Size: {OUTPUT.stat().st_size / 1024:.1f} KB")
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
             "-show_format", "-select_streams", "v", str(OUTPUT)],
            capture_output=True, text=True, timeout=10
        )
        if probe.returncode == 0:
            info = json.loads(probe.stdout)
            s = info.get("streams", [{}])[0]
            f = info.get("format", {})
            print(f"  codec_name:       {s.get('codec_name')}")
            print(f"  profile:          {s.get('profile')}")
            print(f"  pix_fmt:          {s.get('pix_fmt')}")
            print(f"  color_range:      {s.get('color_range')}")
            print(f"  color_primaries:  {s.get('color_primaries')}")
            print(f"  color_transfer:   {s.get('color_transfer')}")
            print(f"  color_space:      {s.get('color_space')}")
            print(f"  duration:         {f.get('duration', 'N/A')}s")

            # Check PASS criteria
            passed = (
                s.get('color_primaries') == 'bt2020'
                and s.get('color_transfer') == 'smpte2084'
                and s.get('color_space') == 'bt2020nc'
                and s.get('color_range') == 'tv'
                and s.get('profile') == 'Main 10'
            )
            print(f"\n  HDR10 SIGNALING: {'PASS ✅' if passed else 'FAIL ❌'}")
    else:
        print("  FILE NOT CREATED!")

    print(f"\n{'='*70}")
    print("P2.10.2 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
