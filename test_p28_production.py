"""P2.8: Production re-baseline + 5s video render."""
import subprocess
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, ShotTransform, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import pq_eotf
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.pipeline.compose import composite_extend

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_DIR = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output")
OUTPUT_B = OUTPUT_DIR / "BR2049_P28_Log_5s.mkv"
PEAK = 10000.0


def main():
    print("=" * 70)
    print("P2.8: PRODUCTION RE-BASELINE + 5s VIDEO")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps

    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256,
                         low_percentile=1.0, high_percentile=99.9)

    # ===== RE-BASELINE =====
    print("\n--- Re-baseline (Log P1-P99.9) ---")
    segments = [(2700.0, 2730.0), (4500.0, 4530.0), (2220.0, 2250.0)]
    all_sdr, all_hdr = [], []
    t0 = time.perf_counter()
    for start, end in segments:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        s = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                     config=config, n_samples=30, proxy_width=960)
        all_sdr.append(s.sdr_luminance)
        all_hdr.append(s.hdr_luminance)
    t1 = time.perf_counter()
    sdr = np.concatenate(all_sdr)
    hdr = np.concatenate(all_hdr)
    print(f"  Samples: {len(sdr):,}, time: {t1-t0:.1f}s")

    rng = np.random.default_rng(42)
    n = len(sdr)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    sdr_train, sdr_val = sdr[idx[:split]], sdr[idx[split:]]
    hdr_train, hdr_val = hdr[idx[:split]], hdr[idx[split:]]

    curve = estimate_luminance_curve(sdr_train, hdr_train, config=config)
    print(f"  Curve: {len(curve)} points")

    # Evaluate
    pred = apply_luminance_curve(sdr_val, curve) * PEAK
    actual = hdr_val * PEAK
    err = pred - actual
    abs_err = np.abs(err)

    def rmae(lo_p, hi_p):
        lo = np.percentile(sdr_val, lo_p) if lo_p > 0 else 0
        hi = np.percentile(sdr_val, hi_p) if hi_p < 100 else sdr_val.max() * 10
        m = (sdr_val >= lo) & (sdr_val < hi)
        return float(np.mean(abs_err[m])) if np.sum(m) > 0 else 0.0

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    signed = float(np.mean(err))

    print(f"\n  PRODUCTION BASELINE (Log P1-P99.9):")
    print(f"    MAE:          {mae:.4f} nits")
    print(f"    RMSE:         {rmse:.4f} nits")
    print(f"    Signed:       {signed:+.4f} nits")
    print(f"    P0-P50:       {rmae(0,50):.4f}")
    print(f"    P50-P90:      {rmae(50,90):.4f}")
    print(f"    P90-P99:      {rmae(90,99):.4f}")
    print(f"    P99-P99.5:    {rmae(99,99.5):.4f}")
    print(f"    P99.5-P99.9:  {rmae(99.5,99.9):.4f}")
    print(f"    P99.9-P100:   {rmae(99.9,100):.4f}")

    # ===== 5s VIDEO RENDER =====
    print(f"\n--- Rendering 5s test video ---")
    start_seconds = 2710.0  # Frame ~2710s (same boundary diagnostic frame)
    n_frames = int(round(5.0 * fps))  # 5 seconds
    hdr_w, hdr_h = 3840, 1600
    om_w, om_h = 3840, 2160

    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.99)
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)

    hdr_start = start_seconds
    om_start = start_seconds + 1167 / fps

    hdr_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{hdr_start:.6f}",
               "-i", str(HDR_PATH), "-frames:v", str(n_frames),
               "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    om_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{om_start:.6f}",
              "-i", str(OM_PATH), "-frames:v", str(n_frames),
              "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    enc_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{om_w}x{om_h}",
               "-r", "24000/1001", "-i", "pipe:0",
               "-c:v", "libx265", "-crf", "18", "-preset", "medium",
               "-pix_fmt", "yuv420p10le",
               "-color_primaries", "bt2020", "-color_trc", "smpte2084",
               "-colorspace", "bt2020nc",
               "-x265-params", "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(40000000,50):max-cll=457,179",
               str(OUTPUT_B)]

    t2 = time.perf_counter()
    hdr_proc = subprocess.Popen(hdr_cmd, stdout=subprocess.PIPE)
    om_proc = subprocess.Popen(om_cmd, stdout=subprocess.PIPE)
    enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    hdr_frame_bytes = hdr_w * hdr_h * 6
    om_frame_bytes = om_w * om_h * 6
    frames_written = 0

    for i in range(n_frames):
        hdr_raw = hdr_proc.stdout.read(hdr_frame_bytes)
        om_raw = om_proc.stdout.read(om_frame_bytes)
        if len(hdr_raw) < hdr_frame_bytes or len(om_raw) < om_frame_bytes:
            break
        hdr_arr = np.frombuffer(hdr_raw, dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
        om_arr = np.frombuffer(om_raw, dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0
        output = composite_extend(hdr_arr, om_arr, transform, geom, ext_mask,
                                  sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
        out_u16 = (np.clip(output, 0.0, 1.0) * 65535.0).astype(np.uint16)
        enc_proc.stdin.write(out_u16.tobytes())
        frames_written += 1

    hdr_proc.stdout.close()
    om_proc.stdout.close()
    enc_proc.stdin.close()
    hdr_proc.wait()
    om_proc.wait()
    enc_proc.wait()
    t3 = time.perf_counter()

    print(f"  Frames: {frames_written}/{n_frames}")
    print(f"  Time: {t3-t2:.1f}s ({frames_written/(t3-t2):.1f} fps)")
    print(f"  Output: {OUTPUT_B}")
    if OUTPUT_B.exists():
        print(f"  Size: {OUTPUT_B.stat().st_size / (1024*1024):.1f} MB")

    # Boundary measurement on output
    print(f"\n--- Boundary verification ---")
    probe_cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", "2.5",
                 "-i", str(OUTPUT_B), "-frames:v", "1", "-pix_fmt", "rgb48le",
                 "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(probe_cmd, capture_output=True, timeout=30)
    if r.returncode == 0 and len(r.stdout) >= om_w * om_h * 6:
        frame = np.frombuffer(r.stdout[:om_w*om_h*6], dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0
        # Decode PQ to nits for boundary
        lum_279 = float(np.mean(pq_eotf(0.2627*frame[279,:,0] + 0.6780*frame[279,:,1] + 0.0593*frame[279,:,2])))
        lum_280 = float(np.mean(pq_eotf(0.2627*frame[280,:,0] + 0.6780*frame[280,:,1] + 0.0593*frame[280,:,2])))
        lum_1879 = float(np.mean(pq_eotf(0.2627*frame[1879,:,0] + 0.6780*frame[1879,:,1] + 0.0593*frame[1879,:,2])))
        lum_1880 = float(np.mean(pq_eotf(0.2627*frame[1880,:,0] + 0.6780*frame[1880,:,1] + 0.0593*frame[1880,:,2])))
        print(f"  Row 279 (ext):  {lum_279:.3f} nits")
        print(f"  Row 280 (HDR):  {lum_280:.3f} nits")
        print(f"  Top jump:       {abs(lum_280-lum_279):.3f} nits")
        print(f"  Row 1879 (HDR): {lum_1879:.3f} nits")
        print(f"  Row 1880 (ext): {lum_1880:.3f} nits")
        print(f"  Bot jump:       {abs(lum_1880-lum_1879):.3f} nits")

    print(f"\n{'='*70}")
    print("P2.8 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
