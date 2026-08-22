"""P2.9.4: Quick 10-frame real-material validation of P2.9.3."""
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
from auto_openmatte.core.transfer_functions import bt1886_oetf, pq_eotf
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance
from auto_openmatte.processing.transform import apply_shot_transform
from auto_openmatte.pipeline.compose import composite_extend

ROOT = Path(__file__).parent

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT = ROOT / "test_output" / "BR2049_P294_quick_10frames.mkv"
PEAK = 10000.0
N_FRAMES = 10
HDR_START = 2714.0


def pq_rgb_to_luminance_nits(frame):
    """Physical BT.2020 Y: EOTF per channel, then linear-light weights."""
    return (
        0.2627 * pq_eotf(frame[..., 0])
        + 0.6780 * pq_eotf(frame[..., 1])
        + 0.0593 * pq_eotf(frame[..., 2])
    )


def main():
    print("=" * 70)
    print(f"P2.9.4: QUICK 10-FRAME VALIDATION (HDR start={HDR_START}s)")
    print("=" * 70)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

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
    config = ColorConfig(samples_per_shot=10, luminance_bins=256,
                         low_percentile=1.0, high_percentile=99.9)

    # Quick curve
    print("\n--- Curve ---")
    sf = int(round(2700.0 * fps))
    ef = int(round(2730.0 * fps))
    shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
    s = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                 config=config, n_samples=10, proxy_width=960)
    curve = estimate_luminance_curve(s.sdr_luminance, s.hdr_luminance, config=config)
    print(f"  {len(curve)} points")

    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.99)
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)

    # Exercise the complete curve → ratio → PQ path at the near-black boundary.
    # This specifically detects a ratio guard that could bypass the low-end bridge.
    print("\n--- Low-end full-chain probe ---")
    probe_linear = np.array([0.0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5, 1e-4, 1e-3])
    probe_signal = bt1886_oetf(probe_linear)[:, None, None]
    probe_signal = np.repeat(probe_signal, 3, axis=2)
    probe_output = apply_shot_transform(probe_signal, transform, peak_nits=PEAK)
    probe_output_nits = pq_rgb_to_luminance_nits(probe_output[:, 0, :])
    probe_curve_nits = apply_luminance_curve(probe_linear, curve) * PEAK
    for value, expected, observed in zip(probe_linear, probe_curve_nits, probe_output_nits):
        print(f"  Y={value:.1e}: curve={expected:.8f} nits, full-chain={observed:.8f} nits, "
              f"delta={observed-expected:+.8f}")

    # Render 10 frames with NVENC
    om_start = HDR_START + 1167 / fps
    print(f"\n--- Rendering {N_FRAMES} frames ---")

    hdr_dec = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{HDR_START:.6f}",
               "-i", str(HDR_PATH), "-frames:v", str(N_FRAMES),
               "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    om_dec = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{om_start:.6f}",
              "-i", str(OM_PATH), "-frames:v", str(N_FRAMES),
              "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    enc_cmd = [
        "ffmpeg", "-v", "warning", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb48le",
        "-s", f"{om_w}x{om_h}", "-r", "24000/1001", "-i", "pipe:0",
        "-vf", "format=p010le,setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc:range=tv",
        "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18",
        str(OUTPUT),
    ]

    hdr_proc = subprocess.Popen(hdr_dec, stdout=subprocess.PIPE)
    om_proc = subprocess.Popen(om_dec, stdout=subprocess.PIPE)
    enc_proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    hdr_fb = hdr_w * hdr_h * 6
    om_fb = om_w * om_h * 6
    t0 = time.perf_counter()

    # Store composites for analysis
    composites = []
    for i in range(N_FRAMES):
        hdr_raw = hdr_proc.stdout.read(hdr_fb)
        om_raw = om_proc.stdout.read(om_fb)
        if len(hdr_raw) < hdr_fb or len(om_raw) < om_fb:
            break
        h = np.frombuffer(hdr_raw, dtype=np.uint16).reshape(hdr_h, hdr_w, 3).astype(np.float64) / 65535.0
        o = np.frombuffer(om_raw, dtype=np.uint16).reshape(om_h, om_w, 3).astype(np.float64) / 65535.0
        out = composite_extend(h, o, transform, geom, ext_mask,
                               sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
        composites.append(out)
        out_u16 = (np.clip(out, 0.0, 1.0) * 65535.0).astype(np.uint16)
        enc_proc.stdin.write(out_u16.tobytes())

    enc_proc.stdin.close()
    hdr_proc.stdout.close()
    om_proc.stdout.close()
    enc_proc.wait()
    hdr_proc.wait()
    om_proc.wait()
    t1 = time.perf_counter()

    print(f"  Frames: {len(composites)}, time: {t1-t0:.1f}s")
    if OUTPUT.exists():
        print(f"  Output: {OUTPUT} ({OUTPUT.stat().st_size/1024:.0f} KB)")

    # === ANALYSIS ===
    print(f"\n{'='*70}")
    print("BOUNDARY ANALYSIS (all 10 frames)")
    print(f"{'='*70}")

    top_jumps = []
    bot_jumps = []
    for i, frame in enumerate(composites):
        y279 = float(np.mean(pq_rgb_to_luminance_nits(frame[279, :, :])))
        y280 = float(np.mean(pq_rgb_to_luminance_nits(frame[280, :, :])))
        y1879 = float(np.mean(pq_rgb_to_luminance_nits(frame[1879, :, :])))
        y1880 = float(np.mean(pq_rgb_to_luminance_nits(frame[1880, :, :])))
        top_jumps.append(abs(y280 - y279))
        bot_jumps.append(abs(y1880 - y1879))

    print(f"  Top boundary (row 279→280):")
    print(f"    Mean jump: {np.mean(top_jumps):.3f} nits")
    print(f"    Max jump:  {np.max(top_jumps):.3f} nits")
    print(f"  Bottom boundary (row 1879→1880):")
    print(f"    Mean jump: {np.mean(bot_jumps):.3f} nits")
    print(f"    Max jump:  {np.max(bot_jumps):.3f} nits")

    # === JACKET ROI ===
    print(f"\n{'='*70}")
    print("JACKET ROI (rows 1900-2050, cols 1500-2500)")
    print(f"{'='*70}")

    all_roi_nits = []
    for frame in composites:
        roi = frame[1900:2050, 1500:2500, :]
        roi_nits = pq_rgb_to_luminance_nits(roi)
        all_roi_nits.append(roi_nits)

    roi_all = np.concatenate([r.flatten() for r in all_roi_nits])
    print(f"\n  Aggregated over 10 frames ({len(roi_all):,} pixels):")
    print(f"    min:  {roi_all.min():.4f} nits")
    print(f"    P1:   {np.percentile(roi_all, 1):.4f}")
    print(f"    P10:  {np.percentile(roi_all, 10):.4f}")
    print(f"    P25:  {np.percentile(roi_all, 25):.4f}")
    print(f"    P50:  {np.percentile(roi_all, 50):.4f}")
    print(f"    P75:  {np.percentile(roi_all, 75):.4f}")
    print(f"    P99:  {np.percentile(roi_all, 99):.4f}")
    print(f"    max:  {roi_all.max():.4f} nits")
    print(f"\n    Pixels > 100 nits: {np.sum(roi_all > 100)}")
    print(f"    Pixels > 200 nits: {np.sum(roi_all > 200)}")
    print(f"    Pixels > 400 nits: {np.sum(roi_all > 400)}")

    # Plateau check
    rounded = np.round(roi_all, 3)
    values, counts = np.unique(rounded, return_counts=True)
    max_count_idx = np.argmax(counts)
    print(f"\n    Unique luminance values (rounded 0.001): {len(values)}")
    print(f"    Most common value: {values[max_count_idx]:.3f} nits ({counts[max_count_idx]} pixels)")
    if counts[max_count_idx] > len(roi_all) * 0.05:
        print(f"    ⚠️ POTENTIAL PLATEAU ({counts[max_count_idx]/len(roi_all)*100:.1f}%)")
    else:
        print(f"    ✅ No plateau (max repetition = {counts[max_count_idx]/len(roi_all)*100:.2f}%)")

    # === WHITE-DOT CHECK ===
    print(f"\n{'='*70}")
    print("WHITE-DOT CHECK (20 brightest in ROI)")
    print(f"{'='*70}")

    # Use frame 5 (middle)
    roi_frame5 = all_roi_nits[min(5, len(all_roi_nits)-1)]
    flat = roi_frame5.flatten()
    top20_idx = np.argsort(flat)[-20:]

    print(f"\n  {'#':<4} {'Nits':<10} {'Status'}")
    print(f"  {'-'*24}")
    for i, idx in enumerate(reversed(top20_idx)):
        nits = flat[idx]
        status = "⚠️ HIGH" if nits > 100 else "OK"
        print(f"  {i+1:<4} {nits:<10.2f} {status}")

    max_roi = float(np.max(flat))
    print(f"\n  Max ROI luminance: {max_roi:.2f} nits")
    if max_roi > 400:
        print(f"  ❌ WHITE DOTS STILL PRESENT")
    elif max_roi > 100:
        print(f"  ⚠️ Some bright pixels (may be legitimate highlights)")
    else:
        print(f"  ✅ No anomalous bright pixels")

    # === FINAL VERDICT ===
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    whitedots_gone = np.sum(roi_all > 400) == 0
    no_plateau = counts[max_count_idx] < len(roi_all) * 0.05
    boundary_ok = np.max(top_jumps) < 10 and np.max(bot_jumps) < 5

    print(f"  1. White dots (>400 nits): {'GONE ✅' if whitedots_gone else 'PRESENT ❌'}")
    print(f"  2. Shadow detail: {'PRESERVED ✅' if no_plateau else 'PLATEAU ❌'}")
    print(f"  3. Boundary: top={np.mean(top_jumps):.2f}, bot={np.mean(bot_jumps):.2f} nits "
          f"{'✅' if boundary_ok else '⚠️'}")
    print(f"  4. P2.9.3 real-material: {'PASS ✅' if (whitedots_gone and no_plateau) else 'NEEDS REVIEW'}")

    print(f"\n{'='*70}")
    print("P2.9.4 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
