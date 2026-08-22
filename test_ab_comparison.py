"""Controlled A/B comparison: PQ-domain vs linear-light compositing.

Both methods use EXACTLY the same:
- luminance curve
- input frames
- geometry
- transform parameters
- resize method

The ONLY difference is WHERE in the pipeline compositing occurs:
A: composite in PQ signal domain (current)
B: composite in linear-light domain (proposed)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, ShotTransform, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import linearize, delinearize, pq_eotf
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")

SYNC_OFFSET = 1167
PEAK_NITS = 10000.0


def extract_frame(path, time_s, width, height):
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin",
           "-ss", f"{time_s:.6f}", "-i", str(path),
           "-frames:v", "1", "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or len(r.stdout) < width * height * 6:
        return None
    return np.frombuffer(r.stdout[:width*height*6], dtype=np.uint16).reshape(
        height, width, 3).astype(np.float64) / 65535.0


def lum_bt709(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def lum_bt2020(rgb):
    return 0.2627 * rgb[..., 0] + 0.6780 * rgb[..., 1] + 0.0593 * rgb[..., 2]


def apply_curve_to_rgb(rgb_linear, curve):
    """Apply luminance curve to RGB via luminance-ratio scaling.
    
    IDENTICAL to what transform.py does internally.
    """
    lum = lum_bt709(rgb_linear)
    lum_mapped = apply_luminance_curve(lum, curve)
    ratio = np.where(lum > 1e-6, lum_mapped / lum, 1.0)
    result = rgb_linear * ratio[..., np.newaxis]
    return np.maximum(result, 0.0)


def method_a_current(hdr_pq, om_sdr, curve):
    """Method A: current pipeline — composite in PQ domain.
    
    1. Transform OM: SDR→linear→curve→delinearize(PQ) → OM_PQ
    2. HDR stays as PQ
    3. Composite: canvas = OM_PQ; canvas[280:1880] = HDR_PQ
    """
    # Transform OM to PQ (exactly as transform.py does)
    om_linear = linearize(om_sdr, "bt709")  # SDR linear [0,1]
    om_hdr_linear = apply_curve_to_rgb(om_linear, curve)  # HDR normalized linear
    om_pq = delinearize(om_hdr_linear, "smpte2084", peak_nits=PEAK_NITS)  # PQ signal
    
    # Composite in PQ domain
    output_pq = om_pq.copy()
    # HDR is already PQ; resize to fit overlap (3840x1600 → rows 280:1880 of 3840x2160)
    from scipy.ndimage import zoom
    hdr_h, hdr_w = hdr_pq.shape[:2]
    target_h = 1600  # 1880-280
    target_w = 3840
    if hdr_h != target_h or hdr_w != target_w:
        zf = (target_h / hdr_h, target_w / hdr_w, 1.0)
        hdr_resized = zoom(hdr_pq, zf, order=1)[:target_h, :target_w, :]
    else:
        hdr_resized = hdr_pq
    
    # Place HDR directly (mask=0 in core, no feather for this test)
    output_pq[280:1880, :, :] = hdr_resized
    return output_pq


def method_b_linear(hdr_pq, om_sdr, curve):
    """Method B: linear-light compositing.
    
    1. HDR PQ → PQ EOTF → HDR linear (normalized [0,1])
    2. OM SDR → BT.709 EOTF → SDR linear → curve → HDR linear
    3. Composite in LINEAR domain
    4. PQ OETF → output PQ
    """
    # HDR: PQ → linear normalized
    hdr_linear = linearize(hdr_pq, "smpte2084", peak_nits=PEAK_NITS)
    
    # OM: SDR → linear → curve → HDR linear (SAME curve as Method A)
    om_linear = linearize(om_sdr, "bt709")
    om_hdr_linear = apply_curve_to_rgb(om_linear, curve)  # IDENTICAL operation
    
    # Composite in LINEAR domain
    output_linear = om_hdr_linear.copy()
    # Resize HDR linear to fit overlap
    from scipy.ndimage import zoom
    hdr_h, hdr_w = hdr_linear.shape[:2]
    target_h = 1600
    target_w = 3840
    if hdr_h != target_h or hdr_w != target_w:
        zf = (target_h / hdr_h, target_w / hdr_w, 1.0)
        hdr_resized = zoom(hdr_linear, zf, order=1)[:target_h, :target_w, :]
    else:
        hdr_resized = hdr_linear
    
    output_linear[280:1880, :, :] = hdr_resized
    
    # Encode entire canvas to PQ
    output_pq = delinearize(output_linear, "smpte2084", peak_nits=PEAK_NITS)
    return output_pq


def decode_to_nits(pq_frame):
    """Decode PQ frame to absolute luminance in nits."""
    linear = linearize(pq_frame, "smpte2084", peak_nits=PEAK_NITS)
    return linear * PEAK_NITS


def analyze_frame(hdr_pq, om_sdr, curve, label):
    """Run both methods and compare."""
    out_a = method_a_current(hdr_pq, om_sdr, curve)
    out_b = method_b_linear(hdr_pq, om_sdr, curve)
    
    # Decode both to nits
    a_nits = decode_to_nits(out_a)
    b_nits = decode_to_nits(out_b)
    
    # HDR master in nits (ground truth for overlap)
    hdr_nits = decode_to_nits(hdr_pq)
    from scipy.ndimage import zoom
    hdr_h = hdr_nits.shape[0]
    if hdr_h != 1600:
        hdr_nits_r = zoom(hdr_nits, (1600/hdr_h, 3840/hdr_nits.shape[1], 1.0), order=1)[:1600,:3840,:]
    else:
        hdr_nits_r = hdr_nits
    
    # Luminance channels
    a_lum_top = lum_bt2020(a_nits[0:280])
    a_lum_mid = lum_bt2020(a_nits[280:1880])
    a_lum_bot = lum_bt2020(a_nits[1880:2160])
    
    b_lum_top = lum_bt2020(b_nits[0:280])
    b_lum_mid = lum_bt2020(b_nits[280:1880])
    b_lum_bot = lum_bt2020(b_nits[1880:2160])
    
    hdr_lum = lum_bt2020(hdr_nits_r)
    
    # Overlap reproduction error (A vs HDR master)
    a_overlap_err = np.abs(a_lum_mid - hdr_lum)
    b_overlap_err = np.abs(b_lum_mid - hdr_lum)
    
    # Boundaries
    a_row279 = float(np.mean(lum_bt2020(a_nits[279:280])))
    a_row280 = float(np.mean(lum_bt2020(a_nits[280:281])))
    a_row1879 = float(np.mean(lum_bt2020(a_nits[1879:1880])))
    a_row1880 = float(np.mean(lum_bt2020(a_nits[1880:1881])))
    
    b_row279 = float(np.mean(lum_bt2020(b_nits[279:280])))
    b_row280 = float(np.mean(lum_bt2020(b_nits[280:281])))
    b_row1879 = float(np.mean(lum_bt2020(b_nits[1879:1880])))
    b_row1880 = float(np.mean(lum_bt2020(b_nits[1880:1881])))
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    print(f"\n  OVERLAP REPRODUCTION (center y=280..1879 vs HDR master):")
    print(f"  {'Metric':<20} {'Method A (PQ)':<20} {'Method B (Linear)':<20}")
    print(f"  {'-'*60}")
    print(f"  {'MAE (nits)':<20} {np.mean(a_overlap_err):<20.4f} {np.mean(b_overlap_err):<20.4f}")
    print(f"  {'RMSE (nits)':<20} {np.sqrt(np.mean(a_overlap_err**2)):<20.4f} {np.sqrt(np.mean(b_overlap_err**2)):<20.4f}")
    
    # Filter > 0.1 nit for relative error
    valid = hdr_lum > 0.1
    if np.any(valid):
        a_rel = float(np.mean(a_overlap_err[valid] / hdr_lum[valid]))
        b_rel = float(np.mean(b_overlap_err[valid] / hdr_lum[valid]))
        print(f"  {'Relative err':<20} {a_rel*100:<20.2f}% {b_rel*100:<20.2f}%")
    
    print(f"\n  EXTENSION LUMINANCE (nits):")
    print(f"  {'Region':<20} {'Method A mean':<15} {'Method B mean':<15} {'Diff (B-A)':<15}")
    print(f"  {'-'*65}")
    print(f"  {'Top (0..279)':<20} {np.mean(a_lum_top):<15.3f} {np.mean(b_lum_top):<15.3f} {np.mean(b_lum_top)-np.mean(a_lum_top):<15.3f}")
    print(f"  {'Center (280..1879)':<20} {np.mean(a_lum_mid):<15.3f} {np.mean(b_lum_mid):<15.3f} {np.mean(b_lum_mid)-np.mean(a_lum_mid):<15.3f}")
    print(f"  {'Bot (1880..2159)':<20} {np.mean(a_lum_bot):<15.3f} {np.mean(b_lum_bot):<15.3f} {np.mean(b_lum_bot)-np.mean(a_lum_bot):<15.3f}")
    
    print(f"\n  BOUNDARY DISCONTINUITY (nits):")
    print(f"  {'Boundary':<25} {'Method A':<15} {'Method B':<15}")
    print(f"  {'-'*55}")
    print(f"  {'|row279-row280|':<25} {abs(a_row280-a_row279):<15.4f} {abs(b_row280-b_row279):<15.4f}")
    print(f"  {'|row1879-row1880|':<25} {abs(a_row1880-a_row1879):<15.4f} {abs(b_row1880-b_row1879):<15.4f}")
    print(f"  {'row279 (ext)':<25} {a_row279:<15.4f} {b_row279:<15.4f}")
    print(f"  {'row280 (hdr)':<25} {a_row280:<15.4f} {b_row280:<15.4f}")
    print(f"  {'row1879 (hdr)':<25} {a_row1879:<15.4f} {b_row1879:<15.4f}")
    print(f"  {'row1880 (ext)':<25} {a_row1880:<15.4f} {b_row1880:<15.4f}")
    
    return {
        "a_boundary_top": abs(a_row280 - a_row279),
        "b_boundary_top": abs(b_row280 - b_row279),
        "a_boundary_bot": abs(a_row1880 - a_row1879),
        "b_boundary_bot": abs(b_row1880 - b_row1879),
        "a_overlap_mae": float(np.mean(a_overlap_err)),
        "b_overlap_mae": float(np.mean(b_overlap_err)),
        "a_ext_top": float(np.mean(a_lum_top)),
        "b_ext_top": float(np.mean(b_lum_top)),
    }


def main():
    print("=" * 70)
    print("CONTROLLED A/B COMPARISON: PQ-domain vs Linear-light Compositing")
    print("=" * 70)
    
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps
    
    # Prepare curve (same for both methods)
    print("\n--- Preparing shared luminance curve ---")
    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256)
    
    sf = int(round(2700.0 * fps))
    ef = int(round(2730.0 * fps))
    shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
    
    samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                       config=config, n_samples=30, proxy_width=960)
    train, _ = train_validation_split(samples, train_ratio=0.8, seed=42)
    curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=config)
    print(f"  Curve: {len(curve)} points (shared by both methods)")
    
    # Test 5 frames
    test_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]
    all_results = []
    
    for t in test_times:
        hdr_time = t
        om_time = t + 1167 / fps
        
        hdr_frame = extract_frame(HDR_PATH, hdr_time, 3840, 1600)
        om_frame = extract_frame(OM_PATH, om_time, 3840, 2160)
        
        if hdr_frame is None or om_frame is None:
            print(f"  Frame {t:.0f}s: extraction failed")
            continue
        
        result = analyze_frame(hdr_frame, om_frame, curve, f"Frame at {t:.0f}s (HDR)")
        all_results.append(result)
    
    # Summary
    if all_results:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE SUMMARY ({len(all_results)} frames)")
        print(f"{'='*70}")
        
        a_top_jumps = [r["a_boundary_top"] for r in all_results]
        b_top_jumps = [r["b_boundary_top"] for r in all_results]
        a_bot_jumps = [r["a_boundary_bot"] for r in all_results]
        b_bot_jumps = [r["b_boundary_bot"] for r in all_results]
        a_maes = [r["a_overlap_mae"] for r in all_results]
        b_maes = [r["b_overlap_mae"] for r in all_results]
        
        print(f"\n  Top boundary jump (nits):  A mean={np.mean(a_top_jumps):.4f}  B mean={np.mean(b_top_jumps):.4f}")
        print(f"  Bot boundary jump (nits):  A mean={np.mean(a_bot_jumps):.4f}  B mean={np.mean(b_bot_jumps):.4f}")
        print(f"  Overlap MAE (nits):        A mean={np.mean(a_maes):.4f}  B mean={np.mean(b_maes):.4f}")
        
        print(f"\n  ANSWERS:")
        print(f"  1. Does B improve boundary continuity?")
        top_improvement = np.mean(a_top_jumps) / np.mean(b_top_jumps) if np.mean(b_top_jumps) > 0 else 0
        print(f"     Top: A={np.mean(a_top_jumps):.3f} vs B={np.mean(b_top_jumps):.3f} nits (ratio {top_improvement:.1f}x)")
        bot_improvement = np.mean(a_bot_jumps) / np.mean(b_bot_jumps) if np.mean(b_bot_jumps) > 0 else 0
        print(f"     Bot: A={np.mean(a_bot_jumps):.3f} vs B={np.mean(b_bot_jumps):.3f} nits (ratio {bot_improvement:.1f}x)")
        
        print(f"\n  2. Does B improve overlap reproduction?")
        print(f"     A MAE={np.mean(a_maes):.4f} vs B MAE={np.mean(b_maes):.4f} nits")
        if np.mean(b_maes) < np.mean(a_maes):
            print(f"     YES — B is better by {(1-np.mean(b_maes)/np.mean(a_maes))*100:.1f}%")
        else:
            print(f"     NO — A is equal or better")
        
        print(f"\n  3. Does B change extension luminance?")
        a_tops = [r["a_ext_top"] for r in all_results]
        b_tops = [r["b_ext_top"] for r in all_results]
        print(f"     Extension top: A={np.mean(a_tops):.3f} vs B={np.mean(b_tops):.3f} nits")
        print(f"     Difference: {np.mean(b_tops)-np.mean(a_tops):.3f} nits")
        
        print(f"\n  4. Source of A/B difference?")
        print(f"     Both use IDENTICAL curve and apply_curve_to_rgb().")
        print(f"     Difference comes from: compositing domain (PQ vs linear).")
        print(f"     In A: HDR PQ placed directly → output stays PQ-encoded")
        print(f"     In B: HDR linearized → placed in linear canvas → re-encoded PQ")
        print(f"     The HDR center goes through EXTRA PQ decode→encode in B")
        print(f"     Extension in B is: linear→(stays linear for compositing)→PQ encode")
        print(f"     Extension in A is: linear→PQ encode→(stays PQ for compositing)")
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
