"""P2.9: Visual artifact source audit.

Tests 1-6: pre/post encode comparison, color boundary analysis,
jacket ROI diagnostics, ratio distribution, source vs transform.
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
from auto_openmatte.core.transfer_functions import linearize, pq_eotf
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.processing.transform import apply_shot_transform
from auto_openmatte.pipeline.compose import composite_extend

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_PATH = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output\BR2049_P28_Log_5s.mkv")
PEAK = 10000.0


def extract_frame(path, time_s, w, h):
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{time_s:.6f}",
           "-i", str(path), "-frames:v", "1", "-pix_fmt", "rgb48le",
           "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or len(r.stdout) < w*h*6:
        return None
    return np.frombuffer(r.stdout[:w*h*6], dtype=np.uint16).reshape(h, w, 3).astype(np.float64) / 65535.0


def main():
    print("=" * 70)
    print("P2.9: VISUAL ARTIFACT SOURCE AUDIT")
    print("=" * 70)

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

    # Get the curve (same as used for the 5s render)
    print("\n--- Preparing curve ---")
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
    split = int(len(sdr_all) * 0.8)
    curve = estimate_luminance_curve(sdr_all[idx[:split]], hdr_all[idx[:split]], config=config)
    print(f"  Curve: {len(curve)} points")

    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.99)
    ext_mask = generate_extension_mask(om_source, geom, feather_width=4)

    # Use frame at ~2.5s into the 5s render (render starts at 2710s HDR)
    frame_time_hdr = 2712.5
    frame_time_om = frame_time_hdr + 1167 / fps
    output_time = 2.5  # 2.5s into output

    print(f"\n--- Extracting frames (HDR t={frame_time_hdr:.1f}s) ---")
    hdr_frame = extract_frame(HDR_PATH, frame_time_hdr, 3840, 1600)
    om_frame = extract_frame(OM_PATH, frame_time_om, 3840, 2160)

    if hdr_frame is None or om_frame is None:
        print("ERROR: frame extraction failed")
        return

    # TEST 1: Generate composite in Python (pre-encode)
    print("\n--- TEST 1: Pre-encode composite ---")
    composite_pre = composite_extend(hdr_frame, om_frame, transform, geom, ext_mask,
                                     sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=PEAK)
    print(f"  Composite shape: {composite_pre.shape}, range: [{composite_pre.min():.4f}, {composite_pre.max():.4f}]")

    # Extract same frame from encoded output
    print("\n--- TEST 1: Post-encode frame ---")
    post_frame = extract_frame(OUTPUT_PATH, output_time, 3840, 2160)
    if post_frame is None:
        print("  WARNING: Could not extract from output. Using pre-encode only.")
        post_frame = composite_pre  # Fallback

    # TEST 2 is deferred (would need lossless encode)

    # ===== TEST 3: COLOR BOUNDARY ANALYSIS =====
    print(f"\n{'='*70}")
    print("TEST 3: COLOR BOUNDARY ANALYSIS (pre-encode)")
    print(f"{'='*70}")

    # Analyze rows 275-284 of the pre-encode composite
    for row in [278, 279, 280, 281, 282]:
        r = composite_pre[row, :, :]
        y = 0.2627*r[:,0] + 0.6780*r[:,1] + 0.0593*r[:,2]
        y_nits = float(np.median(pq_eotf(y)))
        r_med = float(np.median(r[:,0]))
        g_med = float(np.median(r[:,1]))
        b_med = float(np.median(r[:,2]))
        sat = float(np.median(np.sqrt((r[:,0]-y)**2 + (r[:,2]-y)**2)))
        region = "EXT" if row < 280 else "HDR"
        print(f"  Row {row} [{region}]: Y={y_nits:.3f}nits, PQ_R={r_med:.4f}, G={g_med:.4f}, B={b_med:.4f}, sat={sat:.4f}")

    # Bottom boundary
    print()
    for row in [1878, 1879, 1880, 1881, 1882]:
        r = composite_pre[row, :, :]
        y = 0.2627*r[:,0] + 0.6780*r[:,1] + 0.0593*r[:,2]
        y_nits = float(np.median(pq_eotf(y)))
        r_med = float(np.median(r[:,0]))
        g_med = float(np.median(r[:,1]))
        b_med = float(np.median(r[:,2]))
        sat = float(np.median(np.sqrt((r[:,0]-y)**2 + (r[:,2]-y)**2)))
        region = "HDR" if row < 1880 else "EXT"
        print(f"  Row {row} [{region}]: Y={y_nits:.3f}nits, PQ_R={r_med:.4f}, G={g_med:.4f}, B={b_med:.4f}, sat={sat:.4f}")

    # ===== TEST 4: JACKET ROI (bottom extension, dark area) =====
    print(f"\n{'='*70}")
    print("TEST 4: JACKET/DARK AREA ROI (bottom extension)")
    print(f"{'='*70}")

    # ROI in bottom extension: rows 1900-2050, cols 1500-2500 (approximate dark jacket area)
    roi_y1, roi_y2, roi_x1, roi_x2 = 1900, 2050, 1500, 2500
    roi_pre = composite_pre[roi_y1:roi_y2, roi_x1:roi_x2, :]

    y_roi = 0.2627*roi_pre[...,0] + 0.6780*roi_pre[...,1] + 0.0593*roi_pre[...,2]
    y_roi_nits = pq_eotf(y_roi)

    print(f"  ROI: rows {roi_y1}-{roi_y2}, cols {roi_x1}-{roi_x2}")
    print(f"  PQ signal: R=[{roi_pre[...,0].min():.4f},{roi_pre[...,0].max():.4f}], "
          f"G=[{roi_pre[...,1].min():.4f},{roi_pre[...,1].max():.4f}], "
          f"B=[{roi_pre[...,2].min():.4f},{roi_pre[...,2].max():.4f}]")
    print(f"  Luminance (nits): min={y_roi_nits.min():.3f}, max={y_roi_nits.max():.3f}, "
          f"median={np.median(y_roi_nits):.3f}, P95={np.percentile(y_roi_nits,95):.3f}")

    # Check saturation in ROI
    sat_roi = np.sqrt((roi_pre[...,0]-y_roi)**2 + (roi_pre[...,2]-y_roi)**2)
    print(f"  Saturation: min={sat_roi.min():.4f}, max={sat_roi.max():.4f}, "
          f"median={np.median(sat_roi):.4f}")

    # ===== TEST 5: RATIO DISTRIBUTION IN ROI =====
    print(f"\n{'='*70}")
    print("TEST 5: RATIO DISTRIBUTION IN JACKET ROI")
    print(f"{'='*70}")

    # Get the OM source ROI and compute what ratio was applied
    om_roi = om_frame[roi_y1:roi_y2, roi_x1:roi_x2, :]
    # Linearize OM per-channel
    om_lin_r = linearize(om_roi[..., 0], "bt709")
    om_lin_g = linearize(om_roi[..., 1], "bt709")
    om_lin_b = linearize(om_roi[..., 2], "bt709")
    # BT.709→BT.2020
    from auto_openmatte.processing.transform import _M_709_TO_2020, _LUM_R_2020, _LUM_G_2020, _LUM_B_2020
    om_lin_709 = np.stack([om_lin_r, om_lin_g, om_lin_b], axis=-1)
    shape = om_lin_709.shape
    om_lin_2020 = om_lin_709.reshape(-1, 3) @ _M_709_TO_2020.T
    om_lin_2020 = om_lin_2020.reshape(shape)
    om_lin_2020 = np.maximum(om_lin_2020, 0.0)

    lum_sdr = _LUM_R_2020 * om_lin_2020[...,0] + _LUM_G_2020 * om_lin_2020[...,1] + _LUM_B_2020 * om_lin_2020[...,2]
    lum_mapped = apply_luminance_curve(lum_sdr, curve)

    safe = lum_sdr > 1e-6
    ratio = np.ones_like(lum_sdr)
    np.divide(lum_mapped, lum_sdr, out=ratio, where=safe)

    print(f"  SDR lum (linear BT.2020): min={lum_sdr.min():.6f}, median={np.median(lum_sdr):.6f}, max={lum_sdr.max():.6f}")
    print(f"  Mapped lum (normalized):  min={lum_mapped.min():.6f}, median={np.median(lum_mapped):.6f}, max={lum_mapped.max():.6f}")
    print(f"\n  Ratio distribution:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"    P{p}: {np.percentile(ratio, p):.4f}")
    print(f"    Max: {ratio.max():.4f}")
    print(f"    Std: {np.std(ratio):.4f}")
    print(f"    % where lum_sdr < 1e-6: {(~safe).sum() / safe.size * 100:.2f}%")

    # ===== TEST 6: SOURCE SDR vs TRANSFORMED =====
    print(f"\n{'='*70}")
    print("TEST 6: SOURCE SDR vs TRANSFORMED (jacket ROI)")
    print(f"{'='*70}")

    # Original SDR signal in ROI
    print(f"  Original SDR signal (BT.709 nonlinear):")
    print(f"    R: [{om_roi[...,0].min():.4f}, {om_roi[...,0].max():.4f}], median={np.median(om_roi[...,0]):.4f}")
    print(f"    G: [{om_roi[...,1].min():.4f}, {om_roi[...,1].max():.4f}], median={np.median(om_roi[...,1]):.4f}")
    print(f"    B: [{om_roi[...,2].min():.4f}, {om_roi[...,2].max():.4f}], median={np.median(om_roi[...,2]):.4f}")

    # Transformed (PQ-encoded) in ROI — from composite
    print(f"\n  Transformed PQ output (BT.2020 PQ signal):")
    print(f"    R: [{roi_pre[...,0].min():.4f}, {roi_pre[...,0].max():.4f}], median={np.median(roi_pre[...,0]):.4f}")
    print(f"    G: [{roi_pre[...,1].min():.4f}, {roi_pre[...,1].max():.4f}], median={np.median(roi_pre[...,1]):.4f}")
    print(f"    B: [{roi_pre[...,2].min():.4f}, {roi_pre[...,2].max():.4f}], median={np.median(roi_pre[...,2]):.4f}")

    # Check channel variance (indicator of banding/quantization artifacts)
    r_var = float(np.std(roi_pre[...,0]))
    g_var = float(np.std(roi_pre[...,1]))
    b_var = float(np.std(roi_pre[...,2]))
    print(f"\n  Channel std (PQ domain):")
    print(f"    R std: {r_var:.4f}, G std: {g_var:.4f}, B std: {b_var:.4f}")

    # Compare with SDR std (to check if transform increases local contrast)
    r_var_sdr = float(np.std(om_roi[...,0]))
    g_var_sdr = float(np.std(om_roi[...,1]))
    b_var_sdr = float(np.std(om_roi[...,2]))
    print(f"    SDR R std: {r_var_sdr:.4f}, SDR G std: {g_var_sdr:.4f}, SDR B std: {b_var_sdr:.4f}")
    print(f"    Ratio (output/source std): R={r_var/r_var_sdr:.2f}, G={g_var/g_var_sdr:.2f}, B={b_var/b_var_sdr:.2f}")

    # ===== TEST 1 continued: PRE vs POST encode comparison =====
    if post_frame is not composite_pre:
        print(f"\n{'='*70}")
        print("TEST 1: PRE-ENCODE vs POST-ENCODE (jacket ROI)")
        print(f"{'='*70}")
        post_roi = post_frame[roi_y1:roi_y2, roi_x1:roi_x2, :]
        diff_roi = np.abs(roi_pre - post_roi)
        print(f"  Max |pre-post| per channel:")
        print(f"    R: {diff_roi[...,0].max():.6f}")
        print(f"    G: {diff_roi[...,1].max():.6f}")
        print(f"    B: {diff_roi[...,2].max():.6f}")
        print(f"  Mean |pre-post|: {diff_roi.mean():.6f}")
        if diff_roi.mean() > 0.01:
            print(f"  → SIGNIFICANT encoding difference in dark region")
        else:
            print(f"  → Encoding difference is small")

    print(f"\n{'='*70}")
    print("P2.9 AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
