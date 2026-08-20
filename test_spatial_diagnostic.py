"""Spatial/chroma diagnostic for luminance mapping error.

Tests:
1. Row-band scatter (4 bands)
2. Same SDR Y, different HDR Y by region
3. Local curves per band vs global
4. Color/chroma dependency
5. Error map structure
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import linearize
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")

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


def collect_paired_data(hdr_frame, om_frame):
    """Collect paired pixel data from overlap region.
    
    Returns arrays with per-pixel: sdr_lum, hdr_nits, y_position, sdr_r, sdr_g, sdr_b
    Uses proxy (downscale for manageable size).
    """
    from scipy.ndimage import zoom
    
    # Work at 960px width for speed
    proxy_w = 960
    hdr_h, hdr_w = hdr_frame.shape[:2]
    om_h, om_w = om_frame.shape[:2]
    
    scale_h = proxy_w / hdr_w
    hdr_proxy = zoom(hdr_frame, (scale_h, proxy_w/hdr_w, 1.0), order=1)
    
    scale_om = proxy_w / om_w
    om_proxy = zoom(om_frame, (scale_om, proxy_w/om_w, 1.0), order=1)
    
    # OM overlap at proxy scale: rows 280*scale_om to 1880*scale_om
    oy1 = int(round(280 * scale_om))
    oy2 = int(round(1880 * scale_om))
    om_overlap = om_proxy[oy1:oy2, :, :]
    
    # HDR proxy
    ph = hdr_proxy.shape[0]
    pw = hdr_proxy.shape[1]
    
    # Ensure same height
    min_h = min(ph, om_overlap.shape[0])
    min_w = min(pw, om_overlap.shape[1])
    hdr_crop = hdr_proxy[:min_h, :min_w, :]
    om_crop = om_overlap[:min_h, :min_w, :]
    
    # Linearize
    sdr_linear = linearize(om_crop, "bt709")  # [0,1]
    hdr_linear = linearize(hdr_crop, "smpte2084", peak_nits=PEAK_NITS)  # [0,1] normalized
    
    # Luminance
    sdr_lum = 0.2126 * sdr_linear[..., 0] + 0.7152 * sdr_linear[..., 1] + 0.0722 * sdr_linear[..., 2]
    hdr_lum = 0.2627 * hdr_linear[..., 0] + 0.6780 * hdr_linear[..., 1] + 0.0593 * hdr_linear[..., 2]
    hdr_nits = hdr_lum * PEAK_NITS
    
    # Y position array (row index in HDR space, 0=top, 1599=bottom at full res)
    y_positions = np.arange(min_h).reshape(-1, 1) * (1600.0 / min_h)
    y_positions = np.broadcast_to(y_positions, (min_h, min_w))
    
    # Flatten
    sdr_flat = sdr_lum.flatten()
    hdr_flat = hdr_nits.flatten()
    y_flat = y_positions.flatten()
    sdr_r = sdr_linear[..., 0].flatten()
    sdr_g = sdr_linear[..., 1].flatten()
    sdr_b = sdr_linear[..., 2].flatten()
    
    # Filter valid (reject deep black)
    valid = (sdr_flat > 0.001) & (hdr_flat > 0.01) & np.isfinite(sdr_flat) & np.isfinite(hdr_flat)
    
    return (sdr_flat[valid], hdr_flat[valid], y_flat[valid],
            sdr_r[valid], sdr_g[valid], sdr_b[valid])


def main():
    print("=" * 70)
    print("SPATIAL/CHROMA DIAGNOSTIC — Luminance Mapping Error Analysis")
    print("=" * 70)

    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps

    # Prepare global curve
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
    global_curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=config)
    print(f"  Global curve: {len(global_curve)} points")

    # Collect full paired data from 5 frames
    test_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]
    all_sdr, all_hdr, all_y, all_r, all_g, all_b = [], [], [], [], [], []

    print("\n--- Collecting paired pixel data ---")
    for t in test_times:
        hdr_frame = extract_frame(HDR_PATH, t, 3840, 1600)
        om_frame = extract_frame(OM_PATH, t + 1167/fps, 3840, 2160)
        if hdr_frame is None or om_frame is None:
            print(f"  {t:.0f}s: failed")
            continue
        sdr, hdr, y, r, g, b = collect_paired_data(hdr_frame, om_frame)
        all_sdr.append(sdr)
        all_hdr.append(hdr)
        all_y.append(y)
        all_r.append(r)
        all_g.append(g)
        all_b.append(b)
        print(f"  {t:.0f}s: {len(sdr):,} valid pixels")

    sdr = np.concatenate(all_sdr)
    hdr = np.concatenate(all_hdr)
    y_pos = np.concatenate(all_y)
    sdr_r = np.concatenate(all_r)
    sdr_g = np.concatenate(all_g)
    sdr_b = np.concatenate(all_b)
    print(f"  Total: {len(sdr):,} paired pixels")

    # Predicted HDR from global curve (need SDR linear → HDR normalized → nits)
    predicted_norm = apply_luminance_curve(sdr, global_curve)
    predicted_nits = predicted_norm * PEAK_NITS
    error = predicted_nits - hdr  # signed error

    # ===== TEST 1: ROW-BAND SCATTER =====
    print("\n\n" + "=" * 70)
    print("TEST 1: ROW-BAND ANALYSIS")
    print("=" * 70)

    bands = [
        ("TOP (0-199)", 0, 200),
        ("UPPER-MID (200-599)", 200, 600),
        ("LOWER-MID (600-1199)", 600, 1200),
        ("BOTTOM (1200-1599)", 1200, 1600),
    ]

    print(f"\n  {'Band':<22} {'N':<10} {'SDR P50':<9} {'HDR P50':<9} {'Pred P50':<9} "
          f"{'MAE':<8} {'RMSE':<8} {'MeanErr':<9} {'RelErr':<8} {'Corr':<6}")
    print(f"  {'-'*100}")

    for name, y_lo, y_hi in bands:
        mask = (y_pos >= y_lo) & (y_pos < y_hi)
        n = int(np.sum(mask))
        if n < 100:
            print(f"  {name:<22} {n:<10} (insufficient data)")
            continue
        s = sdr[mask]
        h = hdr[mask]
        p = predicted_nits[mask]
        e = error[mask]
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e**2)))
        mean_err = float(np.mean(e))
        rel_err = float(np.mean(np.abs(e) / (h + 0.1)))
        corr = float(np.corrcoef(p, h)[0, 1]) if np.std(p) > 0 and np.std(h) > 0 else 0
        print(f"  {name:<22} {n:<10,} {np.median(s):<9.4f} {np.median(h):<9.2f} {np.median(p):<9.2f} "
              f"{mae:<8.3f} {rmse:<8.3f} {mean_err:<9.3f} {rel_err:<8.3f} {corr:<6.3f}")

    # ===== TEST 2: SAME SDR, DIFFERENT Y =====
    print("\n\n" + "=" * 70)
    print("TEST 2: SAME SDR LUMINANCE, DIFFERENT VERTICAL POSITION")
    print("=" * 70)

    # Find SDR luminance bins and compare HDR across Y regions
    sdr_bins = [(0.01, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40)]
    y_regions = [("TOP", 0, 200), ("CENTER", 600, 1000), ("BOTTOM", 1200, 1600)]

    print(f"\n  For pixels with similar SDR luminance, HDR luminance by vertical region:")
    print(f"  {'SDR range':<15}", end="")
    for rname, _, _ in y_regions:
        print(f"  {rname+' (nits)':<15}", end="")
    print(f"  {'TOP/CENTER':<12}")
    print(f"  {'-'*75}")

    for sdr_lo, sdr_hi in sdr_bins:
        sdr_mask = (sdr >= sdr_lo) & (sdr < sdr_hi)
        if np.sum(sdr_mask) < 100:
            continue
        print(f"  [{sdr_lo:.2f},{sdr_hi:.2f})", end="")
        medians = []
        for rname, y_lo, y_hi in y_regions:
            combined = sdr_mask & (y_pos >= y_lo) & (y_pos < y_hi)
            n = int(np.sum(combined))
            if n > 50:
                med = float(np.median(hdr[combined]))
                medians.append(med)
                print(f"  {med:<15.3f}", end="")
            else:
                medians.append(None)
                print(f"  {'N/A':<15}", end="")
        # Ratio TOP/CENTER
        if medians[0] is not None and medians[1] is not None and medians[1] > 0:
            ratio = medians[0] / medians[1]
            print(f"  {ratio:<12.3f}", end="")
        print()

    # ===== TEST 3: LOCAL CURVES =====
    print("\n\n" + "=" * 70)
    print("TEST 3: LOCAL CURVES vs GLOBAL CURVE")
    print("=" * 70)

    # HDR normalized for curve fitting
    hdr_norm = hdr / PEAK_NITS

    print(f"\n  {'Band':<22} {'Global MAE':<12} {'Local MAE':<12} {'Improvement':<12}")
    print(f"  {'-'*58}")

    for name, y_lo, y_hi in bands:
        mask = (y_pos >= y_lo) & (y_pos < y_hi)
        n = int(np.sum(mask))
        if n < 500:
            print(f"  {name:<22} insufficient data")
            continue
        s_band = sdr[mask]
        h_band = hdr_norm[mask]

        # Split local data 80/20
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        s_train, s_val = s_band[idx[:split]], s_band[idx[split:]]
        h_train, h_val = h_band[idx[:split]], h_band[idx[split:]]

        # Fit local curve
        local_curve = estimate_luminance_curve(s_train, h_train, config=config)

        # Evaluate global on validation
        global_pred = apply_luminance_curve(s_val, global_curve)
        global_mae = float(np.mean(np.abs(global_pred - h_val))) * PEAK_NITS

        # Evaluate local on validation
        local_pred = apply_luminance_curve(s_val, local_curve)
        local_mae = float(np.mean(np.abs(local_pred - h_val))) * PEAK_NITS

        improvement = (1 - local_mae / global_mae) * 100 if global_mae > 0 else 0
        print(f"  {name:<22} {global_mae:<12.4f} {local_mae:<12.4f} {improvement:<12.1f}%")

    # ===== TEST 4: COLOR DEPENDENCY =====
    print("\n\n" + "=" * 70)
    print("TEST 4: COLOR/CHROMA DEPENDENCY OF ERROR")
    print("=" * 70)

    # Chroma: R-Y and B-Y
    sdr_lum_for_chroma = 0.2126 * sdr_r + 0.7152 * sdr_g + 0.0722 * sdr_b
    chroma_r = sdr_r - sdr_lum_for_chroma
    chroma_b = sdr_b - sdr_lum_for_chroma
    saturation = np.sqrt(chroma_r**2 + chroma_b**2)

    abs_error = np.abs(error)

    # Correlations
    from scipy.stats import spearmanr
    corr_sdr_lum, _ = spearmanr(sdr[:10000], abs_error[:10000])
    corr_y_pos, _ = spearmanr(y_pos[:10000], abs_error[:10000])
    corr_sat, _ = spearmanr(saturation[:10000], abs_error[:10000])
    corr_chroma_r, _ = spearmanr(np.abs(chroma_r[:10000]), abs_error[:10000])

    print(f"\n  Spearman correlation of |error| with:")
    print(f"    SDR luminance:    {corr_sdr_lum:.4f}")
    print(f"    Y position:       {corr_y_pos:.4f}")
    print(f"    SDR saturation:   {corr_sat:.4f}")
    print(f"    SDR |chroma_R-Y|: {corr_chroma_r:.4f}")
    print(f"\n  Interpretation:")
    print(f"    Higher |correlation| = stronger predictor of error")

    # ===== TEST 5: ERROR MAP STRUCTURE =====
    print("\n\n" + "=" * 70)
    print("TEST 5: ERROR MAP STRUCTURE")
    print("=" * 70)

    # Compute mean signed error per Y-band of 20 rows
    n_bands_fine = 80  # 1600/20 = 80 bands
    band_size = 1600.0 / n_bands_fine
    band_errors = []
    for i in range(n_bands_fine):
        y_lo = i * band_size
        y_hi = (i + 1) * band_size
        mask = (y_pos >= y_lo) & (y_pos < y_hi)
        if np.sum(mask) > 50:
            band_errors.append(float(np.mean(error[mask])))
        else:
            band_errors.append(0.0)

    # Print profile (every 10th band)
    print(f"\n  Mean signed error by row band (20-row bands, every 4th shown):")
    print(f"  {'Band':<8} {'Y range':<15} {'Mean error (nits)':<20}")
    print(f"  {'-'*43}")
    for i in range(0, n_bands_fine, 4):
        y_lo = int(i * band_size)
        y_hi = int((i + 1) * band_size)
        print(f"  {i:<8} [{y_lo:>4}-{y_hi:>4}]     {band_errors[i]:>+.4f}")

    # Structure analysis
    top_10_mean = float(np.mean(band_errors[:10]))
    mid_mean = float(np.mean(band_errors[30:50]))
    bot_10_mean = float(np.mean(band_errors[70:]))

    print(f"\n  Summary:")
    print(f"    Top 10 bands (rows 0-199):    mean signed error = {top_10_mean:+.4f} nits")
    print(f"    Middle 20 bands (rows 600-999): mean signed error = {mid_mean:+.4f} nits")
    print(f"    Bottom 10 bands (rows 1400-1599): mean signed error = {bot_10_mean:+.4f} nits")

    if top_10_mean < -0.5:
        print(f"    → TOP region: UNDERPREDICTED (curve gives less than HDR master)")
    elif top_10_mean > 0.5:
        print(f"    → TOP region: OVERPREDICTED")
    else:
        print(f"    → TOP region: reasonably matched")

    # ===== FINAL VERDICT =====
    print("\n\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Determine primary error driver
    drivers = [
        ("SDR luminance", abs(corr_sdr_lum)),
        ("Y position (spatial)", abs(corr_y_pos)),
        ("Saturation/chroma", abs(corr_sat)),
    ]
    drivers.sort(key=lambda x: -x[1])
    print(f"\n  Error drivers (ranked by correlation):")
    for name, val in drivers:
        print(f"    {name:<25} |rho| = {val:.4f}")

    print(f"\n  Local vs global curve improvement:")
    print(f"    (see TEST 3 table above)")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
