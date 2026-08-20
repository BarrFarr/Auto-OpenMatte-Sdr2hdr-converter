"""P0.2 RE-BASELINE: FEAT-007 with corrected SDR+HDR ground truth."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
PEAK = 10000.0


def main():
    print("=" * 70)
    print("P0.2 RE-BASELINE: FEAT-007 with corrected ground truth (SDR+HDR)")
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
                         low_percentile=1.0, high_percentile=99.0)

    # Sample from 3 segments
    segments = [(2700.0, 2730.0, "45:00"), (4500.0, 4530.0, "1:15:00"), (2220.0, 2250.0, "37:00")]
    all_samples = []

    t0 = time.perf_counter()
    for start, end, label in segments:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                           config=config, n_samples=30, proxy_width=960)
        all_samples.append(samples)
        print(f"  {label}: {samples.n_valid_pairs:,} pairs")
    t1 = time.perf_counter()

    sdr_all = np.concatenate([s.sdr_luminance for s in all_samples])
    hdr_all = np.concatenate([s.hdr_luminance for s in all_samples])
    print(f"\n  Total: {len(sdr_all):,} pairs, sampling: {t1-t0:.1f}s")

    # Train/val split
    rng = np.random.default_rng(42)
    n = len(sdr_all)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    sdr_train, sdr_val = sdr_all[idx[:split]], sdr_all[idx[split:]]
    hdr_train, hdr_val = hdr_all[idx[:split]], hdr_all[idx[split:]]

    # Fit curve
    curve = estimate_luminance_curve(sdr_train, hdr_train, config=config)
    print(f"  Curve: {len(curve)} control points")

    # Evaluate
    pred_val = apply_luminance_curve(sdr_val, curve) * PEAK
    hdr_val_nits = hdr_val * PEAK
    err = pred_val - hdr_val_nits
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err**2)))
    conf = float(np.corrcoef(pred_val, hdr_val_nits)[0, 1])

    def range_mae(lo_p, hi_p):
        lo = np.percentile(sdr_val, lo_p)
        hi = np.percentile(sdr_val, hi_p) if hi_p < 100 else sdr_val.max() + 1
        m = (sdr_val >= lo) & (sdr_val < hi)
        return float(np.mean(abs_err[m])) if np.sum(m) > 0 else 0.0

    p0_p50 = range_mae(0, 50)
    p50_p90 = range_mae(50, 90)
    p90_p99 = range_mae(90, 99)
    p99_p100 = range_mae(99, 100)

    # Train error
    pred_train = apply_luminance_curve(sdr_train, curve) * PEAK
    hdr_train_nits = hdr_train * PEAK
    train_mae = float(np.mean(np.abs(pred_train - hdr_train_nits)))

    # Print results
    print(f"\n{'='*70}")
    print("NEW RESULTS (P0.2: both SDR + HDR corrected)")
    print(f"{'='*70}")
    print(f"  Validation MAE:    {mae:.4f} nits")
    print(f"  Validation RMSE:   {rmse:.4f} nits")
    print(f"  Train MAE:         {train_mae:.4f} nits")
    print(f"  Confidence:        {conf:.4f}")
    print(f"  Mean signed error: {np.mean(err):.4f} nits")
    print(f"  P0-P50 MAE:        {p0_p50:.4f}")
    print(f"  P50-P90 MAE:       {p50_p90:.4f}")
    print(f"  P90-P99 MAE:       {p90_p99:.4f}")
    print(f"  P99-P100 MAE:      {p99_p100:.4f}")

    # Luminance percentiles
    print(f"\n  SDR luminance percentiles (linear BT.709):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p}: {np.percentile(sdr_val, p):.6f}")
    print(f"\n  HDR luminance percentiles (nits):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p}: {np.percentile(hdr_val_nits, p):.3f}")
    print(f"    Max: {np.max(hdr_val_nits):.3f}")

    # Saturation/error correlation
    # Can't compute directly without RGB in validation, but can check luminance
    from scipy.stats import spearmanr
    n_sub = min(100000, len(sdr_val))
    sub_idx = rng.choice(len(sdr_val), n_sub, replace=False)
    rho_lum, _ = spearmanr(hdr_val_nits[sub_idx], abs_err[sub_idx])
    print(f"\n  Spearman |error| vs HDR luminance: rho = {rho_lum:.4f}")

    # Comparison table
    print(f"\n{'='*70}")
    print("COMPARISON: OLD (original) → P0-only (HDR fix) → P0.2 (SDR+HDR fix)")
    print(f"{'='*70}")
    print(f"  {'Metric':<18} {'Original':<12} {'P0 HDR-only':<12} {'P0.2 Both':<12}")
    print(f"  {'-'*54}")
    print(f"  {'MAE (nits)':<18} {'0.6475':<12} {'0.8700':<12} {mae:<12.4f}")
    print(f"  {'RMSE (nits)':<18} {'2.3526':<12} {'2.4264':<12} {rmse:<12.4f}")
    print(f"  {'Confidence':<18} {'0.9929':<12} {'0.9928':<12} {conf:<12.4f}")
    print(f"  {'P0-P50':<18} {'0.1807':<12} {'0.2234':<12} {p0_p50:<12.4f}")
    print(f"  {'P50-P90':<18} {'0.7004':<12} {'1.1005':<12} {p50_p90:<12.4f}")
    print(f"  {'P90-P99':<18} {'1.4264':<12} {'1.9898':<12} {p90_p99:<12.4f}")
    print(f"  {'P99-P100':<18} {'14.8614':<12} {'13.8959':<12} {p99_p100:<12.4f}")

    print(f"\n{'='*70}")
    print("P0.2 RE-BASELINE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
