"""P0 verification + FEAT-007 re-run with corrected HDR ground truth.

1. Verify the fix (new vs old method)
2. Re-run FEAT-007 baseline with corrected sampling
3. Compare OLD vs NEW results
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import linearize
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.sampling import (
    sample_overlap_luminance, train_validation_split, compute_diagnostics
)

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
PEAK = 10000.0


def main():
    print("=" * 70)
    print("P0 VERIFICATION + FEAT-007 RE-RUN (corrected HDR ground truth)")
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

    # ===== STEP 1: Sample with NEW corrected method =====
    print("\n--- Step 1: Sampling with corrected per-channel PQ EOTF ---")
    segments = [(2700.0, 2730.0), (4500.0, 4530.0), (2220.0, 2250.0)]
    all_samples = []

    t0 = time.perf_counter()
    for start, end in segments:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                           config=config, n_samples=30, proxy_width=960)
        all_samples.append(samples)
        print(f"  Segment {start:.0f}-{end:.0f}s: {samples.n_valid_pairs:,} pairs "
              f"(rejected: black={samples.n_rejected_black:,}, clip={samples.n_rejected_clipped:,})")
    t1 = time.perf_counter()

    # Combine
    sdr_all = np.concatenate([s.sdr_luminance for s in all_samples])
    hdr_all = np.concatenate([s.hdr_luminance for s in all_samples])
    print(f"\n  Total: {len(sdr_all):,} valid pairs, sampling time: {t1-t0:.1f}s")

    # ===== STEP 2: Fit NEW curve =====
    print("\n--- Step 2: Fitting luminance curve (NEW ground truth) ---")
    rng = np.random.default_rng(42)
    n = len(sdr_all)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    sdr_train, sdr_val = sdr_all[idx[:split]], sdr_all[idx[split:]]
    hdr_train, hdr_val = hdr_all[idx[:split]], hdr_all[idx[split:]]

    curve_new = estimate_luminance_curve(sdr_train, hdr_train, config=config)
    print(f"  Curve: {len(curve_new)} points")

    # Evaluate NEW
    pred_new = apply_luminance_curve(sdr_val, curve_new) * PEAK
    hdr_val_nits = hdr_val * PEAK
    err_new = pred_new - hdr_val_nits
    mae_new = float(np.mean(np.abs(err_new)))
    rmse_new = float(np.sqrt(np.mean(err_new**2)))
    conf_new = float(np.corrcoef(pred_new, hdr_val_nits)[0, 1])

    # Per-range errors
    def range_mae(sdr_v, err_v, lo_p, hi_p):
        lo = np.percentile(sdr_v, lo_p)
        hi = np.percentile(sdr_v, hi_p) if hi_p < 100 else sdr_v.max() + 1
        mask = (sdr_v >= lo) & (sdr_v < hi)
        return float(np.mean(np.abs(err_v[mask]))) if np.sum(mask) > 0 else 0.0

    # ===== STEP 3: Report OLD vs NEW =====
    # OLD values from previous FEAT-007 test (before P0 fix)
    old_mae = 0.6475  # from test_curve_resolution.py
    old_rmse = 2.3526
    old_conf = 0.9929
    old_p90_99 = 1.4264
    old_p99_100 = 14.8614

    new_p0_p50 = range_mae(sdr_val, err_new, 0, 50)
    new_p50_p90 = range_mae(sdr_val, err_new, 50, 90)
    new_p90_p99 = range_mae(sdr_val, err_new, 90, 99)
    new_p99_p100 = range_mae(sdr_val, err_new, 99, 100)

    print(f"\n{'='*70}")
    print("COMPARISON: OLD (gray16le) vs NEW (rgb48le per-channel EOTF)")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<20} {'OLD':<15} {'NEW':<15} {'Change':<15}")
    print(f"  {'-'*60}")
    print(f"  {'MAE (nits)':<20} {old_mae:<15.4f} {mae_new:<15.4f} {(mae_new-old_mae)/old_mae*100:+.1f}%")
    print(f"  {'RMSE (nits)':<20} {old_rmse:<15.4f} {rmse_new:<15.4f} {(rmse_new-old_rmse)/old_rmse*100:+.1f}%")
    print(f"  {'Confidence':<20} {old_conf:<15.4f} {conf_new:<15.4f}")
    print(f"  {'P0-P50 MAE':<20} {'0.1807':<15} {new_p0_p50:<15.4f}")
    print(f"  {'P50-P90 MAE':<20} {'0.7004':<15} {new_p50_p90:<15.4f}")
    print(f"  {'P90-P99 MAE':<20} {old_p90_99:<15.4f} {new_p90_p99:<15.4f} {(new_p90_p99-old_p90_99)/old_p90_99*100:+.1f}%")
    print(f"  {'P99-P100 MAE':<20} {old_p99_100:<15.4f} {new_p99_p100:<15.4f} {(new_p99_p100-old_p99_100)/old_p99_100*100:+.1f}%")

    # Saturation proxy: use HDR luminance as proxy (high lum correlates with sat)
    # Actually compute correlation of error with luminance
    from scipy.stats import spearmanr
    n_sub = min(100000, len(sdr_val))
    sub_idx = rng.choice(len(sdr_val), n_sub, replace=False)
    rho_lum, _ = spearmanr(hdr_val_nits[sub_idx], np.abs(err_new[sub_idx]))
    print(f"\n  Spearman |error| vs HDR luminance: rho = {rho_lum:.4f}")
    print(f"  (OLD was ~0.78 for SDR luminance correlation)")

    # HDR percentiles
    print(f"\n  HDR ground truth percentiles (NEW, nits):")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"    P{p}: {np.percentile(hdr_val_nits, p):.3f}")
    print(f"    Max: {np.max(hdr_val_nits):.3f}")

    # Mean signed error
    print(f"\n  Mean signed error: {np.mean(err_new):.4f} nits")
    print(f"  (OLD was always negative due to systematic underprediction)")

    print(f"\n{'='*70}")
    print("AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
