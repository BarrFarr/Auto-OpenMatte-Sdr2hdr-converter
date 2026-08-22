"""P2 VALIDATION: Log domain production candidate.

Tests P2.1–P2.6: identical data, curve comparison, fine-grained error,
extreme highlights, epsilon stability, per-segment stability.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from scipy.interpolate import PchipInterpolator

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import pq_eotf, pq_oetf
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.utils.math_utils import fit_monotonic_spline, percentile_bins

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
PEAK = 10000.0


def fit_log_curve(sdr_train, hdr_train, config, eps=1e-6):
    """Fit curve in log10 domain. Returns apply_fn."""
    sdr_log = np.log10(sdr_train * PEAK + eps)
    hdr_log = np.log10(hdr_train * PEAK + eps)
    pr = (config.low_percentile, config.high_percentile)
    centers, medians = percentile_bins(sdr_log, hdr_log, n_bins=config.luminance_bins,
                                       percentile_range=pr)
    if len(centers) < 5:
        x_pts = np.array([-6.0, 0.0, 4.0])
        y_pts = np.array([-6.0, 0.0, 4.0])
    else:
        x_pts, y_pts = fit_monotonic_spline(centers, medians)
        n_pts = min(64, len(x_pts))
        idx = np.linspace(0, len(x_pts)-1, n_pts, dtype=int)
        x_pts = x_pts[idx]
        y_pts = y_pts[idx]

    def apply_fn(sdr_val):
        sdr_log_v = np.log10(sdr_val * PEAK + eps)
        interp = PchipInterpolator(x_pts, y_pts, extrapolate=True)
        hdr_log_m = interp(sdr_log_v)
        hdr_nits = np.power(10.0, hdr_log_m) - eps
        return np.maximum(hdr_nits, 0.0) / PEAK
    return apply_fn, x_pts, y_pts


def fit_linear_curve(sdr_train, hdr_train, config):
    """Fit in linear domain (current baseline)."""
    curve = estimate_luminance_curve(sdr_train, hdr_train, config=config)
    def apply_fn(sdr_val):
        return apply_luminance_curve(sdr_val, curve)
    return apply_fn, curve


def eval_ranges(apply_fn, sdr_val, hdr_val, ranges):
    """Evaluate MAE for specified percentile ranges."""
    pred = apply_fn(sdr_val) * PEAK
    actual = hdr_val * PEAK
    abs_err = np.abs(pred - actual)
    results = {}
    for label, lo_p, hi_p in ranges:
        lo = np.percentile(sdr_val, lo_p) if lo_p > 0 else 0
        hi = np.percentile(sdr_val, hi_p) if hi_p < 100 else sdr_val.max() * 10
        m = (sdr_val >= lo) & (sdr_val < hi)
        n = int(np.sum(m))
        if n > 0:
            results[label] = {"mae": float(np.mean(abs_err[m])), "n": n}
        else:
            results[label] = {"mae": 0.0, "n": 0}
    return results


def main():
    print("=" * 70)
    print("P2 VALIDATION: LOG DOMAIN PRODUCTION CANDIDATE")
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

    # Collect per-segment samples
    seg_defs = [(2700.0, 2730.0, "45:00"), (4500.0, 4530.0, "1:15:00"), (2220.0, 2250.0, "37:00")]
    seg_samples = []
    print("\n--- Collecting samples ---")
    t0 = time.perf_counter()
    for start, end, label in seg_defs:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        s = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                     config=config, n_samples=30, proxy_width=960)
        seg_samples.append((label, s))
        print(f"  {label}: {s.n_valid_pairs:,}")
    t1 = time.perf_counter()

    sdr_all = np.concatenate([s.sdr_luminance for _, s in seg_samples])
    hdr_all = np.concatenate([s.hdr_luminance for _, s in seg_samples])
    print(f"  Total: {len(sdr_all):,}, time: {t1-t0:.1f}s")

    # Identical split for all
    rng = np.random.default_rng(42)
    n = len(sdr_all)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    sdr_train, sdr_val = sdr_all[idx[:split]], sdr_all[idx[split:]]
    hdr_train, hdr_val = hdr_all[idx[:split]], hdr_all[idx[split:]]

    # ================================================================
    # P2.1 — FIT BOTH CURVES ON IDENTICAL DATA
    # ================================================================
    print(f"\n{'='*70}")
    print("P2.1: Identical data verification")
    print(f"{'='*70}")
    print(f"  Train: {len(sdr_train):,}, Val: {len(sdr_val):,}")
    print(f"  Seed: 42, Split: 80/20, Bins: 256, Range: P1-P99")
    print(f"  Epsilon (Log): 1e-6")

    linear_fn, linear_curve = fit_linear_curve(sdr_train, hdr_train, config)
    log_fn, log_x, log_y = fit_log_curve(sdr_train, hdr_train, config, eps=1e-6)

    # ================================================================
    # P2.2 — CURVE COMPARISON IN NITS
    # ================================================================
    print(f"\n{'='*70}")
    print("P2.2: Curve comparison (control points in nits)")
    print(f"{'='*70}")

    test_sdr = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    linear_out = linear_fn(test_sdr) * PEAK
    log_out = log_fn(test_sdr) * PEAK

    print(f"\n  {'SDR linear':<12} {'SDR nits':<10} {'Linear→nits':<14} {'Log→nits':<14} {'Diff':<10}")
    print(f"  {'-'*60}")
    for i, s in enumerate(test_sdr):
        s_nits = s * PEAK
        print(f"  {s:<12.4f} {s_nits:<10.2f} {linear_out[i]:<14.4f} {log_out[i]:<14.4f} {log_out[i]-linear_out[i]:<10.4f}")

    # ================================================================
    # P2.3 — FINE-GRAINED ERROR BY LUMINANCE
    # ================================================================
    print(f"\n{'='*70}")
    print("P2.3: Error by fine luminance ranges (nits)")
    print(f"{'='*70}")

    fine_ranges = [
        ("P0-P10", 0, 10), ("P10-P25", 10, 25), ("P25-P50", 25, 50),
        ("P50-P75", 50, 75), ("P75-P90", 75, 90), ("P90-P95", 90, 95),
        ("P95-P99", 95, 99), ("P99-P99.5", 99, 99.5),
        ("P99.5-P99.9", 99.5, 99.9), ("P99.9-P100", 99.9, 100),
    ]

    lin_res = eval_ranges(linear_fn, sdr_val, hdr_val, fine_ranges)
    log_res = eval_ranges(log_fn, sdr_val, hdr_val, fine_ranges)

    print(f"\n  {'Range':<14} {'N':<10} {'Linear MAE':<14} {'Log MAE':<14} {'Improvement':<12}")
    print(f"  {'-'*64}")
    for label, _, _ in fine_ranges:
        ln = lin_res[label]
        lg = log_res[label]
        imp = (1 - lg["mae"]/ln["mae"])*100 if ln["mae"] > 0 else 0
        print(f"  {label:<14} {ln['n']:<10,} {ln['mae']:<14.4f} {lg['mae']:<14.4f} {imp:+.1f}%")

    # ================================================================
    # P2.4 — EXTREME HIGHLIGHTS (>P99)
    # ================================================================
    print(f"\n{'='*70}")
    print("P2.4: Extreme highlights (pixels above P99)")
    print(f"{'='*70}")

    p99 = np.percentile(sdr_val, 99)
    mask_99 = sdr_val >= p99
    sdr_hi = sdr_val[mask_99]
    hdr_hi = hdr_val[mask_99]

    lin_hi = linear_fn(sdr_hi) * PEAK
    log_hi = log_fn(sdr_hi) * PEAK
    hdr_hi_nits = hdr_hi * PEAK

    print(f"\n  N pixels above P99: {np.sum(mask_99):,}")
    print(f"  SDR input range: [{sdr_hi.min():.6f}, {sdr_hi.max():.6f}] normalized")
    print(f"  SDR input range: [{sdr_hi.min()*PEAK:.2f}, {sdr_hi.max()*PEAK:.2f}] nits-equivalent")
    print(f"  HDR ground truth: [{hdr_hi_nits.min():.2f}, {hdr_hi_nits.max():.2f}] nits")
    print(f"  Linear prediction: [{lin_hi.min():.2f}, {lin_hi.max():.2f}] nits")
    print(f"  Log prediction:    [{log_hi.min():.2f}, {log_hi.max():.2f}] nits")
    print(f"\n  Linear MAE: {np.mean(np.abs(lin_hi - hdr_hi_nits)):.4f} nits")
    print(f"  Log MAE:    {np.mean(np.abs(log_hi - hdr_hi_nits)):.4f} nits")
    print(f"  Linear signed: {np.mean(lin_hi - hdr_hi_nits):.4f} nits")
    print(f"  Log signed:    {np.mean(log_hi - hdr_hi_nits):.4f} nits")

    # ================================================================
    # P2.5 — EPSILON STABILITY
    # ================================================================
    print(f"\n{'='*70}")
    print("P2.5: Epsilon stability")
    print(f"{'='*70}")

    epsilons = [1e-7, 1e-6, 1e-5, 1e-4]
    print(f"\n  {'Epsilon':<12} {'MAE':<10} {'P0-P10 MAE':<12} {'P99-P100 MAE':<14} {'RMSE':<10}")
    print(f"  {'-'*58}")
    for eps in epsilons:
        fn_e, _, _ = fit_log_curve(sdr_train, hdr_train, config, eps=eps)
        pred_e = fn_e(sdr_val) * PEAK
        actual_e = hdr_val * PEAK
        err_e = np.abs(pred_e - actual_e)
        mae_e = float(np.mean(err_e))
        rmse_e = float(np.sqrt(np.mean((pred_e - actual_e)**2)))
        # P0-P10
        lo10 = np.percentile(sdr_val, 0)
        hi10 = np.percentile(sdr_val, 10)
        m10 = (sdr_val >= lo10) & (sdr_val < hi10)
        p0_10 = float(np.mean(err_e[m10])) if np.sum(m10) > 0 else 0
        # P99-P100
        m99 = sdr_val >= np.percentile(sdr_val, 99)
        p99_100 = float(np.mean(err_e[m99])) if np.sum(m99) > 0 else 0
        print(f"  {eps:<12.0e} {mae_e:<10.4f} {p0_10:<12.4f} {p99_100:<14.4f} {rmse_e:<10.4f}")

    # ================================================================
    # P2.6 — PER-SEGMENT STABILITY
    # ================================================================
    print(f"\n{'='*70}")
    print("P2.6: Per-segment stability (Linear vs Log)")
    print(f"{'='*70}")

    # Fit on FULL aggregate (for per-segment eval)
    # Already have linear_fn and log_fn from aggregate

    print(f"\n  {'Segment':<12} {'Lin MAE':<10} {'Log MAE':<10} {'Lin P99':<10} {'Log P99':<10} {'Lin bias':<10} {'Log bias':<10}")
    print(f"  {'-'*72}")
    for label, s in seg_samples:
        sv = s.sdr_luminance
        hv = s.hdr_luminance
        if len(sv) < 100:
            continue
        lin_p = linear_fn(sv) * PEAK
        log_p = log_fn(sv) * PEAK
        hv_nits = hv * PEAK
        lin_mae = float(np.mean(np.abs(lin_p - hv_nits)))
        log_mae = float(np.mean(np.abs(log_p - hv_nits)))
        # P99 per segment
        p99s = np.percentile(sv, 99)
        m99s = sv >= p99s
        lin_p99 = float(np.mean(np.abs(lin_p[m99s] - hv_nits[m99s]))) if np.sum(m99s) > 0 else 0
        log_p99 = float(np.mean(np.abs(log_p[m99s] - hv_nits[m99s]))) if np.sum(m99s) > 0 else 0
        lin_bias = float(np.mean(lin_p - hv_nits))
        log_bias = float(np.mean(log_p - hv_nits))
        print(f"  {label:<12} {lin_mae:<10.4f} {log_mae:<10.4f} {lin_p99:<10.4f} {log_p99:<10.4f} {lin_bias:<+10.4f} {log_bias:<+10.4f}")

    print(f"\n{'='*70}")
    print("P2 VALIDATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
