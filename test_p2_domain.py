"""P2: Controlled fitting-domain experiment.

Compares three fitting domains using identical data:
A) Linear (current baseline)
B) PQ domain
C) Log domain

All evaluated in physical luminance (nits).
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
LOG_EPS = 1e-6  # Explicit epsilon for log domain


def fit_curve_in_domain(sdr_train, hdr_train, domain, config):
    """Fit a curve in the specified domain. Returns (curve, apply_fn)."""

    if domain == "linear":
        # Current baseline: fit in normalized linear [0,1]
        curve = estimate_luminance_curve(sdr_train, hdr_train, config=config)
        def apply_fn(sdr_val):
            return apply_luminance_curve(sdr_val, curve)
        return curve, apply_fn

    elif domain == "pq":
        # Transform both to PQ domain before fitting
        sdr_pq = pq_oetf(sdr_train * PEAK).astype(np.float64)  # linear [0,1] → nits → PQ
        hdr_pq = pq_oetf(hdr_train * PEAK).astype(np.float64)

        # Fit in PQ domain using same binning/PCHIP methodology
        # Use percentile_bins + fit_monotonic_spline + PCHIP
        pr = (config.low_percentile, config.high_percentile)
        centers, medians = percentile_bins(sdr_pq, hdr_pq, n_bins=config.luminance_bins,
                                           percentile_range=pr)
        if len(centers) < 5:
            curve_pts = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
        else:
            x_mono, y_mono = fit_monotonic_spline(centers, medians)
            n_pts = min(64, len(x_mono))
            idx = np.linspace(0, len(x_mono)-1, n_pts, dtype=int)
            curve_pts = [[float(x_mono[i]), float(y_mono[i])] for i in idx]
            if curve_pts[0][0] > 0.01:
                curve_pts.insert(0, [0.0, 0.0])
            if curve_pts[-1][0] < 0.99:
                curve_pts.append([1.0, curve_pts[-1][1] * 1.02])

        def apply_fn(sdr_val):
            # sdr_val is normalized linear [0,1]
            sdr_pq_v = pq_oetf(sdr_val * PEAK)
            x_pts = np.array([p[0] for p in curve_pts])
            y_pts = np.array([p[1] for p in curve_pts])
            interp = PchipInterpolator(x_pts, y_pts, extrapolate=True)
            hdr_pq_mapped = np.maximum(interp(sdr_pq_v), 0.0)
            # Convert back to linear normalized
            hdr_nits = pq_eotf(hdr_pq_mapped)
            return hdr_nits / PEAK
        return curve_pts, apply_fn

    elif domain == "log":
        # Transform to log domain
        sdr_log = np.log10(sdr_train * PEAK + LOG_EPS)
        hdr_log = np.log10(hdr_train * PEAK + LOG_EPS)

        # Fit in log domain
        pr = (config.low_percentile, config.high_percentile)
        centers, medians = percentile_bins(sdr_log, hdr_log, n_bins=config.luminance_bins,
                                           percentile_range=pr)
        if len(centers) < 5:
            curve_pts = [[-6.0, -6.0], [0.0, 0.0], [4.0, 4.0]]
        else:
            x_mono, y_mono = fit_monotonic_spline(centers, medians)
            n_pts = min(64, len(x_mono))
            idx = np.linspace(0, len(x_mono)-1, n_pts, dtype=int)
            curve_pts = [[float(x_mono[i]), float(y_mono[i])] for i in idx]

        def apply_fn(sdr_val):
            sdr_log_v = np.log10(sdr_val * PEAK + LOG_EPS)
            x_pts = np.array([p[0] for p in curve_pts])
            y_pts = np.array([p[1] for p in curve_pts])
            interp = PchipInterpolator(x_pts, y_pts, extrapolate=True)
            hdr_log_mapped = interp(sdr_log_v)
            hdr_nits = np.power(10.0, hdr_log_mapped) - LOG_EPS
            return np.maximum(hdr_nits, 0.0) / PEAK
        return curve_pts, apply_fn


def evaluate(apply_fn, sdr_val, hdr_val):
    """Evaluate in physical nits."""
    pred_norm = apply_fn(sdr_val)
    pred_nits = pred_norm * PEAK
    hdr_nits = hdr_val * PEAK
    err = pred_nits - hdr_nits
    abs_err = np.abs(err)

    def rmae(lo_p, hi_p):
        lo = np.percentile(sdr_val, lo_p)
        hi = np.percentile(sdr_val, hi_p) if hi_p < 100 else sdr_val.max() + 1
        m = (sdr_val >= lo) & (sdr_val < hi)
        return float(np.mean(abs_err[m])) if np.sum(m) > 0 else 0.0

    from scipy.stats import spearmanr
    n_sub = min(50000, len(sdr_val))
    idx = np.random.default_rng(42).choice(len(sdr_val), n_sub, replace=False)
    rho_lum, _ = spearmanr(hdr_nits[idx], abs_err[idx])

    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "signed": float(np.mean(err)),
        "p0_p50": rmae(0, 50),
        "p50_p90": rmae(50, 90),
        "p90_p99": rmae(90, 99),
        "p99_p100": rmae(99, 100),
        "rho_lum": rho_lum,
    }


def main():
    print("=" * 70)
    print("P2: CONTROLLED FITTING-DOMAIN EXPERIMENT")
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

    # Collect samples (identical for all three variants)
    print("\n--- Collecting samples (3 segments) ---")
    segments = [(2700.0, 2730.0), (4500.0, 4530.0), (2220.0, 2250.0)]
    all_sdr, all_hdr = [], []

    t0 = time.perf_counter()
    for start, end in segments:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                           config=config, n_samples=30, proxy_width=960)
        all_sdr.append(samples.sdr_luminance)
        all_hdr.append(samples.hdr_luminance)
        print(f"  {start:.0f}-{end:.0f}s: {samples.n_valid_pairs:,} pairs")
    t1 = time.perf_counter()

    sdr = np.concatenate(all_sdr)
    hdr = np.concatenate(all_hdr)
    print(f"  Total: {len(sdr):,} pairs, {t1-t0:.1f}s")

    # Train/val split (IDENTICAL for all variants)
    rng = np.random.default_rng(42)
    n = len(sdr)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    sdr_train, sdr_val = sdr[idx[:split]], sdr[idx[split:]]
    hdr_train, hdr_val = hdr[idx[:split]], hdr[idx[split:]]
    print(f"  Train: {len(sdr_train):,}, Val: {len(sdr_val):,}")

    # Run three variants
    domains = ["linear", "pq", "log"]
    results = {}

    for domain in domains:
        print(f"\n--- Fitting in {domain.upper()} domain ---")
        t2 = time.perf_counter()
        _, apply_fn = fit_curve_in_domain(sdr_train, hdr_train, domain, config)
        t3 = time.perf_counter()
        res = evaluate(apply_fn, sdr_val, hdr_val)
        res["fit_time"] = t3 - t2
        results[domain] = res
        print(f"  MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, "
              f"P99-100={res['p99_p100']:.4f}, time={res['fit_time']:.2f}s")

    # Final comparison table
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE (all metrics in nits)")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<22} {'A: Linear':<14} {'B: PQ':<14} {'C: Log':<14}")
    print(f"  {'-'*64}")
    metrics = [
        ("Validation MAE", "mae"),
        ("RMSE", "rmse"),
        ("Mean signed error", "signed"),
        ("P0-P50 MAE", "p0_p50"),
        ("P50-P90 MAE", "p50_p90"),
        ("P90-P99 MAE", "p90_p99"),
        ("P99-P100 MAE", "p99_p100"),
        ("|err| vs lum ρ", "rho_lum"),
        ("Fit time (s)", "fit_time"),
    ]
    for label, key in metrics:
        a = results["linear"][key]
        b = results["pq"][key]
        c = results["log"][key]
        print(f"  {label:<22} {a:<14.4f} {b:<14.4f} {c:<14.4f}")

    # Highlight best per metric
    print(f"\n  BEST per metric:")
    for label, key in metrics:
        vals = {d: results[d][key] for d in domains}
        if key in ("rho_lum",):
            best = min(vals, key=lambda d: abs(vals[d]))  # lower abs correlation = better
        elif key == "fit_time":
            best = min(vals, key=lambda d: vals[d])
        else:
            best = min(vals, key=lambda d: abs(vals[d]) if key == "signed" else vals[d])
        print(f"    {label}: {best.upper()}")

    # Log epsilon
    print(f"\n  Log domain epsilon: {LOG_EPS}")

    print(f"\n{'='*70}")
    print("P2 EXPERIMENT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
