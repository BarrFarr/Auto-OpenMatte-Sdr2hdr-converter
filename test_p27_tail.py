"""P2.7: Log domain tail coverage audit.

Separates domain effect from percentile coverage effect.
A) Log P1-P99 (current Log baseline)
B) Log P1-P99.9
C) Log P1-P100
D) Linear P1-P99.9
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
EPS = 1e-6


def fit_generic(sdr_train, hdr_train, domain, pct_lo, pct_hi, n_bins=256):
    """Fit curve with specified domain and percentile range."""
    if domain == "log":
        x_data = np.log10(sdr_train * PEAK + EPS)
        y_data = np.log10(hdr_train * PEAK + EPS)
    else:  # linear
        x_data = sdr_train
        y_data = hdr_train

    centers, medians = percentile_bins(x_data, y_data, n_bins=n_bins,
                                       percentile_range=(pct_lo, pct_hi))
    if len(centers) < 5:
        # Fallback
        centers = np.linspace(x_data.min(), x_data.max(), 10)
        medians = centers  # identity
    x_mono, y_mono = fit_monotonic_spline(centers, medians)
    n_pts = min(64, len(x_mono))
    idx = np.linspace(0, len(x_mono)-1, n_pts, dtype=int)
    x_pts = x_mono[idx]
    y_pts = y_mono[idx]

    def apply_fn(sdr_val):
        if domain == "log":
            sv = np.log10(sdr_val * PEAK + EPS)
            interp = PchipInterpolator(x_pts, y_pts, extrapolate=True)
            mapped = interp(sv)
            nits = np.power(10.0, mapped) - EPS
            return np.maximum(nits, 0.0) / PEAK
        else:
            interp = PchipInterpolator(x_pts, y_pts, extrapolate=True)
            return np.maximum(interp(sdr_val), 0.0)

    return apply_fn, x_pts, y_pts


def evaluate_full(apply_fn, sdr_val, hdr_val):
    """Full evaluation with fine-grained ranges."""
    pred = apply_fn(sdr_val) * PEAK
    actual = hdr_val * PEAK
    err = pred - actual
    abs_err = np.abs(err)

    def rmae(lo_p, hi_p):
        lo = np.percentile(sdr_val, lo_p) if lo_p > 0 else 0
        hi = np.percentile(sdr_val, hi_p) if hi_p < 100 else sdr_val.max() * 10
        m = (sdr_val >= lo) & (sdr_val < hi)
        return float(np.mean(abs_err[m])) if np.sum(m) > 0 else 0.0

    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "signed": float(np.mean(err)),
        "p0_p50": rmae(0, 50),
        "p50_p90": rmae(50, 90),
        "p90_p99": rmae(90, 99),
        "p99_p995": rmae(99, 99.5),
        "p995_p999": rmae(99.5, 99.9),
        "p999_p100": rmae(99.9, 100),
        "max_abs": float(np.max(abs_err)),
    }


def main():
    print("=" * 70)
    print("P2.7: LOG DOMAIN TAIL COVERAGE AUDIT")
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

    # Collect
    print("\n--- Collecting samples ---")
    seg_defs = [(2700.0, 2730.0), (4500.0, 4530.0), (2220.0, 2250.0)]
    all_sdr, all_hdr = [], []
    t0 = time.perf_counter()
    for start, end in seg_defs:
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
    print(f"  Total: {len(sdr):,} pairs, {t1-t0:.1f}s")

    # Split
    rng = np.random.default_rng(42)
    n = len(sdr)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    sdr_train, sdr_val = sdr[idx[:split]], sdr[idx[split:]]
    hdr_train, hdr_val = hdr[idx[:split]], hdr[idx[split:]]

    # Sample counts in tail
    p99 = np.percentile(sdr_train, 99)
    p999 = np.percentile(sdr_train, 99.9)
    n_above_99 = int(np.sum(sdr_train >= p99))
    n_above_999 = int(np.sum(sdr_train >= p999))
    print(f"\n  Train samples above P99: {n_above_99:,}")
    print(f"  Train samples above P99.9: {n_above_999:,}")
    print(f"  SDR P99: {p99:.6f} ({p99*PEAK:.1f} nits-eq)")
    print(f"  SDR P99.9: {p999:.6f} ({p999*PEAK:.1f} nits-eq)")
    print(f"  SDR max: {sdr_train.max():.6f} ({sdr_train.max()*PEAK:.1f} nits-eq)")

    # Fit variants
    variants = [
        ("A: Log P1-P99", "log", 1.0, 99.0),
        ("B: Log P1-P99.9", "log", 1.0, 99.9),
        ("C: Log P1-P100", "log", 0.0, 100.0),
        ("D: Linear P1-P99.9", "linear", 1.0, 99.9),
    ]

    results = {}
    curve_info = {}

    for name, domain, pct_lo, pct_hi in variants:
        print(f"\n--- Fitting: {name} ---")
        fn, x_pts, y_pts = fit_generic(sdr_train, hdr_train, domain, pct_lo, pct_hi)
        res = evaluate_full(fn, sdr_val, hdr_val)
        results[name] = res

        # Curve endpoint info
        if domain == "log":
            top_x_nits = 10.0**x_pts[-1] - EPS
            top_y_nits = 10.0**y_pts[-1] - EPS
        else:
            top_x_nits = x_pts[-1] * PEAK
            top_y_nits = y_pts[-1] * PEAK
        curve_info[name] = {
            "n_ctrl_pts": len(x_pts),
            "top_x": top_x_nits,
            "top_y": top_y_nits,
        }
        print(f"  Control points: {len(x_pts)}, top: SDR={top_x_nits:.1f} → HDR={top_y_nits:.1f} nits")
        print(f"  MAE={res['mae']:.4f}, P99-99.5={res['p99_p995']:.4f}, P99.9-100={res['p999_p100']:.4f}")

    # Comparison table
    print(f"\n\n{'='*70}")
    print("COMPARISON TABLE (nits)")
    print(f"{'='*70}")

    labels = list(results.keys())
    metrics = [
        ("Validation MAE", "mae"),
        ("RMSE", "rmse"),
        ("Mean signed", "signed"),
        ("P0-P50", "p0_p50"),
        ("P50-P90", "p50_p90"),
        ("P90-P99", "p90_p99"),
        ("P99-P99.5", "p99_p995"),
        ("P99.5-P99.9", "p995_p999"),
        ("P99.9-P100", "p999_p100"),
        ("Max abs error", "max_abs"),
    ]

    header = f"  {'Metric':<16}" + "".join(f"{l:<18}" for l in labels)
    print(f"\n{header}")
    print(f"  {'-'*16}" + "-"*18*4)
    for mlabel, mkey in metrics:
        row = f"  {mlabel:<16}"
        for name in labels:
            row += f"{results[name][mkey]:<18.4f}"
        print(row)

    # Curve endpoint table
    print(f"\n  CURVE ENDPOINTS:")
    print(f"  {'Variant':<20} {'Ctrl pts':<10} {'Top SDR (nits-eq)':<20} {'Top HDR pred (nits)':<20}")
    print(f"  {'-'*70}")
    for name in labels:
        ci = curve_info[name]
        print(f"  {name:<20} {ci['n_ctrl_pts']:<10} {ci['top_x']:<20.2f} {ci['top_y']:<20.2f}")

    # Decomposition: domain effect vs coverage effect
    print(f"\n\n{'='*70}")
    print("EFFECT DECOMPOSITION")
    print(f"{'='*70}")

    a = results["A: Log P1-P99"]
    b = results["B: Log P1-P99.9"]
    c = results["C: Log P1-P100"]
    d = results["D: Linear P1-P99.9"]
    baseline_lin = 14.2958  # Linear P1-P99 from P2

    print(f"\n  P99-P100 MAE decomposition:")
    p99_100_a = a["p99_p995"] * 0.5 + a["p995_p999"] * 0.4 + a["p999_p100"] * 0.1  # weighted
    # Actually just report the combined P99-P100 using sum of sub-ranges weighted by count
    # For simplicity use the P99.9-P100 + P99.5-P99.9 + P99-P99.5 as proxy
    print(f"    Linear P1-P99 (baseline): ~14.30 nits")
    print(f"    A: Log P1-P99:    P99-99.5={a['p99_p995']:.2f}, P99.5-99.9={a['p995_p999']:.2f}, P99.9-100={a['p999_p100']:.2f}")
    print(f"    B: Log P1-P99.9:  P99-99.5={b['p99_p995']:.2f}, P99.5-99.9={b['p995_p999']:.2f}, P99.9-100={b['p999_p100']:.2f}")
    print(f"    C: Log P1-P100:   P99-99.5={c['p99_p995']:.2f}, P99.5-99.9={c['p995_p999']:.2f}, P99.9-100={c['p999_p100']:.2f}")
    print(f"    D: Lin P1-P99.9:  P99-99.5={d['p99_p995']:.2f}, P99.5-99.9={d['p995_p999']:.2f}, P99.9-100={d['p999_p100']:.2f}")

    print(f"\n  DOMAIN EFFECT (same percentile range):")
    print(f"    Log P1-P99.9 MAE = {b['mae']:.4f} vs Linear P1-P99.9 MAE = {d['mae']:.4f}")
    print(f"    → Domain alone: {(1-b['mae']/d['mae'])*100:+.1f}% improvement")

    print(f"\n  COVERAGE EFFECT (same domain):")
    print(f"    Log P1-P99 MAE = {a['mae']:.4f} vs Log P1-P99.9 MAE = {b['mae']:.4f}")
    print(f"    → Coverage P99→P99.9: {(1-b['mae']/a['mae'])*100:+.1f}% improvement")
    print(f"    Log P1-P99.9 MAE = {b['mae']:.4f} vs Log P1-P100 MAE = {c['mae']:.4f}")
    print(f"    → Coverage P99.9→P100: {(1-c['mae']/b['mae'])*100:+.1f}% improvement")

    print(f"\n{'='*70}")
    print("P2.7 AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
