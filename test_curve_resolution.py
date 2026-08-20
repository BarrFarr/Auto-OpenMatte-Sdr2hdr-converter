"""Luminance curve resolution diagnostic.

Tests A-H: bin count, percentile binning, highlight weighting,
piecewise monotonic, bin weight analysis, chroma confounding.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.utils.math_utils import fit_monotonic_spline, percentile_bins
from scipy.interpolate import PchipInterpolator

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
PEAK = 10000.0


def collect_all_samples():
    """Collect paired samples from 3 segments."""
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps

    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256)

    segments = [(2700.0, 2730.0), (4500.0, 4530.0), (2220.0, 2250.0)]
    all_sdr = []
    all_hdr = []

    for start, end in segments:
        sf = int(round(start * fps))
        ef = int(round(end * fps))
        shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                    om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
        samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                           config=config, n_samples=30, proxy_width=960)
        all_sdr.append(samples.sdr_luminance)
        all_hdr.append(samples.hdr_luminance)
        print(f"  Segment {start:.0f}-{end:.0f}s: {samples.n_valid_pairs:,} pairs")

    sdr = np.concatenate(all_sdr)
    hdr = np.concatenate(all_hdr)
    print(f"  Total: {len(sdr):,} pairs")
    return sdr, hdr


def split_data(sdr, hdr, seed=42, ratio=0.8):
    rng = np.random.default_rng(seed)
    n = len(sdr)
    idx = rng.permutation(n)
    split = int(n * ratio)
    return sdr[idx[:split]], hdr[idx[:split]], sdr[idx[split:]], hdr[idx[split:]]


def eval_curve(curve, sdr_val, hdr_val):
    """Evaluate curve on validation data. Returns dict of metrics."""
    pred = apply_luminance_curve(sdr_val, curve)
    pred_nits = pred * PEAK
    hdr_nits = hdr_val * PEAK
    err = pred_nits - hdr_nits
    abs_err = np.abs(err)

    # Per-percentile ranges
    results = {}
    results["overall_mae"] = float(np.mean(abs_err))
    results["overall_rmse"] = float(np.sqrt(np.mean(err**2)))
    results["mean_signed"] = float(np.mean(err))

    # Percentile ranges based on SDR value
    for label, lo_p, hi_p in [("P0-P50", 0, 50), ("P50-P90", 50, 90),
                               ("P90-P99", 90, 99), ("P99-P100", 99, 100)]:
        lo = np.percentile(sdr_val, lo_p)
        hi = np.percentile(sdr_val, hi_p) if hi_p < 100 else sdr_val.max() + 1
        mask = (sdr_val >= lo) & (sdr_val < hi)
        if np.sum(mask) > 10:
            results[f"{label}_mae"] = float(np.mean(abs_err[mask]))
        else:
            results[f"{label}_mae"] = 0.0

    # Confidence
    if np.std(pred_nits) > 0 and np.std(hdr_nits) > 0:
        results["confidence"] = float(np.corrcoef(pred_nits, hdr_nits)[0, 1])
    else:
        results["confidence"] = 0.0

    return results


def fit_custom_curve(sdr_train, hdr_train, n_bins=256, percentile_mode=False,
                     highlight_weighted=False):
    """Fit a curve with custom bin parameters."""
    if percentile_mode:
        # Percentile-based bin edges
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(sdr_train, percentiles)
        # Remove duplicates
        bin_edges = np.unique(bin_edges)
        n_bins_actual = len(bin_edges) - 1
    elif highlight_weighted:
        # More bins in highlights
        edges_shadow = np.linspace(np.percentile(sdr_train, 1), np.percentile(sdr_train, 50), 129)[:-1]
        edges_mid = np.linspace(np.percentile(sdr_train, 50), np.percentile(sdr_train, 90), 129)[:-1]
        edges_high = np.linspace(np.percentile(sdr_train, 90), np.percentile(sdr_train, 99), 129)[:-1]
        edges_top = np.linspace(np.percentile(sdr_train, 99), sdr_train.max(), 129)
        bin_edges = np.concatenate([edges_shadow, edges_mid, edges_high, edges_top])
        bin_edges = np.unique(bin_edges)
        n_bins_actual = len(bin_edges) - 1
    else:
        # Equal-width bins in P1-P99 range
        lo = np.percentile(sdr_train, 1)
        hi = np.percentile(sdr_train, 99)
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        n_bins_actual = n_bins

    # Compute medians
    bin_centers = []
    bin_medians = []
    for i in range(len(bin_edges) - 1):
        mask = (sdr_train >= bin_edges[i]) & (sdr_train < bin_edges[i + 1])
        if np.sum(mask) >= 10:
            bin_centers.append(float((bin_edges[i] + bin_edges[i+1]) / 2))
            bin_medians.append(float(np.median(hdr_train[mask])))

    if len(bin_centers) < 5:
        return [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

    x = np.array(bin_centers)
    y = np.array(bin_medians)

    # Isotonic enforcement
    _, y_mono = fit_monotonic_spline(x, y)

    # Subsample to 64 control points
    n_pts = min(64, len(x))
    indices = np.linspace(0, len(x) - 1, n_pts, dtype=int)
    x_final = x[indices]
    y_final = y_mono[indices]

    curve = [[float(xi), float(yi)] for xi, yi in zip(x_final, y_final)]
    if curve[0][0] > 0.001:
        curve.insert(0, [0.0, 0.0])
    if curve[-1][0] < 0.99:
        curve.append([1.0, curve[-1][1] * 1.02])
    return curve


def fit_piecewise(sdr_train, hdr_train):
    """Fit 3 piecewise monotonic segments: shadow/mid/highlight."""
    p50 = np.percentile(sdr_train, 50)
    p90 = np.percentile(sdr_train, 90)

    segments = [
        (sdr_train < p50, "shadow"),
        ((sdr_train >= p50) & (sdr_train < p90), "mid"),
        (sdr_train >= p90, "highlight"),
    ]

    all_x = []
    all_y = []

    for mask, name in segments:
        s = sdr_train[mask]
        h = hdr_train[mask]
        if len(s) < 100:
            continue
        # Fit within segment
        lo = s.min()
        hi = s.max()
        edges = np.linspace(lo, hi, 128)
        centers = []
        medians = []
        for i in range(len(edges)-1):
            m = (s >= edges[i]) & (s < edges[i+1])
            if np.sum(m) >= 10:
                centers.append(float((edges[i]+edges[i+1])/2))
                medians.append(float(np.median(h[m])))
        if len(centers) >= 3:
            cx = np.array(centers)
            cy = np.array(medians)
            _, cy_mono = fit_monotonic_spline(cx, cy)
            all_x.extend(cx.tolist())
            all_y.extend(cy_mono.tolist())

    if len(all_x) < 5:
        return [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

    # Sort and enforce global monotonicity
    order = np.argsort(all_x)
    x = np.array(all_x)[order]
    y = np.array(all_y)[order]
    _, y_mono = fit_monotonic_spline(x, y)

    # Subsample
    n_pts = min(64, len(x))
    indices = np.linspace(0, len(x)-1, n_pts, dtype=int)
    curve = [[float(x[i]), float(y_mono[i])] for i in indices]
    if curve[0][0] > 0.001:
        curve.insert(0, [0.0, 0.0])
    return curve


def main():
    print("=" * 70)
    print("LUMINANCE CURVE RESOLUTION DIAGNOSTIC")
    print("=" * 70)

    print("\n--- Collecting samples from 3 segments ---")
    sdr, hdr = collect_all_samples()
    sdr_train, hdr_train, sdr_val, hdr_val = split_data(sdr, hdr)
    print(f"  Train: {len(sdr_train):,}, Val: {len(sdr_val):,}")

    # ===== TEST A: CURRENT BASELINE =====
    print("\n" + "=" * 70)
    print("TEST A: Current baseline (256 equal-width bins)")
    config = ColorConfig(luminance_bins=256)
    curve_256 = estimate_luminance_curve(sdr_train, hdr_train, config=config)
    res_a = eval_curve(curve_256, sdr_val, hdr_val)
    print(f"  Overall MAE: {res_a['overall_mae']:.4f} nits, RMSE: {res_a['overall_rmse']:.4f}")
    print(f"  P0-P50 MAE:  {res_a['P0-P50_mae']:.4f}")
    print(f"  P50-P90 MAE: {res_a['P50-P90_mae']:.4f}")
    print(f"  P90-P99 MAE: {res_a['P90-P99_mae']:.4f}")
    print(f"  P99-P100 MAE:{res_a['P99-P100_mae']:.4f}")

    # ===== TEST B: MORE BINS =====
    print("\n" + "=" * 70)
    print("TEST B: More bins (equal-width)")
    results_b = {}
    for n_bins in [256, 512, 1024]:
        curve = fit_custom_curve(sdr_train, hdr_train, n_bins=n_bins)
        res = eval_curve(curve, sdr_val, hdr_val)
        results_b[n_bins] = res
        print(f"  {n_bins} bins: MAE={res['overall_mae']:.4f}, "
              f"P90-99={res['P90-P99_mae']:.4f}, P99-100={res['P99-P100_mae']:.4f}")

    # ===== TEST C: PERCENTILE BINS =====
    print("\n" + "=" * 70)
    print("TEST C: Percentile bins vs equal-width")
    curve_pct = fit_custom_curve(sdr_train, hdr_train, n_bins=256, percentile_mode=True)
    res_pct = eval_curve(curve_pct, sdr_val, hdr_val)
    print(f"  Equal-width 256: MAE={res_a['overall_mae']:.4f}, P90-99={res_a['P90-P99_mae']:.4f}, P99-100={res_a['P99-P100_mae']:.4f}")
    print(f"  Percentile 256:  MAE={res_pct['overall_mae']:.4f}, P90-99={res_pct['P90-P99_mae']:.4f}, P99-100={res_pct['P99-P100_mae']:.4f}")

    # ===== TEST D: HIGHLIGHT-WEIGHTED =====
    print("\n" + "=" * 70)
    print("TEST D: Highlight-weighted binning (128+128+128+128)")
    curve_hw = fit_custom_curve(sdr_train, hdr_train, highlight_weighted=True)
    res_hw = eval_curve(curve_hw, sdr_val, hdr_val)
    print(f"  Highlight-weighted: MAE={res_hw['overall_mae']:.4f}, "
          f"P90-99={res_hw['P90-P99_mae']:.4f}, P99-100={res_hw['P99-P100_mae']:.4f}")

    # ===== TEST E: PIECEWISE MONOTONIC =====
    print("\n" + "=" * 70)
    print("TEST E: Piecewise monotonic (3 segments)")
    curve_pw = fit_piecewise(sdr_train, hdr_train)
    res_pw = eval_curve(curve_pw, sdr_val, hdr_val)
    print(f"  Piecewise: MAE={res_pw['overall_mae']:.4f}, "
          f"P90-99={res_pw['P90-P99_mae']:.4f}, P99-100={res_pw['P99-P100_mae']:.4f}")

    # ===== TEST F: BIN WEIGHT ANALYSIS =====
    print("\n" + "=" * 70)
    print("TEST F: Bin weight analysis (current implementation)")
    print(f"\n  Current `percentile_bins()` mechanism:")
    print(f"    - 256 equal-width bins between P1 and P99 of SDR")
    print(f"    - Each bin computes MEDIAN of HDR values within it")
    print(f"    - Bins with < 10 samples are EXCLUDED")
    print(f"    - No explicit weighting — all valid bins have equal weight")
    print(f"    - Isotonic regression: unweighted (treats all points equally)")
    print(f"    - PCHIP: interpolates through all points")
    print(f"\n  Problem mechanism:")
    # Count samples per SDR percentile range
    p50 = np.percentile(sdr_train, 50)
    p90 = np.percentile(sdr_train, 90)
    p99 = np.percentile(sdr_train, 99)
    n_below_p50 = int(np.sum(sdr_train < p50))
    n_p50_p90 = int(np.sum((sdr_train >= p50) & (sdr_train < p90)))
    n_p90_p99 = int(np.sum((sdr_train >= p90) & (sdr_train < p99)))
    n_above_p99 = int(np.sum(sdr_train >= p99))
    print(f"    SDR < P50:      {n_below_p50:>12,} samples ({n_below_p50/len(sdr_train)*100:.1f}%)")
    print(f"    P50 ≤ SDR < P90:{n_p50_p90:>12,} samples ({n_p50_p90/len(sdr_train)*100:.1f}%)")
    print(f"    P90 ≤ SDR < P99:{n_p90_p99:>12,} samples ({n_p90_p99/len(sdr_train)*100:.1f}%)")
    print(f"    SDR ≥ P99:      {n_above_p99:>12,} samples ({n_above_p99/len(sdr_train)*100:.1f}%)")
    print(f"\n    Equal-width bins in [P1, P99]:")
    print(f"    P1 = {np.percentile(sdr_train, 1):.6f}")
    print(f"    P99 = {np.percentile(sdr_train, 99):.6f}")
    print(f"    Bin width = {(np.percentile(sdr_train, 99) - np.percentile(sdr_train, 1))/256:.6f}")
    # How many bins cover P90-P99?
    p1 = np.percentile(sdr_train, 1)
    bin_width = (p99 - p1) / 256
    bins_below_p50 = int((p50 - p1) / bin_width)
    bins_p50_p90 = int((p90 - p50) / bin_width)
    bins_p90_p99 = int((p99 - p90) / bin_width)
    print(f"    Bins covering P0-P50:  ~{bins_below_p50}")
    print(f"    Bins covering P50-P90: ~{bins_p50_p90}")
    print(f"    Bins covering P90-P99: ~{bins_p90_p99}")
    print(f"\n    ISSUE: P90-P99 SDR range gets {bins_p90_p99} bins BUT has highly variable")
    print(f"    HDR values. Each bin has many samples (median is robust), but")
    print(f"    the variability within highlight bins is larger than in shadows.")

    # ===== TEST G: CHROMA CONFOUNDING =====
    print("\n" + "=" * 70)
    print("TEST G: Chroma confounding (controlling for luminance)")
    # We need RGB data — use the spatial diagnostic approach on proxy
    # For simplicity, use the existing samples (which are luminance-only)
    # and create synthetic saturation proxy from SDR luminance distribution
    # Actually, we can't test chroma without RGB. Report this limitation.
    print(f"\n  NOTE: Current sampling pipeline extracts GRAYSCALE (gray16le).")
    print(f"  RGB chroma information is not available in the sampled data.")
    print(f"  The previous test (test_spatial_diagnostic.py) used full RGB frames")
    print(f"  at proxy resolution and found:")
    print(f"    |ρ(|error|, SDR_lum)| = 0.7809")
    print(f"    |ρ(|error|, saturation)| = 0.7297")
    print(f"    |ρ(|error|, Y_position)| = 0.0291")
    print(f"\n  Since saturation ρ=0.73 is LOWER than luminance ρ=0.78,")
    print(f"  and saturation is strongly correlated with luminance")
    print(f"  (brighter pixels tend to be more saturated in this material),")
    print(f"  the evidence suggests luminance is the PRIMARY predictor.")
    print(f"  Chroma adds marginal information beyond luminance alone.")

    # ===== FINAL COMPARISON TABLE =====
    print("\n\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)

    methods = [
        ("A: 256 equal-width (current)", res_a),
        ("B: 512 equal-width", results_b[512]),
        ("B: 1024 equal-width", results_b[1024]),
        ("C: 256 percentile", res_pct),
        ("D: Highlight-weighted", res_hw),
        ("E: Piecewise monotonic", res_pw),
    ]

    print(f"\n  {'Method':<30} {'MAE':<8} {'P90-99':<8} {'P99-100':<8} {'RMSE':<8} {'Conf':<6}")
    print(f"  {'-'*68}")
    for name, res in methods:
        print(f"  {name:<30} {res['overall_mae']:<8.4f} {res['P90-P99_mae']:<8.4f} "
              f"{res['P99-P100_mae']:<8.4f} {res['overall_rmse']:<8.4f} {res['confidence']:<6.4f}")

    # Best method
    best = min(methods, key=lambda x: x[1]['P90-P99_mae'])
    print(f"\n  Best for P90-P99: {best[0]} (MAE={best[1]['P90-P99_mae']:.4f})")
    best_overall = min(methods, key=lambda x: x[1]['overall_mae'])
    print(f"  Best overall: {best_overall[0]} (MAE={best_overall[1]['overall_mae']:.4f})")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
