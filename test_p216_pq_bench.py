"""P2.16: PQ OETF Performance Microbenchmark.

Measures current PQ OETF implementation and alternative candidates.
No production code changes.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from auto_openmatte.core.transfer_functions import (
    pq_oetf, pq_eotf,
    _PQ_M1, _PQ_M2, _PQ_C1, _PQ_C2, _PQ_C3, _PQ_PEAK_LUMINANCE,
)

rng = np.random.default_rng(42)

# ============================================================
# VARIANT IMPLEMENTATIONS
# ============================================================

def pq_oetf_current(luminance):
    """A: Current production implementation (verbatim copy for isolation)."""
    luminance = np.clip(luminance, 0.0, _PQ_PEAK_LUMINANCE)
    y = luminance / _PQ_PEAK_LUMINANCE
    ym1 = np.power(y, _PQ_M1)
    num = _PQ_C1 + _PQ_C2 * ym1
    den = 1.0 + _PQ_C3 * ym1
    signal = np.power(num / den, _PQ_M2)
    return signal


def pq_oetf_numpy_opt(luminance):
    """B: NumPy optimized — minimize allocations, use in-place ops."""
    # Assume input already in [0, peak] — skip clip if caller guarantees
    y = np.empty_like(luminance)
    np.divide(luminance, _PQ_PEAK_LUMINANCE, out=y)
    np.clip(y, 0.0, 1.0, out=y)
    # y^m1 in-place
    np.power(y, _PQ_M1, out=y)
    # num = c1 + c2 * ym1
    num = y * _PQ_C2
    num += _PQ_C1
    # den = 1 + c3 * ym1
    den = y * _PQ_C3
    den += 1.0
    # num/den in-place
    np.divide(num, den, out=num)
    # (num/den)^m2
    np.power(num, _PQ_M2, out=num)
    return num


def build_pq_lut(n_entries, use_linear_input=True):
    """Build PQ OETF LUT. Input: linear luminance [0,1] normalized → PQ signal [0,1]."""
    if use_linear_input:
        # Uniform in linear normalized space [0, 1]
        lut_input = np.linspace(0.0, 1.0, n_entries)
    else:
        lut_input = np.linspace(0.0, 1.0, n_entries)
    
    # PQ OETF on full range
    lut_output = pq_oetf(lut_input * _PQ_PEAK_LUMINANCE)
    return lut_output.astype(np.float64)


def pq_oetf_lut_nearest(luminance, lut):
    """C/D/E: LUT with nearest-neighbor lookup."""
    n = len(lut)
    y = luminance / _PQ_PEAK_LUMINANCE
    np.clip(y, 0.0, 1.0, out=y)
    idx = (y * (n - 1)).astype(np.int64)
    np.clip(idx, 0, n - 1, out=idx)
    return lut[idx]


def pq_oetf_lut_lerp(luminance, lut):
    """C/D/E: LUT with linear interpolation."""
    n = len(lut)
    y = luminance / _PQ_PEAK_LUMINANCE
    np.clip(y, 0.0, 1.0, out=y)
    # Fractional index
    t = y * (n - 1)
    idx_lo = t.astype(np.int64)
    np.clip(idx_lo, 0, n - 2, out=idx_lo)
    frac = t - idx_lo
    # Linear interpolation
    return lut[idx_lo] * (1.0 - frac) + lut[idx_lo + 1] * frac


def build_pq_lut_sqrt(n_entries):
    """F: Hybrid LUT — sqrt-spaced input for better dark-region resolution."""
    # Input samples are sqrt-spaced: more density near 0
    t = np.linspace(0.0, 1.0, n_entries)
    lut_input_norm = t * t  # quadratic: denser near 0
    lut_output = pq_oetf(lut_input_norm * _PQ_PEAK_LUMINANCE)
    return lut_input_norm, lut_output.astype(np.float64)


def pq_oetf_lut_sqrt_lerp(luminance, lut_input_norm, lut_output):
    """F: Hybrid sqrt-spaced LUT with lerp."""
    n = len(lut_output)
    y = luminance / _PQ_PEAK_LUMINANCE
    np.clip(y, 0.0, 1.0, out=y)
    # Reverse the sqrt mapping: t = sqrt(y)
    t = np.sqrt(y)
    t_idx = t * (n - 1)
    idx_lo = t_idx.astype(np.int64)
    np.clip(idx_lo, 0, n - 2, out=idx_lo)
    frac = t_idx - idx_lo
    return lut_output[idx_lo] * (1.0 - frac) + lut_output[idx_lo + 1] * frac


# ============================================================
# BENCHMARK INFRASTRUCTURE
# ============================================================

def bench(fn, n=10):
    """Benchmark: return median, min, max in ms."""
    # Warmup
    fn()
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times)//2], times[0], times[-1]


def accuracy(ref, candidate):
    """Compute accuracy metrics: PQ signal space + nits space."""
    # PQ signal error
    pq_err = np.abs(ref - candidate)
    
    # Convert both to nits via PQ EOTF
    ref_nits = pq_eotf(ref)
    cand_nits = pq_eotf(candidate)
    nits_err = np.abs(ref_nits - cand_nits)
    
    return {
        "max_pq": float(np.max(pq_err)),
        "mean_pq": float(np.mean(pq_err)),
        "max_nits": float(np.max(nits_err)),
        "mean_nits": float(np.mean(nits_err)),
        "rmse_nits": float(np.sqrt(np.mean(nits_err**2))),
        "p99_nits": float(np.percentile(nits_err, 99)),
        "p999_nits": float(np.percentile(nits_err, 99.9)),
        "p9999_nits": float(np.percentile(nits_err, 99.99)),
    }


def accuracy_ranges(ref, candidate, luminance_norm):
    """Accuracy in specific luminance ranges."""
    ranges = {
        "0-0.001 (0-10 nits)": (0, 0.001),
        "0.001-0.01 (10-100 nits)": (0.001, 0.01),
        "0.01-0.1 (100-1000 nits)": (0.01, 0.1),
        "0.1-1.0 (1000-10000 nits)": (0.1, 1.0),
    }
    results = {}
    ref_nits = pq_eotf(ref)
    cand_nits = pq_eotf(candidate)
    for name, (lo, hi) in ranges.items():
        mask = (luminance_norm >= lo) & (luminance_norm < hi)
        if np.any(mask):
            err = np.abs(ref_nits[mask] - cand_nits[mask])
            results[name] = {
                "max_nits": float(np.max(err)),
                "mean_nits": float(np.mean(err)),
                "count": int(np.sum(mask)),
            }
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 76)
    print("P2.16: PQ OETF PERFORMANCE MICROBENCHMARK")
    print("=" * 76)

    # --- Test data ---
    # ROI-sized data (production use case)
    roi_lum = rng.random(560 * 3840).astype(np.float64) * _PQ_PEAK_LUMINANCE
    
    # 1M random
    rand_1m = rng.random(1_000_000).astype(np.float64) * _PQ_PEAK_LUMINANCE
    
    # Dense grid [0, 1] normalized
    dense_norm = np.linspace(0.0, 1.0, 1_000_000)
    dense_lum = dense_norm * _PQ_PEAK_LUMINANCE
    
    # Extra-dense low range [0, 0.01] = [0, 100 nits]
    low_norm = np.linspace(0.0, 0.01, 500_000)
    low_lum = low_norm * _PQ_PEAK_LUMINANCE

    # --- Build LUTs ---
    lut_16 = build_pq_lut(65536)          # 16-bit
    lut_18 = build_pq_lut(262144)         # 18-bit
    lut_20 = build_pq_lut(1048576)        # 20-bit
    sqrt_input, sqrt_lut_16 = build_pq_lut_sqrt(65536)
    sqrt_input_18, sqrt_lut_18 = build_pq_lut_sqrt(262144)

    # --- Reference output ---
    ref_roi = pq_oetf_current(roi_lum)
    ref_dense = pq_oetf_current(dense_lum)
    ref_low = pq_oetf_current(low_lum)

    # ============================================================
    # ACCURACY
    # ============================================================
    print("\n## ACCURACY (dense 1M grid [0, 10000] nits)")
    print("-" * 76)

    variants = {
        "A: Current": lambda lum: pq_oetf_current(lum),
        "B: NumPy Opt": lambda lum: pq_oetf_numpy_opt(lum),
        "C: LUT 16-bit nearest": lambda lum: pq_oetf_lut_nearest(lum, lut_16),
        "C: LUT 16-bit lerp": lambda lum: pq_oetf_lut_lerp(lum, lut_16),
        "D: LUT 18-bit nearest": lambda lum: pq_oetf_lut_nearest(lum, lut_18),
        "D: LUT 18-bit lerp": lambda lum: pq_oetf_lut_lerp(lum, lut_18),
        "E: LUT 20-bit nearest": lambda lum: pq_oetf_lut_nearest(lum, lut_20),
        "E: LUT 20-bit lerp": lambda lum: pq_oetf_lut_lerp(lum, lut_20),
        "F: Sqrt LUT 16-bit lerp": lambda lum: pq_oetf_lut_sqrt_lerp(lum, sqrt_input, sqrt_lut_16),
        "F: Sqrt LUT 18-bit lerp": lambda lum: pq_oetf_lut_sqrt_lerp(lum, sqrt_input_18, sqrt_lut_18),
    }

    print(f"\n  {'Variant':<28} {'Max PQ':<10} {'Max nits':<10} {'Mean nits':<11} {'RMSE nits':<10} {'P99.9':<10} {'P99.99':<10}")
    print(f"  {'-'*89}")
    
    acc_results = {}
    for name, fn in variants.items():
        out = fn(dense_lum)
        acc = accuracy(ref_dense, out)
        acc_results[name] = acc
        if name == "A: Current":
            print(f"  {name:<28} {'(reference)':<10} {'(reference)':<10} {'(reference)':<11} {'(ref)':<10} {'(ref)':<10} {'(ref)':<10}")
        else:
            print(f"  {name:<28} {acc['max_pq']:<10.2e} {acc['max_nits']:<10.4f} {acc['mean_nits']:<11.6f} {acc['rmse_nits']:<10.6f} {acc['p999_nits']:<10.6f} {acc['p9999_nits']:<10.6f}")

    # Low-range accuracy
    print(f"\n  LOW RANGE [0-100 nits] accuracy:")
    print(f"  {'Variant':<28} {'Max nits':<10} {'Mean nits':<11} {'P99.9':<10}")
    print(f"  {'-'*59}")
    for name, fn in variants.items():
        if name == "A: Current":
            print(f"  {name:<28} {'(reference)':<10} {'(reference)':<11} {'(ref)':<10}")
            continue
        out = fn(low_lum)
        ref_l = pq_oetf_current(low_lum)
        nits_err = np.abs(pq_eotf(ref_l) - pq_eotf(out))
        print(f"  {name:<28} {np.max(nits_err):<10.6f} {np.mean(nits_err):<11.8f} {np.percentile(nits_err, 99.9):<10.6f}")

    # Range-segmented accuracy for key variants
    print(f"\n  RANGE-SEGMENTED ACCURACY (selected variants):")
    for vname in ["C: LUT 16-bit lerp", "D: LUT 18-bit lerp", "E: LUT 20-bit lerp", "F: Sqrt LUT 16-bit lerp"]:
        out = variants[vname](dense_lum)
        ranges = accuracy_ranges(ref_dense, out, dense_norm)
        print(f"\n  {vname}:")
        for rname, rdata in ranges.items():
            print(f"    {rname:<30} max={rdata['max_nits']:.4f} nits  mean={rdata['mean_nits']:.6f} nits")

    # ============================================================
    # BENCHMARK
    # ============================================================
    print(f"\n\n## PERFORMANCE (ROI 560×3840 = {560*3840:,} pixels)")
    print("-" * 76)

    bench_variants = {
        "A: Current": lambda: pq_oetf_current(roi_lum),
        "B: NumPy Opt": lambda: pq_oetf_numpy_opt(roi_lum),
        "C: LUT 16-bit nearest": lambda: pq_oetf_lut_nearest(roi_lum, lut_16),
        "C: LUT 16-bit lerp": lambda: pq_oetf_lut_lerp(roi_lum, lut_16),
        "D: LUT 18-bit nearest": lambda: pq_oetf_lut_nearest(roi_lum, lut_18),
        "D: LUT 18-bit lerp": lambda: pq_oetf_lut_lerp(roi_lum, lut_18),
        "E: LUT 20-bit nearest": lambda: pq_oetf_lut_nearest(roi_lum, lut_20),
        "E: LUT 20-bit lerp": lambda: pq_oetf_lut_lerp(roi_lum, lut_20),
        "F: Sqrt LUT 16-bit lerp": lambda: pq_oetf_lut_sqrt_lerp(roi_lum, sqrt_input, sqrt_lut_16),
        "F: Sqrt LUT 18-bit lerp": lambda: pq_oetf_lut_sqrt_lerp(roi_lum, sqrt_input_18, sqrt_lut_18),
    }

    print(f"\n  {'Variant':<28} {'Median ms':<10} {'Min ms':<8} {'Max ms':<8} {'FPS-eq':<8} {'Speedup':<8}")
    print(f"  {'-'*70}")
    
    timing_results = {}
    ref_median = None
    for name, fn in bench_variants.items():
        median, mn, mx = bench(fn, n=10)
        timing_results[name] = (median, mn, mx)
        if ref_median is None:
            ref_median = median
        speedup = ref_median / median
        fps = 1000.0 / median if median > 0 else 0
        print(f"  {name:<28} {median:<10.1f} {mn:<8.1f} {mx:<8.1f} {fps:<8.1f} {speedup:<8.2f}×")

    # ============================================================
    # MEMORY
    # ============================================================
    print(f"\n\n## MEMORY")
    print("-" * 76)
    print(f"  {'Variant':<28} {'Entries':<10} {'Bytes':<12} {'KB':<8}")
    print(f"  {'-'*58}")
    luts_info = [
        ("C: LUT 16-bit", lut_16),
        ("D: LUT 18-bit", lut_18),
        ("E: LUT 20-bit", lut_20),
        ("F: Sqrt LUT 16-bit", sqrt_lut_16),
        ("F: Sqrt LUT 18-bit", sqrt_lut_18),
    ]
    for name, lut in luts_info:
        print(f"  {name:<28} {len(lut):<10} {lut.nbytes:<12} {lut.nbytes/1024:<8.1f}")

    # ============================================================
    # FULL TRANSFORM IMPACT ESTIMATE
    # ============================================================
    print(f"\n\n## FULL TRANSFORM IMPACT ESTIMATE")
    print("-" * 76)
    # Current apply_shot_transform is ~476ms, of which PQ OETF is a portion
    # Also BT.1886 EOTF is done (linearization)
    # The delinearize() call does: pq_oetf(linear * peak_nits)
    # Let's benchmark delinearize equivalent
    from auto_openmatte.core.transfer_functions import delinearize
    
    bench_frame = rng.random((560, 3840, 3)).astype(np.float64)
    t_delin_med, _, _ = bench(lambda: delinearize(bench_frame, "smpte2084"), n=5)
    print(f"  Current delinearize (PQ, 560×3840×3): {t_delin_med:.1f} ms")
    print(f"  This is the PQ OETF cost inside apply_shot_transform")
    print(f"  Current total transform: ~476 ms")
    print(f"  PQ fraction: {t_delin_med/476*100:.1f}%")
    
    # Best LUT candidate time scaled to 3-channel
    best_lut_name = min(
        [(n,t[0]) for n,t in timing_results.items() if "LUT" in n],
        key=lambda x: x[1]
    )
    # 3 channels = 3× single-channel time
    est_3ch = best_lut_name[1] * 3
    print(f"\n  Best LUT candidate: {best_lut_name[0]}")
    print(f"    Single-channel: {best_lut_name[1]:.1f} ms")
    print(f"    Estimated 3-channel: {est_3ch:.1f} ms")
    print(f"    Estimated savings: {t_delin_med - est_3ch:.1f} ms")
    print(f"    Estimated new transform: {476 - (t_delin_med - est_3ch):.0f} ms")

    # ============================================================
    # FINAL SUMMARY TABLE
    # ============================================================
    print(f"\n\n{'='*76}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*76}")
    
    print(f"\n  {'Variant':<28} {'Med ms':<8} {'FPS':<7} {'Max PQ err':<11} {'Max nits':<10} {'Mean nits':<10} {'P99.9 nits':<10} {'Memory':<8}")
    print(f"  {'-'*92}")
    for name in bench_variants:
        med = timing_results[name][0]
        fps = 1000.0 / med
        acc = acc_results[name]
        if name == "A: Current":
            mem = "0"
            print(f"  {name:<28} {med:<8.1f} {fps:<7.1f} {'ref':<11} {'ref':<10} {'ref':<10} {'ref':<10} {mem:<8}")
        else:
            # Find memory
            mem_str = "-"
            for ln, la in luts_info:
                if ln.split(":")[0] in name.split(":")[0] and ln.split()[-1] in name:
                    mem_str = f"{la.nbytes//1024}KB"
                    break
            if "B:" in name:
                mem_str = "0"
            elif "16" in name and "Sqrt" not in name:
                mem_str = f"{lut_16.nbytes//1024}KB"
            elif "18" in name and "Sqrt" not in name:
                mem_str = f"{lut_18.nbytes//1024}KB"
            elif "20" in name:
                mem_str = f"{lut_20.nbytes//1024}KB"
            elif "Sqrt" in name and "16" in name:
                mem_str = f"{sqrt_lut_16.nbytes//1024}KB"
            elif "Sqrt" in name and "18" in name:
                mem_str = f"{sqrt_lut_18.nbytes//1024}KB"
            
            print(f"  {name:<28} {med:<8.1f} {fps:<7.1f} {acc['max_pq']:<11.2e} {acc['max_nits']:<10.4f} {acc['mean_nits']:<10.6f} {acc['p999_nits']:<10.6f} {mem_str:<8}")

    # ============================================================
    # RECOMMENDATIONS
    # ============================================================
    print(f"\n\n{'='*76}")
    print("RECOMMENDATIONS")
    print(f"{'='*76}")
    
    # Speed ranking
    speed_rank = sorted(timing_results.items(), key=lambda x: x[1][0])
    print(f"\n  Speed ranking (fastest first):")
    for i, (name, (med, _, _)) in enumerate(speed_rank, 1):
        print(f"    {i}. {name}: {med:.1f} ms ({ref_median/med:.2f}×)")
    
    # Accuracy ranking (excluding A which is reference)
    acc_rank = sorted(
        [(n, a) for n, a in acc_results.items() if n != "A: Current"],
        key=lambda x: x[1]["max_nits"]
    )
    print(f"\n  Accuracy ranking (best first, by max nits error):")
    for i, (name, acc) in enumerate(acc_rank, 1):
        print(f"    {i}. {name}: max {acc['max_nits']:.4f} nits, mean {acc['mean_nits']:.6f} nits")
    
    # Trade-off
    print(f"\n  Speed/accuracy trade-off:")
    for name in ["B: NumPy Opt", "C: LUT 16-bit lerp", "D: LUT 18-bit lerp", "E: LUT 20-bit lerp", "F: Sqrt LUT 16-bit lerp"]:
        med = timing_results[name][0]
        acc = acc_results[name]
        print(f"    {name}: {ref_median/med:.2f}× speed, max err {acc['max_nits']:.4f} nits")
    
    print(f"\n  STOP — benchmark complete. No production changes made.")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
