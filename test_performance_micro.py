"""P2.11: Performance microbenchmark on synthetic 4K data."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.core.models import ShotTransform, GeometryModel
from auto_openmatte.core.transfer_functions import linearize, delinearize, pq_oetf, bt1886_eotf
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve, _PEAK_NITS, _LOG_EPS
from auto_openmatte.processing.transform import apply_shot_transform, _M_709_TO_2020, _LUM_R_2020, _LUM_G_2020, _LUM_B_2020
from auto_openmatte.pipeline.compose import composite_extend
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.core.config import ColorConfig

# Generate synthetic data
rng = np.random.default_rng(42)
RESOLUTIONS = [(960, 540), (1920, 1080), (3840, 2160)]

# Generate a realistic log-domain curve
sdr_fit = rng.random(50000) * 0.5 + 0.001
hdr_fit = sdr_fit * 0.002 + rng.random(50000) * 0.0001
config = ColorConfig(luminance_bins=256, low_percentile=1.0, high_percentile=99.9)
curve = estimate_luminance_curve(sdr_fit, hdr_fit, config=config)
transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0, contrast=1.0, saturation=1.0, confidence=0.99)


def bench(fn, label, n_runs=3):
    """Benchmark function, return median ms."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    med = sorted(times)[len(times)//2]
    return med


def main():
    print("=" * 70)
    print("P2.11: PERFORMANCE MICROBENCHMARK")
    print("=" * 70)

    W, H = 3840, 2160
    frame = rng.random((H, W, 3)).astype(np.float64) * 0.5  # SDR-like [0, 0.5]

    print(f"\n--- ETAP 2: Individual operations at {W}x{H} ---")
    print(f"  dtype: float64, shape: ({H},{W},3), size: {H*W*3*8/(1024*1024):.0f} MB")
    print(f"  {'Operation':<40} {'Median ms':<12} {'% of total'}")
    print(f"  {'-'*64}")

    # A. BT.1886 linearization
    t_a = bench(lambda: bt1886_eotf(frame), "BT.1886 EOTF (clip+power)")
    print(f"  {'A. BT.1886 linearize (clip+pow 2.4)':<40} {t_a:<12.1f}")

    # B. 709→2020 matrix
    linear = bt1886_eotf(frame)
    def do_matrix():
        s = linear.shape
        r = linear.reshape(-1, 3) @ _M_709_TO_2020.T
        return r.reshape(s)
    t_b = bench(do_matrix, "709→2020 matrix")
    print(f"  {'B. BT.709→BT.2020 matrix multiply':<40} {t_b:<12.1f}")

    # np.maximum after matrix
    linear_2020 = do_matrix()
    t_b2 = bench(lambda: np.maximum(linear_2020, 0.0), "np.maximum clamp")
    print(f"  {'B2. np.maximum(0) clamp':<40} {t_b2:<12.1f}")

    # C. Luminance computation
    def do_lum():
        return _LUM_R_2020 * linear_2020[...,0] + _LUM_G_2020 * linear_2020[...,1] + _LUM_B_2020 * linear_2020[...,2]
    t_c = bench(do_lum, "BT.2020 luminance")
    print(f"  {'C. BT.2020 luminance (weighted sum)':<40} {t_c:<12.1f}")

    # D. Log-domain curve
    lum = do_lum()
    t_d = bench(lambda: apply_luminance_curve(lum, curve), "Log curve apply")
    print(f"  {'D. apply_luminance_curve (log+PCHIP)':<40} {t_d:<12.1f}")

    # E. Ratio calculation
    lum_mapped = apply_luminance_curve(lum, curve)
    def do_ratio():
        safe = lum > 1e-6
        r = np.ones_like(lum)
        np.divide(lum_mapped, lum, out=r, where=safe)
        return r
    t_e = bench(do_ratio, "ratio")
    print(f"  {'E. Ratio (safe divide)':<40} {t_e:<12.1f}")

    # F. RGB scaling
    ratio = do_ratio()
    def do_scale():
        r = linear_2020 * ratio[..., np.newaxis]
        return np.maximum(r, 0.0)
    t_f = bench(do_scale, "RGB scale")
    print(f"  {'F. RGB × ratio + clamp':<40} {t_f:<12.1f}")

    # G. PQ OETF (delinearize)
    scaled = do_scale()
    t_g = bench(lambda: delinearize(scaled, "smpte2084", peak_nits=10000.0), "PQ OETF")
    print(f"  {'G. PQ OETF (delinearize)':<40} {t_g:<12.1f}")

    # H. scipy.ndimage.zoom (HDR resize 1600→1600, scale=1.0)
    from scipy.ndimage import zoom
    hdr_frame = rng.random((1600, 3840, 3)).astype(np.float64) * 0.3
    def do_zoom():
        return zoom(hdr_frame, (1.0, 1.0, 1.0), order=1)
    t_h = bench(do_zoom, "scipy zoom (identity)")
    print(f"  {'H. scipy.ndimage.zoom (3840x1600)':<40} {t_h:<12.1f}")

    # Total transform
    t_total = bench(lambda: apply_shot_transform(frame, transform), "Total transform")
    print(f"\n  {'TOTAL apply_shot_transform()':<40} {t_total:<12.1f}")

    # Breakdown %
    sum_parts = t_a + t_b + t_b2 + t_c + t_d + t_e + t_f + t_g
    print(f"\n  Sum of parts: {sum_parts:.0f} ms (vs total {t_total:.0f} ms)")
    print(f"  Overhead/other: {t_total - sum_parts:.0f} ms")

    parts = [("A.BT1886", t_a), ("B.Matrix", t_b), ("B2.Clamp", t_b2),
             ("C.Lum", t_c), ("D.Curve", t_d), ("E.Ratio", t_e),
             ("F.Scale", t_f), ("G.PQ", t_g)]
    print(f"\n  {'Op':<12} {'ms':<8} {'%':<6}")
    print(f"  {'-'*26}")
    for name, t in sorted(parts, key=lambda x: -x[1]):
        print(f"  {name:<12} {t:<8.0f} {t/t_total*100:<6.1f}")

    # === ETAP 3: SCALING TEST ===
    print(f"\n\n{'='*70}")
    print("ETAP 3: SCALING TEST")
    print(f"{'='*70}")
    print(f"\n  {'Resolution':<16} {'Median ms':<14} {'FPS':<8} {'Relative'}")
    print(f"  {'-'*50}")
    base_time = None
    for w, h in RESOLUTIONS:
        f = rng.random((h, w, 3)).astype(np.float64) * 0.5
        t = bench(lambda f=f: apply_shot_transform(f, transform), f"{w}x{h}", n_runs=3)
        fps = 1000.0 / t
        if base_time is None:
            base_time = t
        print(f"  {f'{w}x{h}':<16} {t:<14.0f} {fps:<8.2f} {t/base_time:.1f}x")

    # === ETAP 4: MEMORY ===
    print(f"\n\n{'='*70}")
    print("ETAP 4: MEMORY / ALLOCATION AUDIT")
    print(f"{'='*70}")
    pixels = 3840 * 2160
    bytes_per_f64 = 8
    frame_size_mb = pixels * 3 * bytes_per_f64 / (1024*1024)
    lum_size_mb = pixels * bytes_per_f64 / (1024*1024)
    print(f"\n  Input dtype: float64")
    print(f"  Full-frame RGB (H×W×3): {frame_size_mb:.1f} MB")
    print(f"  Single-channel (H×W): {lum_size_mb:.1f} MB")
    print(f"\n  Allocations during apply_shot_transform():")
    print(f"    1. linear_709 = linearize(frame)         → {frame_size_mb:.1f} MB NEW (clip+power creates copy)")
    print(f"    2. reshape(-1,3) @ M.T                   → {frame_size_mb:.1f} MB NEW (matmul output)")
    print(f"    3. linear_2020 = reshape(shape)          → 0 MB (view)")
    print(f"    4. np.maximum(linear_2020, 0)            → {frame_size_mb:.1f} MB NEW (or in-place if same ref)")
    print(f"    5. lum = weighted sum                    → {lum_size_mb:.1f} MB NEW")
    print(f"    6. lum_mapped = apply_curve(lum)         → {lum_size_mb:.1f} MB NEW (internal: log, interp, pow)")
    print(f"    7. ratio = ones_like + divide            → {lum_size_mb:.1f} MB NEW")
    print(f"    8. linear * ratio[...,np.newaxis]        → {frame_size_mb:.1f} MB NEW (broadcast multiply)")
    print(f"    9. np.maximum(linear, 0)                 → {frame_size_mb:.1f} MB NEW")
    print(f"   10. delinearize(linear)                   → {frame_size_mb:.1f} MB NEW (clip+complex math)")
    print(f"   11. np.clip(hdr_signal, 0, 1)            → {frame_size_mb:.1f} MB NEW")
    print(f"\n  Peak ~7 full-frame + 3 single-channel = {7*frame_size_mb + 3*lum_size_mb:.0f} MB")
    print(f"  Estimated peak memory: ~{(7*frame_size_mb + 3*lum_size_mb):.0f} MB per frame")

    # === ETAP 5: COMPOSITE_EXTEND overhead ===
    print(f"\n\n{'='*70}")
    print("ETAP 5: composite_extend() overhead")
    print(f"{'='*70}")
    # The composite_extend calls apply_shot_transform on FULL OM frame
    # then does scipy.ndimage.zoom on HDR + blending
    print(f"\n  composite_extend() performs:")
    print(f"    1. apply_shot_transform(om_frame) on FULL 3840×2160")
    print(f"       → transforms ALL pixels including overlap region")
    print(f"       → only extension region (560 rows) is actually USED")
    print(f"       → 1600/2160 = 74% of computation is WASTED")
    print(f"    2. scipy.ndimage.zoom(hdr_frame, (1.0, 1.0, 1.0))")
    print(f"       → identity zoom when scale=1.0 (NO-OP but still runs)")
    print(f"       → ~{t_h:.0f} ms overhead for nothing")
    print(f"    3. Blending with mask (per-channel loop)")
    print(f"       → 3 iterations × simple arithmetic")

    print(f"\n{'='*70}")
    print("TOP 5 BOTTLENECKS")
    print(f"{'='*70}")
    print(f"""
  1. WASTED COMPUTATION ({int(t_total*0.74)} ms)
     apply_shot_transform() runs on full 3840×2160 but only
     560/2160 = 26% rows are actually used in extension.
     FIX: Transform only extension rows.

  2. LOG-DOMAIN CURVE ({int(t_d)} ms, {t_d/t_total*100:.0f}%)
     apply_luminance_curve: log10 + PchipInterpolator + pow10
     Creates ~3 temporary arrays.
     FIX: Pre-compute LUT (e.g. 65536-entry), replace PCHIP with lookup.

  3. PQ OETF ({int(t_g)} ms, {t_g/t_total*100:.0f}%)
     Complex power operations (clip, pow(1/m2), divisions, pow(1/m1)).
     FIX: Pre-compute LUT (16-bit input → 16-bit output).

  4. BT.1886 LINEARIZATION ({int(t_a)} ms, {t_a/t_total*100:.0f}%)
     clip + pow(2.4) on full frame.
     FIX: LUT (uint16 → float) or process only extension.

  5. MATRIX MULTIPLY ({int(t_b)} ms, {t_b/t_total*100:.0f}%)
     Reshape + matmul. Efficient for NumPy but creates large copy.
     FIX: In-place or reduce to extension only.

  SCIPY ZOOM ({int(t_h)} ms) — identity zoom is wasteful.
     FIX: Skip zoom when scale factors are exactly 1.0.
""")

    print(f"{'='*70}")
    print("P2.11 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
