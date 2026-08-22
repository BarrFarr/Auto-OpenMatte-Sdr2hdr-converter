"""P2.12: Benchmark — ROI optimization + float32 + LUT accuracy tests."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from scipy.interpolate import PchipInterpolator

from auto_openmatte.core.models import ShotTransform, GeometryModel
from auto_openmatte.core.transfer_functions import linearize, delinearize, pq_oetf, bt1886_eotf
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve, _PEAK_NITS, _LOG_EPS
from auto_openmatte.processing.transform import apply_shot_transform
from auto_openmatte.pipeline.compose import composite_extend
from auto_openmatte.processing.overlap import generate_extension_mask
from auto_openmatte.core.config import ColorConfig

rng = np.random.default_rng(42)

# Generate curve
sdr_fit = rng.random(50000) * 0.5 + 0.001
hdr_fit = sdr_fit * 0.002 + rng.random(50000) * 0.0001
config = ColorConfig(luminance_bins=256, low_percentile=1.0, high_percentile=99.9)
curve = estimate_luminance_curve(sdr_fit, hdr_fit, config=config)
transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0, contrast=1.0, saturation=1.0, confidence=0.99)


def bench(fn, n=3):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    return sorted(times)[len(times)//2]


def main():
    print("=" * 70)
    print("P2.12: PERFORMANCE BENCHMARK")
    print("=" * 70)

    # === ETAP 1: ROI-ONLY COMPOSITE ===
    print(f"\n{'='*70}")
    print("ETAP 1: ROI-ONLY composite_extend() benchmark")
    print(f"{'='*70}")

    om_frame = rng.random((2160, 3840, 3)).astype(np.float64) * 0.5
    hdr_frame = rng.random((1600, 3840, 3)).astype(np.float64) * 0.3
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.95, is_global=True)

    # Create minimal extension mask (all 0 in overlap = pure HDR placement)
    ext_mask = np.ones((2160, 3840), dtype=np.float64)
    ext_mask[280:1880, :] = 0.0  # HDR region

    # Benchmark new ROI composite
    t_composite = bench(lambda: composite_extend(hdr_frame, om_frame, transform, geom, ext_mask))
    print(f"  composite_extend() (ROI-only): {t_composite:.0f} ms")

    # Compare: transform only extension (560 rows) vs full frame
    ext_only = om_frame[:280, :, :]  # Top extension
    t_ext_transform = bench(lambda: apply_shot_transform(ext_only, transform))
    t_full_transform = bench(lambda: apply_shot_transform(om_frame, transform))
    print(f"  apply_shot_transform (full 2160): {t_full_transform:.0f} ms")
    print(f"  apply_shot_transform (top 280):   {t_ext_transform:.0f} ms")
    print(f"  Speedup from ROI: {t_full_transform / (t_ext_transform * 2):.1f}x (top+bottom)")

    # === ETAP 3: FLOAT32 ACCURACY ===
    print(f"\n{'='*70}")
    print("ETAP 3: FLOAT32 vs FLOAT64 accuracy")
    print(f"{'='*70}")

    test_frame = rng.random((560, 3840, 3)).astype(np.float64) * 0.5
    ref_output = apply_shot_transform(test_frame, transform)

    test_f32 = test_frame.astype(np.float32)
    out_f32 = apply_shot_transform(test_f32, transform)

    diff = np.abs(ref_output.astype(np.float64) - out_f32.astype(np.float64))
    print(f"  Max RGB difference:  {diff.max():.8f}")
    print(f"  Mean RGB difference: {diff.mean():.10f}")
    # Convert to nits for luminance comparison
    from auto_openmatte.core.transfer_functions import pq_eotf
    ref_lum = pq_eotf(0.2627*ref_output[...,0]+0.6780*ref_output[...,1]+0.0593*ref_output[...,2])
    f32_lum = pq_eotf(0.2627*out_f32[...,0]+0.6780*out_f32[...,1]+0.0593*out_f32[...,2])
    lum_diff = np.abs(ref_lum - f32_lum)
    print(f"  Max lum difference:  {lum_diff.max():.4f} nits")
    print(f"  Mean lum difference: {lum_diff.mean():.6f} nits")
    print(f"  P99 lum difference:  {np.percentile(lum_diff, 99):.6f} nits")

    t_f64 = bench(lambda: apply_shot_transform(test_frame, transform))
    t_f32 = bench(lambda: apply_shot_transform(test_f32, transform))
    print(f"\n  float64: {t_f64:.0f} ms")
    print(f"  float32: {t_f32:.0f} ms")
    print(f"  Speedup: {t_f64/t_f32:.2f}x")

    # === ETAP 4: LUT ACCURACY ===
    print(f"\n{'='*70}")
    print("ETAP 4: LUT ACCURACY TESTS")
    print(f"{'='*70}")

    LUT_SIZE = 65536
    lut_input = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float64)

    # A. BT.1886 EOTF LUT
    lut_bt1886 = bt1886_eotf(lut_input)
    test_vals = rng.random(1000000)
    ref_bt1886 = bt1886_eotf(test_vals)
    lut_indices = np.clip((test_vals * (LUT_SIZE - 1)).astype(np.int64), 0, LUT_SIZE - 1)
    approx_bt1886 = lut_bt1886[lut_indices]
    err_a = np.abs(ref_bt1886 - approx_bt1886)
    print(f"\n  A. BT.1886 EOTF LUT ({LUT_SIZE} entries):")
    print(f"     Max error: {err_a.max():.8f}")
    print(f"     Mean error: {err_a.mean():.10f}")

    # B. PQ OETF LUT
    lut_pq_oetf = pq_oetf(lut_input * _PEAK_NITS).astype(np.float64)
    ref_pq = pq_oetf(test_vals * _PEAK_NITS)
    approx_pq = lut_pq_oetf[lut_indices]
    err_b = np.abs(ref_pq - approx_pq)
    print(f"\n  B. PQ OETF LUT ({LUT_SIZE} entries):")
    print(f"     Max error: {err_b.max():.8f}")
    print(f"     Mean error: {err_b.mean():.10f}")

    # C. PQ EOTF LUT (signal→nits)
    from auto_openmatte.core.transfer_functions import pq_eotf as pq_eotf_fn
    lut_pq_eotf = pq_eotf_fn(lut_input) / _PEAK_NITS
    ref_pq_eotf = pq_eotf_fn(test_vals) / _PEAK_NITS
    approx_pq_eotf = lut_pq_eotf[lut_indices]
    err_c = np.abs(ref_pq_eotf - approx_pq_eotf)
    print(f"\n  C. PQ EOTF LUT ({LUT_SIZE} entries):")
    print(f"     Max error: {err_c.max():.8f} normalized")
    print(f"     Max error: {err_c.max()*_PEAK_NITS:.4f} nits")
    print(f"     Mean error: {err_c.mean()*_PEAK_NITS:.6f} nits")

    # D. Log-domain curve LUT
    # Curve operates on linear normalized [0,1] → [0,1]
    lut_curve = apply_luminance_curve(lut_input, curve)
    ref_curve = apply_luminance_curve(test_vals * 0.5, curve)  # Scale to realistic range
    lut_indices_curve = np.clip((test_vals * 0.5 * (LUT_SIZE-1)).astype(np.int64), 0, LUT_SIZE-1)
    approx_curve = lut_curve[lut_indices_curve]
    err_d = np.abs(ref_curve - approx_curve) * _PEAK_NITS
    print(f"\n  D. Luminance curve LUT ({LUT_SIZE} entries):")
    print(f"     Max error: {err_d.max():.4f} nits")
    print(f"     Mean error: {err_d.mean():.6f} nits")
    print(f"     P99 error: {np.percentile(err_d, 99):.6f} nits")

    # Critical values check
    print(f"\n  Critical value check (LUT curve):")
    critical = np.array([0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5])
    ref_critical = apply_luminance_curve(critical, curve)
    for v, r in zip(critical, ref_critical):
        idx = min(int(v * (LUT_SIZE-1)), LUT_SIZE-1)
        lut_val = lut_curve[idx]
        print(f"     input={v:.2e}: ref={r*_PEAK_NITS:.4f} nits, LUT={lut_val*_PEAK_NITS:.4f} nits, "
              f"err={abs(r-lut_val)*_PEAK_NITS:.4f}")

    # LUT speed benchmark
    print(f"\n  LUT speed benchmark (1M lookups):")
    big_test = rng.random(1000000).astype(np.float64)
    t_ref = bench(lambda: apply_luminance_curve(big_test * 0.5, curve), n=3)
    def lut_lookup():
        idx = np.clip((big_test * 0.5 * (LUT_SIZE-1)).astype(np.int64), 0, LUT_SIZE-1)
        return lut_curve[idx]
    t_lut = bench(lut_lookup, n=3)
    print(f"     PCHIP reference: {t_ref:.1f} ms")
    print(f"     LUT lookup:      {t_lut:.1f} ms")
    print(f"     Speedup:         {t_ref/t_lut:.1f}x")

    # === ETAP 6: FINAL COMPARISON ===
    print(f"\n\n{'='*70}")
    print("ETAP 6: FINAL BENCHMARK COMPARISON")
    print(f"{'='*70}")

    # Extension-only (280 rows top + 280 rows bottom)
    ext_frame_560 = rng.random((560, 3840, 3)).astype(np.float64) * 0.5
    t_ext = bench(lambda: apply_shot_transform(ext_frame_560, transform))

    print(f"\n  {'Configuration':<40} {'ms/frame':<12} {'FPS':<8} {'Speedup'}")
    print(f"  {'-'*68}")
    print(f"  {'OLD: full 3840×2160 transform':<40} {t_full_transform:<12.0f} {1000/t_full_transform:<8.2f} 1.0x")
    print(f"  {'NEW: ROI 3840×560 only':<40} {t_ext:<12.0f} {1000/t_ext:<8.2f} {t_full_transform/t_ext:.1f}x")
    print(f"  {'NEW: composite_extend (ROI+HDR)':<40} {t_composite:<12.0f} {1000/t_composite:<8.2f} {t_full_transform/t_composite:.1f}x")

    # With float32
    ext_f32 = ext_frame_560.astype(np.float32)
    t_ext_f32 = bench(lambda: apply_shot_transform(ext_f32, transform))
    print(f"  {'NEW: ROI float32':<40} {t_ext_f32:<12.0f} {1000/t_ext_f32:<8.2f} {t_full_transform/t_ext_f32:.1f}x")

    print(f"\n  Peak memory estimate:")
    print(f"    OLD (full frame f64): ~1519 MB")
    print(f"    NEW (ROI 560 rows f64): ~{int(1519 * 560/2160)} MB")
    print(f"    NEW (ROI 560 rows f32): ~{int(1519 * 560/2160 / 2)} MB")

    print(f"\n{'='*70}")
    print("P2.12 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
