"""P2.9.2: Regression audit — black crush after P2.9.1 fix.

Compares apply_luminance_curve behavior before and after fix,
analyzing continuity and dark-pixel treatment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve, _PEAK_NITS, _LOG_EPS
from auto_openmatte.core.config import ColorConfig
from scipy.interpolate import PchipInterpolator


def apply_curve_OLD(sdr_luminance, curve):
    """BEFORE P2.9.1: extrapolate=True, no black protection."""
    x_points = np.array([p[0] for p in curve])
    y_points = np.array([p[1] for p in curve])
    sdr_log = np.log10(sdr_luminance * _PEAK_NITS + _LOG_EPS)
    interpolator = PchipInterpolator(x_points, y_points, extrapolate=True)
    hdr_log = interpolator(sdr_log)
    hdr_nits = np.power(10.0, hdr_log) - _LOG_EPS
    return np.maximum(hdr_nits, 0.0) / _PEAK_NITS


def main():
    print("=" * 70)
    print("P2.9.2: REGRESSION AUDIT — BLACK CRUSH")
    print("=" * 70)

    # Generate a realistic curve (same as production)
    print("\n--- Generating realistic Log P1-P99.9 curve ---")
    rng = np.random.default_rng(42)
    # Simulate realistic BR2049 data: mostly dark
    sdr = rng.random(100000) * 0.5 + 0.001
    hdr = sdr * 0.002 + rng.random(100000) * 0.0001  # Very dark HDR
    config = ColorConfig(luminance_bins=256, low_percentile=1.0, high_percentile=99.9)
    curve = estimate_luminance_curve(sdr, hdr, config=config)
    print(f"  Curve: {len(curve)} points")
    print(f"  Curve x range: [{curve[0][0]:.4f}, {curve[-1][0]:.4f}] (log domain)")

    # Convert curve x_min to physical units
    curve_x_min = curve[0][0]
    # What linear normalized value maps to curve_x_min?
    # sdr_log = log10(sdr_linear * 10000 + 1e-6)
    # curve_x_min = log10(sdr_linear_min * 10000 + 1e-6)
    sdr_linear_at_min = (10**curve_x_min - _LOG_EPS) / _PEAK_NITS
    print(f"  curve_x_min = {curve_x_min:.4f} → SDR linear = {sdr_linear_at_min:.6f}")
    print(f"  BLACK_THRESHOLD = 1e-6 → log = {np.log10(1e-6 * _PEAK_NITS + _LOG_EPS):.4f}")
    print(f"  Gap: SDR values in [{1e-6:.2e}, {sdr_linear_at_min:.2e}] get CLAMPED to curve_x_min")

    # ===== CONTINUITY TABLE =====
    print(f"\n{'='*70}")
    print("CONTINUITY TABLE: OLD vs NEW")
    print(f"{'='*70}")

    test_inputs = np.array([
        0, 1e-8, 1e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
        1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2
    ])

    old_output = apply_curve_OLD(test_inputs, curve)
    new_output = apply_luminance_curve(test_inputs, curve)

    print(f"\n  {'Input SDR':<12} {'OLD output':<14} {'NEW output':<14} {'OLD nits':<10} {'NEW nits':<10} {'Change':<10}")
    print(f"  {'-'*70}")
    for i, inp in enumerate(test_inputs):
        old_nits = old_output[i] * _PEAK_NITS
        new_nits = new_output[i] * _PEAK_NITS
        change = "SAME" if abs(old_nits - new_nits) < 0.001 else f"{new_nits-old_nits:+.4f}"
        print(f"  {inp:<12.2e} {old_output[i]:<14.8f} {new_output[i]:<14.8f} {old_nits:<10.4f} {new_nits:<10.4f} {change:<10}")

    # ===== ANALYZE THE GAP =====
    print(f"\n{'='*70}")
    print("GAP ANALYSIS: pixels between BLACK_THRESHOLD and curve_x_min")
    print(f"{'='*70}")

    # The critical range: 1e-6 < sdr < sdr_linear_at_min
    # These pixels pass the black threshold but get CLAMPED to curve_x_min
    gap_inputs = np.logspace(-6, np.log10(sdr_linear_at_min), 20)
    old_gap = apply_curve_OLD(gap_inputs, curve)
    new_gap = apply_luminance_curve(gap_inputs, curve)

    print(f"\n  BLACK_THRESHOLD = 1e-6 normalized ({1e-6*_PEAK_NITS:.4f} nits)")
    print(f"  Curve starts at SDR = {sdr_linear_at_min:.6f} normalized ({sdr_linear_at_min*_PEAK_NITS:.2f} nits)")
    print(f"\n  Pixels in [{1e-6:.2e}, {sdr_linear_at_min:.2e}] (the 'gap'):")
    print(f"  OLD: extrapolates below curve (can produce LARGE values = white dots)")
    print(f"  NEW: clamps to curve_x_min (maps to curve[0] value)")
    print(f"\n  curve[0] = [{curve[0][0]:.4f}, {curve[0][1]:.4f}]")
    print(f"  This means: ALL gap pixels map to {(10**curve[0][1] - _LOG_EPS)/10000:.6f} normalized")
    print(f"  = {10**curve[0][1] - _LOG_EPS:.4f} nits")

    # Is this a DISCONTINUITY?
    # At threshold boundary: sdr=1e-6 → output=0 (black protection)
    # At sdr=1.0001e-6: → output = curve[0] value
    val_at_threshold = apply_luminance_curve(np.array([1e-6]), curve)[0]
    val_just_above = apply_luminance_curve(np.array([1.001e-6]), curve)[0]
    val_well_above = apply_luminance_curve(np.array([1e-5]), curve)[0]
    print(f"\n  DISCONTINUITY CHECK:")
    print(f"    sdr=1e-6:      output = {val_at_threshold:.8f} ({val_at_threshold*_PEAK_NITS:.4f} nits)")
    print(f"    sdr=1.001e-6:  output = {val_just_above:.8f} ({val_just_above*_PEAK_NITS:.4f} nits)")
    print(f"    sdr=1e-5:      output = {val_well_above:.8f} ({val_well_above*_PEAK_NITS:.4f} nits)")
    print(f"    JUMP at threshold: {(val_just_above - val_at_threshold)*_PEAK_NITS:.4f} nits")

    # ===== HOW MANY REAL PIXELS ARE AFFECTED? =====
    print(f"\n{'='*70}")
    print("REAL-WORLD IMPACT: How many pixels in typical dark scene?")
    print(f"{'='*70}")

    # Simulate a dark jacket region (from P2.9 data)
    # SDR luminance of jacket was: median=0.0084, with some pixels near 0.001
    jacket_sdr = rng.exponential(0.005, 100000)  # Typical dark jacket distribution
    jacket_sdr = np.clip(jacket_sdr, 0, 0.1)

    n_below_threshold = np.sum(jacket_sdr <= 1e-6)
    n_in_gap = np.sum((jacket_sdr > 1e-6) & (jacket_sdr < sdr_linear_at_min))
    n_above_curve = np.sum(jacket_sdr >= sdr_linear_at_min)

    print(f"  Simulated jacket (100K pixels, exponential(λ=0.005)):")
    print(f"    Below BLACK_THRESHOLD (1e-6): {n_below_threshold} → mapped to 0")
    print(f"    In gap (1e-6 to {sdr_linear_at_min:.6f}): {n_in_gap} → clamped to curve[0]")
    print(f"    Above curve start: {n_above_curve} → normal PCHIP")
    print(f"    Gap fraction: {n_in_gap/(n_below_threshold+n_in_gap+n_above_curve)*100:.2f}%")

    # What does this look like vs old behavior?
    old_jacket = apply_curve_OLD(jacket_sdr, curve) * _PEAK_NITS
    new_jacket = apply_luminance_curve(jacket_sdr, curve) * _PEAK_NITS

    print(f"\n  Luminance distribution for jacket region:")
    print(f"  {'Percentile':<12} {'OLD (nits)':<12} {'NEW (nits)':<12}")
    print(f"  {'-'*36}")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
        print(f"  P{p:<10} {np.percentile(old_jacket, p):<12.4f} {np.percentile(new_jacket, p):<12.4f}")

    # Count zeros
    print(f"\n  Pixels with output = 0:")
    print(f"    OLD: {np.sum(old_jacket == 0)}")
    print(f"    NEW: {np.sum(new_jacket == 0)}")
    print(f"    OLD < 0.01 nits: {np.sum(old_jacket < 0.01)}")
    print(f"    NEW < 0.01 nits: {np.sum(new_jacket < 0.01)}")

    # ===== VERDICT =====
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"""
  1. BLACK_THRESHOLD = 1e-6 in LINEAR NORMALIZED units
     = 1e-6 * 10000 = 0.01 nits
     This is a reasonable true-black threshold.

  2. The PROBLEM is NOT the threshold itself.
     The problem is the CLAMP to curve_x_min.

  3. curve_x_min = {curve_x_min:.4f}
     = SDR linear {sdr_linear_at_min:.6f}
     = SDR {sdr_linear_at_min*_PEAK_NITS:.2f} nits

  4. For pixels with SDR between 0.01 nits and {sdr_linear_at_min*_PEAK_NITS:.2f} nits:
     OLD: PCHIP extrapolates (bad: white dots for very small values)
     NEW: clamps ALL to curve[0] output = {(10**curve[0][1]-_LOG_EPS):.4f} nits
          This creates a FLAT REGION where different input luminances
          all produce the SAME output = loss of shadow detail.

  5. The correct fix should:
     - Keep black=0 for truly zero pixels
     - Smoothly transition from 0 to curve[0] for near-black
     - NOT clamp a range of luminances to a single value
     - NOT extrapolate in a way that produces white dots
""")

    print(f"{'='*70}")
    print("P2.9.2 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
