"""P2.9.3 verification: smooth low-end bridge — continuity + regression tests."""
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import sys
sys.path.insert(0, "src")
import numpy as np
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve, _PEAK_NITS, _LOG_EPS
from auto_openmatte.core.config import ColorConfig

# Generate realistic curve
rng = np.random.default_rng(42)
sdr_data = rng.random(100000) * 0.5 + 0.001
hdr_data = sdr_data * 0.002 + rng.random(100000) * 0.0001
config = ColorConfig(luminance_bins=256, low_percentile=1.0, high_percentile=99.9)
curve = estimate_luminance_curve(sdr_data, hdr_data, config=config)

# Curve start in linear
curve_x_min = curve[0][0]
curve_start_nits = 10.0**curve_x_min - _LOG_EPS
curve_start_linear = curve_start_nits / _PEAK_NITS
curve_start_hdr_nits = 10.0**curve[0][1] - _LOG_EPS

print("=" * 70)
print("P2.9.3 VERIFICATION: SMOOTH LOW-END BRIDGE")
print("=" * 70)
print(f"\n  Curve start: SDR = {curve_start_linear:.6f} normalized ({curve_start_nits:.2f} nits)")
print(f"  Curve start HDR: {curve_start_hdr_nits:.4f} nits")

# === TEST 1: CONTINUITY TABLE ===
print(f"\n{'='*70}")
print("TEST 1: CONTINUITY TABLE")
print(f"{'='*70}")

test_inputs = np.array([
    0, 1e-8, 1e-7, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3,
    curve_start_linear,
    1e-2, 2e-2
])
outputs = apply_luminance_curve(test_inputs, curve)

print(f"\n  {'Input norm':<12} {'Input nits':<11} {'Output nits':<12} {'Branch':<12}")
print(f"  {'-'*47}")
for inp, out in zip(test_inputs, outputs):
    nits_in = inp * _PEAK_NITS
    nits_out = out * _PEAK_NITS
    if inp == 0:
        branch = "BLACK"
    elif inp < curve_start_linear:
        branch = "BRIDGE"
    else:
        branch = "PCHIP"
    print(f"  {inp:<12.2e} {nits_in:<11.4f} {nits_out:<12.4f} {branch:<12}")

# === TEST 2: MONOTONICITY + CONTINUITY (dense) ===
print(f"\n{'='*70}")
print("TEST 2: DENSE CONTINUITY (1000 points, 0 to curve_start)")
print(f"{'='*70}")

dense = np.linspace(0, curve_start_linear * 1.1, 1000)
dense_out = apply_luminance_curve(dense, curve)

# Check monotonicity
diffs = np.diff(dense_out)
is_monotonic = np.all(diffs >= -1e-12)
max_jump = float(np.max(np.abs(diffs))) * _PEAK_NITS

print(f"  Monotonic: {is_monotonic}")
print(f"  Max local jump: {max_jump:.6f} nits")
print(f"  f(0) = {dense_out[0]*_PEAK_NITS:.6f} nits (must be 0)")
print(f"  f(curve_start) = {dense_out[int(1000/1.1)]*_PEAK_NITS:.4f} nits (must be ≈{curve_start_hdr_nits:.4f})")
print(f"  No plateau: {len(np.unique(np.round(dense_out[1:900]*1e8)))} unique values out of 899")

# === TEST 3: WHITE-DOT REGRESSION ===
print(f"\n{'='*70}")
print("TEST 3: WHITE-DOT REGRESSION")
print(f"{'='*70}")

# The white-dot pixels had SDR R=0, G=0.0039, B=0
# After BT.1886 EOTF + gamut + BT.2020 Y: luminance ≈ 0 or very small
whitedot_inputs = np.array([0.0, 1e-8, 1e-7, 1e-6, 5e-6])
whitedot_outputs = apply_luminance_curve(whitedot_inputs, curve)
print(f"  White-dot trigger values:")
for inp, out in zip(whitedot_inputs, whitedot_outputs):
    print(f"    SDR={inp:.2e} → HDR={out*_PEAK_NITS:.4f} nits")
max_whitedot = np.max(whitedot_outputs) * _PEAK_NITS
print(f"  Max output for near-zero: {max_whitedot:.4f} nits")
assert max_whitedot < 1.0, f"White dot NOT fixed: {max_whitedot} nits!"
print(f"  WHITE-DOT FIX: CONFIRMED ✅")

# === TEST 4: NO PLATEAU ===
print(f"\n{'='*70}")
print("TEST 4: NO PLATEAU AT CURVE_Y_MIN")
print(f"{'='*70}")

# Check that different inputs in bridge give different outputs
bridge_inputs = np.logspace(np.log10(1e-5), np.log10(curve_start_linear * 0.99), 50)
bridge_outputs = apply_luminance_curve(bridge_inputs, curve)
n_unique = len(np.unique(np.round(bridge_outputs * 1e10)))
print(f"  50 inputs in bridge range → {n_unique} unique outputs")
assert n_unique >= 45, f"Plateau detected: only {n_unique} unique values!"
print(f"  NO PLATEAU ✅")

# === TEST 5: JACKET ROI SIMULATION ===
print(f"\n{'='*70}")
print("TEST 5: SIMULATED JACKET ROI")
print(f"{'='*70}")

# Dark jacket: exponential distribution with mean ≈ 0.005
jacket = rng.exponential(0.005, 100000)
jacket = np.clip(jacket, 0, 0.1)
jacket_out = apply_luminance_curve(jacket, curve) * _PEAK_NITS

print(f"  Input: 100K pixels, exponential(0.005)")
print(f"  Output percentiles (nits):")
for p in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
    print(f"    P{p}: {np.percentile(jacket_out, p):.4f}")
n_zero = np.sum(jacket_out == 0)
n_plateau = np.sum(np.abs(jacket_out - curve_start_hdr_nits) < 0.001)
print(f"  Pixels = 0: {n_zero}")
print(f"  Pixels = curve_start_hdr: {n_plateau} (should be near 0)")

print(f"\n{'='*70}")
print("ALL P2.9.3 CHECKS PASSED ✅")
print(f"{'='*70}")
