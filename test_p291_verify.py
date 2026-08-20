"""P2.9.1 verification: black protection + no RuntimeWarning."""
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

import sys
sys.path.insert(0, "src")
import numpy as np
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.core.config import ColorConfig

# Generate a realistic log-domain curve
rng = np.random.default_rng(42)
sdr = rng.random(50000) * 0.5 + 0.01
hdr = sdr * 0.002
config = ColorConfig(luminance_bins=256, low_percentile=1.0, high_percentile=99.9)
curve = estimate_luminance_curve(sdr, hdr, config=config)

# Test black and near-black
test_values = np.array([0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1])
result = apply_luminance_curve(test_values, curve)

print("Black/near-black protection test:")
print(f"  {'Input':<12} {'Output':<15} {'Status'}")
print(f"  {'-'*40}")
for inp, out in zip(test_values, result):
    status = "BLACK→0" if out == 0 else ("OK" if out < 0.01 else "SUSPICIOUS")
    print(f"  {inp:<12.2e} {out:<15.6e} {status}")

# Assertions
assert result[0] == 0.0, f"Black not zero: {result[0]}"
assert result[1] == 0.0, f"1e-8 not zero: {result[1]}"  # below threshold
assert result[2] == 0.0, f"1e-7 not zero: {result[2]}"  # below threshold
assert result[3] == 0.0, f"1e-6 not zero: {result[3]}"  # at threshold
assert np.all(np.isfinite(result))
assert np.all(result >= 0)
assert np.max(result) < 0.1  # No explosion

# Test with the transform pipeline
from auto_openmatte.core.models import ShotTransform
from auto_openmatte.processing.transform import apply_shot_transform

frame = np.zeros((50, 50, 3), dtype=np.float64)
frame[10:40, 10:40, :] = 0.3
frame[0:5, :, :] = 0.0  # Pure black
frame[5:10, :, 0] = 0.0  # R=0, G/B nonzero (the bug trigger)
frame[5:10, :, 1] = 0.004
frame[5:10, :, 2] = 0.0

transform = ShotTransform(shot_id=0, luminance_curve=curve)
output = apply_shot_transform(frame, transform)

# Black rows must stay black
black_lum = output[0:5, :, :].max()
assert black_lum < 0.02, f"Black region not dark: max={black_lum}"

# Near-black rows (the bug trigger: R=0, G=0.004, B=0)
nearbk_max = output[5:10, :, :].max()
assert nearbk_max < 0.15, f"Near-black row too bright: max={nearbk_max}"

print(f"\nTransform test:")
print(f"  Black rows max: {black_lum:.6f} (must be < 0.02)")
print(f"  Near-black rows max: {nearbk_max:.6f} (must be < 0.15)")
print(f"\nALL CHECKS PASSED ✓")
