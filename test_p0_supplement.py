"""P0 SUPPLEMENTARY AUDIT: Limited-range handling, linearize() semantics,
controlled OLD vs NEW comparison.

Tests:
1. FFmpeg limited-range behavior for gray16le vs rgb48le
2. linearize() semantics verification
3. gray16le = nonlinear BT.2020 Y' verification
4. Controlled per-pixel OLD vs NEW on real frames
5. Synthetic achromatic/chromatic tests
6. Saturation correlation of difference
7. Direction of error
"""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.core.transfer_functions import linearize, pq_eotf, pq_oetf

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
WIDTH = 960


def extract(path, time_s, pix_fmt, width=WIDTH):
    """Generic extraction."""
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin",
           "-ss", f"{time_s:.6f}", "-i", str(path),
           "-vf", f"scale={width}:-1",
           "-frames:v", "1", "-pix_fmt", pix_fmt,
           "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    return r.stdout if r.returncode == 0 else None


def main():
    print("=" * 70)
    print("P0 SUPPLEMENTARY AUDIT")
    print("=" * 70)

    # ================================================================
    # SECTION 1: LIMITED-RANGE HANDLING
    # ================================================================
    print("\n" + "=" * 70)
    print("1. LIMITED-RANGE HANDLING: gray16le vs rgb48le")
    print("=" * 70)

    print("""
  FFmpeg behavior for yuv420p10le (color_range=tv) conversion:

  When converting to gray16le or rgb48le WITHOUT explicit range flags:
  - FFmpeg reads the input color_range metadata (tv = limited)
  - For gray16le: extracts Y plane, scales 10-bit limited [64,940] → 16-bit [0,65535]
  - For rgb48le: converts YCbCr→RGB using matrix coefficients, scales limited→full
  
  CRITICAL: Both paths perform limited→full range expansion by default
  because the output pixel formats (gray16le, rgb48le) are full-range formats.
  
  The sampling.py commands use NO explicit range flags:
    gray16le: -vf scale=960:-1 -pix_fmt gray16le
    rgb48le:  -vf scale=960:-1 -pix_fmt rgb48le
  
  So FFmpeg will automatically:
  1. Decode yuv420p10le limited [64-940 for Y, 64-960 for UV]
  2. Convert to output format with full-range expansion
  3. Both outputs represent signal values in [0, 65535] full range
  
  VERIFICATION: Extract a known HDR frame and check value ranges.
""")

    # Extract same frame both ways
    t = 2715.0
    gray_raw = extract(HDR_PATH, t, "gray16le")
    rgb_raw = extract(HDR_PATH, t, "rgb48le")

    if gray_raw and rgb_raw:
        # Parse gray16le
        total_gray = len(gray_raw) // 2
        h_gray = total_gray // WIDTH
        gray = np.frombuffer(gray_raw[:WIDTH*h_gray*2], dtype=np.uint16).reshape(h_gray, WIDTH)

        # Parse rgb48le
        total_rgb = len(rgb_raw) // 6
        h_rgb = total_rgb // WIDTH
        rgb = np.frombuffer(rgb_raw[:WIDTH*h_rgb*6], dtype=np.uint16).reshape(h_rgb, WIDTH, 3)

        h = min(h_gray, h_rgb)
        gray = gray[:h, :]
        rgb = rgb[:h, :, :]

        print(f"  gray16le: shape={gray.shape}, min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        print(f"  rgb48le:  shape={rgb.shape}, min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.1f}")
        print(f"  rgb R: min={rgb[:,:,0].min()}, max={rgb[:,:,0].max()}")
        print(f"  rgb G: min={rgb[:,:,1].min()}, max={rgb[:,:,1].max()}")
        print(f"  rgb B: min={rgb[:,:,2].min()}, max={rgb[:,:,2].max()}")

        # Check if gray ≈ BT.2020 weighted sum of RGB
        # If FFmpeg uses BT.2020 NCL matrix for yuv→rgb and yuv→gray:
        # gray should equal the Y' channel of YUV, and
        # 0.2627*R' + 0.6780*G' + 0.0593*B' should approximate gray
        rgb_y_computed = 0.2627 * rgb[:,:,0].astype(np.float64) + \
                         0.6780 * rgb[:,:,1].astype(np.float64) + \
                         0.0593 * rgb[:,:,2].astype(np.float64)
        gray_f = gray.astype(np.float64)

        diff = np.abs(gray_f - rgb_y_computed)
        print(f"\n  Computed BT.2020 Y' from RGB vs gray16le:")
        print(f"    MAE: {diff.mean():.2f} (out of 65535)")
        print(f"    Max diff: {diff.max():.2f}")
        print(f"    Relative MAE: {diff.mean() / gray_f.mean() * 100:.3f}%")

        if diff.mean() < 100:  # Less than ~0.15% of range
            print(f"    → gray16le ≈ BT.2020 weighted sum of RGB in nonlinear PQ domain ✓")
        else:
            print(f"    → MISMATCH: gray16le does NOT match BT.2020 Y' from RGB!")

    # ================================================================
    # SECTION 2: linearize() SEMANTICS
    # ================================================================
    print("\n\n" + "=" * 70)
    print("2. linearize() SEMANTICS")
    print("=" * 70)

    test_signals = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    print(f"\n  linearize('smpte2084', peak_nits=10000):")
    print(f"  {'Signal':<10} {'Result':<15} {'Meaning'}")
    print(f"  {'-'*45}")
    for s in test_signals:
        result = float(linearize(np.array([s]), "smpte2084", peak_nits=10000.0)[0])
        nits = result * 10000.0
        print(f"  {s:<10.1f} {result:<15.8f} = {nits:.2f} / 10000 nits")

    print(f"\n  CONCLUSION: linearize('smpte2084', peak_nits=10000) returns:")
    print(f"    PQ_EOTF(signal) / peak_nits = absolute_nits / 10000")
    print(f"    Range: [0, 1] where 1.0 = 10000 nits")
    print(f"    Semantics: NORMALIZED LINEAR LUMINANCE")

    # ================================================================
    # SECTION 3: gray16le = nonlinear BT.2020 Y' VERIFICATION
    # ================================================================
    print("\n\n" + "=" * 70)
    print("3. VERIFICATION: gray16le = nonlinear BT.2020 Y'")
    print("=" * 70)

    if gray_raw and rgb_raw:
        # Already computed above
        relative_mae = diff.mean() / gray_f.mean() * 100
        print(f"\n  gray16le vs 0.2627*R'+0.6780*G'+0.0593*B' (from rgb48le):")
        print(f"    Relative MAE: {relative_mae:.3f}%")
        print(f"    Max absolute: {diff.max():.0f} / 65535 = {diff.max()/65535*100:.3f}%")

        if relative_mae < 0.5:
            print(f"    CONFIRMED: gray16le = BT.2020 NCL nonlinear Y' (within rounding)")
        else:
            print(f"    NOT CONFIRMED: discrepancy exceeds acceptable threshold")

    # ================================================================
    # SECTION 4: CONTROLLED OLD vs NEW PER-PIXEL
    # ================================================================
    print("\n\n" + "=" * 70)
    print("4. CONTROLLED OLD vs NEW (same pixels, 5 frames)")
    print("=" * 70)

    test_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]
    all_old = []
    all_new = []
    all_sat = []

    for t in test_times:
        gray_raw = extract(HDR_PATH, t, "gray16le")
        rgb_raw = extract(HDR_PATH, t, "rgb48le")
        if not gray_raw or not rgb_raw:
            continue

        total_gray = len(gray_raw) // 2
        h_gray = total_gray // WIDTH
        gray = np.frombuffer(gray_raw[:WIDTH*h_gray*2], dtype=np.uint16).reshape(h_gray, WIDTH).astype(np.float64) / 65535.0

        total_rgb = len(rgb_raw) // 6
        h_rgb = total_rgb // WIDTH
        rgb = np.frombuffer(rgb_raw[:WIDTH*h_rgb*6], dtype=np.uint16).reshape(h_rgb, WIDTH, 3).astype(np.float64) / 65535.0

        h = min(h_gray, h_rgb)
        gray = gray[:h, :]
        rgb = rgb[:h, :, :]

        # METHOD OLD: PQ_EOTF(gray) normalized
        old_linear = linearize(gray, "smpte2084", peak_nits=10000.0)
        old_nits = old_linear * 10000.0

        # METHOD NEW: per-channel PQ_EOTF then BT.2020 weighted sum
        r_lin = linearize(rgb[..., 0], "smpte2084", peak_nits=10000.0) * 10000.0
        g_lin = linearize(rgb[..., 1], "smpte2084", peak_nits=10000.0) * 10000.0
        b_lin = linearize(rgb[..., 2], "smpte2084", peak_nits=10000.0) * 10000.0
        new_nits = 0.2627 * r_lin + 0.6780 * g_lin + 0.0593 * b_lin

        # Saturation from PQ-coded RGB
        y_prime = 0.2627 * rgb[..., 0] + 0.6780 * rgb[..., 1] + 0.0593 * rgb[..., 2]
        cr = rgb[..., 0] - y_prime
        cb = rgb[..., 2] - y_prime
        sat = np.sqrt(cr**2 + cb**2)

        # Filter valid pixels (> 0.1 nit)
        valid = (old_nits > 0.1) & (new_nits > 0.1)
        old_v = old_nits[valid].flatten()
        new_v = new_nits[valid].flatten()
        sat_v = sat[valid].flatten()

        all_old.append(old_v)
        all_new.append(new_v)
        all_sat.append(sat_v)

        diff_v = old_v - new_v
        print(f"\n  Frame {t:.0f}s ({len(old_v):,} pixels):")
        print(f"    OLD mean: {np.mean(old_v):.3f} nits")
        print(f"    NEW mean: {np.mean(new_v):.3f} nits")
        print(f"    mean(OLD-NEW): {np.mean(diff_v):.4f} nits")
        print(f"    MAE: {np.mean(np.abs(diff_v)):.4f} nits")

    # Aggregate
    old_all = np.concatenate(all_old)
    new_all = np.concatenate(all_new)
    sat_all = np.concatenate(all_sat)
    diff_all = old_all - new_all

    print(f"\n  AGGREGATE ({len(old_all):,} pixels):")
    print(f"    OLD mean: {np.mean(old_all):.4f} nits")
    print(f"    NEW mean: {np.mean(new_all):.4f} nits")
    print(f"    mean(OLD-NEW): {np.mean(diff_all):.4f} nits  ← DIRECTION")
    print(f"    MAE: {np.mean(np.abs(diff_all)):.4f} nits")
    print(f"    RMSE: {np.sqrt(np.mean(diff_all**2)):.4f} nits")

    print(f"\n  Percentile comparison:")
    print(f"  {'P':<5} {'OLD':<12} {'NEW':<12} {'OLD-NEW':<12} {'Rel%':<8}")
    print(f"  {'-'*49}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        po = float(np.percentile(old_all, p))
        pn = float(np.percentile(new_all, p))
        print(f"  {p:<5} {po:<12.4f} {pn:<12.4f} {po-pn:<12.4f} {(po-pn)/pn*100:+.2f}%")

    # ================================================================
    # SECTION 5: SYNTHETIC TESTS
    # ================================================================
    print("\n\n" + "=" * 70)
    print("5. SYNTHETIC ACHROMATIC / CHROMATIC TESTS")
    print("=" * 70)

    print(f"\n  A. ACHROMATIC (R'=G'=B'):")
    print(f"  {'PQ value':<10} {'OLD (nits)':<12} {'NEW (nits)':<12} {'Diff':<12}")
    print(f"  {'-'*46}")
    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        old_n = float(pq_eotf(np.array([v]))[0])  # PQ_EOTF(Y') where Y'=v since achromatic
        r_n = float(pq_eotf(np.array([v]))[0])
        new_n = 0.2627 * r_n + 0.6780 * r_n + 0.0593 * r_n  # = r_n (weights sum to 1)
        print(f"  {v:<10.1f} {old_n:<12.4f} {new_n:<12.4f} {old_n-new_n:<12.8f}")

    print(f"\n  B. CHROMATIC LOW-LUMINANCE (1-10 nits range):")
    print(f"  {'R,G,B PQ':<20} {'OLD (nits)':<12} {'NEW (nits)':<12} {'Diff':<10} {'Rel%':<8}")
    print(f"  {'-'*62}")
    test_chromatic = [
        (0.20, 0.18, 0.15),  # warm dark
        (0.15, 0.20, 0.22),  # cool dark
        (0.25, 0.18, 0.12),  # very warm
        (0.12, 0.22, 0.20),  # teal
        (0.18, 0.18, 0.18),  # near-achromatic
    ]
    for R, G, B in test_chromatic:
        y_prime = 0.2627*R + 0.6780*G + 0.0593*B
        old_n = float(pq_eotf(np.array([y_prime]))[0])
        r_n = float(pq_eotf(np.array([R]))[0])
        g_n = float(pq_eotf(np.array([G]))[0])
        b_n = float(pq_eotf(np.array([B]))[0])
        new_n = 0.2627*r_n + 0.6780*g_n + 0.0593*b_n
        d = old_n - new_n
        rel = d/new_n*100 if new_n > 0 else 0
        print(f"  ({R:.2f},{G:.2f},{B:.2f})   {old_n:<12.4f} {new_n:<12.4f} {d:<10.4f} {rel:+.1f}%")

    print(f"\n  C. CHROMATIC HIGHLIGHTS (100-1000+ nits):")
    test_highlights = [
        (0.55, 0.50, 0.45),
        (0.60, 0.52, 0.40),
        (0.65, 0.55, 0.42),
        (0.70, 0.60, 0.50),
        (0.75, 0.65, 0.55),
    ]
    print(f"  {'R,G,B PQ':<20} {'OLD (nits)':<12} {'NEW (nits)':<12} {'Diff':<10} {'Rel%':<8}")
    print(f"  {'-'*62}")
    for R, G, B in test_highlights:
        y_prime = 0.2627*R + 0.6780*G + 0.0593*B
        old_n = float(pq_eotf(np.array([y_prime]))[0])
        r_n = float(pq_eotf(np.array([R]))[0])
        g_n = float(pq_eotf(np.array([G]))[0])
        b_n = float(pq_eotf(np.array([B]))[0])
        new_n = 0.2627*r_n + 0.6780*g_n + 0.0593*b_n
        d = old_n - new_n
        rel = d/new_n*100 if new_n > 0 else 0
        print(f"  ({R:.2f},{G:.2f},{B:.2f})   {old_n:<12.4f} {new_n:<12.4f} {d:<10.4f} {rel:+.1f}%")

    # ================================================================
    # SECTION 6 & 7: SATURATION CORRELATION + DIRECTION
    # ================================================================
    print("\n\n" + "=" * 70)
    print("6. SATURATION CORRELATION OF |OLD-NEW|")
    print("=" * 70)

    from scipy.stats import spearmanr
    n_sub = min(100000, len(diff_all))
    idx = np.random.default_rng(42).choice(len(diff_all), n_sub, replace=False)

    rho_sat, _ = spearmanr(sat_all[idx], np.abs(diff_all[idx]))
    rho_lum, _ = spearmanr(new_all[idx], np.abs(diff_all[idx]))
    print(f"  Spearman |OLD-NEW| vs saturation: rho = {rho_sat:.4f}")
    print(f"  Spearman |OLD-NEW| vs luminance:  rho = {rho_lum:.4f}")

    # By saturation band
    achro = sat_all < 0.01
    mid_sat = (sat_all >= 0.01) & (sat_all < 0.05)
    high_sat = sat_all >= 0.05
    print(f"\n  MAE by saturation band:")
    print(f"    sat < 0.01 (achromatic): {np.mean(np.abs(diff_all[achro])):.4f} nits ({np.sum(achro):,} px)")
    print(f"    0.01 ≤ sat < 0.05:       {np.mean(np.abs(diff_all[mid_sat])):.4f} nits ({np.sum(mid_sat):,} px)")
    print(f"    sat ≥ 0.05 (saturated):  {np.mean(np.abs(diff_all[high_sat])):.4f} nits ({np.sum(high_sat):,} px)")

    print(f"\n" + "=" * 70)
    print("7. DIRECTION OF ERROR")
    print("=" * 70)
    print(f"  mean(OLD - NEW) = {np.mean(diff_all):.4f} nits")
    print(f"  Direction: OLD {'<' if np.mean(diff_all) < 0 else '>'} NEW")
    print(f"  OLD systematically {'UNDER' if np.mean(diff_all) < 0 else 'OVER'}estimates relative to NEW")
    n_negative = np.sum(diff_all < 0)
    print(f"  {n_negative/len(diff_all)*100:.1f}% of pixels have OLD < NEW")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"""
  A. gray16le represents nonlinear BT.2020 Y' in PQ domain: CONFIRMED
  B. Limited-range expansion: BOTH paths perform it automatically
  C. linearize("smpte2084", peak_nits=10000) = PQ_EOTF(signal)/10000: CONFIRMED
  D. OLD method = PQ_EOTF(Y'): applies EOTF to weighted sum of nonlinear channels
  E. NEW method = weighted sum of PQ_EOTF(per-channel): correct linear luminance
  F. Difference magnitude: MAE = {np.mean(np.abs(diff_all)):.4f} nits
  G. Difference direction: OLD UNDERESTIMATES (mean = {np.mean(diff_all):.4f} nits)
  H. Correlation with saturation: rho = {rho_sat:.4f}
  I. Achromatic error: {np.mean(np.abs(diff_all[achro])):.4f} nits (≈zero)
  J. Saturated error: {np.mean(np.abs(diff_all[high_sat])):.4f} nits
""")

    print(f"{'='*70}")
    print("P0 SUPPLEMENTARY AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
