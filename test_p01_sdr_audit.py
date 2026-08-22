"""P0.1 SDR GROUND-TRUTH LUMINANCE AUDIT.

Controlled comparison:
OLD: gray16le Y' → BT.1886 EOTF(Y') → SDR luminance
NEW: rgb48le → BT.1886 EOTF per channel → 0.2126R + 0.7152G + 0.0722B
"""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.core.transfer_functions import linearize, bt1886_eotf

OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
WIDTH = 960
SYNC_OFFSET = 1167
FPS = 23.976


def extract(path, time_s, pix_fmt):
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin",
           "-ss", f"{time_s:.6f}", "-i", str(path),
           "-vf", f"scale={WIDTH}:-1",
           "-frames:v", "1", "-pix_fmt", pix_fmt,
           "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    return r.stdout if r.returncode == 0 else None


def main():
    print("=" * 70)
    print("P0.1 SDR GROUND-TRUTH LUMINANCE AUDIT")
    print("=" * 70)

    # ================================================================
    # SECTION 1: CURRENT CODE PATH
    # ================================================================
    print("""
  CURRENT SDR PATH IN sampling.py:
    OM source (yuv420p, BT.709, color_range=tv)
      → FFmpeg -pix_fmt gray16le -vf scale=960:-1
      → Y' = BT.709 nonlinear luma (limited→full expanded by FFmpeg)
      → normalize / 65535.0 → [0,1]
      → linearize("bt709") = bt1886_eotf(signal) = signal^2.4
      → SDR linear luminance [0,1]

  CORRECT PATH:
    OM source
      → FFmpeg -pix_fmt rgb48le
      → R'G'B' BT.709 nonlinear (limited→full expanded)
      → normalize / 65535.0 → [0,1]
      → bt1886_eotf per channel: R=R'^2.4, G=G'^2.4, B=B'^2.4
      → Y = 0.2126*R + 0.7152*G + 0.0722*B

  Mathematical question:
    Is (0.2126*R' + 0.7152*G' + 0.0722*B')^2.4
    equal to
    0.2126*R'^2.4 + 0.7152*G'^2.4 + 0.0722*B'^2.4 ?

  Answer: NO (same Jensen's inequality issue as PQ, but gamma 2.4 is less
  nonlinear than PQ, so the error should be smaller).
""")

    # ================================================================
    # SECTION 2: VERIFY gray16le = BT.709 Y' for SDR
    # ================================================================
    print("=" * 70)
    print("2. VERIFICATION: gray16le = BT.709 nonlinear Y'")
    print("=" * 70)

    t = 2715.0 + SYNC_OFFSET / FPS  # OM corresponding time
    gray_raw = extract(OM_PATH, t, "gray16le")
    rgb_raw = extract(OM_PATH, t, "rgb48le")

    if not gray_raw or not rgb_raw:
        print("  ERROR: extraction failed")
        return

    total_gray = len(gray_raw) // 2
    h_gray = total_gray // WIDTH
    gray = np.frombuffer(gray_raw[:WIDTH*h_gray*2], dtype=np.uint16).reshape(h_gray, WIDTH)

    total_rgb = len(rgb_raw) // 6
    h_rgb = total_rgb // WIDTH
    rgb = np.frombuffer(rgb_raw[:WIDTH*h_rgb*6], dtype=np.uint16).reshape(h_rgb, WIDTH, 3)

    h = min(h_gray, h_rgb)
    gray = gray[:h, :]
    rgb = rgb[:h, :, :]

    # Compute BT.709 Y' from RGB
    rgb_y = 0.2126 * rgb[:,:,0].astype(np.float64) + \
             0.7152 * rgb[:,:,1].astype(np.float64) + \
             0.0722 * rgb[:,:,2].astype(np.float64)
    gray_f = gray.astype(np.float64)

    diff_raw = np.abs(gray_f - rgb_y)
    print(f"\n  gray16le vs 0.2126*R'+0.7152*G'+0.0722*B' (from rgb48le):")
    print(f"    gray range: [{gray.min()}, {gray.max()}]")
    print(f"    rgb range: [{rgb.min()}, {rgb.max()}]")
    print(f"    MAE: {diff_raw.mean():.2f} (out of 65535)")
    print(f"    Max diff: {diff_raw.max():.2f}")
    print(f"    Relative MAE: {diff_raw.mean()/gray_f.mean()*100:.4f}%")
    if diff_raw.mean() < 100:
        print(f"    CONFIRMED: gray16le = BT.709 Y' (nonlinear) within rounding")
    else:
        print(f"    NOT CONFIRMED: significant discrepancy")

    # ================================================================
    # SECTION 3: SYNTHETIC TESTS
    # ================================================================
    print("\n\n" + "=" * 70)
    print("3. SYNTHETIC TESTS")
    print("=" * 70)

    print(f"\n  A. ACHROMATIC (R'=G'=B'):")
    print(f"  {'Signal':<10} {'OLD=Y^2.4':<15} {'NEW=Σw·c^2.4':<15} {'Diff':<15}")
    print(f"  {'-'*55}")
    for v in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        y_prime = v  # achromatic: Y'=R'=G'=B'
        old = y_prime ** 2.4
        new = 0.2126 * (v**2.4) + 0.7152 * (v**2.4) + 0.0722 * (v**2.4)
        # new = v^2.4 * (0.2126+0.7152+0.0722) = v^2.4 * 1.0 = old
        print(f"  {v:<10.1f} {old:<15.8f} {new:<15.8f} {old-new:<15.12f}")

    print(f"\n  B. CHROMATIC LOW-LUMINANCE:")
    print(f"  {'R,G,B signal':<20} {'OLD':<12} {'NEW':<12} {'Diff':<10} {'Rel%':<8}")
    print(f"  {'-'*62}")
    test_chrom = [
        (0.30, 0.20, 0.10),
        (0.10, 0.30, 0.35),
        (0.40, 0.20, 0.05),
        (0.05, 0.35, 0.30),
        (0.25, 0.25, 0.25),  # achromatic
    ]
    for R, G, B in test_chrom:
        y_prime = 0.2126*R + 0.7152*G + 0.0722*B
        old = y_prime ** 2.4
        new = 0.2126*(R**2.4) + 0.7152*(G**2.4) + 0.0722*(B**2.4)
        d = old - new
        rel = d/new*100 if new > 0 else 0
        print(f"  ({R:.2f},{G:.2f},{B:.2f})   {old:<12.6f} {new:<12.6f} {d:<10.6f} {rel:+.2f}%")

    print(f"\n  C. CHROMATIC HIGHLIGHTS:")
    test_high = [
        (0.80, 0.60, 0.40),
        (0.90, 0.70, 0.50),
        (0.95, 0.80, 0.60),
        (0.70, 0.90, 0.85),
    ]
    print(f"  {'R,G,B signal':<20} {'OLD':<12} {'NEW':<12} {'Diff':<10} {'Rel%':<8}")
    print(f"  {'-'*62}")
    for R, G, B in test_high:
        y_prime = 0.2126*R + 0.7152*G + 0.0722*B
        old = y_prime ** 2.4
        new = 0.2126*(R**2.4) + 0.7152*(G**2.4) + 0.0722*(B**2.4)
        d = old - new
        rel = d/new*100 if new > 0 else 0
        print(f"  ({R:.2f},{G:.2f},{B:.2f})   {old:<12.6f} {new:<12.6f} {d:<10.6f} {rel:+.2f}%")

    # ================================================================
    # SECTION 4: REAL-MATERIAL CONTROLLED COMPARISON (5 frames)
    # ================================================================
    print("\n\n" + "=" * 70)
    print("4. REAL-MATERIAL: OLD vs NEW (5 frames)")
    print("=" * 70)

    # Use OM frames corresponding to HDR times 2705-2725
    test_hdr_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]
    all_old = []
    all_new = []
    all_sat = []

    for hdr_t in test_hdr_times:
        om_time = hdr_t + SYNC_OFFSET / FPS

        gray_raw = extract(OM_PATH, om_time, "gray16le")
        rgb_raw = extract(OM_PATH, om_time, "rgb48le")
        if not gray_raw or not rgb_raw:
            continue

        total_gray = len(gray_raw) // 2
        h_g = total_gray // WIDTH
        gray_frame = np.frombuffer(gray_raw[:WIDTH*h_g*2], dtype=np.uint16).reshape(h_g, WIDTH).astype(np.float64) / 65535.0

        total_rgb = len(rgb_raw) // 6
        h_r = total_rgb // WIDTH
        rgb_frame = np.frombuffer(rgb_raw[:WIDTH*h_r*6], dtype=np.uint16).reshape(h_r, WIDTH, 3).astype(np.float64) / 65535.0

        h = min(h_g, h_r)
        gray_frame = gray_frame[:h, :]
        rgb_frame = rgb_frame[:h, :, :]

        # Crop to overlap region (proxy coords: 280*960/3840 ≈ 70, 1880*960/3840 ≈ 470)
        py1 = int(round(280 * WIDTH / 3840))
        py2 = int(round(1880 * WIDTH / 3840))
        gray_ov = gray_frame[py1:py2, :]
        rgb_ov = rgb_frame[py1:py2, :, :]

        # METHOD OLD: Y'^2.4
        old = linearize(gray_ov, "bt709")

        # METHOD NEW: per-channel EOTF + BT.709 weights
        r_lin = linearize(rgb_ov[..., 0], "bt709")
        g_lin = linearize(rgb_ov[..., 1], "bt709")
        b_lin = linearize(rgb_ov[..., 2], "bt709")
        new = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

        # Saturation from signal-domain RGB
        y_prime_sig = 0.2126 * rgb_ov[..., 0] + 0.7152 * rgb_ov[..., 1] + 0.0722 * rgb_ov[..., 2]
        cr = rgb_ov[..., 0] - y_prime_sig
        cb = rgb_ov[..., 2] - y_prime_sig
        sat = np.sqrt(cr**2 + cb**2)

        # Filter
        valid = (old > 0.001) & (new > 0.001) & np.isfinite(old) & np.isfinite(new)
        old_v = old[valid].flatten()
        new_v = new[valid].flatten()
        sat_v = sat[valid].flatten()

        all_old.append(old_v)
        all_new.append(new_v)
        all_sat.append(sat_v)

        diff_v = old_v - new_v
        print(f"\n  Frame HDR={hdr_t:.0f}s ({len(old_v):,} pixels):")
        print(f"    OLD mean: {np.mean(old_v):.6f}")
        print(f"    NEW mean: {np.mean(new_v):.6f}")
        print(f"    mean(OLD-NEW): {np.mean(diff_v):.6f}")
        print(f"    MAE: {np.mean(np.abs(diff_v)):.6f}")
        print(f"    Relative MAE: {np.mean(np.abs(diff_v)/new_v)*100:.3f}%")

    # Aggregate
    old_all = np.concatenate(all_old)
    new_all = np.concatenate(all_new)
    sat_all = np.concatenate(all_sat)
    diff_all = old_all - new_all

    print(f"\n  AGGREGATE ({len(old_all):,} pixels):")
    print(f"    OLD mean: {np.mean(old_all):.6f}")
    print(f"    NEW mean: {np.mean(new_all):.6f}")
    print(f"    mean(OLD-NEW): {np.mean(diff_all):.6f}")
    print(f"    MAE: {np.mean(np.abs(diff_all)):.6f}")
    print(f"    RMSE: {np.sqrt(np.mean(diff_all**2)):.6f}")
    print(f"    Relative MAE: {np.mean(np.abs(diff_all)/(new_all+1e-6))*100:.4f}%")

    print(f"\n  Percentiles:")
    print(f"  {'P':<5} {'OLD':<12} {'NEW':<12} {'Diff':<12} {'Rel%':<8}")
    print(f"  {'-'*49}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        po = float(np.percentile(old_all, p))
        pn = float(np.percentile(new_all, p))
        print(f"  {p:<5} {po:<12.6f} {pn:<12.6f} {po-pn:<12.6f} {(po-pn)/pn*100:+.3f}%")

    # ================================================================
    # SECTION 5: SATURATION CORRELATION
    # ================================================================
    print("\n\n" + "=" * 70)
    print("5. SATURATION CORRELATION")
    print("=" * 70)

    from scipy.stats import spearmanr
    n_sub = min(100000, len(diff_all))
    idx = np.random.default_rng(42).choice(len(diff_all), n_sub, replace=False)

    rho_sat, _ = spearmanr(sat_all[idx], np.abs(diff_all[idx]))
    rho_lum, _ = spearmanr(new_all[idx], np.abs(diff_all[idx]))
    print(f"  Spearman |OLD-NEW| vs saturation: rho = {rho_sat:.4f}")
    print(f"  Spearman |OLD-NEW| vs luminance:  rho = {rho_lum:.4f}")

    achro = sat_all < 0.01
    mid_s = (sat_all >= 0.01) & (sat_all < 0.05)
    high_s = sat_all >= 0.05
    print(f"\n  MAE by saturation band:")
    if np.sum(achro) > 0:
        print(f"    sat < 0.01: {np.mean(np.abs(diff_all[achro])):.8f} ({np.sum(achro):,} px)")
    if np.sum(mid_s) > 0:
        print(f"    0.01 ≤ sat < 0.05: {np.mean(np.abs(diff_all[mid_s])):.6f} ({np.sum(mid_s):,} px)")
    if np.sum(high_s) > 0:
        print(f"    sat ≥ 0.05: {np.mean(np.abs(diff_all[high_s])):.6f} ({np.sum(high_s):,} px)")

    # ================================================================
    # SECTION 6: DIRECTION
    # ================================================================
    print(f"\n  DIRECTION:")
    print(f"    mean(OLD-NEW) = {np.mean(diff_all):.6f}")
    n_neg = np.sum(diff_all < 0)
    print(f"    {n_neg/len(diff_all)*100:.1f}% pixels: OLD < NEW")

    # ================================================================
    # SECTION 7: COMPARISON WITH HDR P0
    # ================================================================
    print(f"\n\n{'='*70}")
    print("7. COMPARISON WITH HDR P0")
    print(f"{'='*70}")
    hdr_mae = 0.4438  # nits from P0
    # SDR MAE in same units? SDR is [0,1] normalized. For comparison:
    sdr_mae_normalized = float(np.mean(np.abs(diff_all)))
    # In relative terms:
    sdr_rel = float(np.mean(np.abs(diff_all) / (new_all + 1e-6))) * 100
    print(f"  HDR P0 error: MAE = {hdr_mae:.4f} nits (relative ~5%)")
    print(f"  SDR P0.1 error: MAE = {sdr_mae_normalized:.6f} normalized (relative {sdr_rel:.2f}%)")
    print(f"  SDR error is {'SMALLER' if sdr_rel < 5.0 else 'COMPARABLE'} than HDR error")

    # ================================================================
    print(f"\n\n{'='*70}")
    print("P0.1 SDR AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
