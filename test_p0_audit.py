"""P0 AUDIT: HDR ground-truth luminance correctness.

Compares:
- Method A (current): FFmpeg gray16le → PQ_EOTF(Y') 
- Method B (correct): FFmpeg rgb48le → PQ_EOTF(R',G',B') → weighted_sum

On real BR2049 frames.
"""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.core.transfer_functions import pq_eotf, linearize

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")

WIDTH = 960  # Same proxy as sampling.py uses


def extract_gray16(path, time_s, width):
    """Extract as gray16le (what current pipeline does)."""
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin",
           "-ss", f"{time_s:.6f}", "-i", str(path),
           "-vf", f"scale={width}:-1",
           "-frames:v", "1", "-pix_fmt", "gray16le", "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or not r.stdout:
        return None
    total_pixels = len(r.stdout) // 2
    height = total_pixels // width
    if height <= 0:
        return None
    arr = np.frombuffer(r.stdout[:width*height*2], dtype=np.uint16)
    return arr.reshape(height, width).astype(np.float64) / 65535.0


def extract_rgb48(path, time_s, width):
    """Extract as rgb48le (correct method for per-channel EOTF)."""
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin",
           "-ss", f"{time_s:.6f}", "-i", str(path),
           "-vf", f"scale={width}:-1",
           "-frames:v", "1", "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or not r.stdout:
        return None
    total_pixels = len(r.stdout) // 6
    height = total_pixels // width
    if height <= 0:
        return None
    arr = np.frombuffer(r.stdout[:width*height*6], dtype=np.uint16)
    return arr.reshape(height, width, 3).astype(np.float64) / 65535.0


def main():
    print("=" * 70)
    print("P0 AUDIT: HDR GROUND-TRUTH LUMINANCE")
    print("=" * 70)

    test_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]
    
    all_a = []
    all_b = []
    all_diff = []
    all_sat = []

    for t in test_times:
        # Method A: gray16le → linearize("smpte2084")
        gray = extract_gray16(HDR_PATH, t, WIDTH)
        # Method B: rgb48le → per-channel PQ EOTF → BT.2020 weighted sum
        rgb = extract_rgb48(HDR_PATH, t, WIDTH)

        if gray is None or rgb is None:
            print(f"  {t:.0f}s: extraction failed")
            continue

        # Ensure same resolution
        h = min(gray.shape[0], rgb.shape[0])
        w = min(gray.shape[1], rgb.shape[1])
        gray = gray[:h, :w]
        rgb = rgb[:h, :w, :]

        # METHOD A (current pipeline):
        # gray is Y' (PQ-encoded luma from FFmpeg's YUV→gray conversion)
        # Then linearize applies PQ EOTF: nits = PQ_EOTF(Y') / peak * peak... 
        # Actually linearize("smpte2084") returns PQ_EOTF(signal) / peak_nits
        a_linear = linearize(gray, "smpte2084", peak_nits=10000.0)  # normalized [0,1]
        a_nits = a_linear * 10000.0

        # METHOD B (correct):
        # Apply PQ EOTF to each channel independently, then weighted sum
        r_linear = linearize(rgb[..., 0], "smpte2084", peak_nits=10000.0) * 10000.0
        g_linear = linearize(rgb[..., 1], "smpte2084", peak_nits=10000.0) * 10000.0
        b_linear = linearize(rgb[..., 2], "smpte2084", peak_nits=10000.0) * 10000.0
        b_nits = 0.2627 * r_linear + 0.6780 * g_linear + 0.0593 * b_linear

        # Compute saturation from PQ-encoded RGB
        y_prime = 0.2627 * rgb[..., 0] + 0.6780 * rgb[..., 1] + 0.0593 * rgb[..., 2]
        chroma_r = rgb[..., 0] - y_prime
        chroma_b = rgb[..., 2] - y_prime
        sat = np.sqrt(chroma_r**2 + chroma_b**2)

        # Flatten and filter
        a_flat = a_nits.flatten()
        b_flat = b_nits.flatten()
        sat_flat = sat.flatten()

        valid = (a_flat > 0.1) & (b_flat > 0.1) & np.isfinite(a_flat) & np.isfinite(b_flat)
        a_v = a_flat[valid]
        b_v = b_flat[valid]
        s_v = sat_flat[valid]
        diff = a_v - b_v

        all_a.append(a_v)
        all_b.append(b_v)
        all_diff.append(diff)
        all_sat.append(s_v)

        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        rel = float(np.mean(np.abs(diff) / b_v))
        print(f"\n  Frame {t:.0f}s ({len(a_v):,} valid pixels):")
        print(f"    Method A mean: {np.mean(a_v):.3f} nits")
        print(f"    Method B mean: {np.mean(b_v):.3f} nits")
        print(f"    MAE: {mae:.4f} nits")
        print(f"    RMSE: {rmse:.4f} nits")
        print(f"    Relative error: {rel*100:.2f}%")
        print(f"    Mean signed (A-B): {np.mean(diff):.4f} nits")

    # Aggregate
    a_all = np.concatenate(all_a)
    b_all = np.concatenate(all_b)
    diff_all = np.concatenate(all_diff)
    sat_all = np.concatenate(all_sat)

    print(f"\n{'='*70}")
    print(f"AGGREGATE ({len(a_all):,} pixels)")
    print(f"{'='*70}")

    print(f"\n  Overall statistics:")
    print(f"    MAE:  {np.mean(np.abs(diff_all)):.4f} nits")
    print(f"    RMSE: {np.sqrt(np.mean(diff_all**2)):.4f} nits")
    print(f"    Mean signed (A-B): {np.mean(diff_all):.4f} nits")
    print(f"    Relative error: {np.mean(np.abs(diff_all) / (b_all + 0.01))*100:.2f}%")

    # Percentile comparison
    print(f"\n  Percentile comparison:")
    print(f"  {'Percentile':<12} {'Method A':<12} {'Method B':<12} {'Diff':<12} {'Rel%':<8}")
    print(f"  {'-'*56}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        pa = float(np.percentile(a_all, p))
        pb = float(np.percentile(b_all, p))
        print(f"  P{p:<10} {pa:<12.4f} {pb:<12.4f} {pa-pb:<12.4f} {abs(pa-pb)/pb*100:<8.2f}")

    # Saturation correlation
    print(f"\n  Saturation correlation:")
    abs_diff = np.abs(diff_all)
    # Subsample for speed
    n = min(100000, len(abs_diff))
    idx = np.random.default_rng(42).choice(len(abs_diff), n, replace=False)
    from scipy.stats import spearmanr
    rho_sat, _ = spearmanr(sat_all[idx], abs_diff[idx])
    rho_lum, _ = spearmanr(b_all[idx], abs_diff[idx])
    print(f"    |diff(A,B)| vs saturation: rho = {rho_sat:.4f}")
    print(f"    |diff(A,B)| vs luminance:  rho = {rho_lum:.4f}")

    # For achromatic pixels (low saturation)
    achro_mask = sat_all < 0.01
    if np.sum(achro_mask) > 100:
        achro_mae = float(np.mean(np.abs(diff_all[achro_mask])))
        print(f"    Achromatic (sat<0.01): MAE = {achro_mae:.6f} nits ({np.sum(achro_mask):,} pixels)")

    # For saturated pixels
    sat_mask = sat_all > 0.05
    if np.sum(sat_mask) > 100:
        sat_mae = float(np.mean(np.abs(diff_all[sat_mask])))
        print(f"    Saturated (sat>0.05):  MAE = {sat_mae:.4f} nits ({np.sum(sat_mask):,} pixels)")

    # Check if this error explains the FEAT-007 error pattern
    print(f"\n  IMPLICATIONS FOR FEAT-007:")
    print(f"    FEAT-007 overall MAE was ~0.65 nits")
    print(f"    Ground-truth error (A vs B) MAE = {np.mean(np.abs(diff_all)):.4f} nits")
    ratio = np.mean(np.abs(diff_all)) / 0.65 * 100
    print(f"    Ground-truth error accounts for {ratio:.1f}% of FEAT-007 error")

    print(f"\n{'='*70}")
    print("P0 AUDIT COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
