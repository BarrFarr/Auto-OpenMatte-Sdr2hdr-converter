"""P2.9 continued: White/gray dots on jacket at 3.5s.

Extracts the exact frame, finds anomalous bright pixels in the dark jacket region,
traces them back to source SDR to determine origin.
"""
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from auto_openmatte.core.transfer_functions import linearize, pq_eotf

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_PATH = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output\BR2049_P28_Log_5s.mkv")
PEAK = 10000.0


def extract_frame(path, time_s, w, h):
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin", "-ss", f"{time_s:.6f}",
           "-i", str(path), "-frames:v", "1", "-pix_fmt", "rgb48le",
           "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or len(r.stdout) < w * h * 6:
        return None
    return np.frombuffer(r.stdout[:w*h*6], dtype=np.uint16).reshape(h, w, 3).astype(np.float64) / 65535.0


def main():
    print("=" * 70)
    print("P2.9 DOTS DIAGNOSIS: White/gray dots on jacket at 3.5s")
    print("=" * 70)

    # The render starts at HDR time 2710.0s
    # 3.5s into render = HDR time 2713.5s
    hdr_time = 2713.5
    om_time = hdr_time + 1167 / 23.976
    output_time = 3.5

    print(f"\n--- Extracting frame at 3.5s into render ---")
    print(f"  HDR time: {hdr_time:.1f}s")
    print(f"  OM time: {om_time:.2f}s")

    # Extract from output
    out_frame = extract_frame(OUTPUT_PATH, output_time, 3840, 2160)
    # Extract SDR source at corresponding time
    om_frame = extract_frame(OM_PATH, om_time, 3840, 2160)

    if out_frame is None:
        print("ERROR: could not extract output frame")
        return
    if om_frame is None:
        print("ERROR: could not extract OM frame")
        return

    # Focus on bottom extension where jacket is (rows 1880-2100)
    # Look for anomalous bright pixels in dark region
    ext_region_out = out_frame[1880:2100, :, :]
    ext_region_om = om_frame[1880:2100, :, :]

    # Compute luminance of output (PQ → nits)
    y_pq = 0.2627 * ext_region_out[..., 0] + 0.6780 * ext_region_out[..., 1] + 0.0593 * ext_region_out[..., 2]
    y_nits = pq_eotf(y_pq)

    # Statistics of the region
    median_nits = float(np.median(y_nits))
    print(f"\n--- Extension region (rows 1880-2100) ---")
    print(f"  Output luminance: median={median_nits:.2f} nits, P99={np.percentile(y_nits, 99):.2f}")

    # Find pixels that are significantly brighter than median (anomalous dots)
    # "Dots" would be pixels much brighter than their surroundings
    threshold = median_nits * 5  # 5x brighter than median = likely a dot
    bright_mask = y_nits > threshold
    n_bright = int(np.sum(bright_mask))
    print(f"  Pixels > {threshold:.1f} nits (5x median): {n_bright} ({n_bright/(220*3840)*100:.3f}%)")

    # Also check for pixels > 10 nits in a region with median ~2 nits
    threshold2 = max(10.0, median_nits * 3)
    bright_mask2 = y_nits > threshold2
    n_bright2 = int(np.sum(bright_mask2))
    print(f"  Pixels > {threshold2:.1f} nits (3x or 10 nits): {n_bright2} ({n_bright2/(220*3840)*100:.2f}%)")

    # Find positions of brightest outlier pixels
    flat_nits = y_nits.flatten()
    top_indices = np.argsort(flat_nits)[-20:]  # Top 20 brightest
    h_region = 220
    w_region = 3840

    print(f"\n--- Top 20 brightest pixels in extension ---")
    print(f"  {'#':<4} {'Row':<6} {'Col':<6} {'Nits':<10} {'OM_R':<8} {'OM_G':<8} {'OM_B':<8} {'Out_R':<8} {'Out_G':<8} {'Out_B':<8}")
    print(f"  {'-'*70}")
    for i, idx in enumerate(reversed(top_indices)):
        row = idx // w_region
        col = idx % w_region
        nits_val = flat_nits[idx]
        om_r = ext_region_om[row, col, 0]
        om_g = ext_region_om[row, col, 1]
        om_b = ext_region_om[row, col, 2]
        out_r = ext_region_out[row, col, 0]
        out_g = ext_region_out[row, col, 1]
        out_b = ext_region_out[row, col, 2]
        abs_row = row + 1880
        print(f"  {i+1:<4} {abs_row:<6} {col:<6} {nits_val:<10.2f} {om_r:<8.4f} {om_g:<8.4f} {om_b:<8.4f} {out_r:<8.4f} {out_g:<8.4f} {out_b:<8.4f}")

    # Check if these bright pixels exist in the SDR SOURCE
    print(f"\n--- SDR source analysis of bright pixel locations ---")
    # Get the SDR luminance at these locations
    om_y_signal = 0.2126 * ext_region_om[..., 0] + 0.7152 * ext_region_om[..., 1] + 0.0722 * ext_region_om[..., 2]
    om_median = float(np.median(om_y_signal))
    print(f"  OM SDR signal median in region: {om_median:.4f}")

    # Check if the bright output pixels come from bright SDR pixels
    for i, idx in enumerate(reversed(top_indices[:5])):
        row = idx // w_region
        col = idx % w_region
        om_val = om_y_signal[row, col]
        region_med = float(np.median(om_y_signal[max(0,row-2):row+3, max(0,col-2):col+3]))
        print(f"  Bright pixel {i+1}: OM signal={om_val:.4f}, local_median={region_med:.4f}, "
              f"ratio_to_local={om_val/region_med:.2f}x")

    # Check if SDR source already has these bright spots
    om_bright = om_y_signal > (om_median * 3)
    print(f"\n  SDR pixels > 3x median in extension region: {np.sum(om_bright)} ({np.sum(om_bright)/(220*3840)*100:.2f}%)")

    # Local analysis: 5x5 patch around brightest pixel
    idx_max = top_indices[-1]
    row_max = idx_max // w_region
    col_max = idx_max % w_region
    print(f"\n--- 5x5 patch around brightest pixel (row={row_max+1880}, col={col_max}) ---")
    print(f"  SDR source patch:")
    patch_om = ext_region_om[max(0,row_max-2):row_max+3, max(0,col_max-2):col_max+3, :]
    patch_om_y = 0.2126*patch_om[...,0] + 0.7152*patch_om[...,1] + 0.0722*patch_om[...,2]
    print(f"    Y signal:\n{np.array2string(patch_om_y, precision=4, suppress_small=True)}")

    print(f"\n  Output patch (nits):")
    patch_out = ext_region_out[max(0,row_max-2):row_max+3, max(0,col_max-2):col_max+3, :]
    patch_y_pq = 0.2627*patch_out[...,0] + 0.6780*patch_out[...,1] + 0.0593*patch_out[...,2]
    patch_nits = pq_eotf(patch_y_pq)
    print(f"    Nits:\n{np.array2string(patch_nits, precision=2, suppress_small=True)}")

    print(f"\n{'='*70}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
