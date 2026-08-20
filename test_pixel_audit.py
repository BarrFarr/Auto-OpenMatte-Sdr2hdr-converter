"""Pixel-domain audit of the rendered output.

Extracts one frame from HDR source, OM source, and output file,
measures values in the three vertical regions to identify the tonal mismatch.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np
from auto_openmatte.core.transfer_functions import pq_eotf

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")
OUTPUT_PATH = Path(r"G:\Auto-OpenMatte-Sdr2hdr-converter\test_output\BR2049_FEAT007_45m_30s.mkv")

# Extract frame at 45:15 (middle of 30s segment = 15s into the output)
HDR_TIME = 2715.0  # 45:15 in HDR
OM_TIME = HDR_TIME + 1167 / 23.976  # OM corresponding time
OUTPUT_TIME = 15.0  # 15s into the output file


def extract_frame_rgb48(path, time_s, width, height):
    """Extract one frame as rgb48le."""
    cmd = [
        "ffmpeg", "-v", "quiet", "-nostdin",
        "-ss", f"{time_s:.6f}",
        "-i", str(path),
        "-frames:v", "1",
        "-pix_fmt", "rgb48le",
        "-f", "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or len(result.stdout) < width * height * 6:
        return None
    arr = np.frombuffer(result.stdout[:width*height*6], dtype=np.uint16)
    return arr.reshape(height, width, 3).astype(np.float64) / 65535.0


def analyze_region(frame, y_start, y_end, label):
    """Analyze PQ code values in a region."""
    region = frame[y_start:y_end, :, :]
    r_mean = float(np.mean(region[:, :, 0]))
    g_mean = float(np.mean(region[:, :, 1]))
    b_mean = float(np.mean(region[:, :, 2]))
    rgb_mean = float(np.mean(region))

    # Compute luminance in nits (assuming PQ signal)
    # Use green channel as proxy for luminance (BT.2020: G has highest weight)
    lum_signal = 0.2627 * region[:, :, 0] + 0.6780 * region[:, :, 1] + 0.0593 * region[:, :, 2]
    lum_nits = pq_eotf(lum_signal)

    print(f"  {label} (y={y_start}..{y_end-1}):")
    print(f"    PQ signal: R={r_mean:.4f} G={g_mean:.4f} B={b_mean:.4f} mean={rgb_mean:.4f}")
    print(f"    Luminance (nits): mean={float(np.mean(lum_nits)):.2f} "
          f"P50={float(np.median(lum_nits)):.2f} "
          f"P95={float(np.percentile(lum_nits, 95)):.2f} "
          f"max={float(np.max(lum_nits)):.2f}")
    return rgb_mean


def main():
    print("=" * 70)
    print("PIXEL-DOMAIN AUDIT — BR2049 FEAT-007 Output")
    print("=" * 70)

    # Extract frames
    print("\n--- Extracting frames ---")

    hdr_frame = extract_frame_rgb48(HDR_PATH, HDR_TIME, 3840, 1600)
    if hdr_frame is None:
        print("ERROR: Could not extract HDR frame")
        return
    print(f"  HDR frame: {hdr_frame.shape} (signal range [{hdr_frame.min():.4f}, {hdr_frame.max():.4f}])")

    om_frame = extract_frame_rgb48(OM_PATH, OM_TIME, 3840, 2160)
    if om_frame is None:
        print("ERROR: Could not extract OM frame")
        return
    print(f"  OM frame: {om_frame.shape} (signal range [{om_frame.min():.4f}, {om_frame.max():.4f}])")

    out_frame = extract_frame_rgb48(OUTPUT_PATH, OUTPUT_TIME, 3840, 2160)
    if out_frame is None:
        print("ERROR: Could not extract output frame")
        return
    print(f"  Output frame: {out_frame.shape} (signal range [{out_frame.min():.4f}, {out_frame.max():.4f}])")

    # Analyze OUTPUT regions
    print("\n--- OUTPUT FRAME ANALYSIS (should be PQ-encoded) ---")
    top_mean = analyze_region(out_frame, 0, 280, "TOP EXTENSION")
    mid_mean = analyze_region(out_frame, 280, 1880, "HDR CENTER")
    bot_mean = analyze_region(out_frame, 1880, 2160, "BOTTOM EXTENSION")

    print(f"\n  TONAL JUMP: top/mid ratio = {top_mean/mid_mean:.3f}" if mid_mean > 0 else "")
    print(f"  TONAL JUMP: bot/mid ratio = {bot_mean/mid_mean:.3f}" if mid_mean > 0 else "")

    # Boundary analysis
    print("\n--- BOUNDARY ANALYSIS ---")
    row_279 = float(np.mean(out_frame[279, :, :]))
    row_280 = float(np.mean(out_frame[280, :, :]))
    row_1879 = float(np.mean(out_frame[1879, :, :]))
    row_1880 = float(np.mean(out_frame[1880, :, :]))
    print(f"  Row 279 (last extension): mean PQ = {row_279:.4f}")
    print(f"  Row 280 (first HDR):      mean PQ = {row_280:.4f}")
    print(f"  JUMP at y=280:            {abs(row_280 - row_279):.4f}")
    print(f"  Row 1879 (last HDR):      mean PQ = {row_1879:.4f}")
    print(f"  Row 1880 (first extension): mean PQ = {row_1880:.4f}")
    print(f"  JUMP at y=1880:           {abs(row_1880 - row_1879):.4f}")

    # Analyze HDR source (what values does the original master have?)
    print("\n--- HDR SOURCE (original PQ values) ---")
    hdr_mean = float(np.mean(hdr_frame))
    hdr_lum = pq_eotf(0.2627 * hdr_frame[:,:,0] + 0.6780 * hdr_frame[:,:,1] + 0.0593 * hdr_frame[:,:,2])
    print(f"  PQ signal mean: {hdr_mean:.4f}")
    print(f"  Luminance (nits): mean={float(np.mean(hdr_lum)):.2f} "
          f"P50={float(np.median(hdr_lum)):.2f} "
          f"P95={float(np.percentile(hdr_lum, 95)):.2f}")

    # Analyze OM source
    print("\n--- OM SOURCE (original SDR values) ---")
    om_mean = float(np.mean(om_frame))
    # OM overlap region (what goes into center)
    om_overlap = om_frame[280:1880, :, :]
    om_overlap_mean = float(np.mean(om_overlap))
    om_ext_top = float(np.mean(om_frame[0:280, :, :]))
    om_ext_bot = float(np.mean(om_frame[1880:2160, :, :]))
    print(f"  SDR signal mean (full): {om_mean:.4f}")
    print(f"  SDR signal mean (overlap 280:1880): {om_overlap_mean:.4f}")
    print(f"  SDR signal mean (top 0:280): {om_ext_top:.4f}")
    print(f"  SDR signal mean (bot 1880:2160): {om_ext_bot:.4f}")

    # KEY DIAGNOSTIC: What does composite_extend actually do to HDR frame?
    print("\n--- DIAGNOSTIC: HDR frame in rendering pipeline ---")
    print(f"  HDR frame is decoded as rgb48le → float [0,1] (values ARE PQ-encoded)")
    print(f"  These PQ values are placed DIRECTLY into the overlap region of output")
    print(f"  The output canvas already contains PQ-encoded transformed OM")
    print(f"  So compositing blends: PQ_HDR × (1-mask) + PQ_OM_transformed × mask")
    print(f"  In the center (mask≈0): output = HDR PQ values directly → CORRECT")
    print(f"  In extension (mask=1): output = transformed OM → should be PQ-encoded")

    # The transform pipeline:
    # om_frame [0,1] SDR signal
    #   → linearize("bt709") → SDR linear
    #   → apply_luminance_curve → mapped luminance
    #   → delinearize("smpte2084") → PQ signal
    # So the extension IS PQ-encoded. But what about the luminance curve values?

    print("\n--- LUMINANCE CURVE ANALYSIS ---")
    print(f"  The curve maps SDR_linear → HDR_normalized")
    print(f"  HDR_normalized = absolute_nits / peak_nits")
    print(f"  For BR2049 at 45:00: HDR P99 ≈ 69 nits → normalized ≈ 0.0069")
    print(f"  SDR linear P99 ≈ 0.335 → curve maps to ≈ 0.007")
    print(f"  After delinearize PQ: 0.007 * 10000 = 70 nits → PQ signal ≈ 0.39")
    print(f"")
    print(f"  But HDR original PQ signal mean ≈ {hdr_mean:.4f}")
    print(f"  And output center mean ≈ {mid_mean:.4f}")
    print(f"  And extension top mean ≈ {top_mean:.4f}")
    print(f"  And extension bot mean ≈ {bot_mean:.4f}")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
