"""Boundary diagnostic: vertical luminance profile across OM/HDR boundary.

Tests whether boundary discontinuity is:
A) natural content difference
B) luminance mapping artifact
C) geometry problem
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import linearize, pq_eotf
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")


def extract_frame(path, time_s, width, height):
    cmd = ["ffmpeg", "-v", "quiet", "-nostdin",
           "-ss", f"{time_s:.6f}", "-i", str(path),
           "-frames:v", "1", "-pix_fmt", "rgb48le", "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode != 0 or len(r.stdout) < width * height * 6:
        return None
    return np.frombuffer(r.stdout[:width*height*6], dtype=np.uint16).reshape(
        height, width, 3).astype(np.float64) / 65535.0


def row_median_luma_bt709(frame, row):
    """Median BT.709 luma of a single row (signal domain)."""
    r = frame[row, :, :]
    luma = 0.2126 * r[:, 0] + 0.7152 * r[:, 1] + 0.0722 * r[:, 2]
    return float(np.median(luma))


def row_median_nits_pq(frame, row):
    """Median luminance in nits from PQ-encoded frame."""
    r = frame[row, :, :]
    luma_pq = 0.2627 * r[:, 0] + 0.6780 * r[:, 1] + 0.0593 * r[:, 2]
    nits = pq_eotf(luma_pq)
    return float(np.median(nits))


def apply_curve_to_row(om_signal_row, curve):
    """Apply luminance transform to a single OM row. Returns nits."""
    # om_signal_row is [W, 3] in SDR signal domain [0,1]
    luma_signal = 0.2126 * om_signal_row[:, 0] + 0.7152 * om_signal_row[:, 1] + 0.0722 * om_signal_row[:, 2]
    # Linearize BT.709
    luma_linear = linearize(luma_signal, "bt709")
    # Apply curve (SDR linear → HDR normalized [0,1] where 1=10000 nits)
    luma_mapped = apply_luminance_curve(luma_linear, curve)
    # Convert to nits
    nits = luma_mapped * 10000.0
    return float(np.median(nits))


def analyze_frame(hdr_frame, om_frame, curve, label):
    """Full boundary analysis for one frame pair."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # --- RAW OM vertical profile around boundaries ---
    print(f"\n  RAW OM SDR signal (median luma per row):")
    print(f"  {'Row':<8} {'SDR signal':<12} {'Region'}")
    print(f"  {'-'*40}")
    for row in [275, 276, 277, 278, 279, 280, 281, 282, 283, 284]:
        val = row_median_luma_bt709(om_frame, row)
        region = "extension" if row < 280 else "overlap"
        marker = " <<<" if row in (279, 280) else ""
        print(f"  {row:<8} {val:<12.6f} {region}{marker}")

    print(f"\n  {'Row':<8} {'SDR signal':<12} {'Region'}")
    print(f"  {'-'*40}")
    for row in [1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884]:
        val = row_median_luma_bt709(om_frame, row)
        region = "overlap" if row < 1880 else "extension"
        marker = " <<<" if row in (1879, 1880) else ""
        print(f"  {row:<8} {val:<12.6f} {region}{marker}")

    # --- HDR master profile (first/last rows) ---
    print(f"\n  HDR MASTER (median nits per row):")
    print(f"  {'HDR row':<10} {'nits':<12} {'= OM row'}")
    print(f"  {'-'*35}")
    for hdr_row in [0, 1, 2, 3, 4, 5, 10, 15, 19]:
        val = row_median_nits_pq(hdr_frame, hdr_row)
        print(f"  {hdr_row:<10} {val:<12.3f} = OM {hdr_row + 280}")
    print(f"  ...")
    for hdr_row in [1580, 1585, 1590, 1594, 1595, 1596, 1597, 1598, 1599]:
        val = row_median_nits_pq(hdr_frame, hdr_row)
        print(f"  {hdr_row:<10} {val:<12.3f} = OM {hdr_row + 280}")

    # --- Transformed OM profile around boundaries (in nits) ---
    print(f"\n  TRANSFORMED OM (median nits per row, via luminance mapping):")
    print(f"  {'OM row':<10} {'nits':<12} {'Region'}")
    print(f"  {'-'*40}")
    for row in [275, 276, 277, 278, 279, 280, 281, 282, 283, 284]:
        val = apply_curve_to_row(om_frame[row:row+1, :, :].reshape(-1, 3), curve)
        region = "extension" if row < 280 else "overlap"
        marker = " <<<" if row in (279, 280) else ""
        print(f"  {row:<10} {val:<12.3f} {region}{marker}")

    for row in [1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884]:
        val = apply_curve_to_row(om_frame[row:row+1, :, :].reshape(-1, 3), curve)
        region = "overlap" if row < 1880 else "extension"
        marker = " <<<" if row in (1879, 1880) else ""
        print(f"  {row:<10} {val:<12.3f} {region}{marker}")

    # --- COMPARISON: transformed OM overlap vs HDR master ---
    # First 20 rows, middle 20, last 20
    errors_first20 = []
    errors_mid20 = []
    errors_last20 = []

    for i in range(20):
        hdr_nits = row_median_nits_pq(hdr_frame, i)
        om_nits = apply_curve_to_row(om_frame[280+i:281+i, :, :].reshape(-1, 3), curve)
        errors_first20.append(abs(hdr_nits - om_nits))

    mid_start = 790  # middle of 1600
    for i in range(20):
        hdr_nits = row_median_nits_pq(hdr_frame, mid_start + i)
        om_nits = apply_curve_to_row(om_frame[280+mid_start+i:281+mid_start+i, :, :].reshape(-1, 3), curve)
        errors_mid20.append(abs(hdr_nits - om_nits))

    for i in range(20):
        row = 1580 + i
        hdr_nits = row_median_nits_pq(hdr_frame, row)
        om_nits = apply_curve_to_row(om_frame[280+row:281+row, :, :].reshape(-1, 3), curve)
        errors_last20.append(abs(hdr_nits - om_nits))

    print(f"\n  OVERLAP REPRODUCTION ERROR BY REGION (MAE in nits):")
    print(f"    First 20 rows (HDR 0-19 = OM 280-299):   MAE = {np.mean(errors_first20):.4f} nits")
    print(f"    Middle 20 rows (HDR 790-809 = OM 1070-1089): MAE = {np.mean(errors_mid20):.4f} nits")
    print(f"    Last 20 rows (HDR 1580-1599 = OM 1860-1879): MAE = {np.mean(errors_last20):.4f} nits")

    # --- RAW CORRELATION: OM signal vs HDR signal (spatial match) ---
    # Check if OM[280] truly corresponds to HDR[0]
    om_row280_signal = row_median_luma_bt709(om_frame, 280)
    om_row279_signal = row_median_luma_bt709(om_frame, 279)
    hdr_row0_pq = float(np.median(0.2627*hdr_frame[0,:,0] + 0.6780*hdr_frame[0,:,1] + 0.0593*hdr_frame[0,:,2]))

    # Compute correlation between OM overlap rows and HDR rows
    om_overlap_profile = np.array([row_median_luma_bt709(om_frame, r) for r in range(280, 300)])
    hdr_first_profile_pq = np.array([float(np.median(0.2627*hdr_frame[r,:,0]+0.6780*hdr_frame[r,:,1]+0.0593*hdr_frame[r,:,2])) for r in range(0, 20)])

    corr_first20 = float(np.corrcoef(om_overlap_profile, hdr_first_profile_pq)[0, 1])

    om_last_profile = np.array([row_median_luma_bt709(om_frame, r) for r in range(1860, 1880)])
    hdr_last_profile_pq = np.array([float(np.median(0.2627*hdr_frame[r,:,0]+0.6780*hdr_frame[r,:,1]+0.0593*hdr_frame[r,:,2])) for r in range(1580, 1600)])
    corr_last20 = float(np.corrcoef(om_last_profile, hdr_last_profile_pq)[0, 1])

    print(f"\n  SPATIAL CORRELATION (row profiles):")
    print(f"    First 20 rows (OM 280-299 vs HDR 0-19):    r = {corr_first20:.4f}")
    print(f"    Last 20 rows (OM 1860-1879 vs HDR 1580-1599): r = {corr_last20:.4f}")

    # --- KEY BOUNDARY VALUES ---
    hdr_row0_nits = row_median_nits_pq(hdr_frame, 0)
    hdr_row1599_nits = row_median_nits_pq(hdr_frame, 1599)
    om_row279_nits = apply_curve_to_row(om_frame[279:280, :, :].reshape(-1, 3), curve)
    om_row280_nits = apply_curve_to_row(om_frame[280:281, :, :].reshape(-1, 3), curve)
    om_row1879_nits = apply_curve_to_row(om_frame[1879:1880, :, :].reshape(-1, 3), curve)
    om_row1880_nits = apply_curve_to_row(om_frame[1880:1881, :, :].reshape(-1, 3), curve)

    top_jump_raw = abs(om_row279_signal - om_row280_signal)
    top_jump_nits = abs(om_row279_nits - hdr_row0_nits)
    bot_jump_nits = abs(hdr_row1599_nits - om_row1880_nits)

    print(f"\n  BOUNDARY SUMMARY:")
    print(f"    TOP: OM row 279 (ext) = {om_row279_nits:.3f} nits")
    print(f"    TOP: HDR row 0 = OM row 280 (overlap) = {hdr_row0_nits:.3f} nits")
    print(f"    TOP JUMP: {top_jump_nits:.3f} nits ({top_jump_nits/max(hdr_row0_nits,0.01)*100:.1f}% relative)")
    print(f"    RAW SDR jump at 279/280: {top_jump_raw:.6f} signal")
    print(f"")
    print(f"    BOT: HDR row 1599 = OM row 1879 (overlap) = {hdr_row1599_nits:.3f} nits")
    print(f"    BOT: OM row 1880 (ext) = {om_row1880_nits:.3f} nits")
    print(f"    BOT JUMP: {bot_jump_nits:.3f} nits ({bot_jump_nits/max(hdr_row1599_nits,0.01)*100:.1f}% relative)")

    return {
        "top_jump_nits": top_jump_nits,
        "bot_jump_nits": bot_jump_nits,
        "top_raw_jump": top_jump_raw,
        "first20_mae": np.mean(errors_first20),
        "mid20_mae": np.mean(errors_mid20),
        "last20_mae": np.mean(errors_last20),
        "corr_first20": corr_first20,
        "corr_last20": corr_last20,
    }


def main():
    print("=" * 70)
    print("BOUNDARY DIAGNOSTIC — Vertical Luminance Profile Analysis")
    print("=" * 70)

    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps

    # Prepare curve
    print("\n--- Preparing luminance curve ---")
    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256)

    sf = int(round(2700.0 * fps))
    ef = int(round(2730.0 * fps))
    shot = Shot(shot_id=0, hdr_start_frame=sf, hdr_end_frame=ef,
                om_start_frame=sf+1167, om_end_frame=ef+1167, duration_frames=ef-sf)
    samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                       config=config, n_samples=30, proxy_width=960)
    train, _ = train_validation_split(samples, train_ratio=0.8, seed=42)
    curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=config)
    print(f"  Curve: {len(curve)} points")

    # Test frames
    test_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]
    all_results = []

    for t in test_times:
        hdr_time = t
        om_time = t + 1167 / fps

        hdr_frame = extract_frame(HDR_PATH, hdr_time, 3840, 1600)
        om_frame = extract_frame(OM_PATH, om_time, 3840, 2160)

        if hdr_frame is None or om_frame is None:
            print(f"\n  Frame {t:.0f}s: extraction failed")
            continue

        result = analyze_frame(hdr_frame, om_frame, curve, f"Frame at {t:.0f}s")
        all_results.append(result)

    # Summary
    if all_results:
        print(f"\n\n{'='*70}")
        print(f"AGGREGATE SUMMARY ({len(all_results)} frames)")
        print(f"{'='*70}")

        print(f"\n  {'Frame':<12} {'Top jump':<12} {'Bot jump':<12} {'Raw SDR':<12} "
              f"{'1st20 MAE':<12} {'Mid20 MAE':<12} {'Last20 MAE':<12} {'Corr 1st':<10} {'Corr last':<10}")
        print(f"  {'-'*100}")
        for i, r in enumerate(all_results):
            print(f"  {test_times[i]:<12.0f} {r['top_jump_nits']:<12.3f} {r['bot_jump_nits']:<12.3f} "
                  f"{r['top_raw_jump']:<12.6f} {r['first20_mae']:<12.4f} {r['mid20_mae']:<12.4f} "
                  f"{r['last20_mae']:<12.4f} {r['corr_first20']:<10.4f} {r['corr_last20']:<10.4f}")

        mean_top = np.mean([r['top_jump_nits'] for r in all_results])
        mean_bot = np.mean([r['bot_jump_nits'] for r in all_results])
        mean_first = np.mean([r['first20_mae'] for r in all_results])
        mean_mid = np.mean([r['mid20_mae'] for r in all_results])
        mean_last = np.mean([r['last20_mae'] for r in all_results])
        mean_raw = np.mean([r['top_raw_jump'] for r in all_results])

        print(f"\n  MEANS:")
        print(f"    Top boundary jump:       {mean_top:.3f} nits")
        print(f"    Bot boundary jump:       {mean_bot:.3f} nits")
        print(f"    Raw SDR jump (279/280):  {mean_raw:.6f}")
        print(f"    First 20 rows MAE:       {mean_first:.4f} nits")
        print(f"    Middle 20 rows MAE:      {mean_mid:.4f} nits")
        print(f"    Last 20 rows MAE:        {mean_last:.4f} nits")

        print(f"\n  Q3 — Does boundary jump exist in RAW OM (before mapping)?")
        print(f"    Mean raw SDR signal jump at 279/280: {mean_raw:.6f}")
        if mean_raw > 0.01:
            print(f"    YES — significant raw content discontinuity at crop boundary")
        else:
            print(f"    MINIMAL — raw content is mostly continuous at boundary")

        print(f"\n  Q2 — Is mapping equally good at boundary vs center?")
        ratio = mean_first / mean_mid if mean_mid > 0 else 0
        print(f"    First20/Mid20 MAE ratio: {ratio:.2f}")
        if ratio > 2.0:
            print(f"    WORSE at boundary — mapping may have bias near edges")
        else:
            print(f"    SIMILAR — mapping quality is uniform across overlap")

        print(f"\n  Q4 — Does mapping increase the jump?")
        print(f"    Jump before mapping (raw SDR): {mean_raw:.6f} signal units")
        print(f"    Jump after mapping (nits):     {mean_top:.3f} nits")
        print(f"    These are in different units — compare transformed OM 279 vs 280:")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
