"""Deep pixel-domain audit: PQ-domain vs linear-light compositing.

Answers:
1. Does extension reproduce HDR master in overlap region?
2. Is PQ-domain compositing physically correct?
3. Linear-light compositing comparison (experiment B).
4. Boundary gradients in nits.
5. All measurements in absolute cd/m² (nits), not normalized.
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, ShotTransform, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import linearize, delinearize, pq_eotf, pq_oetf
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.sampling import sample_overlap_luminance, train_validation_split
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream

HDR_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv")
OM_PATH = Path(r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv")

SYNC_OFFSET = 1167
FPS = 23.976


def extract_frame_rgb48(path, time_s, width, height):
    cmd = [
        "ffmpeg", "-v", "quiet", "-nostdin",
        "-ss", f"{time_s:.6f}", "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "rgb48le",
        "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0 or len(result.stdout) < width * height * 6:
        return None
    arr = np.frombuffer(result.stdout[:width*height*6], dtype=np.uint16)
    return arr.reshape(height, width, 3).astype(np.float64) / 65535.0


def pq_to_nits(pq_signal):
    """Convert PQ signal [0,1] → absolute luminance in nits."""
    return pq_eotf(pq_signal)


def luminance_from_rgb_bt2020(rgb):
    """BT.2020 luminance from linear RGB."""
    return 0.2627 * rgb[..., 0] + 0.6780 * rgb[..., 1] + 0.0593 * rgb[..., 2]


def luminance_from_rgb_bt709(rgb):
    """BT.709 luminance from linear RGB."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def main():
    print("=" * 70)
    print("DEEP PIXEL AUDIT — PQ vs Linear Compositing")
    print("=" * 70)

    # First, get the luminance curve from the segment
    print("\n--- Preparing luminance curve (from 45:00-45:30) ---")
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)
    fps = hdr_source.selected_stream.fps

    sync = SyncModel(frame_offset=1167, confidence=0.9745, status=SyncStatus.LOCKED,
                     frame_locked=True, offset_seconds=1167/fps)
    geom = GeometryModel(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=280.0,
                         overlap_bbox=[0.0, 280.0, 3840.0, 1880.0], confidence=0.9155, is_global=True)
    config = ColorConfig(samples_per_shot=30, luminance_bins=256)

    start_frame = int(round(2700.0 * fps))
    end_frame = int(round(2730.0 * fps))
    shot = Shot(shot_id=0, hdr_start_frame=start_frame, hdr_end_frame=end_frame,
                om_start_frame=start_frame+1167, om_end_frame=end_frame+1167,
                duration_frames=end_frame - start_frame)

    samples = sample_overlap_luminance(hdr_source, om_source, sync, geom, shot,
                                       config=config, n_samples=30, proxy_width=960)
    train, _ = train_validation_split(samples, train_ratio=0.8, seed=42)
    curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=config)
    print(f"  Curve ready: {len(curve)} points")

    # Extract 5 frame pairs at different times
    test_times = [2705.0, 2710.0, 2715.0, 2720.0, 2725.0]

    print("\n" + "=" * 70)
    print("OVERLAP REPRODUCTION TEST (5 frames)")
    print("Does transform(SDR_overlap) reproduce HDR_master in overlap?")
    print("=" * 70)

    all_hdr_nits = []
    all_transformed_nits = []
    all_errors_nits = []

    for frame_time in test_times:
        hdr_time = frame_time
        om_time = frame_time + 1167 / fps

        hdr_frame = extract_frame_rgb48(HDR_PATH, hdr_time, 3840, 1600)
        om_frame = extract_frame_rgb48(OM_PATH, om_time, 3840, 2160)

        if hdr_frame is None or om_frame is None:
            print(f"  Frame {frame_time:.0f}s: extraction failed, skipping")
            continue

        # --- HDR MASTER: PQ → linear nits ---
        # hdr_frame is PQ-encoded RGB [0,1]
        # Convert to linear: linearize("smpte2084") returns normalized [0,1] where 1=peak_nits
        hdr_linear = linearize(hdr_frame, "smpte2084", peak_nits=10000.0)
        hdr_nits = hdr_linear * 10000.0  # absolute nits
        hdr_lum_nits = luminance_from_rgb_bt2020(hdr_nits)

        # --- SDR OM OVERLAP: BT.709 → linear → luminance mapping → HDR nits ---
        om_overlap = om_frame[280:1880, :, :]  # The region that corresponds to HDR

        # Step 1: Linearize SDR (BT.1886 EOTF)
        om_linear = linearize(om_overlap, "bt709")  # [0,1] linear
        om_lum_linear = luminance_from_rgb_bt709(om_linear)  # SDR linear luminance

        # Step 2: Apply luminance curve (SDR linear → HDR normalized)
        om_lum_mapped = apply_luminance_curve(om_lum_linear, curve)  # HDR normalized [0,1]
        om_lum_nits = om_lum_mapped * 10000.0  # absolute nits

        # --- COMPARISON in NITS ---
        # Both should represent the same scene — compare luminance
        hdr_flat = hdr_lum_nits.flatten()
        om_flat = om_lum_nits.flatten()

        # Reject very dark pixels (< 0.1 nit) for error calculation
        valid = (hdr_flat > 0.1) & (om_flat > 0.0)
        hdr_v = hdr_flat[valid]
        om_v = om_flat[valid]

        mae_nits = float(np.mean(np.abs(hdr_v - om_v)))
        rmse_nits = float(np.sqrt(np.mean((hdr_v - om_v)**2)))
        rel_err = float(np.mean(np.abs(hdr_v - om_v) / (hdr_v + 0.01)))
        log_err = float(np.mean(np.abs(np.log10(om_v + 0.01) - np.log10(hdr_v + 0.01))))

        print(f"\n  Frame {frame_time:.0f}s:")
        print(f"    HDR master (nits):     mean={np.mean(hdr_v):.2f} P50={np.median(hdr_v):.2f} "
              f"P95={np.percentile(hdr_v, 95):.2f} P99={np.percentile(hdr_v, 99):.2f}")
        print(f"    Transformed OM (nits): mean={np.mean(om_v):.2f} P50={np.median(om_v):.2f} "
              f"P95={np.percentile(om_v, 95):.2f} P99={np.percentile(om_v, 99):.2f}")
        print(f"    MAE: {mae_nits:.3f} nits")
        print(f"    RMSE: {rmse_nits:.3f} nits")
        print(f"    Relative error: {rel_err*100:.1f}%")
        print(f"    Log-luminance error: {log_err:.4f}")

        all_hdr_nits.append(hdr_v)
        all_transformed_nits.append(om_v)
        all_errors_nits.append(np.abs(hdr_v - om_v))

    # Aggregate
    if all_hdr_nits:
        agg_hdr = np.concatenate(all_hdr_nits)
        agg_om = np.concatenate(all_transformed_nits)
        agg_err = np.concatenate(all_errors_nits)
        print(f"\n  AGGREGATE (5 frames):")
        print(f"    Total valid pixels: {len(agg_hdr):,}")
        print(f"    MAE: {np.mean(agg_err):.3f} nits")
        print(f"    RMSE: {np.sqrt(np.mean(agg_err**2)):.3f} nits")
        print(f"    Relative error: {np.mean(agg_err / (agg_hdr + 0.01))*100:.1f}%")
        print(f"    P50 error: {np.median(agg_err):.3f} nits")
        print(f"    P95 error: {np.percentile(agg_err, 95):.3f} nits")
        print(f"    P99 error: {np.percentile(agg_err, 99):.3f} nits")

    # === EXPERIMENT B: PQ-domain vs Linear-domain compositing ===
    print("\n\n" + "=" * 70)
    print("EXPERIMENT B: PQ-domain vs Linear-domain compositing")
    print("Single frame at 45:15")
    print("=" * 70)

    frame_time = 2715.0
    hdr_frame = extract_frame_rgb48(HDR_PATH, frame_time, 3840, 1600)
    om_frame = extract_frame_rgb48(OM_PATH, frame_time + 1167/fps, 3840, 2160)

    if hdr_frame is None or om_frame is None:
        print("ERROR: frame extraction failed")
        return

    # --- Method A: PQ-domain compositing (current implementation) ---
    # Transform OM to PQ: om → linearize(bt709) → lum_curve → delinearize(smpte2084)
    from auto_openmatte.processing.transform import apply_shot_transform
    transform = ShotTransform(shot_id=0, luminance_curve=curve, exposure=1.0,
                              contrast=1.0, saturation=1.0, confidence=0.97)

    om_pq = apply_shot_transform(om_frame, transform,
                                  sdr_transfer="bt709", hdr_transfer="smpte2084", peak_nits=10000.0)

    # Composite A (PQ-domain): HDR PQ + OM PQ
    composite_a = om_pq.copy()  # Start with transformed OM
    # Place HDR in overlap (scale HDR to fit OM overlap: 1600 rows → 1600 rows, trivial)
    composite_a[280:1880, :, :] = hdr_frame  # HDR PQ values directly

    # --- Method B: Linear-domain compositing ---
    # HDR → linear
    hdr_linear = linearize(hdr_frame, "smpte2084", peak_nits=10000.0)  # [0,1] normalized

    # OM → linear → luminance mapping → linear HDR
    om_linear = linearize(om_frame, "bt709")  # SDR linear [0,1]
    # Apply luminance curve per-channel via luminance ratio
    lum = luminance_from_rgb_bt709(om_linear)
    lum_mapped = apply_luminance_curve(lum, curve)
    ratio = np.where(lum > 1e-6, lum_mapped / lum, 1.0)
    om_hdr_linear = om_linear * ratio[..., np.newaxis]
    om_hdr_linear = np.maximum(om_hdr_linear, 0.0)

    # Composite B (linear-domain): HDR linear + OM linear
    composite_b_linear = om_hdr_linear.copy()
    composite_b_linear[280:1880, :, :] = hdr_linear  # HDR linear values

    # Encode B to PQ
    composite_b_pq = delinearize(composite_b_linear, "smpte2084", peak_nits=10000.0)

    # --- Compare A and B ---
    # Convert both to nits for comparison
    a_nits_top = pq_to_nits(luminance_from_rgb_bt2020(composite_a[0:280, :, :]))
    a_nits_mid = pq_to_nits(luminance_from_rgb_bt2020(composite_a[280:1880, :, :]))
    a_nits_bot = pq_to_nits(luminance_from_rgb_bt2020(composite_a[1880:2160, :, :]))

    b_linear_top = composite_b_linear[0:280, :, :]
    b_linear_mid = composite_b_linear[280:1880, :, :]
    b_linear_bot = composite_b_linear[1880:2160, :, :]
    b_nits_top = luminance_from_rgb_bt2020(b_linear_top) * 10000.0
    b_nits_mid = luminance_from_rgb_bt2020(b_linear_mid) * 10000.0
    b_nits_bot = luminance_from_rgb_bt2020(b_linear_bot) * 10000.0

    print(f"\n  METHOD A (PQ-domain compositing):")
    print(f"    Top extension:  mean={np.mean(a_nits_top):.2f} nits, P50={np.median(a_nits_top):.2f}")
    print(f"    HDR center:     mean={np.mean(a_nits_mid):.2f} nits, P50={np.median(a_nits_mid):.2f}")
    print(f"    Bot extension:  mean={np.mean(a_nits_bot):.2f} nits, P50={np.median(a_nits_bot):.2f}")

    print(f"\n  METHOD B (Linear-domain compositing):")
    print(f"    Top extension:  mean={np.mean(b_nits_top):.2f} nits, P50={np.median(b_nits_top):.2f}")
    print(f"    HDR center:     mean={np.mean(b_nits_mid):.2f} nits, P50={np.median(b_nits_mid):.2f}")
    print(f"    Bot extension:  mean={np.mean(b_nits_bot):.2f} nits, P50={np.median(b_nits_bot):.2f}")

    print(f"\n  DIFFERENCE (A vs B) in nits:")
    diff_top = float(np.mean(np.abs(a_nits_top - b_nits_top)))
    diff_mid = float(np.mean(np.abs(a_nits_mid - b_nits_mid)))
    diff_bot = float(np.mean(np.abs(a_nits_bot - b_nits_bot)))
    print(f"    Top:    MAE = {diff_top:.4f} nits")
    print(f"    Center: MAE = {diff_mid:.4f} nits")
    print(f"    Bottom: MAE = {diff_bot:.4f} nits")

    # Boundary gradient comparison
    print(f"\n  BOUNDARY GRADIENT (nits):")
    # Method A
    a_row279 = float(np.mean(pq_to_nits(luminance_from_rgb_bt2020(composite_a[279:280, :, :]))))
    a_row280 = float(np.mean(pq_to_nits(luminance_from_rgb_bt2020(composite_a[280:281, :, :]))))
    a_row1879 = float(np.mean(pq_to_nits(luminance_from_rgb_bt2020(composite_a[1879:1880, :, :]))))
    a_row1880 = float(np.mean(pq_to_nits(luminance_from_rgb_bt2020(composite_a[1880:1881, :, :]))))

    b_row279 = float(np.mean(luminance_from_rgb_bt2020(b_linear_top[-1:, :, :]) * 10000.0))
    b_row280 = float(np.mean(luminance_from_rgb_bt2020(b_linear_mid[0:1, :, :]) * 10000.0))
    b_row1879 = float(np.mean(luminance_from_rgb_bt2020(b_linear_mid[-1:, :, :]) * 10000.0))
    b_row1880 = float(np.mean(luminance_from_rgb_bt2020(b_linear_bot[0:1, :, :]) * 10000.0))

    print(f"    Method A: row279={a_row279:.3f}, row280={a_row280:.3f}, jump={abs(a_row280-a_row279):.3f}")
    print(f"    Method A: row1879={a_row1879:.3f}, row1880={a_row1880:.3f}, jump={abs(a_row1880-a_row1879):.3f}")
    print(f"    Method B: row279={b_row279:.3f}, row280={b_row280:.3f}, jump={abs(b_row280-b_row279):.3f}")
    print(f"    Method B: row1879={b_row1879:.3f}, row1880={b_row1880:.3f}, jump={abs(b_row1880-b_row1879):.3f}")

    # === KEY QUESTION: Does blending in PQ domain cause error? ===
    # In the FEATHER region (mask between 0 and 1), PQ blending is wrong
    # because PQ is nonlinear. Linear blending in PQ ≠ linear blending in luminance.
    # However, with feather_width=4, this only affects 4 pixels at the boundary.
    # In the CORE regions (mask=0 or mask=1), there's no blending — just selection.
    # So the question is really about the feather.

    print(f"\n  NOTE ON PQ BLENDING:")
    print(f"    Feather width = 4 pixels")
    print(f"    In core regions (mask=0 or mask=1): pure selection, NO blending")
    print(f"    PQ nonlinear blending only affects the 4px feather strip")
    print(f"    For 4K content, 4 pixels is < 0.2% of image height")

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
