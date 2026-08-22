"""Real-material luminance mapping test — Blade Runner 2049 (45:00–45:30).

Extracts overlap samples, fits luminance curve, validates, reports diagnostics.
No rendering — diagnostic only.
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.processing.luminance import estimate_luminance_curve, apply_luminance_curve
from auto_openmatte.processing.sampling import (
    OverlapSamples,
    compute_diagnostics,
    sample_overlap_luminance,
    train_validation_split,
)

import numpy as np

HDR_PATH = Path(
    r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
    r"\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"
)
OM_PATH = Path(
    r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
    r"\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
)


def main() -> None:
    print("=" * 70)
    print("REAL-MATERIAL LUMINANCE MAPPING TEST — Blade Runner 2049")
    print("Range: 45:00 – 45:30 (HDR timeline)")
    print("=" * 70)

    if not HDR_PATH.exists() or not OM_PATH.exists():
        print("ERROR: Source files not found")
        return

    # Inspect
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)

    fps = hdr_source.selected_stream.fps

    # Validated models
    sync_model = SyncModel(
        frame_offset=1167, confidence=0.9745,
        status=SyncStatus.LOCKED, frame_locked=True,
        offset_seconds=1167 / fps,
    )
    geometry = GeometryModel(
        scale_x=1.0, scale_y=1.0,
        offset_x=0.0, offset_y=280.0,
        overlap_bbox=[0.0, 280.0, 3840.0, 1880.0],
        confidence=0.9155, is_global=True,
    )

    # Shot: 45:00 – 45:30
    start_frame = int(round(2700.0 * fps))
    end_frame = int(round(2730.0 * fps))
    shot = Shot(
        shot_id=0,
        hdr_start_frame=start_frame,
        hdr_end_frame=end_frame,
        om_start_frame=start_frame + sync_model.frame_offset,
        om_end_frame=end_frame + sync_model.frame_offset,
        duration_frames=end_frame - start_frame,
    )

    config = ColorConfig(
        samples_per_shot=30,
        luminance_bins=256,
        low_percentile=1.0,
        high_percentile=99.0,
    )

    # ===== SAMPLE =====
    print(f"\n--- Sampling overlap ({config.samples_per_shot} frames) ---")
    t0 = time.perf_counter()
    samples = sample_overlap_luminance(
        hdr_source, om_source, sync_model, geometry, shot,
        config=config, n_samples=config.samples_per_shot,
        proxy_width=960, peak_nits=10000.0,
    )
    t1 = time.perf_counter()

    print(f"  Frames sampled:     {samples.n_frames}")
    print(f"  Raw pixel pairs:    {samples.n_raw_pairs:,}")
    print(f"  Valid pairs:        {samples.n_valid_pairs:,}")
    print(f"  Rejected black:     {samples.n_rejected_black:,}")
    print(f"  Rejected clipped:   {samples.n_rejected_clipped:,}")
    print(f"  Rejected NaN:       {samples.n_rejected_nan:,}")
    print(f"  Sampling time:      {t1 - t0:.1f}s")

    if samples.n_valid_pairs < 1000:
        print("\nERROR: Too few valid samples. Aborting.")
        return

    # ===== TRAIN/VAL SPLIT =====
    train, val = train_validation_split(samples, train_ratio=0.8, seed=42)
    print(f"\n--- Train/Validation Split ---")
    print(f"  Train samples:      {train.n_valid_pairs:,}")
    print(f"  Validation samples: {val.n_valid_pairs:,}")

    # ===== FIT LUMINANCE CURVE =====
    print(f"\n--- Fitting luminance curve ---")
    t2 = time.perf_counter()
    curve = estimate_luminance_curve(
        train.sdr_luminance, train.hdr_luminance, config=config
    )
    t3 = time.perf_counter()
    print(f"  Control points:     {len(curve)}")
    print(f"  Fit time:           {t3 - t2:.3f}s")

    # ===== DIAGNOSTICS =====
    diag = compute_diagnostics(samples, curve, train_samples=train, val_samples=val)

    print(f"\n--- Diagnostics ---")
    print(f"  Monotonic:          {diag.monotonic}")
    print(f"  Confidence:         {diag.confidence:.4f}")
    print(f"  Train MAE:          {diag.train_mae:.6f}")
    print(f"  Train RMSE:         {diag.train_rmse:.6f}")
    print(f"  Val MAE:            {diag.val_mae:.6f}")
    print(f"  Val RMSE:           {diag.val_rmse:.6f}")

    print(f"\n--- SDR Luminance Percentiles ---")
    for k, v in sorted(diag.sdr_percentiles.items()):
        print(f"    {k}: {v:.6f}")

    print(f"\n--- HDR Luminance Percentiles (normalized to [0,1]) ---")
    for k, v in sorted(diag.hdr_percentiles.items()):
        print(f"    {k}: {v:.6f}")

    # ===== LUMINANCE CURVE SUMMARY =====
    print(f"\n--- Luminance Curve (SDR linear → HDR normalized) ---")
    print(f"  First 5 points: {curve[:5]}")
    print(f"  Mid point:      {curve[len(curve)//2]}")
    print(f"  Last 5 points:  {curve[-5:]}")

    # ===== PER-RANGE ERRORS =====
    sdr = samples.sdr_luminance
    hdr = samples.hdr_luminance
    mapped = apply_luminance_curve(sdr, curve)
    errors = np.abs(mapped - hdr)

    print(f"\n--- Per-Range Error Analysis ---")
    ranges = [
        ("Shadows (P0-P25)", 0.0, np.percentile(sdr, 25)),
        ("Midtones (P25-P75)", np.percentile(sdr, 25), np.percentile(sdr, 75)),
        ("Highlights (P75-P100)", np.percentile(sdr, 75), 1.0),
    ]
    for name, lo, hi in ranges:
        mask = (sdr >= lo) & (sdr < hi)
        if np.sum(mask) > 0:
            range_mae = float(np.mean(errors[mask]))
            range_rmse = float(np.sqrt(np.mean(errors[mask] ** 2)))
            range_rel = float(np.mean(errors[mask] / (hdr[mask] + 1e-6)))
            print(f"  {name}:")
            print(f"    N={np.sum(mask):,}, MAE={range_mae:.6f}, "
                  f"RMSE={range_rmse:.6f}, RelErr={range_rel:.4f}")

    # ===== CHECK BOUNDS =====
    print(f"\n--- Output Bounds Check ---")
    print(f"  min(mapped):  {float(np.min(mapped)):.8f}")
    print(f"  max(mapped):  {float(np.max(mapped)):.8f}")
    print(f"  any negative: {bool(np.any(mapped < 0))}")
    print(f"  any NaN:      {bool(np.any(np.isnan(mapped)))}")
    print(f"  any Inf:      {bool(np.any(np.isinf(mapped)))}")

    # ===== TOTAL TIME =====
    total = t3 - t0
    print(f"\n--- Performance ---")
    print(f"  Sampling:     {t1 - t0:.1f}s")
    print(f"  Fitting:      {t3 - t2:.3f}s")
    print(f"  Total:        {total:.1f}s")

    # Save curve to JSON
    out_path = Path("luminance_curve_br2049.json")
    with open(out_path, "w") as f:
        json.dump({
            "curve": curve,
            "diagnostics": {
                "confidence": diag.confidence,
                "train_mae": diag.train_mae,
                "val_mae": diag.val_mae,
                "monotonic": diag.monotonic,
                "n_samples": samples.n_valid_pairs,
                "sdr_percentiles": diag.sdr_percentiles,
                "hdr_percentiles": diag.hdr_percentiles,
            },
        }, f, indent=2)
    print(f"\n  Curve saved to: {out_path}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
