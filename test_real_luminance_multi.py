"""Multi-segment real-material luminance mapping test — Blade Runner 2049.

Tests:
1. 45:00–45:30 (known dark segment)
2. 01:15:00–01:15:30 (independent segment)
3. Brightest segment (scan for high-luminance scene)
4. Comparison of all three transforms
5. Global transform from combined samples
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

import numpy as np

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ColorConfig
from auto_openmatte.core.models import GeometryModel, Shot, SyncModel, SyncStatus
from auto_openmatte.core.transfer_functions import pq_eotf
from auto_openmatte.processing.luminance import apply_luminance_curve, estimate_luminance_curve
from auto_openmatte.processing.sampling import (
    OverlapSamples,
    compute_diagnostics,
    sample_overlap_luminance,
    train_validation_split,
)

HDR_PATH = Path(
    r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
    r"\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"
)
OM_PATH = Path(
    r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
    r"\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
)

SYNC_OFFSET = 1167
GEOMETRY = GeometryModel(
    scale_x=1.0, scale_y=1.0,
    offset_x=0.0, offset_y=280.0,
    overlap_bbox=[0.0, 280.0, 3840.0, 1880.0],
    confidence=0.9155, is_global=True,
)
CONFIG = ColorConfig(
    samples_per_shot=30,
    luminance_bins=256,
    low_percentile=1.0,
    high_percentile=99.0,
)


def run_segment_test(
    hdr_source, om_source, sync_model, start_seconds, end_seconds, label
):
    """Run luminance sampling + fitting for one segment. Returns results dict."""
    fps = hdr_source.selected_stream.fps
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))

    shot = Shot(
        shot_id=0,
        hdr_start_frame=start_frame,
        hdr_end_frame=end_frame,
        om_start_frame=start_frame + SYNC_OFFSET,
        om_end_frame=end_frame + SYNC_OFFSET,
        duration_frames=end_frame - start_frame,
    )

    t0 = time.perf_counter()
    samples = sample_overlap_luminance(
        hdr_source, om_source, sync_model, GEOMETRY, shot,
        config=CONFIG, n_samples=30, proxy_width=960, peak_nits=10000.0,
    )
    t1 = time.perf_counter()

    if samples.n_valid_pairs < 1000:
        print(f"  [{label}] ERROR: Only {samples.n_valid_pairs} valid pairs. Skipping.")
        return None

    train, val = train_validation_split(samples, train_ratio=0.8, seed=42)

    t2 = time.perf_counter()
    curve = estimate_luminance_curve(train.sdr_luminance, train.hdr_luminance, config=CONFIG)
    t3 = time.perf_counter()

    diag = compute_diagnostics(samples, curve, train_samples=train, val_samples=val)

    # Compute absolute HDR nits from normalized
    hdr_nits = samples.hdr_luminance * 10000.0

    # Percentiles
    sdr = samples.sdr_luminance
    results = {
        "label": label,
        "start": start_seconds,
        "end": end_seconds,
        "n_frames": samples.n_frames,
        "n_raw": samples.n_raw_pairs,
        "n_valid": samples.n_valid_pairs,
        "n_rejected_black": samples.n_rejected_black,
        "n_rejected_clipped": samples.n_rejected_clipped,
        "curve_points": len(curve),
        "confidence": diag.confidence,
        "train_mae": diag.train_mae,
        "train_rmse": diag.train_rmse,
        "val_mae": diag.val_mae,
        "val_rmse": diag.val_rmse,
        "monotonic": diag.monotonic,
        "sampling_time": t1 - t0,
        "fitting_time": t3 - t2,
        "total_time": t3 - t0,
        "curve": curve,
        "samples": samples,
        # SDR percentiles
        "sdr_P1": float(np.percentile(sdr, 1)),
        "sdr_P10": float(np.percentile(sdr, 10)),
        "sdr_P25": float(np.percentile(sdr, 25)),
        "sdr_P50": float(np.percentile(sdr, 50)),
        "sdr_P75": float(np.percentile(sdr, 75)),
        "sdr_P90": float(np.percentile(sdr, 90)),
        "sdr_P95": float(np.percentile(sdr, 95)),
        "sdr_P99": float(np.percentile(sdr, 99)),
        # HDR absolute nits percentiles
        "hdr_P1_nits": float(np.percentile(hdr_nits, 1)),
        "hdr_P10_nits": float(np.percentile(hdr_nits, 10)),
        "hdr_P25_nits": float(np.percentile(hdr_nits, 25)),
        "hdr_P50_nits": float(np.percentile(hdr_nits, 50)),
        "hdr_P75_nits": float(np.percentile(hdr_nits, 75)),
        "hdr_P90_nits": float(np.percentile(hdr_nits, 90)),
        "hdr_P95_nits": float(np.percentile(hdr_nits, 95)),
        "hdr_P99_nits": float(np.percentile(hdr_nits, 99)),
        "hdr_max_nits": float(np.max(hdr_nits)),
    }
    return results


def print_segment_results(r):
    """Print results for one segment."""
    print(f"\n  {'='*60}")
    print(f"  SEGMENT: {r['label']} ({r['start']:.0f}s – {r['end']:.0f}s)")
    print(f"  {'='*60}")
    print(f"  Frames sampled:     {r['n_frames']}")
    print(f"  Raw pairs:          {r['n_raw']:,}")
    print(f"  Valid pairs:        {r['n_valid']:,}")
    print(f"  Rejected black:     {r['n_rejected_black']:,}")
    print(f"  Rejected clipped:   {r['n_rejected_clipped']:,}")
    print(f"  Curve points:       {r['curve_points']}")
    print(f"  Confidence:         {r['confidence']:.4f}")
    print(f"  Monotonic:          {r['monotonic']}")
    print(f"  Train MAE:          {r['train_mae']:.6f}")
    print(f"  Train RMSE:         {r['train_rmse']:.6f}")
    print(f"  Val MAE:            {r['val_mae']:.6f}")
    print(f"  Val RMSE:           {r['val_rmse']:.6f}")
    print(f"  Sampling time:      {r['sampling_time']:.1f}s")
    print(f"  Fitting time:       {r['fitting_time']:.2f}s")
    print(f"  Total time:         {r['total_time']:.1f}s")
    print(f"\n  SDR Linear Luminance Percentiles:")
    for p in ['P1', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']:
        print(f"    {p}: {r[f'sdr_{p}']:.6f}")
    print(f"\n  HDR Absolute Luminance (nits):")
    for p in ['P1', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']:
        print(f"    {p}: {r[f'hdr_{p}_nits']:.2f}")
    print(f"    MAX: {r['hdr_max_nits']:.2f}")


def main():
    print("=" * 70)
    print("MULTI-SEGMENT LUMINANCE MAPPING TEST — Blade Runner 2049")
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

    sync_model = SyncModel(
        frame_offset=SYNC_OFFSET, confidence=0.9745,
        status=SyncStatus.LOCKED, frame_locked=True,
        offset_seconds=SYNC_OFFSET / fps,
    )

    print(f"  HDR: {hdr_source.selected_stream.width}x{hdr_source.selected_stream.height}")
    print(f"  OM:  {om_source.selected_stream.width}x{om_source.selected_stream.height}")
    print(f"  FPS: {fps:.3f}")
    print(f"  Sync offset: {SYNC_OFFSET} frames")

    # ===== TEST 1: 45:00–45:30 =====
    print("\n\n" + "#" * 70)
    print("TEST 1: 45:00–45:30 (known dark segment)")
    print("#" * 70)
    r1 = run_segment_test(hdr_source, om_source, sync_model, 2700.0, 2730.0, "45:00–45:30")
    if r1:
        print_segment_results(r1)

    # ===== TEST 2: 01:15:00–01:15:30 =====
    print("\n\n" + "#" * 70)
    print("TEST 2: 01:15:00–01:15:30 (independent segment)")
    print("#" * 70)
    r2 = run_segment_test(hdr_source, om_source, sync_model, 4500.0, 4530.0, "1:15:00–1:15:30")
    if r2:
        print_segment_results(r2)

    # ===== TEST 3: Bright segment =====
    # Based on film knowledge: outdoor/desert scenes around 00:35:00-00:40:00
    # and Las Vegas scene ~01:40:00. Try 00:37:00–00:37:30 (outdoor flying scene)
    print("\n\n" + "#" * 70)
    print("TEST 3: 00:37:00–00:37:30 (outdoor/bright scene candidate)")
    print("#" * 70)
    r3 = run_segment_test(hdr_source, om_source, sync_model, 2220.0, 2250.0, "37:00–37:30")
    if r3:
        print_segment_results(r3)

    # ===== TEST 4: Comparison =====
    segments = [r for r in [r1, r2, r3] if r is not None]
    if len(segments) >= 2:
        print("\n\n" + "#" * 70)
        print("TEST 4: TRANSFORM COMPARISON")
        print("#" * 70)
        print(f"\n  {'Segment':<20} {'SDR P50':<10} {'SDR P99':<10} "
              f"{'HDR P50 nits':<13} {'HDR P99 nits':<13} {'HDR MAX nits':<13} "
              f"{'Confidence':<11} {'Val MAE':<10}")
        print(f"  {'-'*100}")
        for r in segments:
            print(f"  {r['label']:<20} {r['sdr_P50']:<10.4f} {r['sdr_P99']:<10.4f} "
                  f"{r['hdr_P50_nits']:<13.2f} {r['hdr_P99_nits']:<13.2f} "
                  f"{r['hdr_max_nits']:<13.2f} "
                  f"{r['confidence']:<11.4f} {r['val_mae']:<10.6f}")

    # ===== TEST 5: Global transform =====
    if len(segments) >= 2:
        print("\n\n" + "#" * 70)
        print("TEST 5: GLOBAL TRANSFORM (combined samples)")
        print("#" * 70)

        # Combine all samples
        all_sdr = np.concatenate([r['samples'].sdr_luminance for r in segments])
        all_hdr = np.concatenate([r['samples'].hdr_luminance for r in segments])

        # 80/20 split
        n = len(all_sdr)
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        split = int(n * 0.8)
        train_sdr, val_sdr = all_sdr[idx[:split]], all_sdr[idx[split:]]
        train_hdr, val_hdr = all_hdr[idx[:split]], all_hdr[idx[split:]]

        global_curve = estimate_luminance_curve(train_sdr, train_hdr, config=CONFIG)

        # Global metrics
        mapped_train = apply_luminance_curve(train_sdr, global_curve)
        mapped_val = apply_luminance_curve(val_sdr, global_curve)
        global_train_mae = float(np.mean(np.abs(mapped_train - train_hdr)))
        global_val_mae = float(np.mean(np.abs(mapped_val - val_hdr)))
        global_val_rmse = float(np.sqrt(np.mean((mapped_val - val_hdr) ** 2)))
        global_conf = float(np.corrcoef(mapped_val, val_hdr)[0, 1])

        print(f"\n  Global curve points:  {len(global_curve)}")
        print(f"  Total samples:        {n:,}")
        print(f"  Train MAE:            {global_train_mae:.6f}")
        print(f"  Val MAE:              {global_val_mae:.6f}")
        print(f"  Val RMSE:             {global_val_rmse:.6f}")
        print(f"  Confidence (corr):    {global_conf:.4f}")

        # Per-segment validation with global curve
        print(f"\n  Per-segment validation with global curve:")
        print(f"  {'Segment':<20} {'MAE':<12} {'RMSE':<12}")
        print(f"  {'-'*44}")
        for r in segments:
            seg_mapped = apply_luminance_curve(r['samples'].sdr_luminance, global_curve)
            seg_mae = float(np.mean(np.abs(seg_mapped - r['samples'].hdr_luminance)))
            seg_rmse = float(np.sqrt(np.mean((seg_mapped - r['samples'].hdr_luminance) ** 2)))
            print(f"  {r['label']:<20} {seg_mae:<12.6f} {seg_rmse:<12.6f}")

    print("\n\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
