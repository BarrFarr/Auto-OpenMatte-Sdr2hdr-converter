"""Real-material synchronization test — Blade Runner 2049.

This is a one-off test harness. It bypasses the full pipeline (which would
block on DV detection) and calls find_global_offset / validate_sync directly.

NOT production code. NOT part of the test suite.
"""

import logging
import sys
import time
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.analysis.sync import find_global_offset, validate_sync
from auto_openmatte.core.config import SyncConfig
from auto_openmatte.core.exceptions import SyncDriftError

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
    print("REAL-MATERIAL SYNC TEST — Blade Runner 2049")
    print("=" * 70)

    # Verify files exist
    if not HDR_PATH.exists():
        print(f"ERROR: HDR file not found: {HDR_PATH}")
        return
    if not OM_PATH.exists():
        print(f"ERROR: OM file not found: {OM_PATH}")
        return

    print(f"\nHDR: {HDR_PATH.name}")
    print(f"OM:  {OM_PATH.name}")

    # ===== INSPECT =====
    print("\n--- Inspecting sources ---")
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)

    hdr_stream = hdr_source.selected_stream
    om_stream = om_source.selected_stream

    print(f"\nHDR stream: {hdr_stream.width}x{hdr_stream.height}, "
          f"{hdr_stream.fps:.3f} fps, "
          f"frames={hdr_stream.frame_count}, "
          f"duration={hdr_stream.duration_seconds:.2f}s, "
          f"transfer={hdr_stream.transfer.value}, "
          f"primaries={hdr_stream.color_primaries.value}")
    print(f"OM stream:  {om_stream.width}x{om_stream.height}, "
          f"{om_stream.fps:.3f} fps, "
          f"frames={om_stream.frame_count}, "
          f"duration={om_stream.duration_seconds:.2f}s, "
          f"transfer={om_stream.transfer.value}, "
          f"primaries={om_stream.color_primaries.value}")
    print(f"\nHDR metadata: {hdr_source.hdr_metadata.format.value}")

    # ===== SYNC CONFIG =====
    config = SyncConfig(
        search_range_seconds=120.0,
        validation_points=20,
        min_confidence=0.95,
        max_drift_frames=0.5,
        proxy_width=480,
    )

    # ===== FIND GLOBAL OFFSET =====
    print("\n" + "=" * 70)
    print("STAGE: find_global_offset")
    print("=" * 70)

    t0 = time.perf_counter()
    try:
        sync_model = find_global_offset(hdr_source, om_source, config=config)
    except Exception as e:
        t1 = time.perf_counter()
        print(f"\nERROR in find_global_offset after {t1 - t0:.1f}s:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        return
    t1 = time.perf_counter()
    find_time = t1 - t0

    print(f"\n--- GLOBAL SYNCHRONIZATION ---")
    print(f"  status:         {sync_model.status.value}")
    print(f"  offset_frames:  {sync_model.frame_offset}")
    print(f"  offset_seconds: {sync_model.offset_seconds:.6f}")
    print(f"  confidence:     {sync_model.confidence:.6f}")
    print(f"  method:         {sync_model.method}")
    print(f"  frame_locked:   {sync_model.frame_locked}")
    print(f"  time:           {find_time:.1f}s")

    # ===== VALIDATE SYNC =====
    # Always validate — even if initial confidence was below threshold,
    # validation can confirm or reject the candidate offset.
    print("\n" + "=" * 70)
    print("STAGE: validate_sync (20 checkpoints)")
    print("=" * 70)

    t2 = time.perf_counter()
    try:
        sync_model = validate_sync(hdr_source, om_source, sync_model, config=config)
    except SyncDriftError as e:
        t3 = time.perf_counter()
        validate_time = t3 - t2
        print(f"\n  DRIFT DETECTED after {validate_time:.1f}s:")
        print(f"  {e}")
        print(f"\n  Model state:")
        print(f"    status:           {sync_model.status.value}")
        print(f"    drift_frames:     {sync_model.drift_frames:.4f}")
        print(f"    mean_error:       {sync_model.mean_error_frames:.4f}")
        print(f"    max_error:        {sync_model.max_error_frames:.4f}")
        # Print checkpoints
        if sync_model.checkpoints:
            print(f"\n  --- 20-POINT VALIDATION ---")
            for i, cp in enumerate(sync_model.checkpoints, 1):
                print(f"    {i:02d}: hdr_frame={cp['hdr_frame']:>7}, "
                      f"offset={cp['measured_offset']:>5}, "
                      f"error={cp['error_frames']:.3f}, "
                      f"conf={cp['confidence']:.3f}")
        print(f"\n  TOTAL TIME: {find_time + validate_time:.1f}s")
        return
    except Exception as e:
        t3 = time.perf_counter()
        print(f"\nERROR in validate_sync after {t3 - t2:.1f}s:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        return
    t3 = time.perf_counter()
    validate_time = t3 - t2

    # ===== RESULTS =====
    print(f"\n--- VALIDATION COMPLETE ---")
    print(f"  status:           {sync_model.status.value}")
    print(f"  drift_frames:     {sync_model.drift_frames:.4f}")
    print(f"  mean_error:       {sync_model.mean_error_frames:.4f}")
    print(f"  max_error:        {sync_model.max_error_frames:.4f}")
    print(f"  frame_locked:     {sync_model.frame_locked}")

    # Print all checkpoints
    if sync_model.checkpoints:
        print(f"\n  --- 20-POINT VALIDATION ---")
        for i, cp in enumerate(sync_model.checkpoints, 1):
            print(f"    {i:02d}: hdr_frame={cp['hdr_frame']:>7}, "
                  f"offset={cp['measured_offset']:>5}, "
                  f"error={cp['error_frames']:.3f}, "
                  f"conf={cp['confidence']:.3f}")

    print(f"\n--- PERFORMANCE ---")
    print(f"  find_global_offset: {find_time:.1f}s")
    print(f"  validate_sync:      {validate_time:.1f}s")
    print(f"  TOTAL:              {find_time + validate_time:.1f}s")
    print(f"  FFmpeg calls:       NOT MEASURED (subprocess count not instrumented)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
