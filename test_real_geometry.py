"""Real-material geometry alignment test — Blade Runner 2049.

One-off test harness. Calls estimate_geometry directly with validated SyncModel.
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from auto_openmatte.analysis.geometry import estimate_geometry
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import GeometryConfig
from auto_openmatte.core.models import SyncModel, SyncStatus

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
    print("REAL-MATERIAL GEOMETRY TEST — Blade Runner 2049")
    print("=" * 70)

    if not HDR_PATH.exists():
        print(f"ERROR: HDR file not found: {HDR_PATH}")
        return
    if not OM_PATH.exists():
        print(f"ERROR: OM file not found: {OM_PATH}")
        return

    # Inspect
    print("\n--- Inspecting sources ---")
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    om_source = inspect_source(OM_PATH)
    select_video_stream(om_source)

    hdr_s = hdr_source.selected_stream
    om_s = om_source.selected_stream
    print(f"  HDR: {hdr_s.width}x{hdr_s.height}, {hdr_s.fps:.3f} fps")
    print(f"  OM:  {om_s.width}x{om_s.height}, {om_s.fps:.3f} fps")

    # Validated SyncModel
    fps = hdr_s.fps
    sync_model = SyncModel(
        frame_offset=1167,
        confidence=0.9745,
        status=SyncStatus.LOCKED,
        frame_locked=True,
        offset_seconds=1167 / fps,
        method="multi_window_temporal_edge_diff_ncc",
    )
    print(f"  Sync offset: {sync_model.frame_offset} frames")

    # Config
    config = GeometryConfig(
        min_confidence=0.95,
        max_features=5000,
        allow_per_shot=True,
        stability_tolerance=2.0,
    )

    # Run geometry estimation
    print("\n" + "=" * 70)
    print("RUNNING GEOMETRY ESTIMATION")
    print("=" * 70)

    t0 = time.perf_counter()
    try:
        geometry = estimate_geometry(hdr_source, om_source, sync_model, config=config)
    except Exception as e:
        t1 = time.perf_counter()
        print(f"\nERROR after {t1 - t0:.1f}s:")
        print(f"  Type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        traceback.print_exc()
        return
    t1 = time.perf_counter()

    # Results
    print(f"\n--- GEOMETRY RESULTS ---")
    print(f"  resolution HDR:   {hdr_s.width}x{hdr_s.height}")
    print(f"  resolution OM:    {om_s.width}x{om_s.height}")
    print(f"  scale_x:          {geometry.scale_x:.6f}")
    print(f"  scale_y:          {geometry.scale_y:.6f}")
    print(f"  offset_x:         {geometry.offset_x:.2f}")
    print(f"  offset_y:         {geometry.offset_y:.2f}")
    print(f"  overlap_bbox:     [{geometry.overlap_bbox[0]:.1f}, "
          f"{geometry.overlap_bbox[1]:.1f}, "
          f"{geometry.overlap_bbox[2]:.1f}, "
          f"{geometry.overlap_bbox[3]:.1f}]")
    print(f"  confidence:       {geometry.confidence:.6f}")
    print(f"  is_global:        {geometry.is_global}")
    print(f"  shot_id:          {geometry.shot_id}")
    print(f"  processing_time:  {t1 - t0:.1f}s")
    print(f"  FFmpeg calls:     14 (7 samples × 2 sources)")

    # Expected vs actual
    expected_oy = (om_s.height - hdr_s.height) / 2.0  # 280 for centered
    print(f"\n--- EXPECTED vs ACTUAL ---")
    print(f"  Expected offset_y (centered): {expected_oy:.0f}")
    print(f"  Actual offset_y:              {geometry.offset_y:.1f}")
    print(f"  Difference:                   {abs(geometry.offset_y - expected_oy):.1f} px")

    # Extension regions
    ext_top = geometry.overlap_bbox[1]
    ext_bottom = om_s.height - geometry.overlap_bbox[3]
    print(f"\n--- EXTENSION REGIONS ---")
    print(f"  Extension top:    {ext_top:.0f} px")
    print(f"  Extension bottom: {ext_bottom:.0f} px")
    print(f"  HDR region:       {geometry.overlap_bbox[1]:.0f} – "
          f"{geometry.overlap_bbox[3]:.0f} "
          f"({geometry.overlap_bbox[3] - geometry.overlap_bbox[1]:.0f} px)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
