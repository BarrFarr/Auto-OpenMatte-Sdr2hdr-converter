"""Real-material shot detection test — Blade Runner 2049 (45:00–45:30).

One-off test harness. Calls detect_shots directly on HDR source
with the validated SyncModel (offset=1167).
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

from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.shots import detect_shots
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.core.config import ShotConfig
from auto_openmatte.core.models import SyncModel, SyncStatus

HDR_PATH = Path(
    r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
    r"\Blade.Runner.2049.2017.UHD.BluRay.2160p.DDP.7.1.DV.HDR.x265-hallowed.mkv"
)


def main() -> None:
    print("=" * 70)
    print("REAL-MATERIAL SHOT DETECTION TEST — Blade Runner 2049")
    print("Range: 45:00 – 45:30 (HDR timeline)")
    print("=" * 70)

    if not HDR_PATH.exists():
        print(f"ERROR: HDR file not found: {HDR_PATH}")
        return

    # Inspect source
    print("\n--- Inspecting HDR source ---")
    hdr_source = inspect_source(HDR_PATH)
    select_video_stream(hdr_source)
    stream = hdr_source.selected_stream
    fps = stream.fps
    print(f"  {stream.width}x{stream.height}, {fps:.3f} fps, "
          f"duration={stream.duration_seconds:.2f}s")

    # Build SyncModel from validated result
    sync_model = SyncModel(
        frame_offset=1167,
        confidence=0.9745,
        status=SyncStatus.LOCKED,
        frame_locked=True,
        offset_seconds=1167 / fps,
        method="multi_window_temporal_edge_diff_ncc",
    )

    # Range: 45:00 – 45:30
    start_seconds = 45 * 60.0  # 2700s
    end_seconds = 45 * 60.0 + 30.0  # 2730s
    start_frame = int(round(start_seconds * fps))
    end_frame = int(round(end_seconds * fps))

    print(f"\n  HDR range: {start_seconds:.0f}s – {end_seconds:.0f}s")
    print(f"  Frame range: {start_frame} – {end_frame}")
    print(f"  Sync offset: {sync_model.frame_offset} frames")

    # Shot detection config
    config = ShotConfig(
        threshold_multiplier=3.0,
        min_shot_frames=6,
        detect_transitions=True,
    )

    # Run shot detection
    print("\n" + "=" * 70)
    print("RUNNING SHOT DETECTION")
    print("=" * 70)

    t0 = time.perf_counter()
    try:
        shots = detect_shots(
            hdr_source, sync_model,
            config=config,
            frame_range=(start_frame, end_frame),
        )
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
    print(f"\n--- RESULTS ---")
    print(f"  Shots detected: {len(shots)}")
    print(f"  Time: {t1 - t0:.1f}s")
    print()

    for shot in shots:
        hdr_start_t = shot.hdr_start_frame / fps
        hdr_end_t = shot.hdr_end_frame / fps
        duration_s = shot.duration_frames / fps
        print(f"  Shot {shot.shot_id:3d}: "
              f"HDR [{shot.hdr_start_frame:>7}–{shot.hdr_end_frame:>7}] "
              f"({hdr_start_t:>8.3f}s–{hdr_end_t:>8.3f}s) "
              f"dur={duration_s:.3f}s "
              f"type={shot.cut_type:<9} "
              f"conf={shot.confidence:.3f} "
              f"OM [{shot.om_start_frame:>7}–{shot.om_end_frame:>7}]")

    print(f"\n--- SUMMARY ---")
    print(f"  Total shots: {len(shots)}")
    print(f"  Hard cuts: {sum(1 for s in shots if s.cut_type == 'hard')}")
    print(f"  Fades: {sum(1 for s in shots if s.cut_type == 'fade')}")
    print(f"  Dissolves: {sum(1 for s in shots if s.cut_type == 'dissolve')}")
    print(f"  Time: {t1 - t0:.1f}s")
    print(f"  FFmpeg calls: NOT MEASURED")

    # Verify OM mapping
    offset = sync_model.frame_offset
    mapping_ok = all(
        s.om_start_frame == s.hdr_start_frame + offset
        and s.om_end_frame == s.hdr_end_frame + offset
        for s in shots
    )
    print(f"  OM mapping correct: {mapping_ok}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
