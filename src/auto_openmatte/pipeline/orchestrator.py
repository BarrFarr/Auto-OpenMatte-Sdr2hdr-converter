"""Pipeline orchestrator — coordinates all analysis stages.

Executes the pipeline in the correct order:
1. Inspect sources
2. Identify streams and HDR
3. Synchronize (frame offset) — uses cache if available
4. Validate sync (drift check)
5. Build range spec (map HDR range → OM range)
6. Detect shots on HDR (within range + context)
7. Estimate geometry (within range)
8. Estimate luminance + color per shot
9. Validate transforms
10. Save project.json

When a range is specified, detailed processing (shots, geometry, color)
is limited to the selected range + context window. Synchronization
remains global (but lightweight and cached).
"""

from __future__ import annotations

import logging
from pathlib import Path

from auto_openmatte.analysis.geometry import estimate_geometry
from auto_openmatte.analysis.hdr_detect import assign_roles
from auto_openmatte.analysis.inspect import inspect_source
from auto_openmatte.analysis.shots import detect_shots
from auto_openmatte.analysis.stream_select import select_video_stream
from auto_openmatte.analysis.sync import find_global_offset, validate_sync
from auto_openmatte.analysis.sync_debug import write_sync_debug
from auto_openmatte.core.config import PipelineConfig
from auto_openmatte.core.models import ProjectData, SyncStatus
from auto_openmatte.core.project import save_project
from auto_openmatte.core.range_spec import RangeSpec, build_range_spec
from auto_openmatte.core.sync_cache import load_sync_cache, save_sync_cache
from auto_openmatte.output.console import print_analysis_report

logger = logging.getLogger(__name__)


def run_analysis(
    hdr_path: Path,
    openmatte_path: Path,
    config: PipelineConfig | None = None,
    range_spec: RangeSpec | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    duration_seconds: float | None = None,
    context_seconds: float = 2.0,
) -> ProjectData:
    """Run the analysis pipeline, optionally limited to a time range.

    All time arguments refer to the HDR timeline exclusively.
    The Open Matte range is derived automatically from the sync model.

    Order of operations:
    1. Inspect → 2. HDR detect → 3. Sync (cached) → 4. Validate
    5. Build range → 6. Shots (in range) → 7. Geometry → 8. Save

    Args:
        hdr_path: Path to HDR source (or either source — will auto-detect).
        openmatte_path: Path to Open Matte source.
        config: Pipeline configuration.
        range_spec: Pre-built RangeSpec (overrides start/end/duration).
        start_seconds: Start time on HDR timeline (seconds).
        end_seconds: End time on HDR timeline (seconds).
        duration_seconds: Duration in seconds (alternative to end_seconds).
        context_seconds: Context window for range processing (default 2.0s).

    Returns:
        ProjectData with all analysis results.

    Raises:
        AutoOpenMatteError: If any stage fails critically.
    """
    if config is None:
        config = PipelineConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    project = ProjectData()

    # ===== STAGE 1: INSPECT SOURCES =====
    logger.info("=" * 60)
    logger.info("STAGE 1: Source Inspection")
    logger.info("=" * 60)

    source_a = inspect_source(hdr_path)
    source_b = inspect_source(openmatte_path)

    # Select best video stream for each
    select_video_stream(source_a)
    select_video_stream(source_b)

    # ===== STAGE 2: HDR IDENTIFICATION =====
    logger.info("=" * 60)
    logger.info("STAGE 2: HDR Identification")
    logger.info("=" * 60)

    hdr_source, om_source = assign_roles(source_a, source_b)
    project.hdr_source = hdr_source
    project.openmatte_source = om_source

    logger.info(f"HDR: {hdr_source.path.name} ({hdr_source.hdr_metadata.format.value})")
    logger.info(f"Open Matte: {om_source.path.name} (SDR)")

    # Verify FPS match
    hdr_fps = hdr_source.selected_stream.fps if hdr_source.selected_stream else 0
    om_fps = om_source.selected_stream.fps if om_source.selected_stream else 0

    if hdr_fps > 0 and om_fps > 0:
        fps_diff = abs(hdr_fps - om_fps) / max(hdr_fps, om_fps)
        if fps_diff > 0.005:
            project.warnings.append(
                f"FPS mismatch: HDR={hdr_fps:.3f}, OM={om_fps:.3f}. "
                "Frame-locked sync may be inaccurate."
            )
            logger.warning(project.warnings[-1])

    # Check VFR
    if hdr_source.selected_stream and hdr_source.selected_stream.frame_rate_type.value == "VFR":
        project.warnings.append("HDR source appears to be VFR. Frame-locked sync may drift.")
    if om_source.selected_stream and om_source.selected_stream.frame_rate_type.value == "VFR":
        project.warnings.append(
            "Open Matte source appears to be VFR. Frame-locked sync may drift."
        )

    # ===== STAGE 3: SYNCHRONIZATION (with cache) =====
    logger.info("=" * 60)
    logger.info("STAGE 3: Frame-Offset Synchronization")
    logger.info("=" * 60)

    # Try loading from cache first
    cached_sync = load_sync_cache(output_dir, hdr_source, om_source)

    if cached_sync is not None:
        sync_model = cached_sync
        logger.info(
            f"Using cached sync: offset={sync_model.frame_offset} frames, "
            f"confidence={sync_model.confidence:.4f}"
        )
    else:
        sync_model = find_global_offset(hdr_source, om_source, config=config.sync)
        logger.info(
            f"Initial offset found: {sync_model.frame_offset} frames "
            f"({sync_model.offset_seconds:.3f}s), score={sync_model.confidence:.4f}"
        )

        # ===== STAGE 4: SYNC VALIDATION =====
        logger.info("=" * 60)
        logger.info("STAGE 4: Synchronization Validation (Drift Check)")
        logger.info("=" * 60)

        sync_model = validate_sync(hdr_source, om_source, sync_model, config=config.sync)

        # Save to cache if successful
        if sync_model.status == SyncStatus.LOCKED:
            save_sync_cache(output_dir, hdr_source, om_source, sync_model)

    project.sync_model = sync_model

    if config.debug_sync:
        write_sync_debug(sync_model, output_dir)

    if sync_model.status != SyncStatus.LOCKED:
        logger.error(f"Synchronization status: {sync_model.status.value}")
        # Save partial project and stop
        save_project(project, output_dir / "project.json")
        return project

    logger.info(
        f"SYNC LOCKED: offset={sync_model.frame_offset} frames, "
        f"confidence={sync_model.confidence:.4f}, drift={sync_model.drift_frames:.3f}"
    )

    # ===== STAGE 5: BUILD RANGE SPEC =====
    # Sync is now locked — we can map HDR frames to OM frames
    hdr_stream = hdr_source.selected_stream
    total_frames = 0
    total_duration = 0.0
    if hdr_stream:
        total_frames = hdr_stream.frame_count or int(
            hdr_stream.duration_seconds * hdr_stream.fps
        )
        total_duration = hdr_stream.duration_seconds

    if range_spec is None:
        # Build from time arguments (or full range if none specified)
        range_spec = build_range_spec(
            start=start_seconds,
            end=end_seconds,
            duration=duration_seconds,
            context=context_seconds,
            fps=hdr_fps,
            total_frames=total_frames,
            total_duration=total_duration,
            frame_offset=sync_model.frame_offset,
        )

    project.range_spec = range_spec.to_dict()

    if not range_spec.is_full_range:
        logger.info("=" * 60)
        logger.info("RANGE MODE: Processing limited to selected range")
        logger.info("=" * 60)
        logger.info(
            f"  HDR range: {range_spec.start_seconds:.3f}s — "
            f"{range_spec.end_seconds:.3f}s "
            f"(frames {range_spec.start_frame}–{range_spec.end_frame})"
        )
        logger.info(
            f"  Analysis range (with context): "
            f"{range_spec.analysis_start_seconds:.3f}s — "
            f"{range_spec.analysis_end_seconds:.3f}s "
            f"(frames {range_spec.analysis_start_frame}–{range_spec.analysis_end_frame})"
        )
        logger.info(
            f"  OM range: frames {range_spec.om_start_frame}–{range_spec.om_end_frame}"
        )

    # ===== STAGE 6: SHOT DETECTION (within range) =====
    logger.info("=" * 60)
    logger.info("STAGE 5: Shot Detection (on HDR timeline)")
    logger.info("=" * 60)

    # If range mode, limit shot detection to analysis range (includes context)
    shot_frame_range = None
    if not range_spec.is_full_range:
        shot_frame_range = (range_spec.analysis_start_frame, range_spec.analysis_end_frame)

    shots = detect_shots(
        hdr_source, sync_model, config=config.shots, frame_range=shot_frame_range
    )
    project.shots = shots
    logger.info(f"Detected {len(shots)} shots")

    # ===== STAGE 7: GEOMETRY (within range) =====
    logger.info("=" * 60)
    logger.info("STAGE 6: Geometric Alignment")
    logger.info("=" * 60)

    geometry = estimate_geometry(hdr_source, om_source, sync_model, config=config.geometry)
    project.global_geometry = geometry
    logger.info(
        f"Geometry: offset=({geometry.offset_x:.1f}, {geometry.offset_y:.1f}), "
        f"scale=({geometry.scale_x:.4f}, {geometry.scale_y:.4f}), "
        f"confidence={geometry.confidence:.4f}"
    )

    # ===== STAGE 8: LUMINANCE + COLOR =====
    logger.info("=" * 60)
    logger.info("STAGE 7-8: Transform Estimation (per-shot)")
    logger.info("=" * 60)
    logger.info("Transform estimation requires full frame extraction — deferred to render stage.")

    # Mark project status
    project.analysis_complete = True
    project.ready_for_render = sync_model.status == SyncStatus.LOCKED

    # ===== SAVE PROJECT =====
    project_path = output_dir / "project.json"
    save_project(project, project_path)
    logger.info(f"Project saved to: {project_path}")

    # Print report to console
    print_analysis_report(project)

    return project
