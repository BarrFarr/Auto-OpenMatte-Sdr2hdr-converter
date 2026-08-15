"""Pipeline orchestrator — coordinates all analysis stages.

Executes the pipeline in the correct order:
1. Inspect sources
2. Identify streams and HDR
3. Synchronize (frame offset)
4. Validate sync (drift check)
5. Detect shots on HDR
6. Map shots to Open Matte
7. Estimate geometry
8. Estimate luminance + color per shot
9. Validate transforms
10. Save project.json
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
from auto_openmatte.output.console import print_analysis_report

logger = logging.getLogger(__name__)


def run_analysis(
    hdr_path: Path,
    openmatte_path: Path,
    config: PipelineConfig | None = None,
) -> ProjectData:
    """Run the complete analysis pipeline.

    Args:
        hdr_path: Path to HDR source (or either source — will auto-detect roles).
        openmatte_path: Path to Open Matte source.
        config: Pipeline configuration.

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
        project.warnings.append("Open Matte source appears to be VFR. Frame-locked sync may drift.")

    # ===== STAGE 3: SYNCHRONIZATION =====
    logger.info("=" * 60)
    logger.info("STAGE 3: Frame-Offset Synchronization")
    logger.info("=" * 60)

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

    # ===== STAGE 5: SHOT DETECTION =====
    logger.info("=" * 60)
    logger.info("STAGE 5: Shot Detection (on HDR timeline)")
    logger.info("=" * 60)

    shots = detect_shots(hdr_source, sync_model, config=config.shots)
    project.shots = shots
    logger.info(f"Detected {len(shots)} shots")

    # ===== STAGE 6: GEOMETRY =====
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

    # ===== STAGE 7-8: LUMINANCE + COLOR (placeholder for full implementation) =====
    # The full luminance and color estimation requires frame extraction in bulk.
    # For now, mark the project as analysis-complete up to geometry.
    # Full transform estimation will be implemented in the render stage.

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
