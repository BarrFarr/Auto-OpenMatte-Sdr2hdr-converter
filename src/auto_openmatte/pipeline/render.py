"""Final video rendering via FFmpeg.

Frame-locked render loop:
    for hdr_frame_index in range(total_frames):
        om_frame_index = hdr_frame_index + frame_offset
        hdr_frame = read_hdr(hdr_frame_index)
        om_frame = read_openmatte(om_frame_index)
        transform = transform_for_shot(hdr_frame_index)
        output = composite(hdr_frame, om_frame, transform, mask)
        write_output(output)

No temporal synchronization inside the render loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

from auto_openmatte.core.config import RenderConfig
from auto_openmatte.core.exceptions import RenderError
from auto_openmatte.core.models import ProjectData, Shot, ShotTransform

logger = logging.getLogger(__name__)


def _find_shot_for_frame(shots: list[Shot], frame: int) -> Shot | None:
    """Find which shot a given frame belongs to."""
    for shot in shots:
        if shot.hdr_start_frame <= frame < shot.hdr_end_frame:
            return shot
    # After the last shot boundary
    if shots:
        return shots[-1]
    return None


def _find_transform_for_shot(
    transforms: list[ShotTransform], shot_id: int
) -> ShotTransform | None:
    """Find transform for a given shot ID."""
    for t in transforms:
        if t.shot_id == shot_id:
            return t
    return None


def render_extend(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
) -> bool:
    """Render Mode A: Extended HDR (16:9 HDR from 21:9 HDR + OM).

    Frame-locked processing loop. No temporal re-synchronization.

    Args:
        project: Complete project data (must be ready_for_render).
        output_path: Path for output video.
        config: Render configuration.

    Returns:
        True if successful.

    Raises:
        RenderError: If rendering fails.
    """
    if config is None:
        config = RenderConfig()

    if not project.ready_for_render:
        raise RenderError("Project is not ready for render. Run analysis first.")

    if not project.hdr_source or not project.openmatte_source:
        raise RenderError("Project missing source information.")

    if not project.sync_model.frame_locked:
        raise RenderError("Synchronization is not frame-locked. Cannot render.")

    logger.info("Render Mode A: Extended HDR")
    logger.info(f"Output: {output_path}")

    # TODO: Implement frame-by-frame render loop
    # This requires:
    # 1. FFmpeg decode pipeline for both sources (parallel)
    # 2. Frame-locked reading using frame_offset
    # 3. Per-frame composition using shot transforms
    # 4. FFmpeg encode pipeline for output
    #
    # The architecture for this is:
    # - Two FFmpeg decode subprocesses (HDR and OM)
    # - One FFmpeg encode subprocess (output)
    # - Python orchestrates reading/processing/writing
    #
    # Full implementation pending after analysis pipeline is validated.

    logger.info("Render: placeholder — full implementation pending")
    return False


def render_convert_hdr(
    project: ProjectData,
    output_path: Path,
    config: RenderConfig | None = None,
) -> bool:
    """Render Mode B: Standalone SDR → HDR conversion of Open Matte.

    Transforms the entire OM frame to HDR using transforms derived from
    the overlap region analysis.

    Args:
        project: Complete project data.
        output_path: Path for output video.
        config: Render configuration.

    Returns:
        True if successful.

    Raises:
        RenderError: If rendering fails.
    """
    if config is None:
        config = RenderConfig()

    if not project.ready_for_render:
        raise RenderError("Project is not ready for render. Run analysis first.")

    logger.info("Render Mode B: Convert SDR → HDR")
    logger.info(f"Output: {output_path}")

    # TODO: Implement Mode B render
    # Similar to Mode A but simpler — no HDR placement, just full-frame transform.

    logger.info("Render: placeholder — full implementation pending")
    return False
