"""Preview generation — contact sheets and preview video.

Generates visual previews of the analysis results without full rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path

from auto_openmatte.core.models import ProjectData

logger = logging.getLogger(__name__)


def generate_preview(
    project: ProjectData,
    output_dir: Path,
) -> Path | None:
    """Generate preview artifacts from project data.

    Creates:
    - Contact sheet showing before/after for representative shots
    - Preview summary report

    Args:
        project: Complete project data.
        output_dir: Directory for preview output.

    Returns:
        Path to preview output directory, or None on failure.
    """
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preview output directory: {preview_dir}")

    # TODO: Implement full preview generation
    # This requires frame extraction and composition for sample frames
    # Will be implemented after core analysis pipeline is validated

    logger.info("Preview generation: placeholder — full implementation pending")
    return preview_dir


def generate_contact_sheet(
    project: ProjectData,
    output_path: Path,
    shots_to_show: int = 12,
) -> bool:
    """Generate a contact sheet image with representative shots.

    Shows before/after pairs for a selection of shots.

    Args:
        project: Project data with shots and transforms.
        output_path: Path to write the contact sheet image.
        shots_to_show: Number of shots to include.

    Returns:
        True if successful.
    """
    # TODO: Implement contact sheet generation
    # Requires: numpy image composition, optional: PIL/Pillow for output
    logger.info("Contact sheet generation: placeholder — full implementation pending")
    return False
