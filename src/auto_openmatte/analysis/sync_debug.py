"""Synchronization debug output — writes debug artifacts for sync analysis.

Stub implementation — full debug visualization deferred to later iteration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from auto_openmatte.core.models import SyncModel

logger = logging.getLogger(__name__)


def write_sync_debug(sync_model: SyncModel, output_dir: Path) -> None:
    """Write synchronization debug artifacts to output directory.

    Args:
        sync_model: The validated sync model.
        output_dir: Directory to write debug files into.
    """
    debug_dir = output_dir / "analysis" / "sync_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Write sync summary JSON
    summary = {
        "frame_offset": sync_model.frame_offset,
        "confidence": sync_model.confidence,
        "status": sync_model.status.value,
        "drift_frames": sync_model.drift_frames,
        "mean_error_frames": sync_model.mean_error_frames,
        "max_error_frames": sync_model.max_error_frames,
        "offset_seconds": sync_model.offset_seconds,
        "frame_locked": sync_model.frame_locked,
        "method": sync_model.method,
        "checkpoints": sync_model.checkpoints,
    }

    summary_path = debug_dir / "synchronization.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Sync debug written to {debug_dir}")
