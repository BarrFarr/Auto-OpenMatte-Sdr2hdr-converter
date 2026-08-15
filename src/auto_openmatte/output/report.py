"""HTML report generation.

Stub implementation — full HTML report deferred to later iteration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from auto_openmatte.core.models import ProjectData

logger = logging.getLogger(__name__)


def generate_html_report(project: ProjectData, output_path: Path) -> Path:
    """Generate an HTML analysis report.

    Stub: writes a minimal JSON summary as HTML placeholder.
    Full implementation deferred.

    Args:
        project: Completed project data.
        output_path: Path to write the HTML file.

    Returns:
        Path to the generated report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sync = project.sync_model
    summary = {
        "sync_status": sync.status.value,
        "frame_offset": sync.frame_offset,
        "confidence": sync.confidence,
        "drift_frames": sync.drift_frames,
        "shots": len(project.shots),
        "analysis_complete": project.analysis_complete,
        "ready_for_render": project.ready_for_render,
        "warnings": project.warnings,
    }

    html = f"""<!DOCTYPE html>
<html>
<head><title>Auto OpenMatte Analysis Report</title></head>
<body>
<h1>Analysis Report</h1>
<pre>{json.dumps(summary, indent=2)}</pre>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {output_path}")
    return output_path
