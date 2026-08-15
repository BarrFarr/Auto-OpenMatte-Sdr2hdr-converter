"""Console output — prints analysis report summary to stdout.

Stub implementation — full formatting deferred to later iteration.
"""

from __future__ import annotations

from auto_openmatte.core.models import ProjectData


def print_analysis_report(project: ProjectData) -> None:
    """Print a summary of the analysis results to the console.

    Args:
        project: Completed project data.
    """
    print()
    print("=" * 60)
    print("ANALYSIS REPORT")
    print("=" * 60)

    if project.hdr_source:
        print(f"  HDR: {project.hdr_source.path.name}")
    if project.openmatte_source:
        print(f"  Open Matte: {project.openmatte_source.path.name}")

    sync = project.sync_model
    print(f"  Sync status: {sync.status.value}")
    print(f"  Frame offset: {sync.frame_offset}")
    print(f"  Confidence: {sync.confidence:.4f}")
    print(f"  Drift: {sync.drift_frames:.3f} frames")

    if project.global_geometry:
        geo = project.global_geometry
        print(f"  Geometry: scale=({geo.scale_x:.4f}, {geo.scale_y:.4f})")

    print(f"  Shots: {len(project.shots)}")
    print(f"  Analysis complete: {project.analysis_complete}")
    print(f"  Ready for render: {project.ready_for_render}")

    if project.warnings:
        print()
        print("  Warnings:")
        for w in project.warnings:
            print(f"    - {w}")

    print("=" * 60)
    print()
