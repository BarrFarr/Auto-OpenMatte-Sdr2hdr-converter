"""Command-line interface for Auto OpenMatte.

Commands:
    analyze       — Run full analysis pipeline
    extend        — Mode A: HDR center + OM extension → 16:9 HDR
    convert-hdr   — Mode B: Standalone SDR → HDR conversion
    preview       — Generate preview from project.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from auto_openmatte import __version__
from auto_openmatte.core.config import (
    ColorConfig,
    GeometryConfig,
    PipelineConfig,
    RenderConfig,
    ShotConfig,
    SyncConfig,
)
from auto_openmatte.core.exceptions import AutoOpenMatteError
from auto_openmatte.core.project import load_project


def _setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Run analysis pipeline."""
    from auto_openmatte.output.report import generate_html_report
    from auto_openmatte.pipeline.orchestrator import run_analysis

    config = PipelineConfig(
        sync=SyncConfig(
            search_range_seconds=args.sync_search,
            min_confidence=args.alignment_confidence,
        ),
        shots=ShotConfig(),
        geometry=GeometryConfig(min_confidence=args.alignment_confidence),
        color=ColorConfig(
            samples_per_shot=args.samples_per_shot,
            min_confidence=args.color_confidence,
        ),
        debug=args.debug,
        debug_sync=args.debug_sync,
        output_dir=args.output_dir,
    )

    hdr_path = Path(args.hdr)
    om_path = Path(args.openmatte)

    if not hdr_path.exists():
        print(f"ERROR: HDR file not found: {hdr_path}", file=sys.stderr)
        return 1
    if not om_path.exists():
        print(f"ERROR: Open Matte file not found: {om_path}", file=sys.stderr)
        return 1

    project = run_analysis(hdr_path, om_path, config=config)

    # Generate HTML report
    report_path = Path(config.output_dir) / "analysis" / "reports" / "report.html"
    generate_html_report(project, report_path)

    return 0 if project.analysis_complete else 1


def _cmd_extend(args: argparse.Namespace) -> int:
    """Mode A: Extend HDR with Open Matte."""
    from auto_openmatte.pipeline.render import render_extend

    if args.project:
        project = load_project(Path(args.project))
    elif args.hdr and args.openmatte:
        # Run analysis first, then render
        from auto_openmatte.pipeline.orchestrator import run_analysis

        config = PipelineConfig(output_dir=args.output_dir or "./project")
        project = run_analysis(Path(args.hdr), Path(args.openmatte), config=config)
    else:
        print("ERROR: Provide either --project or both --hdr and --openmatte", file=sys.stderr)
        return 1

    if not project.ready_for_render:
        print("ERROR: Project is not ready for render. Check analysis report.", file=sys.stderr)
        return 1

    output = Path(args.output)
    config = RenderConfig()
    success = render_extend(project, output, config=config)
    return 0 if success else 1


def _cmd_convert_hdr(args: argparse.Namespace) -> int:
    """Mode B: Convert SDR Open Matte to standalone HDR."""
    from auto_openmatte.pipeline.render import render_convert_hdr

    if args.project:
        project = load_project(Path(args.project))
    elif args.input and args.reference:
        # Run analysis first
        from auto_openmatte.pipeline.orchestrator import run_analysis

        config = PipelineConfig(output_dir=args.output_dir or "./project")
        project = run_analysis(Path(args.reference), Path(args.input), config=config)
    else:
        print(
            "ERROR: Provide either --project or both --input and --reference",
            file=sys.stderr,
        )
        return 1

    if not project.ready_for_render:
        print("ERROR: Project is not ready for render. Check analysis report.", file=sys.stderr)
        return 1

    output = Path(args.output)
    config = RenderConfig()
    success = render_convert_hdr(project, output, config=config)
    return 0 if success else 1


def _cmd_preview(args: argparse.Namespace) -> int:
    """Generate preview from project."""
    from auto_openmatte.pipeline.preview import generate_preview

    project_path = Path(args.project)
    if not project_path.exists():
        print(f"ERROR: Project file not found: {project_path}", file=sys.stderr)
        return 1

    project = load_project(project_path)
    output_dir = project_path.parent
    result = generate_preview(project, output_dir)
    return 0 if result else 1


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="auto_openmatte",
        description="Auto OpenMatte — HDR Open Matte extension and SDR-to-HDR conversion",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === ANALYZE ===
    p_analyze = subparsers.add_parser("analyze", help="Run full analysis pipeline")
    p_analyze.add_argument("--hdr", required=True, help="Path to HDR source video")
    p_analyze.add_argument("--openmatte", required=True, help="Path to Open Matte source video")
    p_analyze.add_argument("--output-dir", default="./project", help="Output directory")
    p_analyze.add_argument(
        "--sync-search", type=float, default=120.0, help="Sync search range (seconds)"
    )
    p_analyze.add_argument(
        "--samples-per-shot", type=int, default=30, help="Sample frames per shot"
    )
    p_analyze.add_argument(
        "--alignment-confidence", type=float, default=0.95, help="Min alignment confidence"
    )
    p_analyze.add_argument(
        "--color-confidence", type=float, default=0.90, help="Min color confidence"
    )
    p_analyze.add_argument("--debug", action="store_true", help="Enable debug output")
    p_analyze.add_argument("--debug-sync", action="store_true", help="Write sync debug artifacts")

    # === EXTEND (Mode A) ===
    p_extend = subparsers.add_parser("extend", help="Mode A: Extend HDR with Open Matte")
    p_extend.add_argument("--project", help="Path to project.json (skip analysis)")
    p_extend.add_argument("--hdr", help="Path to HDR source (triggers analysis)")
    p_extend.add_argument("--openmatte", help="Path to Open Matte source (triggers analysis)")
    p_extend.add_argument("--output", required=True, help="Output video path")
    p_extend.add_argument("--output-dir", default="./project", help="Output directory for analysis")

    # === CONVERT-HDR (Mode B) ===
    p_convert = subparsers.add_parser("convert-hdr", help="Mode B: Convert Open Matte SDR to HDR")
    p_convert.add_argument("--project", help="Path to project.json (skip analysis)")
    p_convert.add_argument("--input", help="Path to Open Matte SDR source")
    p_convert.add_argument("--reference", help="Path to HDR reference")
    p_convert.add_argument("--output", required=True, help="Output video path")
    p_convert.add_argument(
        "--output-dir", default="./project", help="Output directory for analysis"
    )

    # === PREVIEW ===
    p_preview = subparsers.add_parser("preview", help="Generate preview from project")
    p_preview.add_argument("--project", required=True, help="Path to project.json")

    # Parse
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    debug = getattr(args, "debug", False) or getattr(args, "debug_sync", False)
    _setup_logging(debug=debug)

    # Dispatch
    try:
        if args.command == "analyze":
            code = _cmd_analyze(args)
        elif args.command == "extend":
            code = _cmd_extend(args)
        elif args.command == "convert-hdr":
            code = _cmd_convert_hdr(args)
        elif args.command == "preview":
            code = _cmd_preview(args)
        else:
            parser.print_help()
            code = 1
    except AutoOpenMatteError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130

    sys.exit(code)
