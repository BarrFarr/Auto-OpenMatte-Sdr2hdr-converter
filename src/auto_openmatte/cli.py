"""Command-line interface for Auto OpenMatte.

Commands:
    analyze       — Run full analysis pipeline (or range-limited)
    extend        — Mode A: HDR center + OM extension → 16:9 HDR
    convert-hdr   — Mode B: Standalone SDR → HDR conversion
    preview       — Generate preview from project.json
    test          — Quick test: process a short sample from the real pipeline
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
from auto_openmatte.core.range_spec import RangeError, parse_time


def _setup_logging(debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_time_arg(value: str | None) -> float | None:
    """Parse a time CLI argument, returning None if not provided."""
    if value is None:
        return None
    return parse_time(value)


def _add_range_args(parser: argparse.ArgumentParser) -> None:
    """Add time range arguments to a subparser."""
    group = parser.add_argument_group("time range (HDR timeline)")
    group.add_argument(
        "--start", type=str, default=None,
        help="Start time on HDR timeline (HH:MM:SS.mmm or seconds)"
    )
    group.add_argument(
        "--end", type=str, default=None,
        help="End time on HDR timeline (HH:MM:SS.mmm or seconds)"
    )
    group.add_argument(
        "--duration", type=float, default=None,
        help="Duration in seconds (alternative to --end)"
    )
    group.add_argument(
        "--start-frame", type=int, default=None,
        help="Start frame number (alternative to --start)"
    )
    group.add_argument(
        "--end-frame", type=int, default=None,
        help="End frame number (alternative to --end)"
    )
    group.add_argument(
        "--context", type=float, default=2.0,
        help="Context window in seconds (default: 2.0)"
    )


def _get_range_kwargs(args: argparse.Namespace) -> dict:
    """Extract range keyword arguments from parsed args."""
    kwargs: dict = {}

    start_str = getattr(args, "start", None)
    end_str = getattr(args, "end", None)
    duration = getattr(args, "duration", None)
    context = getattr(args, "context", 2.0)

    if start_str is not None:
        kwargs["start_seconds"] = parse_time(start_str)
    if end_str is not None:
        kwargs["end_seconds"] = parse_time(end_str)
    if duration is not None:
        kwargs["duration_seconds"] = duration
    if context is not None:
        kwargs["context_seconds"] = context

    # Validate: cannot have both time and frame if inconsistent
    # (the build_range_spec function handles the actual validation)
    # We just pass them through here

    return kwargs


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

    # Get range arguments
    range_kwargs = _get_range_kwargs(args)

    project = run_analysis(hdr_path, om_path, config=config, **range_kwargs)

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
        from auto_openmatte.pipeline.orchestrator import run_analysis

        config = PipelineConfig(output_dir=args.output_dir or "./project")
        range_kwargs = _get_range_kwargs(args)
        project = run_analysis(
            Path(args.hdr), Path(args.openmatte), config=config, **range_kwargs
        )
    else:
        print(
            "ERROR: Provide either --project or both --hdr and --openmatte",
            file=sys.stderr,
        )
        return 1

    if not project.ready_for_render:
        print(
            "ERROR: Project is not ready for render. Check analysis report.",
            file=sys.stderr,
        )
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
        from auto_openmatte.pipeline.orchestrator import run_analysis

        config = PipelineConfig(output_dir=args.output_dir or "./project")
        range_kwargs = _get_range_kwargs(args)
        project = run_analysis(
            Path(args.reference), Path(args.input), config=config, **range_kwargs
        )
    else:
        print(
            "ERROR: Provide either --project or both --input and --reference",
            file=sys.stderr,
        )
        return 1

    if not project.ready_for_render:
        print(
            "ERROR: Project is not ready for render. Check analysis report.",
            file=sys.stderr,
        )
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


def _cmd_test(args: argparse.Namespace) -> int:
    """Quick test: run the real pipeline on a short sample.

    Produces a real processed sample clip using the production pipeline.
    Intended for rapid development iteration and quality testing.
    """
    from auto_openmatte.output.report import generate_html_report
    from auto_openmatte.pipeline.orchestrator import run_analysis

    hdr_path = Path(args.hdr)
    om_path = Path(args.openmatte)

    if not hdr_path.exists():
        print(f"ERROR: HDR file not found: {hdr_path}", file=sys.stderr)
        return 1
    if not om_path.exists():
        print(f"ERROR: Open Matte file not found: {om_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir

    # Handle --random-samples mode
    random_samples = getattr(args, "random_samples", None)
    if random_samples:
        return _cmd_test_random(args, hdr_path, om_path)

    # Single sample mode — require --start
    start_str = args.start
    if start_str is None:
        print(
            "ERROR: --start is required for test command "
            "(or use --random-samples N)",
            file=sys.stderr,
        )
        return 1

    config = PipelineConfig(
        sync=SyncConfig(),
        shots=ShotConfig(),
        geometry=GeometryConfig(),
        color=ColorConfig(samples_per_shot=args.samples_per_shot),
        output_dir=output_dir,
        debug=getattr(args, "debug", False),
        debug_sync=getattr(args, "debug_sync", False),
    )

    # Parse range
    start_seconds = parse_time(start_str)
    end_str = getattr(args, "end", None)
    end_seconds = parse_time(end_str) if end_str else None
    duration = args.duration
    context = args.context

    # Run analysis with range
    project = run_analysis(
        hdr_path, om_path, config=config,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        duration_seconds=duration,
        context_seconds=context,
    )

    # Generate report
    report_path = Path(output_dir) / "report.html"
    generate_html_report(project, report_path)

    if not project.analysis_complete:
        print("ERROR: Analysis failed. Check logs.", file=sys.stderr)
        return 1

    # Print summary
    print()
    print("=" * 60)
    print("TEST SAMPLE — COMPLETE")
    print("=" * 60)
    print(f"  Output dir: {output_dir}")
    print(f"  Report: {report_path}")
    print(f"  Project: {Path(output_dir) / 'project.json'}")

    range_data = project.range_spec
    if range_data and not range_data.get("is_full_range", True):
        from auto_openmatte.core.range_spec import format_time

        print(f"  HDR range: {format_time(range_data['start_seconds'])} — "
              f"{format_time(range_data['end_seconds'])}")
        print(f"  Duration: {range_data['end_seconds'] - range_data['start_seconds']:.3f}s")
        print(f"  HDR frames: {range_data['start_frame']}–{range_data['end_frame']}")
        print(f"  OM frames: {range_data['om_start_frame']}–{range_data['om_end_frame']}")

    print("=" * 60)
    return 0


def _cmd_test_random(
    args: argparse.Namespace, hdr_path: Path, om_path: Path
) -> int:
    """Handle --random-samples mode: select N random ranges and test each."""
    import random

    from auto_openmatte.analysis.inspect import inspect_source
    from auto_openmatte.analysis.stream_select import select_video_stream
    from auto_openmatte.core.range_spec import format_time
    from auto_openmatte.pipeline.orchestrator import run_analysis

    n_samples = args.random_samples
    duration = args.duration or 30.0
    context = args.context
    output_base = Path(args.output_dir)

    # Quick inspect to get total duration
    source = inspect_source(hdr_path)
    select_video_stream(source)
    if not source.selected_stream:
        print("ERROR: Cannot read HDR source", file=sys.stderr)
        return 1

    total_duration = source.selected_stream.duration_seconds
    if total_duration <= 0:
        print("ERROR: Cannot determine HDR duration", file=sys.stderr)
        return 1

    # Select random ranges (avoid first/last 60s)
    margin = 60.0
    available_start = margin
    available_end = total_duration - margin - duration

    if available_end <= available_start:
        print(
            f"ERROR: Source too short for {n_samples} random samples "
            f"of {duration}s with {margin}s margin",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(42)  # Deterministic for reproducibility
    starts = sorted(
        rng.uniform(available_start, available_end) for _ in range(n_samples)
    )

    print(f"Testing {n_samples} random samples of {duration:.0f}s each:")
    print()

    results: list[dict] = []
    for i, sample_start in enumerate(starts, 1):
        sample_dir = str(output_base / f"sample_{i:03d}")
        print(f"  [{i}/{n_samples}] {format_time(sample_start)} — "
              f"{format_time(sample_start + duration)}")

        config = PipelineConfig(
            color=ColorConfig(samples_per_shot=args.samples_per_shot),
            output_dir=sample_dir,
        )

        try:
            project = run_analysis(
                hdr_path, om_path, config=config,
                start_seconds=sample_start,
                duration_seconds=duration,
                context_seconds=context,
            )
            status = "PASS" if project.analysis_complete else "FAIL"
        except AutoOpenMatteError as e:
            status = "ERROR"
            print(f"    ERROR: {e}", file=sys.stderr)

        results.append({
            "sample": i,
            "start": sample_start,
            "end": sample_start + duration,
            "status": status,
        })

    # Print summary
    print()
    print("=" * 60)
    print("RANDOM SAMPLE SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] != "PASS")
    print(f"  Passed: {passed}/{n_samples}")
    print(f"  Failed: {failed}/{n_samples}")
    for r in results:
        marker = "✓" if r["status"] == "PASS" else "✗"
        from auto_openmatte.core.range_spec import format_time as _ft

        print(f"  {marker} Sample {r['sample']}: "
              f"{_ft(r['start'])} — {_ft(r['end'])} [{r['status']}]")
    print("=" * 60)

    return 0 if failed == 0 else 1


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="auto_openmatte",
        description="Auto OpenMatte — HDR Open Matte extension and SDR-to-HDR conversion",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === ANALYZE ===
    p_analyze = subparsers.add_parser(
        "analyze", help="Run full analysis pipeline"
    )
    p_analyze.add_argument(
        "--hdr", required=True, help="Path to HDR source video"
    )
    p_analyze.add_argument(
        "--openmatte", required=True, help="Path to Open Matte source video"
    )
    p_analyze.add_argument(
        "--output-dir", default="./project", help="Output directory"
    )
    p_analyze.add_argument(
        "--sync-search", type=float, default=120.0,
        help="Sync search range (seconds)"
    )
    p_analyze.add_argument(
        "--samples-per-shot", type=int, default=30,
        help="Sample frames per shot"
    )
    p_analyze.add_argument(
        "--alignment-confidence", type=float, default=0.95,
        help="Min alignment confidence"
    )
    p_analyze.add_argument(
        "--color-confidence", type=float, default=0.90,
        help="Min color confidence"
    )
    p_analyze.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )
    p_analyze.add_argument(
        "--debug-sync", action="store_true",
        help="Write sync debug artifacts"
    )
    _add_range_args(p_analyze)

    # === EXTEND (Mode A) ===
    p_extend = subparsers.add_parser(
        "extend", help="Mode A: Extend HDR with Open Matte"
    )
    p_extend.add_argument(
        "--project", help="Path to project.json (skip analysis)"
    )
    p_extend.add_argument(
        "--hdr", help="Path to HDR source (triggers analysis)"
    )
    p_extend.add_argument(
        "--openmatte", help="Path to Open Matte source (triggers analysis)"
    )
    p_extend.add_argument(
        "--output", required=True, help="Output video path"
    )
    p_extend.add_argument(
        "--output-dir", default="./project",
        help="Output directory for analysis"
    )
    _add_range_args(p_extend)

    # === CONVERT-HDR (Mode B) ===
    p_convert = subparsers.add_parser(
        "convert-hdr", help="Mode B: Convert Open Matte SDR to HDR"
    )
    p_convert.add_argument(
        "--project", help="Path to project.json (skip analysis)"
    )
    p_convert.add_argument(
        "--input", help="Path to Open Matte SDR source"
    )
    p_convert.add_argument(
        "--reference", help="Path to HDR reference"
    )
    p_convert.add_argument(
        "--output", required=True, help="Output video path"
    )
    p_convert.add_argument(
        "--output-dir", default="./project",
        help="Output directory for analysis"
    )
    _add_range_args(p_convert)

    # === PREVIEW ===
    p_preview = subparsers.add_parser(
        "preview", help="Generate preview from project"
    )
    p_preview.add_argument(
        "--project", required=True, help="Path to project.json"
    )

    # === TEST (Quick sample) ===
    p_test = subparsers.add_parser(
        "test",
        help="Quick test: process a short sample using the real pipeline",
    )
    p_test.add_argument(
        "--hdr", required=True, help="Path to HDR source video"
    )
    p_test.add_argument(
        "--openmatte", required=True, help="Path to Open Matte source video"
    )
    p_test.add_argument(
        "--output-dir", default="./test_output",
        help="Output directory for test artifacts"
    )
    p_test.add_argument(
        "--samples-per-shot", type=int, default=20,
        help="Sample frames per shot (default: 20 for speed)"
    )
    p_test.add_argument(
        "--random-samples", type=int, default=None,
        help="Select N random ranges instead of a specific range"
    )
    p_test.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )
    p_test.add_argument(
        "--debug-sync", action="store_true",
        help="Write sync debug artifacts"
    )
    _add_range_args(p_test)

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
        elif args.command == "test":
            code = _cmd_test(args)
        else:
            parser.print_help()
            code = 1
    except RangeError as e:
        print(f"\nRANGE ERROR: {e}", file=sys.stderr)
        code = 1
    except AutoOpenMatteError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        code = 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130

    sys.exit(code)
