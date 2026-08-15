"""Command-line interface for Auto Open-Matte HDR Extender."""

import argparse
import sys

from auto_openmatte import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="auto_openmatte",
        description="Auto Open-Matte HDR Extender - Combine HDR 21:9 with SDR 16:9 Open Matte",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze HDR and Open Matte sources for synchronization and alignment",
    )
    analyze_parser.add_argument(
        "--hdr",
        required=True,
        help="Path to HDR 21:9 reference video",
    )
    analyze_parser.add_argument(
        "--openmatte",
        required=True,
        help="Path to SDR 16:9 Open Matte video",
    )
    analyze_parser.add_argument(
        "--sync-search",
        type=int,
        default=120,
        help="Sync search range in frames (default: 120)",
    )
    analyze_parser.add_argument(
        "--samples-per-shot",
        type=int,
        default=30,
        help="Number of samples per shot for analysis (default: 30)",
    )
    analyze_parser.add_argument(
        "--alignment-confidence",
        type=float,
        default=0.95,
        help="Minimum confidence threshold for alignment (default: 0.95)",
    )
    analyze_parser.add_argument(
        "--color-confidence",
        type=float,
        default=0.90,
        help="Minimum confidence threshold for color matching (default: 0.90)",
    )
    analyze_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    analyze_parser.add_argument(
        "--debug-sync",
        action="store_true",
        help="Enable debug output for sync process",
    )

    # preview subcommand
    preview_parser = subparsers.add_parser(
        "preview",
        help="Generate a preview of the composited output",
    )
    preview_parser.add_argument(
        "--project",
        required=True,
        help="Path to project JSON file from analyze step",
    )

    # render subcommand
    render_parser = subparsers.add_parser(
        "render",
        help="Render the final composited HDR output",
    )
    render_parser.add_argument(
        "--project",
        required=True,
        help="Path to project JSON file from analyze step",
    )
    render_parser.add_argument(
        "--output",
        required=True,
        help="Path for the output video file",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "analyze":
        print(f"Analyzing HDR: {args.hdr}")
        print(f"Open Matte: {args.openmatte}")
        print("Analysis pipeline not yet implemented.")
    elif args.command == "preview":
        print(f"Previewing project: {args.project}")
        print("Preview pipeline not yet implemented.")
    elif args.command == "render":
        print(f"Rendering project: {args.project}")
        print(f"Output: {args.output}")
        print("Render pipeline not yet implemented.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
