"""Tests for CLI argument parsing."""

import pytest

from auto_openmatte.cli import create_parser, main


class TestCLIParser:
    """Test CLI argument parser configuration."""

    def test_parser_creation(self):
        """Parser is created without errors."""
        parser = create_parser()
        assert parser is not None

    def test_version_flag(self, capsys):
        """--version prints version and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out

    def test_no_command_shows_help(self, capsys):
        """Running with no subcommand shows help."""
        result = main([])
        assert result == 0

    def test_analyze_requires_hdr(self):
        """analyze subcommand requires --hdr argument."""
        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", "--openmatte", "om.mkv"])
        assert exc_info.value.code == 2

    def test_analyze_requires_openmatte(self):
        """analyze subcommand requires --openmatte argument."""
        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", "--hdr", "hdr.mkv"])
        assert exc_info.value.code == 2

    def test_analyze_all_args(self):
        """analyze subcommand parses all arguments correctly."""
        result = main([
            "analyze",
            "--hdr", "hdr.mkv",
            "--openmatte", "om.mkv",
            "--sync-search", "200",
            "--samples-per-shot", "50",
            "--alignment-confidence", "0.98",
            "--color-confidence", "0.85",
            "--debug",
            "--debug-sync",
        ])
        assert result == 0

    def test_analyze_defaults(self):
        """analyze subcommand uses correct defaults."""
        parser = create_parser()
        args = parser.parse_args([
            "analyze",
            "--hdr", "hdr.mkv",
            "--openmatte", "om.mkv",
        ])
        assert args.sync_search == 120
        assert args.samples_per_shot == 30
        assert args.alignment_confidence == 0.95
        assert args.color_confidence == 0.90
        assert args.debug is False
        assert args.debug_sync is False

    def test_preview_requires_project(self):
        """preview subcommand requires --project argument."""
        with pytest.raises(SystemExit) as exc_info:
            main(["preview"])
        assert exc_info.value.code == 2

    def test_preview_parses_project(self):
        """preview subcommand parses --project correctly."""
        result = main(["preview", "--project", "analysis.json"])
        assert result == 0

    def test_render_requires_project(self):
        """render subcommand requires --project argument."""
        with pytest.raises(SystemExit) as exc_info:
            main(["render", "--output", "out.mkv"])
        assert exc_info.value.code == 2

    def test_render_requires_output(self):
        """render subcommand requires --output argument."""
        with pytest.raises(SystemExit) as exc_info:
            main(["render", "--project", "analysis.json"])
        assert exc_info.value.code == 2

    def test_render_all_args(self):
        """render subcommand parses all arguments correctly."""
        result = main([
            "render",
            "--project", "analysis.json",
            "--output", "output.mkv",
        ])
        assert result == 0
