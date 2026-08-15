"""Tests for CLI interface."""

from __future__ import annotations

import subprocess
import sys


class TestCLI:
    """Tests for CLI entry points."""

    def test_version(self) -> None:
        """--version should print version."""
        result = subprocess.run(
            [sys.executable, "-m", "auto_openmatte", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_help(self) -> None:
        """--help should print usage."""
        result = subprocess.run(
            [sys.executable, "-m", "auto_openmatte", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "auto_openmatte" in result.stdout
        assert "analyze" in result.stdout
        assert "extend" in result.stdout
        assert "convert-hdr" in result.stdout

    def test_analyze_help(self) -> None:
        """analyze --help should show options."""
        result = subprocess.run(
            [sys.executable, "-m", "auto_openmatte", "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--hdr" in result.stdout
        assert "--openmatte" in result.stdout
        assert "--sync-search" in result.stdout
        assert "--debug-sync" in result.stdout

    def test_extend_help(self) -> None:
        """extend --help should show options."""
        result = subprocess.run(
            [sys.executable, "-m", "auto_openmatte", "extend", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--project" in result.stdout
        assert "--output" in result.stdout

    def test_convert_hdr_help(self) -> None:
        """convert-hdr --help should show options."""
        result = subprocess.run(
            [sys.executable, "-m", "auto_openmatte", "convert-hdr", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--input" in result.stdout
        assert "--reference" in result.stdout
        assert "--output" in result.stdout

    def test_no_command(self) -> None:
        """Running without command should show help and exit with 1."""
        result = subprocess.run(
            [sys.executable, "-m", "auto_openmatte"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1

    def test_analyze_missing_file(self) -> None:
        """analyze with nonexistent file should report error."""
        result = subprocess.run(
            [
                sys.executable, "-m", "auto_openmatte", "analyze",
                "--hdr", "/nonexistent/hdr.mkv",
                "--openmatte", "/nonexistent/om.mkv",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()
