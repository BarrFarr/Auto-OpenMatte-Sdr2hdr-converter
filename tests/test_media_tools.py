"""Tests for bundled FFmpeg/ffprobe discovery and probing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from auto_openmatte.utils.ffmpeg import (
    MediaToolLocator,
    get_media_tool_config,
    get_media_tool_status,
    run_ffprobe,
)


def _tool_filename(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def test_bundled_tools_have_priority_and_use_absolute_application_root(
    tmp_path: Path,
) -> None:
    """The application-local bin directory wins over development and PATH."""
    app_root = (tmp_path / "installed-app").resolve()
    bundled_bin = app_root / "bin"
    bundled_bin.mkdir(parents=True)
    ffmpeg = bundled_bin / _tool_filename("ffmpeg")
    ffprobe = bundled_bin / _tool_filename("ffprobe")
    ffmpeg.write_bytes(b"bundled ffmpeg")
    ffprobe.write_bytes(b"bundled ffprobe")

    config = MediaToolLocator(
        app_root=app_root,
        repository_root=tmp_path / "repository-without-tools",
    ).resolve()

    assert config.application_root == app_root
    assert config.ffmpeg_path == ffmpeg.resolve()
    assert config.ffprobe_path == ffprobe.resolve()
    assert config.ffmpeg_source == "bundled"
    assert config.ffprobe_source == "bundled"
    assert config.complete


def test_development_tree_is_the_second_resolution_tier(tmp_path: Path) -> None:
    """The repository-relative development tree is used when no bundle exists."""
    repository_root = (tmp_path / "repository").resolve()
    development_bin = repository_root / "dev" / "ffmpeg-build" / "install" / "bin"
    development_bin.mkdir(parents=True)
    ffmpeg = development_bin / _tool_filename("ffmpeg")
    ffprobe = development_bin / _tool_filename("ffprobe")
    ffmpeg.write_bytes(b"development ffmpeg")
    ffprobe.write_bytes(b"development ffprobe")

    config = MediaToolLocator(
        app_root=(tmp_path / "installed-app").resolve(),
        repository_root=repository_root,
    ).resolve()

    assert config.ffmpeg_path == ffmpeg.resolve()
    assert config.ffprobe_path == ffprobe.resolve()
    assert config.ffmpeg_source == "development"
    assert config.ffprobe_source == "development"


def test_default_application_root_is_derived_from_executable_not_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Changing CWD cannot change the application-root calculation."""
    monkeypatch.chdir(tmp_path)
    locator = MediaToolLocator()

    assert locator.application_root == Path(sys.executable).resolve().parent
    assert locator.application_root != tmp_path


def test_status_reports_exact_paths_and_versions() -> None:
    """The diagnostic status exposes capabilities, paths, and version banners."""
    status = get_media_tool_status()
    if not status.complete:
        pytest.skip("bundled, development, and PATH media tools are unavailable")

    assert status.ffmpeg_path is not None
    assert status.ffprobe_path is not None
    assert status.ffmpeg_path.is_absolute()
    assert status.ffprobe_path.is_absolute()
    assert status.ffmpeg_version is not None
    assert status.ffprobe_version is not None


def test_ffprobe_can_probe_a_generated_sample(tmp_path: Path) -> None:
    """A small generated sample is probed through the central resolver."""
    config = get_media_tool_config()
    if config.ffmpeg_path is None or config.ffprobe_path is None:
        pytest.skip("bundled, development, and PATH media tools are unavailable")

    raw_frame = tmp_path / "frame.rgb"
    raw_frame.write_bytes(bytes(32 * 18 * 3))
    sample = tmp_path / "sample.mkv"
    encode = subprocess.run(
        [
            str(config.ffmpeg_path),
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            "32x18",
            "-framerate",
            "1",
            "-i",
            str(raw_frame),
            "-frames:v",
            "1",
            "-c:v",
            "ffv1",
            "-y",
            str(sample),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if encode.returncode != 0:
        pytest.fail(f"resolved ffmpeg could not create the probe sample: {encode.stderr}")

    probe = run_ffprobe(sample)
    stream = probe["streams"][0]
    assert stream["codec_type"] == "video"
    assert stream["width"] == 32
    assert stream["height"] == 18
