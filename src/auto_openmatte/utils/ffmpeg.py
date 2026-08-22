"""FFmpeg and ffprobe subprocess helpers.

All operations explicitly select video streams and ignore audio.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from auto_openmatte.core.exceptions import InspectionError

ToolName = Literal["ffmpeg", "ffprobe"]


@dataclass(frozen=True)
class MediaToolConfig:
    """Resolved media-tool paths and capability information.

    Paths are resolved in this order: ``APP_ROOT/bin``, the repository's
    development FFmpeg tree, and finally PATH.  A missing path is kept as
    ``None`` so callers can report capabilities without silently claiming that
    a tool is available.
    """

    application_root: Path
    development_root: Path | None
    ffmpeg_path: Path | None
    ffprobe_path: Path | None
    ffmpeg_source: str | None = None
    ffprobe_source: str | None = None
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None

    @property
    def ffmpeg_found(self) -> bool:
        """Whether an FFmpeg executable was resolved."""
        return self.ffmpeg_path is not None

    @property
    def ffprobe_found(self) -> bool:
        """Whether an ffprobe executable was resolved."""
        return self.ffprobe_path is not None

    @property
    def complete(self) -> bool:
        """Whether both required media tools are available."""
        return self.ffmpeg_found and self.ffprobe_found

    @property
    def ffmpeg_command(self) -> str:
        """Executable argument for FFmpeg subprocess commands."""
        return str(self.ffmpeg_path) if self.ffmpeg_path else "ffmpeg"

    @property
    def ffprobe_command(self) -> str:
        """Executable argument for ffprobe subprocess commands."""
        return str(self.ffprobe_path) if self.ffprobe_path else "ffprobe"

    def require(self, tool: ToolName) -> Path:
        """Return a resolved tool path or raise a useful ``FileNotFoundError``."""
        path = self.ffmpeg_path if tool == "ffmpeg" else self.ffprobe_path
        if path is None:
            raise FileNotFoundError(
                f"{tool} was not found in the bundled application bin directory, "
                "the development FFmpeg tree, or PATH"
            )
        return path

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly capability/status summary."""
        return {
            "application_root": str(self.application_root),
            "development_root": str(self.development_root) if self.development_root else None,
            "ffmpeg_found": self.ffmpeg_found,
            "ffmpeg_path": str(self.ffmpeg_path) if self.ffmpeg_path else None,
            "ffmpeg_source": self.ffmpeg_source,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_found": self.ffprobe_found,
            "ffprobe_path": str(self.ffprobe_path) if self.ffprobe_path else None,
            "ffprobe_source": self.ffprobe_source,
            "ffprobe_version": self.ffprobe_version,
        }


class MediaToolLocator:
    """Locate bundled, development, and PATH media tools.

    ``app_root`` is normally derived from the running executable.  Supplying
    it explicitly is useful for packaging validation and tests, not for normal
    application workflow.
    """

    _DEVELOPMENT_RELATIVE_BIN = Path("dev") / "ffmpeg-build" / "install" / "bin"

    def __init__(
        self,
        app_root: Path | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.application_root = (
            Path(app_root).expanduser().resolve()
            if app_root is not None
            else Path(sys.executable).resolve().parent
        )
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root is not None
            else self._discover_repository_root()
        )

    @staticmethod
    def _discover_repository_root() -> Path | None:
        """Find a source-tree root by its relative development-tool layout."""
        for parent in Path(__file__).resolve().parents:
            if (parent / MediaToolLocator._DEVELOPMENT_RELATIVE_BIN).is_dir():
                return parent
        return None

    @staticmethod
    def _tool_names(tool: ToolName) -> tuple[str, ...]:
        if os.name == "nt":
            return (f"{tool}.exe", tool)
        return (tool,)

    def _bundled_path(self, tool: ToolName) -> Path | None:
        directory = self.application_root / "bin"
        for name in self._tool_names(tool):
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _development_path(self, tool: ToolName) -> Path | None:
        if self.repository_root is None:
            return None
        directory = self.repository_root / self._DEVELOPMENT_RELATIVE_BIN
        for name in self._tool_names(tool):
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _path_path(self, tool: ToolName) -> Path | None:
        found = shutil.which(tool)
        return Path(found).resolve() if found else None

    def _resolve_one(self, tool: ToolName) -> tuple[Path | None, str | None]:
        for source, resolver in (
            ("bundled", self._bundled_path),
            ("development", self._development_path),
            ("PATH", self._path_path),
        ):
            path = resolver(tool)
            if path is not None:
                return path, source
        return None, None

    @staticmethod
    def _read_version(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            result = subprocess.run(
                [str(path), "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = (result.stdout or result.stderr or "").strip()
        return output.splitlines()[0] if output else None

    def resolve(self, *, include_versions: bool = False) -> MediaToolConfig:
        """Resolve both tools and optionally collect their version banners."""
        ffmpeg_path, ffmpeg_source = self._resolve_one("ffmpeg")
        ffprobe_path, ffprobe_source = self._resolve_one("ffprobe")
        return MediaToolConfig(
            application_root=self.application_root,
            development_root=self.repository_root,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            ffmpeg_source=ffmpeg_source,
            ffprobe_source=ffprobe_source,
            ffmpeg_version=self._read_version(ffmpeg_path) if include_versions else None,
            ffprobe_version=self._read_version(ffprobe_path) if include_versions else None,
        )


@lru_cache(maxsize=1)
def get_media_tool_config() -> MediaToolConfig:
    """Return the cached no-side-effect configuration used by app subprocesses."""
    return MediaToolLocator().resolve()


def get_media_tool_status() -> MediaToolConfig:
    """Resolve tools and collect capability/version status for diagnostics."""
    return MediaToolLocator().resolve(include_versions=True)


def run_ffprobe(file_path: Path) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output.

    Args:
        file_path: Path to the media file.

    Returns:
        Parsed JSON dict with format and stream information.

    Raises:
        InspectionError: If ffprobe fails or returns invalid JSON.
    """
    cmd = [
        get_media_tool_config().ffprobe_command,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_frames",
        "-read_intervals", "%+#1",  # Only read first frame for side_data
        "-select_streams", "v",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode != 0:
            raise InspectionError(
                f"ffprobe failed for {file_path}: {result.stderr.strip()}"
            )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired as e:
        raise InspectionError(f"ffprobe timed out for {file_path}") from e
    except json.JSONDecodeError as e:
        raise InspectionError(f"ffprobe returned invalid JSON for {file_path}") from e
    except FileNotFoundError as e:
        raise InspectionError(
            "ffprobe not found. Ensure FFmpeg is installed and on PATH."
        ) from e


def run_ffprobe_streams_only(file_path: Path) -> dict[str, Any]:
    """Run ffprobe for stream info only (faster, no frame decoding).

    Args:
        file_path: Path to the media file.

    Returns:
        Parsed JSON dict with stream information.
    """
    cmd = [
        get_media_tool_config().ffprobe_command,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-select_streams", "v",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode != 0:
            raise InspectionError(
                f"ffprobe failed for {file_path}: {result.stderr.strip()}"
            )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired as e:
        raise InspectionError(f"ffprobe timed out for {file_path}") from e
    except json.JSONDecodeError as e:
        raise InspectionError(f"ffprobe returned invalid JSON for {file_path}") from e
    except FileNotFoundError as e:
        raise InspectionError(
            "ffprobe not found. Ensure FFmpeg is installed and on PATH."
        ) from e


def extract_frame_to_numpy(
    file_path: Path,
    frame_number: int,
    stream_index: int = 0,
    width: int | None = None,
    pix_fmt: str = "gray16le",
) -> bytes:
    """Extract a single frame as raw pixel data using FFmpeg.

    Args:
        file_path: Path to the media file.
        frame_number: Frame number to extract (0-indexed).
        stream_index: Video stream index.
        width: Target width for scaling (None = original).
        pix_fmt: Output pixel format.

    Returns:
        Raw pixel data as bytes.
    """
    vf_parts = []
    if width:
        vf_parts.append(f"scale={width}:-1")

    cmd = [
        get_media_tool_config().ffmpeg_command,
        "-v", "quiet",
        "-nostdin",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
        "-vf", f"select=eq(n\\,{frame_number})" + (("," + ",".join(vf_parts)) if vf_parts else ""),
        "-frames:v", "1",
        "-pix_fmt", pix_fmt,
        "-f", "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30, check=False
        )
        if result.returncode != 0:
            return b""
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return b""


def extract_frame_at_time(
    file_path: Path,
    time_seconds: float,
    stream_index: int = 0,
    width: int | None = None,
    pix_fmt: str = "gray16le",
) -> bytes:
    """Extract a frame at a specific timestamp.

    Uses seeking for performance on large files.

    Args:
        file_path: Path to the media file.
        time_seconds: Timestamp in seconds.
        stream_index: Video stream index.
        width: Target width for scaling (None = original).
        pix_fmt: Output pixel format.

    Returns:
        Raw pixel data as bytes.
    """
    vf_parts = []
    if width:
        vf_parts.append(f"scale={width}:-1")

    vf = ",".join(vf_parts) if vf_parts else None

    cmd = [
        get_media_tool_config().ffmpeg_command,
        "-v", "quiet",
        "-nostdin",
        "-ss", f"{time_seconds:.6f}",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-frames:v", "1",
        "-pix_fmt", pix_fmt,
        "-f", "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=30, check=False
        )
        if result.returncode != 0:
            return b""
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return b""


def generate_proxy(
    file_path: Path,
    output_path: Path,
    stream_index: int = 0,
    width: int = 480,
    fps: float | None = None,
) -> bool:
    """Generate a low-resolution grayscale proxy video.

    Args:
        file_path: Input video path.
        output_path: Output proxy path.
        stream_index: Video stream index to use.
        width: Target width.
        fps: Target framerate (None = same as source).

    Returns:
        True if successful.
    """
    vf_parts = [f"scale={width}:-1", "format=gray"]
    if fps:
        vf_parts.insert(0, f"fps={fps}")

    cmd = [
        get_media_tool_config().ffmpeg_command,
        "-v", "quiet",
        "-nostdin",
        "-i", str(file_path),
        "-map", f"0:v:{stream_index}",
        "-vf", ",".join(vf_parts),
        "-an",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-y",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=600, check=False
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
