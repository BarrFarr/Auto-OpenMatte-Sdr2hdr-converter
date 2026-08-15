"""Synchronization cache — save, load, and invalidate sync results.

Avoids expensive re-synchronization when sources haven't changed.
Invalidates automatically if any relevant source property differs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from auto_openmatte.core.models import SourceInfo, SyncModel, SyncStatus

logger = logging.getLogger(__name__)

CACHE_FILENAME = "sync_cache.json"


@dataclass
class SourceFingerprint:
    """Fingerprint of a source file for cache invalidation.

    If any of these properties change, the sync cache is invalidated.
    """

    path: str = ""
    file_size: int = 0
    mtime: float = 0.0
    stream_index: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    codec: str = ""
    transfer: str = ""
    color_primaries: str = ""
    bit_depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceFingerprint:
        """Deserialize from dict."""
        return cls(
            path=data.get("path", ""),
            file_size=data.get("file_size", 0),
            mtime=data.get("mtime", 0.0),
            stream_index=data.get("stream_index", 0),
            width=data.get("width", 0),
            height=data.get("height", 0),
            fps=data.get("fps", 0.0),
            duration=data.get("duration", 0.0),
            codec=data.get("codec", ""),
            transfer=data.get("transfer", ""),
            color_primaries=data.get("color_primaries", ""),
            bit_depth=data.get("bit_depth", 0),
        )

    def compute_hash(self) -> str:
        """Compute a stable hash of the fingerprint for quick comparison."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SyncCache:
    """Cached synchronization result with source fingerprints."""

    hdr_fingerprint: SourceFingerprint = field(default_factory=SourceFingerprint)
    om_fingerprint: SourceFingerprint = field(default_factory=SourceFingerprint)
    sync_model: SyncModel = field(default_factory=SyncModel)
    cache_version: str = "1.0"


def build_fingerprint(source: SourceInfo) -> SourceFingerprint:
    """Build a fingerprint from a SourceInfo.

    Args:
        source: Inspected source information.

    Returns:
        SourceFingerprint capturing all cache-relevant properties.
    """
    fp = SourceFingerprint()
    fp.path = str(source.path.resolve())

    # File system properties
    try:
        stat = source.path.stat()
        fp.file_size = stat.st_size
        fp.mtime = stat.st_mtime
    except OSError:
        pass

    # Stream properties
    stream = source.selected_stream
    if stream:
        fp.stream_index = stream.index
        fp.width = stream.width
        fp.height = stream.height
        fp.fps = stream.fps
        fp.duration = stream.duration_seconds
        fp.codec = stream.codec
        fp.transfer = stream.transfer.value
        fp.color_primaries = stream.color_primaries.value
        fp.bit_depth = stream.bit_depth

    return fp


def save_sync_cache(
    cache_dir: Path,
    hdr_source: SourceInfo,
    om_source: SourceInfo,
    sync_model: SyncModel,
) -> Path:
    """Save synchronization result to cache.

    Args:
        cache_dir: Directory to store the cache file.
        hdr_source: HDR source info (for fingerprinting).
        om_source: Open Matte source info (for fingerprinting).
        sync_model: The validated sync model to cache.

    Returns:
        Path to the saved cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILENAME

    hdr_fp = build_fingerprint(hdr_source)
    om_fp = build_fingerprint(om_source)

    cache_data = {
        "cache_version": "1.0",
        "hdr_fingerprint": hdr_fp.to_dict(),
        "om_fingerprint": om_fp.to_dict(),
        "sync_model": {
            "frame_offset": sync_model.frame_offset,
            "confidence": sync_model.confidence,
            "status": sync_model.status.value,
            "mean_error_frames": sync_model.mean_error_frames,
            "max_error_frames": sync_model.max_error_frames,
            "drift_frames": sync_model.drift_frames,
            "offset_seconds": sync_model.offset_seconds,
            "frame_locked": sync_model.frame_locked,
            "method": sync_model.method,
            "checkpoints": sync_model.checkpoints,
        },
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    logger.info(f"Sync cache saved to {cache_path}")
    return cache_path


def load_sync_cache(
    cache_dir: Path,
    hdr_source: SourceInfo,
    om_source: SourceInfo,
) -> SyncModel | None:
    """Load and validate cached synchronization result.

    Returns None if:
    - Cache file doesn't exist
    - Cache is stale (source fingerprints don't match)
    - Cache data is corrupted

    Args:
        cache_dir: Directory containing the cache file.
        hdr_source: Current HDR source info.
        om_source: Current Open Matte source info.

    Returns:
        Cached SyncModel if valid, None if cache miss or stale.
    """
    cache_path = cache_dir / CACHE_FILENAME

    if not cache_path.exists():
        logger.debug("No sync cache found")
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read sync cache: {e}")
        return None

    # Check version
    if cache_data.get("cache_version") != "1.0":
        logger.info("Sync cache version mismatch — invalidating")
        return None

    # Build current fingerprints
    current_hdr_fp = build_fingerprint(hdr_source)
    current_om_fp = build_fingerprint(om_source)

    # Load cached fingerprints
    cached_hdr_fp = SourceFingerprint.from_dict(
        cache_data.get("hdr_fingerprint", {})
    )
    cached_om_fp = SourceFingerprint.from_dict(
        cache_data.get("om_fingerprint", {})
    )

    # Compare fingerprints
    if not _fingerprints_match(current_hdr_fp, cached_hdr_fp, "HDR"):
        return None
    if not _fingerprints_match(current_om_fp, cached_om_fp, "Open Matte"):
        return None

    # Rebuild SyncModel from cache
    sync_data = cache_data.get("sync_model", {})
    sync_model = SyncModel(
        frame_offset=sync_data.get("frame_offset", 0),
        confidence=sync_data.get("confidence", 0.0),
        status=SyncStatus(sync_data.get("status", "NOT_RUN")),
        mean_error_frames=sync_data.get("mean_error_frames", 0.0),
        max_error_frames=sync_data.get("max_error_frames", 0.0),
        drift_frames=sync_data.get("drift_frames", 0.0),
        offset_seconds=sync_data.get("offset_seconds", 0.0),
        frame_locked=sync_data.get("frame_locked", False),
        method=sync_data.get("method", "image_based_frame_offset"),
        checkpoints=sync_data.get("checkpoints", []),
    )

    # Only return if the cached result was successful
    if sync_model.status != SyncStatus.LOCKED:
        logger.info(
            f"Cached sync status is {sync_model.status.value} — "
            "will re-run synchronization"
        )
        return None

    logger.info(
        f"Sync cache HIT: offset={sync_model.frame_offset}, "
        f"confidence={sync_model.confidence:.4f}"
    )
    return sync_model


def invalidate_sync_cache(cache_dir: Path) -> None:
    """Delete the sync cache file.

    Args:
        cache_dir: Directory containing the cache file.
    """
    cache_path = cache_dir / CACHE_FILENAME
    if cache_path.exists():
        cache_path.unlink()
        logger.info("Sync cache invalidated")


def _fingerprints_match(
    current: SourceFingerprint,
    cached: SourceFingerprint,
    label: str,
) -> bool:
    """Compare two fingerprints and log differences.

    Returns True if they match (cache is valid).
    """
    mismatches: list[str] = []

    # Path must match (resolved absolute path)
    if current.path != cached.path:
        mismatches.append(f"path: {cached.path} -> {current.path}")

    # File size
    if current.file_size != cached.file_size and cached.file_size > 0:
        mismatches.append(
            f"file_size: {cached.file_size} -> {current.file_size}"
        )

    # Modification time
    if current.mtime != cached.mtime and cached.mtime > 0:
        mismatches.append("mtime changed")

    # Stream properties
    if current.stream_index != cached.stream_index:
        mismatches.append(
            f"stream_index: {cached.stream_index} -> {current.stream_index}"
        )
    if current.width != cached.width and cached.width > 0:
        mismatches.append(f"width: {cached.width} -> {current.width}")
    if current.height != cached.height and cached.height > 0:
        mismatches.append(f"height: {cached.height} -> {current.height}")
    if abs(current.fps - cached.fps) > 0.001 and cached.fps > 0:
        mismatches.append(f"fps: {cached.fps} -> {current.fps}")
    if abs(current.duration - cached.duration) > 0.1 and cached.duration > 0:
        mismatches.append(
            f"duration: {cached.duration:.1f} -> {current.duration:.1f}"
        )
    if current.codec != cached.codec and cached.codec:
        mismatches.append(f"codec: {cached.codec} -> {current.codec}")
    if current.transfer != cached.transfer and cached.transfer:
        mismatches.append(
            f"transfer: {cached.transfer} -> {current.transfer}"
        )
    if current.bit_depth != cached.bit_depth and cached.bit_depth > 0:
        mismatches.append(
            f"bit_depth: {cached.bit_depth} -> {current.bit_depth}"
        )

    if mismatches:
        logger.info(
            f"Sync cache INVALIDATED — {label} source changed: "
            + "; ".join(mismatches[:3])
        )
        return False

    return True
