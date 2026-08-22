"""Tests for synchronization cache system."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_openmatte.core.models import (
    ColorPrimaries,
    SourceInfo,
    SyncModel,
    SyncStatus,
    TransferFunction,
    VideoStreamInfo,
)
from auto_openmatte.core.sync_cache import (
    SourceFingerprint,
    build_fingerprint,
    invalidate_sync_cache,
    load_sync_cache,
    save_sync_cache,
)


@pytest.fixture
def hdr_source(tmp_path: Path) -> SourceInfo:
    """Create a mock HDR source with a real file on disk."""
    # Create a dummy file so stat() works
    dummy = tmp_path / "hdr.mkv"
    dummy.write_bytes(b"\x00" * 1024)

    stream = VideoStreamInfo(
        index=0, codec="hevc", width=3840, height=1600,
        fps=23.976, duration_seconds=7200.0, frame_count=172800,
        bit_depth=10, transfer=TransferFunction.PQ,
        color_primaries=ColorPrimaries.BT2020,
    )
    return SourceInfo(
        path=dummy,
        video_streams=[stream],
        selected_stream=stream,
    )


@pytest.fixture
def om_source(tmp_path: Path) -> SourceInfo:
    """Create a mock OM source with a real file on disk."""
    dummy = tmp_path / "om.mkv"
    dummy.write_bytes(b"\x00" * 2048)

    stream = VideoStreamInfo(
        index=0, codec="hevc", width=3840, height=2160,
        fps=23.976, duration_seconds=7202.4, frame_count=172858,
        bit_depth=8, transfer=TransferFunction.BT709,
        color_primaries=ColorPrimaries.BT709,
    )
    return SourceInfo(
        path=dummy,
        video_streams=[stream],
        selected_stream=stream,
    )


@pytest.fixture
def locked_sync() -> SyncModel:
    """A locked sync model."""
    return SyncModel(
        frame_offset=58,
        confidence=0.998,
        status=SyncStatus.LOCKED,
        frame_locked=True,
        drift_frames=0.0,
        offset_seconds=58 / 23.976,
        method="image_based_frame_offset",
    )


class TestSourceFingerprint:
    """Tests for SourceFingerprint."""

    def test_build_from_source(self, hdr_source: SourceInfo) -> None:
        fp = build_fingerprint(hdr_source)
        assert fp.width == 3840
        assert fp.height == 1600
        assert fp.fps == pytest.approx(23.976)
        assert fp.codec == "hevc"
        assert fp.transfer == "smpte2084"
        assert fp.bit_depth == 10
        assert fp.file_size == 1024

    def test_hash_deterministic(self, hdr_source: SourceInfo) -> None:
        fp1 = build_fingerprint(hdr_source)
        fp2 = build_fingerprint(hdr_source)
        assert fp1.compute_hash() == fp2.compute_hash()

    def test_serialization(self, hdr_source: SourceInfo) -> None:
        fp = build_fingerprint(hdr_source)
        data = fp.to_dict()
        restored = SourceFingerprint.from_dict(data)
        assert restored.width == fp.width
        assert restored.fps == fp.fps
        assert restored.codec == fp.codec


class TestSyncCache:
    """Tests for cache save/load/invalidate."""

    def test_save_and_load(
        self, tmp_path: Path, hdr_source: SourceInfo,
        om_source: SourceInfo, locked_sync: SyncModel
    ) -> None:
        """Save then load should return the same sync model."""
        cache_dir = tmp_path / "cache"
        save_sync_cache(cache_dir, hdr_source, om_source, locked_sync)

        loaded = load_sync_cache(cache_dir, hdr_source, om_source)
        assert loaded is not None
        assert loaded.frame_offset == 58
        assert loaded.status == SyncStatus.LOCKED
        assert loaded.confidence == pytest.approx(0.998)
        assert loaded.frame_locked is True

    def test_cache_miss_no_file(
        self, tmp_path: Path, hdr_source: SourceInfo, om_source: SourceInfo
    ) -> None:
        """No cache file → None."""
        result = load_sync_cache(tmp_path / "empty", hdr_source, om_source)
        assert result is None

    def test_invalidate_on_source_change(
        self, tmp_path: Path, hdr_source: SourceInfo,
        om_source: SourceInfo, locked_sync: SyncModel
    ) -> None:
        """Changing source file should invalidate cache."""
        cache_dir = tmp_path / "cache"
        save_sync_cache(cache_dir, hdr_source, om_source, locked_sync)

        # Modify the HDR source file (change size)
        hdr_source.path.write_bytes(b"\x00" * 9999)

        loaded = load_sync_cache(cache_dir, hdr_source, om_source)
        assert loaded is None  # Cache invalidated

    def test_invalidate_on_resolution_change(
        self, tmp_path: Path, hdr_source: SourceInfo,
        om_source: SourceInfo, locked_sync: SyncModel
    ) -> None:
        """Changing resolution should invalidate cache."""
        cache_dir = tmp_path / "cache"
        save_sync_cache(cache_dir, hdr_source, om_source, locked_sync)

        # Change resolution in the source info
        assert hdr_source.selected_stream is not None
        hdr_source.selected_stream.width = 1920
        hdr_source.selected_stream.height = 800

        loaded = load_sync_cache(cache_dir, hdr_source, om_source)
        assert loaded is None  # Invalidated

    def test_invalidate_command(
        self, tmp_path: Path, hdr_source: SourceInfo,
        om_source: SourceInfo, locked_sync: SyncModel
    ) -> None:
        """invalidate_sync_cache should delete the cache file."""
        cache_dir = tmp_path / "cache"
        save_sync_cache(cache_dir, hdr_source, om_source, locked_sync)

        assert (cache_dir / "sync_cache.json").exists()
        invalidate_sync_cache(cache_dir)
        assert not (cache_dir / "sync_cache.json").exists()

    def test_failed_sync_not_cached(
        self, tmp_path: Path, hdr_source: SourceInfo, om_source: SourceInfo
    ) -> None:
        """A FAILED sync model, if saved, should not be returned on load."""
        cache_dir = tmp_path / "cache"
        failed_sync = SyncModel(
            frame_offset=0, confidence=0.3, status=SyncStatus.FAILED
        )
        save_sync_cache(cache_dir, hdr_source, om_source, failed_sync)

        loaded = load_sync_cache(cache_dir, hdr_source, om_source)
        assert loaded is None  # Only LOCKED results are reusable

    def test_corrupted_json(
        self, tmp_path: Path, hdr_source: SourceInfo, om_source: SourceInfo
    ) -> None:
        """Corrupted cache file → None (not a crash)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "sync_cache.json").write_text("not valid json {{{")

        loaded = load_sync_cache(cache_dir, hdr_source, om_source)
        assert loaded is None
