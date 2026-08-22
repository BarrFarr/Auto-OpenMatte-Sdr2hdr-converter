"""Tests for HDR detection and role assignment."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_openmatte.analysis.hdr_detect import _hdr_score, assign_roles
from auto_openmatte.analysis.inspect import _detect_hdr_metadata, _parse_stream
from auto_openmatte.core.exceptions import HDRDetectionError, UnsupportedHDRError
from auto_openmatte.core.models import (
    HDRFormat,
    HDRMetadata,
    SourceInfo,
    SourceRole,
    TransferFunction,
    VideoStreamInfo,
)


class TestHDRScore:
    """Tests for HDR confidence scoring."""

    def test_pq_bt2020_high_score(self, ffprobe_hdr10: dict) -> None:
        """PQ + BT.2020 + mastering display should score high."""
        stream = _parse_stream(ffprobe_hdr10["streams"][0])
        metadata = _detect_hdr_metadata(ffprobe_hdr10)
        source = SourceInfo(
            path=Path("hdr.mkv"),
            selected_stream=stream,
            hdr_metadata=metadata,
        )
        score = _hdr_score(source)
        assert score >= 100  # Very confident HDR

    def test_sdr_low_score(self, ffprobe_sdr_openmatte: dict) -> None:
        """SDR source should score very low."""
        stream = _parse_stream(ffprobe_sdr_openmatte["streams"][0])
        source = SourceInfo(
            path=Path("sdr.mkv"),
            selected_stream=stream,
            hdr_metadata=HDRMetadata(),
        )
        score = _hdr_score(source)
        assert score < 30  # Clearly not HDR

    def test_hlg_high_score(self, ffprobe_hlg: dict) -> None:
        """HLG + BT.2020 should score high."""
        stream = _parse_stream(ffprobe_hlg["streams"][0])
        source = SourceInfo(
            path=Path("hlg.mkv"),
            selected_stream=stream,
            hdr_metadata=HDRMetadata(),
        )
        score = _hdr_score(source)
        assert score >= 90


class TestRoleAssignment:
    """Tests for automatic role assignment."""

    def test_correct_assignment(self, ffprobe_hdr10: dict, ffprobe_sdr_openmatte: dict) -> None:
        """Should correctly identify HDR and Open Matte."""
        hdr_stream = _parse_stream(ffprobe_hdr10["streams"][0])
        hdr_metadata = _detect_hdr_metadata(ffprobe_hdr10)
        source_a = SourceInfo(
            path=Path("film_hdr.mkv"),
            selected_stream=hdr_stream,
            hdr_metadata=hdr_metadata,
        )

        sdr_stream = _parse_stream(ffprobe_sdr_openmatte["streams"][0])
        source_b = SourceInfo(
            path=Path("film_openmatte.mkv"),
            selected_stream=sdr_stream,
            hdr_metadata=HDRMetadata(),
        )

        hdr, om = assign_roles(source_a, source_b)
        assert hdr.role == SourceRole.HDR_REFERENCE
        assert om.role == SourceRole.OPEN_MATTE
        assert hdr.path.name == "film_hdr.mkv"
        assert om.path.name == "film_openmatte.mkv"

    def test_reversed_order(self, ffprobe_hdr10: dict, ffprobe_sdr_openmatte: dict) -> None:
        """Should detect correct roles even when given in reverse order."""
        hdr_stream = _parse_stream(ffprobe_hdr10["streams"][0])
        hdr_metadata = _detect_hdr_metadata(ffprobe_hdr10)
        source_hdr = SourceInfo(
            path=Path("film_hdr.mkv"),
            selected_stream=hdr_stream,
            hdr_metadata=hdr_metadata,
        )

        sdr_stream = _parse_stream(ffprobe_sdr_openmatte["streams"][0])
        source_sdr = SourceInfo(
            path=Path("film_openmatte.mkv"),
            selected_stream=sdr_stream,
            hdr_metadata=HDRMetadata(),
        )

        # Pass SDR first, HDR second — should still work
        hdr, om = assign_roles(source_sdr, source_hdr)
        assert hdr.path.name == "film_hdr.mkv"
        assert om.path.name == "film_openmatte.mkv"

    def test_both_sdr_raises(self) -> None:
        """Should fail if both sources are SDR."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.BT709, bit_depth=8,
        )
        source_a = SourceInfo(
            path=Path("a.mkv"),
            selected_stream=stream,
            hdr_metadata=HDRMetadata(),
        )
        source_b = SourceInfo(
            path=Path("b.mkv"),
            selected_stream=stream,
            hdr_metadata=HDRMetadata(),
        )

        with pytest.raises(HDRDetectionError, match="Cannot determine HDR roles"):
            assign_roles(source_a, source_b)

    def test_dolby_vision_unsupported(self) -> None:
        """Should raise UnsupportedHDRError for Dolby Vision."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.PQ, bit_depth=10,
        )
        dv_metadata = HDRMetadata(format=HDRFormat.DOLBY_VISION)
        source_a = SourceInfo(
            path=Path("dv.mkv"),
            selected_stream=stream,
            hdr_metadata=dv_metadata,
        )

        sdr_stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.BT709, bit_depth=8,
        )
        source_b = SourceInfo(
            path=Path("sdr.mkv"),
            selected_stream=sdr_stream,
            hdr_metadata=HDRMetadata(),
        )

        with pytest.raises(UnsupportedHDRError, match="Dolby Vision"):
            assign_roles(source_a, source_b)
