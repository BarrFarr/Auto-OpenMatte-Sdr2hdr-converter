"""Tests for video stream selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_openmatte.analysis.inspect import _parse_stream
from auto_openmatte.analysis.stream_select import _stream_quality_score, select_video_stream
from auto_openmatte.core.exceptions import StreamSelectionError
from auto_openmatte.core.models import SourceInfo, VideoStreamInfo


class TestStreamQualityScore:
    """Tests for stream quality scoring."""

    def test_hdr_stream_scores_higher(self, ffprobe_multi_stream: dict) -> None:
        """HDR stream should score higher than SDR proxy."""
        streams = ffprobe_multi_stream["streams"]
        hdr_stream = _parse_stream(streams[0])
        proxy_stream = _parse_stream(streams[1])

        hdr_score = _stream_quality_score(hdr_stream)
        proxy_score = _stream_quality_score(proxy_stream)

        assert hdr_score > proxy_score

    def test_high_res_scores_higher(self) -> None:
        """Higher resolution should score higher."""
        stream_4k = VideoStreamInfo(index=0, codec="hevc", width=3840, height=2160, bit_depth=10)
        stream_hd = VideoStreamInfo(index=1, codec="hevc", width=1920, height=1080, bit_depth=10)

        assert _stream_quality_score(stream_4k) > _stream_quality_score(stream_hd)

    def test_low_res_penalized(self) -> None:
        """Very low resolution streams should be penalized."""
        stream_proxy = VideoStreamInfo(index=0, codec="h264", width=160, height=90, bit_depth=8)
        score = _stream_quality_score(stream_proxy)
        assert score < 0  # Penalized below zero


class TestStreamSelection:
    """Tests for stream selection logic."""

    def test_single_stream(self) -> None:
        """Single stream should be selected automatically."""
        stream = VideoStreamInfo(index=0, codec="hevc", width=3840, height=2160)
        source = SourceInfo(path=Path("test.mkv"), video_streams=[stream])

        result = select_video_stream(source)
        assert result.selected_stream == stream

    def test_multi_stream_selects_best(self, ffprobe_multi_stream: dict) -> None:
        """Should select the highest quality stream from multiple options."""
        streams = [_parse_stream(s) for s in ffprobe_multi_stream["streams"]]
        source = SourceInfo(path=Path("test.mkv"), video_streams=streams)

        result = select_video_stream(source)
        assert result.selected_stream is not None
        assert result.selected_stream.width == 3840
        assert result.selected_stream.height == 1600

    def test_no_streams_raises(self) -> None:
        """Should raise if no video streams."""
        source = SourceInfo(path=Path("test.mkv"), video_streams=[])
        with pytest.raises(StreamSelectionError):
            select_video_stream(source)
