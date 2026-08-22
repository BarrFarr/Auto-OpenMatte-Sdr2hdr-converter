"""Tests for source inspection and metadata detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from auto_openmatte.analysis.inspect import (
    _classify_hdr_format,
    _detect_hdr_metadata,
    _parse_stream,
    inspect_source,
)
from auto_openmatte.core.models import (
    ColorPrimaries,
    FrameRateType,
    HDRFormat,
    HDRMetadata,
    TransferFunction,
)


class TestParseStream:
    """Tests for video stream parsing."""

    def test_parse_hdr10_stream(self, ffprobe_hdr10: dict) -> None:
        stream_data = ffprobe_hdr10["streams"][0]
        stream = _parse_stream(stream_data)

        assert stream.codec == "hevc"
        assert stream.profile == "Main 10"
        assert stream.width == 3840
        assert stream.height == 1600
        assert abs(stream.fps - 23.976) < 0.01
        assert stream.bit_depth == 10
        assert stream.color_primaries == ColorPrimaries.BT2020
        assert stream.transfer == TransferFunction.PQ
        assert stream.color_range == "tv"
        assert stream.chroma_subsampling == "4:2:0"
        assert stream.frame_rate_type == FrameRateType.CFR
        assert stream.is_default is True

    def test_parse_sdr_stream(self, ffprobe_sdr_openmatte: dict) -> None:
        stream_data = ffprobe_sdr_openmatte["streams"][0]
        stream = _parse_stream(stream_data)

        assert stream.codec == "hevc"
        assert stream.width == 3840
        assert stream.height == 2160
        assert abs(stream.fps - 23.976) < 0.01
        assert stream.bit_depth == 8
        assert stream.color_primaries == ColorPrimaries.BT709
        assert stream.transfer == TransferFunction.BT709
        assert stream.frame_rate_type == FrameRateType.CFR

    def test_parse_hlg_stream(self, ffprobe_hlg: dict) -> None:
        stream_data = ffprobe_hlg["streams"][0]
        stream = _parse_stream(stream_data)

        assert stream.transfer == TransferFunction.HLG
        assert stream.color_primaries == ColorPrimaries.BT2020
        assert stream.bit_depth == 10
        assert stream.width == 3840
        assert stream.height == 1608  # Non-standard aspect ratio

    def test_frame_count(self, ffprobe_hdr10: dict) -> None:
        stream_data = ffprobe_hdr10["streams"][0]
        stream = _parse_stream(stream_data)
        assert stream.frame_count == 172800

    def test_duration(self, ffprobe_hdr10: dict) -> None:
        stream_data = ffprobe_hdr10["streams"][0]
        stream = _parse_stream(stream_data)
        assert abs(stream.duration_seconds - 7200.0) < 0.01


class TestHDRDetection:
    """Tests for HDR metadata detection."""

    def test_detect_hdr10_metadata(self, ffprobe_hdr10: dict) -> None:
        metadata = _detect_hdr_metadata(ffprobe_hdr10)
        assert metadata.mastering_display is not None
        assert metadata.max_cll == 1000
        assert metadata.max_fall == 400

    def test_detect_no_hdr_metadata(self, ffprobe_sdr_openmatte: dict) -> None:
        metadata = _detect_hdr_metadata(ffprobe_sdr_openmatte)
        assert metadata.mastering_display is None
        assert metadata.max_cll is None
        assert metadata.max_fall is None

    def test_classify_hdr10(self, ffprobe_hdr10: dict) -> None:
        stream_data = ffprobe_hdr10["streams"][0]
        stream = _parse_stream(stream_data)
        metadata = _detect_hdr_metadata(ffprobe_hdr10)
        fmt = _classify_hdr_format(stream, metadata)
        assert fmt == HDRFormat.HDR10

    def test_classify_hlg(self, ffprobe_hlg: dict) -> None:
        stream_data = ffprobe_hlg["streams"][0]
        stream = _parse_stream(stream_data)
        metadata = HDRMetadata()
        fmt = _classify_hdr_format(stream, metadata)
        assert fmt == HDRFormat.HLG

    def test_classify_sdr(self, ffprobe_sdr_openmatte: dict) -> None:
        stream_data = ffprobe_sdr_openmatte["streams"][0]
        stream = _parse_stream(stream_data)
        metadata = HDRMetadata()
        fmt = _classify_hdr_format(stream, metadata)
        assert fmt == HDRFormat.SDR


class TestInspectSource:
    """Tests for the full inspect_source function."""

    def test_file_not_found(self) -> None:
        from auto_openmatte.core.exceptions import InspectionError
        with pytest.raises(InspectionError, match="File not found"):
            inspect_source(Path("/nonexistent/file.mkv"))
