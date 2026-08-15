"""Tests for source inspection, HDR detection, and stream selection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from auto_openmatte.analysis.hdr_detect import (
    detect_hdr_standard,
    detect_hdr_standard_from_side_data,
    identify_roles,
    validate_hdr_support,
)
from auto_openmatte.analysis.inspect import (
    extract_source_info,
    run_ffprobe,
)
from auto_openmatte.analysis.stream_select import (
    list_video_streams,
    select_best_video_stream,
)
from auto_openmatte.core.exceptions import (
    InvalidSourceError,
    UnsupportedHDRFormatError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    """Load a JSON fixture file."""
    fixture_path = FIXTURES_DIR / name
    with open(fixture_path) as f:
        return json.load(f)


# ============================================================
# Tests for extract_source_info
# ============================================================


class TestExtractSourceInfoHDR10:
    """Test parsing HDR10 ffprobe fixture."""

    @pytest.fixture()
    def hdr10_data(self):
        return _load_fixture("ffprobe_hdr10.json")

    def test_extract_source_info_hdr10(self, hdr10_data):
        """Parse HDR10 fixture and verify all fields."""
        info = extract_source_info(hdr10_data, "/media/movies/sample_hdr10.mkv")

        # Basic properties
        assert info.path == "/media/movies/sample_hdr10.mkv"
        assert info.container == "matroska,webm"
        assert info.codec == "hevc"
        assert info.profile == "Main 10"
        assert info.level == "150"
        assert info.width == 3840
        assert info.height == 1600
        assert info.fps == pytest.approx(23.976, rel=1e-3)
        assert info.nominal_fps == "24000/1001"
        assert info.vfr_cfr == "CFR"
        assert info.duration == pytest.approx(166.833333, rel=1e-4)
        assert info.frame_count == 4004
        assert info.pixel_format == "yuv420p10le"
        assert info.bit_depth == 10
        assert info.chroma_subsampling == "4:2:0"
        assert info.color_range == "tv"
        assert info.color_primaries == "bt2020"
        assert info.transfer_characteristics == "smpte2084"
        assert info.matrix_coefficients == "bt2020nc"

        # HDR metadata
        assert info.mastering_display != ""
        assert "red_x" in info.mastering_display
        assert info.max_cll == 1000
        assert info.max_fall == 400


class TestExtractSourceInfoSDR:
    """Test parsing SDR Open Matte ffprobe fixture."""

    @pytest.fixture()
    def sdr_data(self):
        return _load_fixture("ffprobe_sdr_openmatte.json")

    def test_extract_source_info_sdr(self, sdr_data):
        """Parse SDR fixture and verify all fields."""
        info = extract_source_info(sdr_data, "/media/movies/sample_openmatte.mkv")

        assert info.path == "/media/movies/sample_openmatte.mkv"
        assert info.container == "matroska,webm"
        assert info.codec == "hevc"
        assert info.profile == "Main 10"
        assert info.level == "150"
        assert info.width == 3840
        assert info.height == 2160
        assert info.fps == pytest.approx(23.976, rel=1e-3)
        assert info.nominal_fps == "24000/1001"
        assert info.vfr_cfr == "CFR"
        assert info.duration == pytest.approx(166.833333, rel=1e-4)
        assert info.frame_count == 4004
        assert info.pixel_format == "yuv420p10le"
        assert info.bit_depth == 10
        assert info.chroma_subsampling == "4:2:0"
        assert info.color_range == "tv"
        assert info.color_primaries == "bt709"
        assert info.transfer_characteristics == "bt709"
        assert info.matrix_coefficients == "bt709"

        # No HDR metadata for SDR source
        assert info.mastering_display == ""
        assert info.max_cll == 0
        assert info.max_fall == 0


class TestExtractSourceInfoHLG:
    """Test parsing HLG ffprobe fixture."""

    @pytest.fixture()
    def hlg_data(self):
        return _load_fixture("ffprobe_hlg.json")

    def test_extract_source_info_hlg(self, hlg_data):
        """Parse HLG fixture and verify key fields."""
        info = extract_source_info(hlg_data, "/media/broadcast/sample_hlg.ts")

        assert info.path == "/media/broadcast/sample_hlg.ts"
        assert info.container == "mpegts"
        assert info.codec == "hevc"
        assert info.width == 3840
        assert info.height == 2160
        assert info.fps == pytest.approx(50.0, rel=1e-3)
        assert info.duration == pytest.approx(300.0, rel=1e-4)
        assert info.frame_count == 15000
        assert info.bit_depth == 10
        assert info.color_primaries == "bt2020"
        assert info.transfer_characteristics == "arib-std-b67"
        assert info.matrix_coefficients == "bt2020nc"


# ============================================================
# Tests for detect_hdr_standard
# ============================================================


class TestDetectHDRStandard:
    """Test HDR format detection from source metadata."""

    @pytest.fixture()
    def hdr10_info(self):
        data = _load_fixture("ffprobe_hdr10.json")
        return extract_source_info(data, "hdr10.mkv")

    @pytest.fixture()
    def sdr_info(self):
        data = _load_fixture("ffprobe_sdr_openmatte.json")
        return extract_source_info(data, "sdr.mkv")

    @pytest.fixture()
    def hlg_info(self):
        data = _load_fixture("ffprobe_hlg.json")
        return extract_source_info(data, "hlg.ts")

    def test_detect_hdr_standard_pq_bt2020(self, hdr10_info):
        """PQ transfer + BT.2020 primaries + mastering display = HDR10."""
        result = detect_hdr_standard(hdr10_info)
        assert result == "HDR10"

    def test_detect_hdr_standard_hlg(self, hlg_info):
        """arib-std-b67 transfer = HLG."""
        result = detect_hdr_standard(hlg_info)
        assert result == "HLG"

    def test_detect_hdr_standard_bt709(self, sdr_info):
        """BT.709 transfer = SDR."""
        result = detect_hdr_standard(sdr_info)
        assert result == "SDR"

    def test_detect_hdr_standard_with_dolby_vision_side_data(self, hdr10_info):
        """Side data with Dolby Vision flag = Dolby Vision."""
        side_data = {"dolby_vision": "true"}
        result = detect_hdr_standard_from_side_data(hdr10_info, side_data)
        assert result == "Dolby Vision"

    def test_detect_hdr_standard_with_hdr10plus_side_data(self, hdr10_info):
        """Side data with HDR10+ flag = HDR10+."""
        side_data = {"hdr10plus": "true"}
        result = detect_hdr_standard_from_side_data(hdr10_info, side_data)
        assert result == "HDR10+"


# ============================================================
# Tests for identify_roles
# ============================================================


class TestIdentifyRoles:
    """Test source role identification logic."""

    @pytest.fixture()
    def hdr_info(self):
        data = _load_fixture("ffprobe_hdr10.json")
        return extract_source_info(data, "hdr_source.mkv")

    @pytest.fixture()
    def sdr_info(self):
        data = _load_fixture("ffprobe_sdr_openmatte.json")
        return extract_source_info(data, "openmatte_source.mkv")

    @pytest.fixture()
    def hlg_info(self):
        data = _load_fixture("ffprobe_hlg.json")
        return extract_source_info(data, "hlg_source.ts")

    def test_identify_roles_hdr_first(self, hdr_info, sdr_info):
        """When HDR is first arg, returns (HDR, SDR)."""
        hdr_result, om_result = identify_roles(hdr_info, sdr_info)
        assert hdr_result.path == "hdr_source.mkv"
        assert om_result.path == "openmatte_source.mkv"

    def test_identify_roles_hdr_second(self, hdr_info, sdr_info):
        """When HDR is second arg, still returns (HDR, SDR) - swap detection."""
        hdr_result, om_result = identify_roles(sdr_info, hdr_info)
        assert hdr_result.path == "hdr_source.mkv"
        assert om_result.path == "openmatte_source.mkv"

    def test_identify_roles_both_hdr_raises(self, hdr_info, hlg_info):
        """Both sources HDR raises InvalidSourceError."""
        with pytest.raises(InvalidSourceError, match="Both sources appear to be HDR"):
            identify_roles(hdr_info, hlg_info)

    def test_identify_roles_both_sdr_raises(self, sdr_info):
        """Both sources SDR raises InvalidSourceError."""
        # Create a second SDR source with different path
        data = _load_fixture("ffprobe_sdr_openmatte.json")
        sdr_info_2 = extract_source_info(data, "another_sdr.mkv")

        with pytest.raises(InvalidSourceError, match="Both sources appear to be SDR"):
            identify_roles(sdr_info, sdr_info_2)

    def test_identify_roles_hlg_as_hdr(self, hlg_info, sdr_info):
        """HLG source is identified as the HDR reference."""
        hdr_result, om_result = identify_roles(hlg_info, sdr_info)
        assert hdr_result.path == "hlg_source.ts"
        assert om_result.path == "openmatte_source.mkv"


# ============================================================
# Tests for validate_hdr_support
# ============================================================


class TestValidateHDRSupport:
    """Test HDR format validation against supported formats."""

    def test_validate_hdr_support_hdr10_ok(self):
        """HDR10 is a supported format."""
        # Should not raise
        validate_hdr_support("HDR10")

    def test_validate_hdr_support_hlg_ok(self):
        """HLG is a supported format."""
        # Should not raise
        validate_hdr_support("HLG")

    def test_validate_hdr_support_dolby_vision_unsupported(self):
        """Dolby Vision raises UnsupportedHDRFormatError."""
        with pytest.raises(UnsupportedHDRFormatError, match="Dolby Vision"):
            validate_hdr_support("Dolby Vision")

    def test_validate_hdr_support_hdr10plus_unsupported(self):
        """HDR10+ raises UnsupportedHDRFormatError."""
        with pytest.raises(UnsupportedHDRFormatError, match="HDR10\\+"):
            validate_hdr_support("HDR10+")


# ============================================================
# Tests for stream selection
# ============================================================


class TestStreamSelection:
    """Test video stream selection from multi-stream containers."""

    @pytest.fixture()
    def multi_stream_data(self):
        return _load_fixture("ffprobe_multi_stream.json")

    def test_select_best_stream_multi(self, multi_stream_data):
        """Picks highest quality HDR stream from multi-stream container."""
        streams = multi_stream_data["streams"]
        best_idx = select_best_video_stream(streams)

        # Stream 0 is 4K HDR10 with mastering display metadata - best choice
        assert best_idx == 0
        assert streams[best_idx]["width"] == 3840
        assert streams[best_idx]["height"] == 2160

    def test_select_best_stream_single(self):
        """Single stream always returns index 0."""
        streams = [{"width": 1920, "height": 1080, "pix_fmt": "yuv420p"}]
        assert select_best_video_stream(streams) == 0

    def test_select_best_stream_empty(self):
        """Empty list returns 0."""
        assert select_best_video_stream([]) == 0

    def test_list_video_streams(self, multi_stream_data):
        """List all video streams with summary info."""
        streams = multi_stream_data["streams"]
        summaries = list_video_streams(streams)

        assert len(summaries) == 3

        # First stream: 4K HDR
        assert summaries[0]["width"] == 3840
        assert summaries[0]["height"] == 2160
        assert summaries[0]["codec"] == "hevc"
        assert summaries[0]["bit_depth"] == 10
        assert summaries[0]["has_hdr_metadata"] is True
        assert summaries[0]["is_default"] is True
        assert summaries[0]["resolution"] == "3840x2160"

        # Second stream: 720p proxy
        assert summaries[1]["width"] == 1280
        assert summaries[1]["height"] == 720
        assert summaries[1]["codec"] == "h264"
        assert summaries[1]["has_hdr_metadata"] is False
        assert summaries[1]["is_default"] is False

        # Third stream: thumbnail
        assert summaries[2]["width"] == 320
        assert summaries[2]["height"] == 180
        assert summaries[2]["codec"] == "mjpeg"


# ============================================================
# Tests for run_ffprobe error handling
# ============================================================


class TestRunFfprobe:
    """Test ffprobe execution and error handling."""

    def test_run_ffprobe_file_not_found(self):
        """Raises InvalidSourceError for non-existent file."""
        with pytest.raises(InvalidSourceError, match="File not found"):
            run_ffprobe("/nonexistent/path/video.mkv")

    @patch("auto_openmatte.analysis.inspect.subprocess.run")
    def test_run_ffprobe_ffprobe_not_installed(self, mock_run, tmp_path):
        """Raises InvalidSourceError when ffprobe binary is not found."""
        # Create a temp file so the path check passes
        test_file = tmp_path / "test.mkv"
        test_file.write_bytes(b"\x00" * 100)

        mock_run.side_effect = FileNotFoundError("No such file or directory: 'ffprobe'")

        with pytest.raises(InvalidSourceError, match="ffprobe not found"):
            run_ffprobe(str(test_file))

    @patch("auto_openmatte.analysis.inspect.subprocess.run")
    def test_run_ffprobe_process_failure(self, mock_run, tmp_path):
        """Raises InvalidSourceError when ffprobe returns non-zero exit."""
        test_file = tmp_path / "test.mkv"
        test_file.write_bytes(b"\x00" * 100)

        mock_run.return_value = type("Result", (), {
            "returncode": 1,
            "stdout": "",
            "stderr": "Invalid data found when processing input",
        })()

        with pytest.raises(InvalidSourceError, match="ffprobe failed"):
            run_ffprobe(str(test_file))

    @patch("auto_openmatte.analysis.inspect.subprocess.run")
    def test_run_ffprobe_success(self, mock_run, tmp_path):
        """Successful ffprobe returns parsed JSON dict."""
        test_file = tmp_path / "test.mkv"
        test_file.write_bytes(b"\x00" * 100)

        fixture_data = _load_fixture("ffprobe_hdr10.json")
        mock_run.return_value = type("Result", (), {
            "returncode": 0,
            "stdout": json.dumps(fixture_data),
            "stderr": "",
        })()

        result = run_ffprobe(str(test_file))
        assert "streams" in result
        assert result["streams"][0]["codec_name"] == "hevc"

    @patch("auto_openmatte.analysis.inspect.subprocess.run")
    def test_run_ffprobe_invalid_json(self, mock_run, tmp_path):
        """Raises InvalidSourceError when output is not valid JSON."""
        test_file = tmp_path / "test.mkv"
        test_file.write_bytes(b"\x00" * 100)

        mock_run.return_value = type("Result", (), {
            "returncode": 0,
            "stdout": "not valid json {{{",
            "stderr": "",
        })()

        with pytest.raises(InvalidSourceError, match="Failed to parse ffprobe JSON"):
            run_ffprobe(str(test_file))

    @patch("auto_openmatte.analysis.inspect.subprocess.run")
    def test_run_ffprobe_no_streams(self, mock_run, tmp_path):
        """Raises InvalidSourceError when no video streams in output."""
        test_file = tmp_path / "test.mkv"
        test_file.write_bytes(b"\x00" * 100)

        mock_run.return_value = type("Result", (), {
            "returncode": 0,
            "stdout": json.dumps({"streams": [], "format": {}}),
            "stderr": "",
        })()

        with pytest.raises(InvalidSourceError, match="No video streams"):
            run_ffprobe(str(test_file))
