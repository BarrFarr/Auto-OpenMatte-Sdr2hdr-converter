"""Shared pytest fixtures for auto_openmatte tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_hdr_source_info() -> dict:
    """Sample HDR source metadata as returned by ffprobe."""
    return {
        "path": "/videos/hdr_master.mkv",
        "width": 3840,
        "height": 1620,
        "fps": 23.976,
        "duration": 7200.0,
        "frame_count": 172627,
        "codec": "hevc",
        "bit_depth": 10,
        "color_space": "bt2020nc",
        "color_transfer": "smpte2084",
        "color_primaries": "bt2020",
        "pixel_format": "yuv420p10le",
        "aspect_ratio": "2.37:1",
    }


@pytest.fixture
def sample_openmatte_source_info() -> dict:
    """Sample Open Matte source metadata as returned by ffprobe."""
    return {
        "path": "/videos/openmatte_sdr.mkv",
        "width": 1920,
        "height": 1080,
        "fps": 23.976,
        "duration": 7200.0,
        "frame_count": 172627,
        "codec": "h264",
        "bit_depth": 8,
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "pixel_format": "yuv420p",
        "aspect_ratio": "16:9",
    }


@pytest.fixture
def mock_ffprobe_output() -> dict:
    """Mock ffprobe JSON output for testing source inspection."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 1620,
                "r_frame_rate": "24000/1001",
                "duration": "7200.000000",
                "nb_frames": "172627",
                "bits_per_raw_sample": "10",
                "color_space": "bt2020nc",
                "color_transfer": "smpte2084",
                "color_primaries": "bt2020",
                "pix_fmt": "yuv420p10le",
                "display_aspect_ratio": "128:54",
                "side_data_list": [
                    {
                        "side_data_type": "Mastering display metadata",
                        "red_x": "34000/50000",
                        "red_y": "16000/50000",
                        "green_x": "13250/50000",
                        "green_y": "34500/50000",
                        "blue_x": "7500/50000",
                        "blue_y": "3000/50000",
                        "white_point_x": "15635/50000",
                        "white_point_y": "16450/50000",
                        "min_luminance": "50/10000",
                        "max_luminance": "10000000/10000",
                    },
                    {
                        "side_data_type": "Content light level metadata",
                        "max_content": 1000,
                        "max_average": 400,
                    },
                ],
            }
        ],
        "format": {
            "duration": "7200.000000",
        },
    }


@pytest.fixture
def tmp_project_dir():
    """Create a temporary directory for project output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
