"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def ffprobe_hdr10() -> dict:
    """ffprobe output for an HDR10 source."""
    with open(FIXTURES_DIR / "ffprobe_hdr10.json") as f:
        return json.load(f)


@pytest.fixture
def ffprobe_hlg() -> dict:
    """ffprobe output for an HLG source."""
    with open(FIXTURES_DIR / "ffprobe_hlg.json") as f:
        return json.load(f)


@pytest.fixture
def ffprobe_sdr_openmatte() -> dict:
    """ffprobe output for an SDR Open Matte source."""
    with open(FIXTURES_DIR / "ffprobe_sdr_openmatte.json") as f:
        return json.load(f)


@pytest.fixture
def ffprobe_multi_stream() -> dict:
    """ffprobe output for a multi-stream container."""
    with open(FIXTURES_DIR / "ffprobe_multi_stream.json") as f:
        return json.load(f)
