"""Unit tests for the replaceable device-side resample contract."""

from __future__ import annotations

from typing import Any

import pytest

from tools.openmatte_hdr.resample_backend import ResampleBackend, ResampleRequest


class _RecordingBackend:
    """Small device-array stand-in used to validate the protocol boundary."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ResampleRequest, Any]] = []

    def process(self, rgb: Any, request: ResampleRequest, stream: Any) -> Any:
        self.calls.append((rgb, request, stream))
        return {"rgb": rgb, "request": request, "stream": stream}


def test_noop_request_is_exactly_inactive() -> None:
    request = ResampleRequest(
        source_width=1920,
        source_height=800,
        target_width=1920,
        target_height=800,
    )

    assert request.resample_active is False
    assert request.source_geometry == request.target_geometry
    assert request.as_dict()["resample_active"] is False


def test_active_request_preserves_geometry_format_and_method_contract() -> None:
    request = ResampleRequest(
        source_width=3840,
        source_height=1600,
        target_width=1920,
        target_height=800,
        source_format="RGB32F",
        target_format="RGB32F",
        method="INTER_LINEAR",
    )
    backend: ResampleBackend = _RecordingBackend()
    stream = object()
    result = backend.process("device-rgb", request, stream)

    assert request.resample_active is True
    assert request.source_geometry == (3840, 1600)
    assert request.target_geometry == (1920, 800)
    assert result["request"] is request
    assert result["stream"] is stream
    assert len(backend.calls) == 1


def test_request_rejects_invalid_geometry_and_missing_method() -> None:
    with pytest.raises(ValueError, match="source_width"):
        ResampleRequest(0, 1600, 1920, 800)
    with pytest.raises(ValueError, match="method"):
        ResampleRequest(3840, 1600, 1920, 800, method="")
