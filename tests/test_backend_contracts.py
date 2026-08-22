"""Regression tests for the neutral common backend architecture."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from auto_openmatte.backends import (
    Frame,
    FrameBatch,
    FrameMemory,
    FrameTimestamps,
    MemoryDomain,
    Ownership,
)
from auto_openmatte.backends import ResampleRequest as NeutralResampleRequest
from tools.openmatte_hdr.cuda_backend import (
    CUDA_CAPABILITIES,
    CudaComputeBackend,
    CudaDecoderBackend,
    CudaEncoderBackend,
    CudaResampleBackend,
    make_cuda_frame,
)
from tools.openmatte_hdr.resample_backend import ResampleBackend, ResampleRequest


def _frame(*, domain: MemoryDomain = MemoryDomain.HOST) -> Frame:
    return Frame(
        width=4,
        height=2,
        pixel_format="RGB32F",
        bit_depth=32,
        color_range="full",
        primaries="BT.2020",
        transfer="linear",
        matrix="RGB",
        timestamps=FrameTimestamps(pts=7, time_base=(1, 24)),
        memory_domain=domain,
        backend="test",
        device=None,
        ownership=Ownership.OWNED,
        memory=FrameMemory(
            payload=np.zeros((2, 4, 3), dtype=np.float32),
            domain=domain,
            nbytes=96,
            ownership=Ownership.OWNED,
        ),
    )


def test_frame_contract_carries_all_neutral_metadata_and_lifecycle() -> None:
    releases: list[object | None] = []
    memory = FrameMemory(
        payload=object(),
        domain=MemoryDomain.HOST,
        nbytes=12,
        ownership=Ownership.BORROWED,
        release_callback=releases.append,
    )
    frame = Frame(
        width=1920,
        height=800,
        pixel_format="RGB32F",
        bit_depth=32,
        color_range="full",
        primaries="BT.2020",
        transfer="linear",
        matrix="RGB",
        timestamps=FrameTimestamps(pts=10, dts=9, duration=1, time_base=(1001, 24000)),
        memory_domain=MemoryDomain.HOST,
        backend="test",
        device="host",
        ownership=Ownership.BORROWED,
        memory=memory,
    )

    assert frame.as_dict()["width"] == 1920
    assert frame.as_dict()["timestamps"]["time_base"] == [1001, 24000]
    assert frame.view() is memory.payload
    frame.release("consumer-context")
    frame.release("ignored")
    assert releases == ["consumer-context"]
    assert frame.released is True


def test_frame_batch_wait_release_and_capacity_credit_are_bounded() -> None:
    waits: list[object | None] = []
    releases: list[object | None] = []
    credits: list[str] = []
    batch = FrameBatch(
        frames=(_frame(),),
        sequence=3,
        release_callback=releases.append,
        metadata={"wait_ready": waits.append},
    )
    batch.attach_release_credit(lambda: credits.append("returned"))

    batch.wait_ready("compute-context")
    batch.release("consumer-context")
    batch.release("ignored")

    assert waits == ["compute-context"]
    assert releases == ["consumer-context"]
    assert credits == ["returned"]
    assert batch.frames[0].released is True
    assert batch.released is True


def test_neutral_resample_contract_is_the_compatibility_export() -> None:
    assert ResampleRequest is NeutralResampleRequest
    no_op = ResampleRequest(8, 4, 8, 4)
    active = ResampleRequest(8, 4, 4, 2)
    assert no_op.resample_active is False
    assert active.resample_active is True


def test_cuda_capability_report_declares_only_the_existing_backend() -> None:
    report = CUDA_CAPABILITIES.as_dict()

    assert report == {
        "backend": "cuda-v05",
        "supported_codecs": ["h264", "hevc"],
        "supported_pixel_formats": ["NV12", "P010LE", "RGB32F"],
        "bit_depths": [8, 10, 32],
        "hardware_decode": True,
        "hardware_encode": True,
        "resampling": True,
        "zero_copy": True,
        "device_memory_type": "cuda-device",
    }
    assert "amd" not in report["backend"].lower()
    assert "cpu" not in report["backend"].lower()


def test_cuda_compute_facade_delegates_without_changing_numeric_backend() -> None:
    class FakeFastcore:
        gpu = True
        xp = object()

        def asarray(self, value: Any) -> tuple[str, Any]:
            return ("asarray", value)

        def tohost(self, value: Any) -> tuple[str, Any]:
            return ("tohost", value)

    implementation = FakeFastcore()
    backend = CudaComputeBackend(implementation)

    assert backend.name == "cuda-v05-compute"
    assert backend.xp is implementation.xp
    assert backend.asarray("device-value") == ("asarray", "device-value")
    assert backend.tohost("device-value") == ("tohost", "device-value")


def test_cuda_frame_wrapper_is_zero_copy_and_declares_p010() -> None:
    payload = np.zeros((3, 4), dtype=np.uint16)
    frame = make_cuda_frame(
        payload,
        width=4,
        height=2,
        pixel_format="P010LE",
        bit_depth=10,
        pts=11,
        device=0,
    )

    assert frame.view() is payload
    assert frame.memory_domain is MemoryDomain.DEVICE
    assert frame.pixel_format == "P010LE"
    assert frame.bit_depth == 10
    assert frame.timestamps.pts == 11
    assert frame.memory.nbytes == payload.nbytes


@dataclass
class _NativeFrame:
    array: Any
    sequence: int
    software_format: int

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])


@dataclass
class _NativePair:
    sequence: int
    hdr: _NativeFrame
    sdr: _NativeFrame
    ready_event: object


class _DecoderSource:
    def __init__(self) -> None:
        self.config = SimpleNamespace(fps_text="24000/1001")
        self.device_index = 0
        self.count = 1
        self.hdr_decoder = "hevc_cuvid"
        self.om_decoder = "h264_cuvid"
        self.ring_size = 4
        self._pair = _NativePair(
            sequence=0,
            hdr=_NativeFrame(np.zeros((2, 4, 3), dtype=np.float32), 0, 158),
            sdr=_NativeFrame(np.ones((2, 4, 3), dtype=np.float32), 0, 23),
            ready_event=object(),
        )
        self.released: list[tuple[object, object | None]] = []
        self.opened = False
        self.closed = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {"source": "fake-cuda"}

    @property
    def lifecycle(self) -> dict[str, Any]:
        return {"in_flight": 0, "ring_size": self.ring_size}

    def open_pairs(self) -> None:
        self.opened = True

    def next_pair(self) -> _NativePair | None:
        return self._pair if not self.released else None

    def release_pair(self, pair: _NativePair, context: object | None = None) -> None:
        self.released.append((pair, context))

    def close_pairs(self) -> None:
        self.closed = True


def test_cuda_decoder_facade_preserves_format_metadata_and_release() -> None:
    source = _DecoderSource()
    decoder = CudaDecoderBackend(source)

    decoder.open()
    batch = decoder.read()
    assert batch is not None
    assert batch.sequence == 0
    assert batch.frames[0].view() is source._pair.hdr.array
    assert batch.frames[1].view() is source._pair.sdr.array
    assert batch.frames[0].metadata["source_pixel_format"] == "P010LE"
    assert batch.frames[1].metadata["source_pixel_format"] == "NV12"
    assert batch.frames[0].memory_domain is MemoryDomain.DEVICE
    assert batch.frames[0].timestamps.time_base == (1001, 24000)

    context = object()
    batch.release(context)
    decoder.close()
    assert source.released == [(source._pair, context)]
    assert batch.frames[0].released is True
    assert batch.frames[1].released is True
    assert source.closed is True


def test_cuda_resample_facade_preserves_replaceable_call_boundary() -> None:
    class FakeResampler:
        def process(self, frame: Any, request: ResampleRequest, stream: object) -> tuple[Any, ResampleRequest, object]:
            return frame, request, stream

    backend: ResampleBackend = CudaResampleBackend(FakeResampler())
    request = ResampleRequest(8, 4, 4, 2)
    stream = object()
    assert backend.process("device-rgb", request, stream) == (
        "device-rgb",
        request,
        stream,
    )
    assert backend.capabilities.zero_copy is True


def test_cuda_encoder_facade_consumes_neutral_device_frame_and_lifecycle() -> None:
    class FakeEncoder:
        def __init__(self) -> None:
            self.writes: list[tuple[Any, int]] = []
            self.closed = False
            self.aborted = False

        def write(self, payload: Any, pts: int) -> None:
            self.writes.append((payload, pts))

        def close(self) -> None:
            self.closed = True

        def abort(self) -> None:
            self.aborted = True

    implementation = FakeEncoder()
    encoder = CudaEncoderBackend()
    encoder._implementation = implementation
    frame = make_cuda_frame(
        np.zeros((3, 4), dtype=np.uint16),
        width=4,
        height=2,
        pixel_format="P010LE",
        bit_depth=10,
        pts=5,
        device=0,
    )

    encoder.write(frame, 5)
    encoder.close()
    assert implementation.writes == [(frame.view(), 5)]
    assert implementation.closed is True
    assert implementation.aborted is False
