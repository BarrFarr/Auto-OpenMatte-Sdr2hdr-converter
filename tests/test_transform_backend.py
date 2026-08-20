"""Focused validation for the optional float64 transform backends."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from auto_openmatte.core.models import GeometryModel, ShotTransform
from auto_openmatte.pipeline.compose import composite_extend
from auto_openmatte.processing.transform import apply_shot_transform
from auto_openmatte.processing.transform_backend import (
    BackendRuntimeError,
    BackendUnavailableError,
    CPUTransformBackend,
    FallbackTransformBackend,
    GPUTransformBackend,
    create_transform_backend,
)

CURVE = [
    [-2.0, -1.5],
    [0.0, 0.3],
    [2.0, 2.0],
    [4.0, 4.0],
]

_REAL_OM_SOURCES = (
    Path(
        r"G:\Filmy\IMAX format (open matte)\Blade Runner 2049  Open Matte 2160p"
        r"\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
    ),
    Path(
        r"G:\Filmy\Blade.Runner.2049.2017.2160p.UHD.Bluray.x265.HDR10.HEVC.DTS-HDMA.5.1-4K4U"
        r"\Blade.Runner.2049.2017.Open.Matte.RUS.SDR.2160p.mkv"
    ),
)


def _load_real_material_roi() -> np.ndarray:
    source = next((path for path in _REAL_OM_SOURCES if path.exists()), None)
    if source is None:
        pytest.skip("configured real-material Open Matte source is unavailable")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is unavailable for the real-material ROI test")

    om_time = 2715.0 + 1167.0 / 23.976
    command = [
        "ffmpeg",
        "-v",
        "quiet",
        "-nostdin",
        "-ss",
        f"{om_time:.6f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb48le",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"real-material frame extraction failed: {exc}")

    expected = 3840 * 2160 * 3 * 2
    if result.returncode != 0 or len(result.stdout) < expected:
        pytest.skip("ffmpeg did not return one complete real-material frame")
    frame = np.frombuffer(result.stdout[:expected], dtype=np.uint16).reshape(
        2160, 3840, 3
    )
    frame = frame.astype(np.float64) / 65535.0
    return np.concatenate((frame[:280], frame[1880:]), axis=0)


def _make_transform(
    *,
    color_matrix: list[list[float]] | None = None,
    saturation: float = 1.0,
) -> ShotTransform:
    return ShotTransform(
        shot_id=220,
        luminance_curve=CURVE,
        color_matrix=color_matrix
        or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        saturation=saturation,
    )


def _neutral_frame(target_linear: np.ndarray) -> np.ndarray:
    """Create neutral BT.709 signal pixels for selected linear luminances."""
    signal = np.power(np.asarray(target_linear, dtype=np.float64), 1.0 / 2.4)
    return np.repeat(signal.reshape(-1, 1), 3, axis=1).reshape(-1, 1, 3)


def _gpu_pair(
    transform: ShotTransform,
) -> tuple[GPUTransformBackend, object]:
    backend = GPUTransformBackend()
    try:
        workspace = backend.prepare_shot(transform)
    except BackendUnavailableError as exc:
        pytest.skip(f"CUDA/CuPy unavailable: {exc}")
    return backend, workspace


def _assert_cpu_gpu_parity(
    frame: np.ndarray,
    transform: ShotTransform,
) -> tuple[np.ndarray, np.ndarray, GPUTransformBackend, object]:
    cpu = apply_shot_transform(frame, transform)
    gpu_backend, workspace = _gpu_pair(transform)
    gpu = gpu_backend.transform_roi(frame, transform, workspace=workspace)
    np.testing.assert_allclose(cpu, gpu, rtol=0.0, atol=1e-5)
    return cpu, gpu, gpu_backend, workspace


def test_cpu_is_the_default_backend() -> None:
    backend = create_transform_backend()
    assert isinstance(backend, CPUTransformBackend)
    assert backend.name == "cpu"


@pytest.mark.parametrize(
    ("label", "target_linear"),
    [
        ("black", 0.0),
        ("near_black_bridge", 1e-8),
        ("curve_start", 1e-6),
        ("midtones", 1e-3),
        ("p99", 9e-3),
        ("p99_9", 1.1e-2),
        ("above_curve_max", 0.1),
    ],
)
def test_cpu_gpu_edge_case_parity(label: str, target_linear: float) -> None:
    """The isolated GPU path matches each required luminance branch."""
    del label
    frame = _neutral_frame(np.array([target_linear]))
    cpu, gpu, _, _ = _assert_cpu_gpu_parity(frame, _make_transform())
    if target_linear == 0.0:
        np.testing.assert_allclose(cpu, gpu, rtol=0.0, atol=1e-12)
        assert float(np.max(cpu)) < 1e-5


def test_cpu_gpu_material_like_roi_parity() -> None:
    """A varied, textured ROI exercises all channels and luminance ranges."""
    yy, xx = np.mgrid[0:48, 0:64]
    base = 0.025 + 0.55 * (xx / 63.0) * (0.35 + 0.65 * yy / 47.0)
    texture = 0.035 * np.sin(xx * 0.47) * np.cos(yy * 0.31)
    roi = np.stack(
        [
            np.clip(base + texture, 0.0, 1.0),
            np.clip(base * 0.82 - texture * 0.4, 0.0, 1.0),
            np.clip(base * 0.63 + texture * 0.7, 0.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float64)
    _assert_cpu_gpu_parity(roi, _make_transform())


def test_cpu_gpu_real_material_roi_parity() -> None:
    """Compare the backends on one real Open Matte extension ROI when available."""
    roi = _load_real_material_roi()
    _assert_cpu_gpu_parity(roi, _make_transform())


def test_cpu_gpu_non_identity_color_and_saturation_parity() -> None:
    transform = _make_transform(
        color_matrix=[
            [1.02, -0.01, 0.00],
            [0.01, 0.98, 0.01],
            [0.00, 0.02, 0.97],
        ],
        saturation=1.12,
    )
    frame = np.array(
        [
            [[0.04, 0.16, 0.32], [0.28, 0.52, 0.76]],
            [[0.02, 0.41, 0.18], [0.91, 0.63, 0.12]],
        ],
        dtype=np.float64,
    )
    _assert_cpu_gpu_parity(frame, transform)


def test_gpu_workspace_reuses_luts_and_shape_buffers() -> None:
    transform = _make_transform()
    frame = np.full((6, 8, 3), 0.22, dtype=np.float64)
    backend, workspace = _gpu_pair(transform)
    first = backend.transform_roi(frame, transform, workspace=workspace)
    buffers = workspace.buffers
    second = backend.transform_roi(frame * 0.8, transform, workspace=workspace)

    assert first.shape == second.shape == frame.shape
    assert workspace.buffers is buffers
    assert workspace.metadata["workspace_allocations"] == 1
    assert workspace.metadata["curve_lut_built_once"] is True
    assert workspace.metadata["curve_lut_copied_once"] is True
    assert workspace.metadata["pq_lut_copied_once_per_backend"] is True
    assert workspace.curve_lut_gpu.dtype == np.float64
    assert workspace.pq_lut_gpu.dtype == np.float64


def test_hdr_overlap_is_not_transformed_by_backend() -> None:
    transform = _make_transform()
    hdr = np.full((4, 6, 3), 0.73, dtype=np.float64)
    om = np.linspace(0.01, 0.8, 8 * 6 * 3, dtype=np.float64).reshape(8, 6, 3)
    geometry = GeometryModel(overlap_bbox=[0.0, 2.0, 6.0, 6.0])
    extension_mask = np.zeros((8, 6), dtype=np.float64)

    cpu_result = composite_extend(hdr, om, transform, geometry, extension_mask)
    gpu_backend, workspace = _gpu_pair(transform)
    gpu_result = composite_extend(
        hdr,
        om,
        transform,
        geometry,
        extension_mask,
        backend=gpu_backend,
        backend_workspace=workspace,
    )

    np.testing.assert_array_equal(cpu_result[2:6], hdr)
    np.testing.assert_array_equal(gpu_result[2:6], hdr)
    np.testing.assert_allclose(cpu_result, gpu_result, rtol=0.0, atol=1e-5)


def test_unavailable_gpu_falls_back_and_stays_on_cpu() -> None:
    transform = _make_transform()
    frame = np.full((3, 4, 3), 0.2, dtype=np.float64)
    expected = apply_shot_transform(frame, transform)
    backend = create_transform_backend(
        experimental_gpu=True,
        cupy_loader=lambda: None,
    )

    result = backend.transform_roi(frame, transform)
    second = backend.transform_roi(frame * 0.5, transform)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(second, apply_shot_transform(frame * 0.5, transform))
    assert backend.using_gpu is False
    assert backend.name == "cpu-fallback"
    assert backend.fallback_reason is not None


class _RuntimeFailingBackend:
    name = "gpu"

    def transform_roi(self, *args: object, **kwargs: object) -> np.ndarray:
        raise BackendRuntimeError("injected runtime failure")



def test_runtime_gpu_failure_falls_back_stickily() -> None:
    transform = _make_transform()
    frame = np.full((2, 2, 3), 0.3, dtype=np.float64)
    backend = FallbackTransformBackend(_RuntimeFailingBackend())

    result = backend.transform_roi(frame, transform)
    expected = apply_shot_transform(frame, transform)

    np.testing.assert_array_equal(result, expected)
    assert backend.using_gpu is False
    assert backend.fallback_reason == "injected runtime failure"
    np.testing.assert_array_equal(backend.transform_roi(frame, transform), expected)


def test_gpu_prepare_rejects_unsupported_transfer_without_changing_cpu() -> None:
    transform = _make_transform()
    cpu = apply_shot_transform(
        np.full((2, 2, 3), 0.2, dtype=np.float64),
        transform,
        sdr_transfer="bt1886",
    )
    backend = GPUTransformBackend()
    try:
        backend.prepare_shot(transform, sdr_transfer="bt1886")
    except BackendUnavailableError:
        pass
    else:
        pytest.fail("unsupported GPU transfer pair was accepted")
    assert cpu.shape == (2, 2, 3)

