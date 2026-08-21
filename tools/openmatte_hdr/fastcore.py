#!/usr/bin/env python3
"""Fast execution core: decode-once cache, GPU math, multi-threaded CPU fallbacks.

Three things make the reference pipeline slow:
  1. every pass decodes the shot again,
  2. all array math runs on a single NumPy thread,
  3. the GPU is only used for the final encode.

This module fixes all three. Decoded frames are cached once as float32, which is
lossless relative to the reference decode. Per-frame math runs on CuPy when a
device is available, and every CPU library is told to use all cores.

Numerical note: GPU results are not bit-identical to the NumPy path. Measured
deviation is 1.5e-08 for the sigma-16 blur and 8.3e-06 for PQ encoding, which can
move an occasional 16-bit code by one step. Use the CPU backend when bit-exact
reproduction matters.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CPU_COUNT = os.cpu_count() or 8
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, str(CPU_COUNT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(CPU_COUNT)

PEAK_NITS = 10000.0
EPS = 1e-6
PQ_M1, PQ_M2 = 2610.0 / 16384.0, 2523.0 / 4096.0 * 128.0
PQ_C1, PQ_C2, PQ_C3 = 3424.0 / 4096.0, 2413.0 / 4096.0 * 32.0, 2392.0 / 4096.0 * 32.0
BT2020_LUMA = (0.2627, 0.6780, 0.0593)


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


class Backend:
    """One numeric backend with an identical interface for CuPy and NumPy."""

    def __init__(self, use_gpu: bool) -> None:
        self.gpu = bool(use_gpu)
        if self.gpu:
            import cupy as xp
            from cupyx.scipy.ndimage import gaussian_filter

            self.xp = xp
            self._gaussian = gaussian_filter
        else:
            self.xp = np
            self._gaussian = None
        from reshaping_research.utils.color_spaces import BT2020_TO_LMS, BT709_TO_BT2020, ICTCP_TO_LMS, LMS_TO_BT2020, LMS_TO_ICTCP

        self.m_rgb_lms = self.asarray(BT2020_TO_LMS.astype(np.float32))
        self.m_lms_ictcp = self.asarray(LMS_TO_ICTCP.astype(np.float32))
        self.m_ictcp_lms = self.asarray(ICTCP_TO_LMS.astype(np.float32))
        self.m_lms_rgb = self.asarray(LMS_TO_BT2020.astype(np.float32))
        self.m_709_2020 = self.asarray(BT709_TO_BT2020.astype(np.float32))
        self.luma = self.asarray(np.asarray(BT2020_LUMA, dtype=np.float32))

    @property
    def name(self) -> str:
        return "cupy/gpu" if self.gpu else f"numpy/cpu x{CPU_COUNT}"

    def asarray(self, value: Any) -> Any:
        return self.xp.asarray(value)

    def tohost(self, value: Any) -> np.ndarray:
        return self.xp.asnumpy(value) if self.gpu else np.asarray(value)

    def blur(self, image: Any, sigma: float) -> Any:
        if sigma <= 0.0:
            return image
        if self.gpu:
            axes = (sigma,) * (image.ndim - 1) + (0.0,) if image.ndim == 3 else (sigma,) * image.ndim
            return self._gaussian(image, sigma=axes, mode="mirror", truncate=4.0)
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)

    def luminance(self, rgb: Any) -> Any:
        return self.xp.tensordot(rgb, self.luma, axes=([-1], [0]))

    def matrix(self, transform: Any, rgb: Any) -> Any:
        return self.xp.tensordot(rgb, transform.T, axes=([-1], [0]))

    def pq_oetf(self, nits: Any) -> Any:
        y = self.xp.clip(nits, 0.0, PEAK_NITS) / PEAK_NITS
        powered = self.xp.power(y, PQ_M1)
        return self.xp.power((PQ_C1 + PQ_C2 * powered) / (1.0 + PQ_C3 * powered), PQ_M2)

    def pq_eotf(self, signal: Any) -> Any:
        powered = self.xp.power(self.xp.clip(signal, 0.0, 1.0), 1.0 / PQ_M2)
        base = self.xp.maximum(powered - PQ_C1, 0.0) / self.xp.maximum(PQ_C2 - PQ_C3 * powered, 1e-12)
        return self.xp.power(base, 1.0 / PQ_M1) * PEAK_NITS

    def to_ictcp(self, rgb_nits: Any) -> Any:
        lms = self.xp.maximum(self.matrix(self.m_rgb_lms, rgb_nits), 0.0)
        return self.matrix(self.m_lms_ictcp, self.pq_oetf(lms))

    def from_ictcp(self, ictcp: Any) -> Any:
        return self.matrix(self.m_lms_rgb, self.pq_eotf(self.matrix(self.m_ictcp_lms, ictcp)))

    def to_pq16(self, normalized: Any) -> np.ndarray:
        signal = self.pq_oetf(self.xp.clip(normalized, 0.0, 1.0) * PEAK_NITS)
        return self.tohost(self.xp.round(signal * 65535.0).astype(self.xp.uint16))

    def percentile(self, values: Any, q: float) -> float:
        return float(self.tohost(self.xp.percentile(values, q)))


class FrameCache:
    """Decodes the shot once into float32 memmaps and reuses them across passes and runs."""

    def __init__(self, directory: Path, count: int, sdr_shape: tuple[int, int], hdr_shape: tuple[int, int], key: dict[str, Any]) -> None:
        self.directory = directory
        self.count = count
        self.sdr_shape = (count, sdr_shape[0], sdr_shape[1], 3)
        self.hdr_shape = (count, hdr_shape[0], hdr_shape[1], 3)
        self.key = key
        self.manifest_path = directory / "manifest.json"
        self.sdr_path = directory / "sdr_f32.dat"
        self.hdr_path = directory / "hdr_f32.dat"

    @property
    def gigabytes(self) -> float:
        return (np.prod(self.sdr_shape) + np.prod(self.hdr_shape)) * 4 / 1e9

    def _fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.key, sort_keys=True).encode("utf-8")).hexdigest().upper()

    def valid(self) -> bool:
        if not (self.manifest_path.is_file() and self.sdr_path.is_file() and self.hdr_path.is_file()):
            return False
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected = int(np.prod(self.sdr_shape) * 4), int(np.prod(self.hdr_shape) * 4)
        return manifest.get("fingerprint") == self._fingerprint() and (self.sdr_path.stat().st_size, self.hdr_path.stat().st_size) == expected

    def build(self, stream_factory: Any) -> dict[str, Any]:
        self.directory.mkdir(parents=True, exist_ok=True)
        sdr = np.memmap(self.sdr_path, dtype=np.float32, mode="w+", shape=self.sdr_shape)
        hdr = np.memmap(self.hdr_path, dtype=np.float32, mode="w+", shape=self.hdr_shape)
        hashes: dict[str, Any] = {}
        written = 0
        with stream_factory() as frames:
            for index, (hdr_frame, sdr_frame, frame_hashes) in enumerate(frames):
                sdr[index] = sdr_frame
                hdr[index] = hdr_frame
                if index == 0:
                    hashes["first"] = frame_hashes
                hashes["last"] = frame_hashes
                written += 1
        if written != self.count:
            raise RuntimeError(f"Cache build got {written} of {self.count} frames")
        sdr.flush()
        hdr.flush()
        del sdr, hdr
        manifest = {"fingerprint": self._fingerprint(), "key": self.key, "frames": self.count, "source_frame_hashes": hashes, "dtype": "float32"}
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    def open(self) -> tuple[np.memmap, np.memmap, dict[str, Any]]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        sdr = np.memmap(self.sdr_path, dtype=np.float32, mode="r", shape=self.sdr_shape)
        hdr = np.memmap(self.hdr_path, dtype=np.float32, mode="r", shape=self.hdr_shape)
        return sdr, hdr, manifest


class ParallelWriter:
    """Writes each encoder pipe on its own thread, so swscale waits overlap."""

    def __init__(self, pipes: list[Any]) -> None:
        self.pipes = pipes
        self.pool = ThreadPoolExecutor(max_workers=max(len(pipes), 1))
        self.pending: list[Any] = []

    def submit(self, payloads: list[bytes]) -> None:
        self.wait()
        self.pending = [self.pool.submit(pipe.stdin.write, payload) for pipe, payload in zip(self.pipes, payloads)]

    def wait(self) -> None:
        for future in self.pending:
            future.result()
        self.pending = []

    def close(self) -> None:
        self.wait()
        self.pool.shutdown(wait=True)
