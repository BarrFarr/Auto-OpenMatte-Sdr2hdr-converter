"""Compatibility exports for the neutral resampling contract.

The interface lives in ``auto_openmatte.backends``.  The V5 CUDA implementation
continues to live in ``v05_gpu_native.GpuOutputResampler`` and is selected by
the CUDA adapter without changing its kernel or no-op behavior.
"""

from auto_openmatte.backends import ResampleBackend, ResampleRequest

__all__ = ["ResampleBackend", "ResampleRequest"]
