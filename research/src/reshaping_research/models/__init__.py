"""Reshaping models - 7 candidate approaches for scene-based SDR-to-HDR conversion.

All models follow a common interface:
- fit(sdr_linear, hdr_linear, sdr_chroma=None, hdr_chroma=None)
- apply(sdr_image_linear, params=None)
- param_count() -> int
- name() -> str
"""

from __future__ import annotations

from typing import List

from .model_a_linear import ModelALinear
from .model_b_piecewise import ModelBPiecewise
from .model_c_polynomial import ModelCPolynomial
from .model_d_cdf import ModelDCdf
from .model_e_cdf_regularized import ModelECdfRegularized
from .model_f_luma_chroma_regression import ModelFLumaChromaRegression
from .model_g_hybrid import ModelGHybrid

__all__ = [
    "ModelALinear",
    "ModelBPiecewise",
    "ModelCPolynomial",
    "ModelDCdf",
    "ModelECdfRegularized",
    "ModelFLumaChromaRegression",
    "ModelGHybrid",
    "get_all_models",
]


def get_all_models() -> List[object]:
    """Return instances of all candidate reshaping models.

    Returns:
        List of model instances, each implementing the common interface.
    """
    return [
        ModelALinear(),
        ModelBPiecewise(),
        ModelCPolynomial(),
        ModelDCdf(),
        ModelECdfRegularized(),
        ModelFLumaChromaRegression(),
        ModelGHybrid(),
    ]
