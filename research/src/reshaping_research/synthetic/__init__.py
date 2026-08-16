"""Synthetic scene generation for controlled reshaping experiments."""

from __future__ import annotations

from .scene_generator import generate_scene, SceneType
from .sdr_simulator import simulate_sdr
from .open_matte_simulator import create_open_matte_pair

__all__ = [
    "generate_scene",
    "SceneType",
    "simulate_sdr",
    "create_open_matte_pair",
]
