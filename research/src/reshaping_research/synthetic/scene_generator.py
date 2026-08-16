"""Synthetic HDR scene generator - 6 scene types for controlled testing.

Generates small HDR images in linear BT.2020 color space with known
luminance characteristics for evaluating reshaping algorithms.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


class SceneType(Enum):
    """Scene type identifiers for synthetic generation."""

    NEUTRAL = "neutral"
    HIGH_KEY = "high_key"
    LOW_KEY = "low_key"
    COLORFUL = "colorful"
    MIXED = "mixed"
    DIFFICULT = "difficult"


def generate_scene(
    scene_type: SceneType,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> NDArray[np.floating]:
    """Generate a synthetic HDR scene in linear BT.2020 RGB (nits).

    Args:
        scene_type: Type of scene to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        seed: Random seed for deterministic generation.

    Returns:
        HDR image array of shape (height, width, 3) in linear BT.2020 nits.
        Luminance range depends on scene type (typically 0 to 4000-10000 nits).
    """
    rng = np.random.default_rng(seed)

    generators = {
        SceneType.NEUTRAL: _generate_neutral,
        SceneType.HIGH_KEY: _generate_high_key,
        SceneType.LOW_KEY: _generate_low_key,
        SceneType.COLORFUL: _generate_colorful,
        SceneType.MIXED: _generate_mixed,
        SceneType.DIFFICULT: _generate_difficult,
    }

    return generators[scene_type](height, width, rng)


def _generate_neutral(
    height: int, width: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """Neutral: uniform mid-gray gradients, moderate contrast, no color extremes.

    Luminance range: approximately 10-500 nits.
    """
    # Horizontal luminance gradient
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Base luminance: smooth gradient from 10 to 500 nits
    base_lum = 10.0 + 490.0 * (xx * 0.6 + yy * 0.4)

    # Add subtle noise for texture
    noise = rng.normal(0.0, 5.0, (height, width))
    base_lum = np.maximum(base_lum + noise, 0.1)

    # Near-neutral color: slight warm tint
    img = np.zeros((height, width, 3), dtype=np.float64)
    img[..., 0] = base_lum * 1.02  # Slightly warm
    img[..., 1] = base_lum * 1.00
    img[..., 2] = base_lum * 0.98

    return np.maximum(img, 0.0)


def _generate_high_key(
    height: int, width: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """High-key: bright overall, simulating sky/bright surfaces.

    Highlights reach 2000-4000 nits. Overall bright scene.
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Sky-like gradient: brighter at top
    sky_lum = 2000.0 + 2000.0 * (1.0 - yy) * np.cos(xx * np.pi * 0.5)

    # Add some cloud-like structure
    cloud = rng.normal(0.0, 200.0, (height, width))
    sky_lum = np.maximum(sky_lum + cloud, 100.0)

    # Blue sky tint
    img = np.zeros((height, width, 3), dtype=np.float64)
    img[..., 0] = sky_lum * 0.85
    img[..., 1] = sky_lum * 0.92
    img[..., 2] = sky_lum * 1.15

    # Bright surface region (bottom half)
    surface_mask = yy > 0.6
    surface_lum = 800.0 + 400.0 * xx
    img[surface_mask, 0] = surface_lum[surface_mask] * 1.05
    img[surface_mask, 1] = surface_lum[surface_mask] * 1.00
    img[surface_mask, 2] = surface_lum[surface_mask] * 0.90

    return np.maximum(img, 0.0)


def _generate_low_key(
    height: int, width: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """Low-key: dark scene, minimal highlights, shadows dominate.

    Most values under 50 nits, small specular highlights up to 500 nits.
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Dark base: 1-30 nits
    base_lum = 1.0 + 29.0 * (0.3 + 0.7 * np.sin(xx * np.pi) * np.sin(yy * np.pi))

    # Add shadow texture
    noise = rng.normal(0.0, 2.0, (height, width))
    base_lum = np.maximum(base_lum + noise, 0.1)

    # Small specular highlights (point lights)
    n_highlights = 5
    for _ in range(n_highlights):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        radius = rng.integers(3, 10)
        dist = np.sqrt((xx * width - cx) ** 2 + (yy * height - cy) ** 2)
        highlight = 500.0 * np.exp(-dist**2 / (2.0 * radius**2))
        base_lum += highlight

    # Near-neutral, slightly cool
    img = np.zeros((height, width, 3), dtype=np.float64)
    img[..., 0] = base_lum * 0.95
    img[..., 1] = base_lum * 1.00
    img[..., 2] = base_lum * 1.05

    return np.maximum(img, 0.0)


def _generate_colorful(
    height: int, width: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """Colorful: saturated reds, greens, blues at various luminance levels.

    Tests gamut handling with highly saturated primaries.
    Luminance range: 20-2000 nits.
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    img = np.zeros((height, width, 3), dtype=np.float64)

    # Divide into color bands with varying luminance
    band_height = height // 6

    # Band 1: Saturated reds (BT.2020 primary red)
    r1 = slice(0, band_height)
    lum = 100.0 + 400.0 * xx[r1, :]
    img[r1, :, 0] = lum * 3.0  # Highly saturated red
    img[r1, :, 1] = lum * 0.1
    img[r1, :, 2] = lum * 0.05

    # Band 2: Saturated greens
    r2 = slice(band_height, 2 * band_height)
    lum = 200.0 + 800.0 * xx[r2, :]
    img[r2, :, 0] = lum * 0.1
    img[r2, :, 1] = lum * 2.5
    img[r2, :, 2] = lum * 0.1

    # Band 3: Saturated blues
    r3 = slice(2 * band_height, 3 * band_height)
    lum = 50.0 + 200.0 * xx[r3, :]
    img[r3, :, 0] = lum * 0.05
    img[r3, :, 1] = lum * 0.1
    img[r3, :, 2] = lum * 4.0

    # Band 4: Cyan
    r4 = slice(3 * band_height, 4 * band_height)
    lum = 300.0 + 700.0 * xx[r4, :]
    img[r4, :, 0] = lum * 0.1
    img[r4, :, 1] = lum * 1.8
    img[r4, :, 2] = lum * 2.0

    # Band 5: Magenta
    r5 = slice(4 * band_height, 5 * band_height)
    lum = 80.0 + 300.0 * xx[r5, :]
    img[r5, :, 0] = lum * 2.5
    img[r5, :, 1] = lum * 0.1
    img[r5, :, 2] = lum * 2.5

    # Band 6: Yellow
    r6 = slice(5 * band_height, height)
    lum = 500.0 + 1500.0 * xx[r6, :]
    img[r6, :, 0] = lum * 1.8
    img[r6, :, 1] = lum * 1.7
    img[r6, :, 2] = lum * 0.1

    # Add some noise for texture
    noise = rng.normal(0.0, 10.0, (height, width, 3))
    img = np.maximum(img + noise, 0.0)

    return img


def _generate_mixed(
    height: int, width: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """Mixed: combination of skin tones, sky, vegetation, metal.

    Realistic content variety. Luminance range: 5-3000 nits.
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    img = np.zeros((height, width, 3), dtype=np.float64)

    # Sky region (top 30%)
    sky_mask = yy < 0.3
    sky_lum = 1500.0 + 1000.0 * (0.3 - yy[sky_mask]) / 0.3
    img[sky_mask, 0] = sky_lum * 0.7
    img[sky_mask, 1] = sky_lum * 0.85
    img[sky_mask, 2] = sky_lum * 1.3

    # Vegetation (30-50%)
    veg_mask = (yy >= 0.3) & (yy < 0.5)
    veg_lum = 100.0 + 200.0 * rng.random((int(np.sum(veg_mask)),))
    img[veg_mask, 0] = veg_lum * 0.4
    img[veg_mask, 1] = veg_lum * 1.5
    img[veg_mask, 2] = veg_lum * 0.3

    # Skin tones region (50-70%)
    skin_mask = (yy >= 0.5) & (yy < 0.7)
    skin_lum = 80.0 + 100.0 * xx[skin_mask]
    img[skin_mask, 0] = skin_lum * 1.4
    img[skin_mask, 1] = skin_lum * 1.0
    img[skin_mask, 2] = skin_lum * 0.7

    # Metal/specular (70-100%)
    metal_mask = yy >= 0.7
    metal_base = 50.0 + 200.0 * xx[metal_mask]
    # Add specular highlights
    spec = 2000.0 * np.exp(
        -((xx[metal_mask] - 0.5) ** 2 + (yy[metal_mask] - 0.85) ** 2) / 0.02
    )
    metal_lum = metal_base + spec
    img[metal_mask, 0] = metal_lum * 1.1
    img[metal_mask, 1] = metal_lum * 1.1
    img[metal_mask, 2] = metal_lum * 1.0

    # Add subtle noise
    noise = rng.normal(0.0, 5.0, (height, width, 3))
    img = np.maximum(img + noise, 0.0)

    return img


def _generate_difficult(
    height: int, width: int, rng: np.random.Generator
) -> NDArray[np.floating]:
    """Difficult: extreme dynamic range, clipping, near-gamut-boundary colors.

    HDR highlights up to 8000+ nits, wide luminance range,
    near-gamut-boundary colors that will clip in BT.709.
    """
    x = np.linspace(0.0, 1.0, width, dtype=np.float64)
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    img = np.zeros((height, width, 3), dtype=np.float64)

    # Dark base with extreme contrast
    base_lum = 2.0 + 20.0 * yy

    # Multiple bright specular highlights (up to 9000 nits)
    highlights_params = [
        (0.2, 0.3, 8000.0, 0.01),
        (0.7, 0.5, 9000.0, 0.015),
        (0.5, 0.8, 6000.0, 0.008),
        (0.3, 0.6, 7000.0, 0.012),
    ]
    for cx, cy, peak, sigma in highlights_params:
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        highlight = peak * np.exp(-dist_sq / (2.0 * sigma))
        base_lum += highlight

    # Near-neutral base
    img[..., 0] = base_lum * 1.0
    img[..., 1] = base_lum * 1.0
    img[..., 2] = base_lum * 1.0

    # Add near-gamut-boundary saturated patches
    # Saturated BT.2020 red patch (will clip heavily in BT.709)
    patch_mask = (xx > 0.05) & (xx < 0.25) & (yy > 0.1) & (yy < 0.25)
    img[patch_mask, 0] = 2000.0
    img[patch_mask, 1] = 50.0
    img[patch_mask, 2] = 20.0

    # Saturated BT.2020 green patch
    patch_mask = (xx > 0.75) & (xx < 0.95) & (yy > 0.1) & (yy < 0.25)
    img[patch_mask, 0] = 30.0
    img[patch_mask, 1] = 1500.0
    img[patch_mask, 2] = 30.0

    # Deep blue patch
    patch_mask = (xx > 0.4) & (xx < 0.6) & (yy > 0.05) & (yy < 0.15)
    img[patch_mask, 0] = 20.0
    img[patch_mask, 1] = 20.0
    img[patch_mask, 2] = 800.0

    # Add noise
    noise = rng.normal(0.0, 3.0, (height, width, 3))
    img = np.maximum(img + noise, 0.0)

    return img
