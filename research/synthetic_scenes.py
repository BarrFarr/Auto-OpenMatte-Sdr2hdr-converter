"""Synthetic test scene generator for SDR-to-HDR reshaping research.

Generates 6 test scenes, each returning:
  (hdr_full, sdr_openmatte, hdr_region_slice)

HDR: 1920x1600 in BT.2020 linear (0-1 range, representing 0-10000 nits)
SDR Open Matte: 1920x2160 in BT.709 gamma (0-1 range)
HDR maps to SDR rows 280:1880 (the overlap region)

A known/controlled tone mapping is applied HDR->SDR so ground truth is exact.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from color_utils import (
    bt2020_to_bt709,
    linear_to_sdr_gamma,
    _BT2020_LUMA,
)

# =============================================================================
# Dimensions
# =============================================================================
HDR_WIDTH = 1920
HDR_HEIGHT = 1600
SDR_WIDTH = 1920
SDR_HEIGHT = 2160
OVERLAP_TOP = 280
OVERLAP_BOTTOM = 1880  # HDR maps to SDR rows 280:1880

SceneTuple = Tuple[NDArray[np.floating], NDArray[np.floating], slice]


# =============================================================================
# Controlled tone mapping (HDR linear BT.2020 -> SDR gamma BT.709)
# =============================================================================

def _tonemap_reinhard(
    hdr_linear_bt2020: NDArray[np.floating],
    max_nits: float = 4000.0,
) -> NDArray[np.floating]:
    """Apply controlled Reinhard-style tone mapping preserving color ratios.

    Maps BT.2020 linear [0, 1] (representing 0-10000 nits) to BT.709 gamma [0, 1].
    """
    lum = np.einsum("...c,c->...", hdr_linear_bt2020, _BT2020_LUMA)
    lum = np.maximum(lum, 1e-10)

    max_norm = max_nits / 10000.0
    L = lum / max_norm
    L_mapped = L / (1.0 + L)
    L_mapped = np.clip(L_mapped * 2.0, 0.0, 1.0)

    ratio = np.where(lum > 1e-10, L_mapped / (lum / max_norm), 0.0)
    ratio = ratio[..., np.newaxis]

    tm_bt2020 = hdr_linear_bt2020 * ratio
    tm_bt709 = bt2020_to_bt709(tm_bt2020)
    tm_bt709 = np.clip(tm_bt709, 0.0, 1.0)
    sdr_gamma = linear_to_sdr_gamma(tm_bt709)
    return sdr_gamma


def _generate_extension(
    edge_row: NDArray[np.floating],
    num_rows: int,
    fade_from_top: bool,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Generate smooth SDR extension above/below the overlap region."""
    W = edge_row.shape[0]
    extension = np.tile(edge_row[np.newaxis, :, :], (num_rows, 1, 1))
    fade = np.linspace(0.4, 1.0, num_rows)
    if not fade_from_top:
        fade = fade[::-1]
    extension = extension * fade[:, np.newaxis, np.newaxis]
    noise = rng.normal(0.0, 0.01, (num_rows, W, 3))
    extension = np.clip(extension + noise, 0.0, 1.0)
    return extension


# =============================================================================
# Scene generators
# =============================================================================

def _make_scene(
    hdr_bt2020_linear: NDArray[np.floating],
    seed: int = 42,
) -> SceneTuple:
    """Convert raw HDR scene into (hdr_full, sdr_openmatte, hdr_region_slice)."""
    rng = np.random.default_rng(seed)
    sdr_overlap = _tonemap_reinhard(hdr_bt2020_linear)

    top_edge = sdr_overlap[0, :, :]
    bottom_edge = sdr_overlap[-1, :, :]

    top_ext = _generate_extension(top_edge, OVERLAP_TOP, fade_from_top=True, rng=rng)
    bot_ext = _generate_extension(
        bottom_edge, SDR_HEIGHT - OVERLAP_BOTTOM, fade_from_top=False, rng=rng
    )

    sdr_openmatte = np.zeros((SDR_HEIGHT, SDR_WIDTH, 3), dtype=np.float64)
    sdr_openmatte[:OVERLAP_TOP, :, :] = top_ext
    sdr_openmatte[OVERLAP_TOP:OVERLAP_BOTTOM, :, :] = sdr_overlap
    sdr_openmatte[OVERLAP_BOTTOM:, :, :] = bot_ext

    hdr_region_slice = slice(OVERLAP_TOP, OVERLAP_BOTTOM)
    return (hdr_bt2020_linear, sdr_openmatte, hdr_region_slice)


def generate_scene_neutral(seed: int = 42) -> SceneTuple:
    """Scene 1: Neutral - gray ramp + color patches, moderate contrast."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HDR_HEIGHT, HDR_WIDTH, 3), dtype=np.float64)

    x = np.linspace(0.0, 1.0, HDR_WIDTH)
    y = np.linspace(0.0, 1.0, HDR_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    base_nits = 50.0 + 1950.0 * (xx * 0.7 + yy * 0.3)
    base = base_nits / 10000.0

    img[..., 0] = base * 1.01
    img[..., 1] = base * 1.00
    img[..., 2] = base * 0.99

    patch_size = 100
    colors = [(1.2, 0.8, 0.8), (0.8, 1.2, 0.8), (0.8, 0.8, 1.2), (1.1, 1.1, 0.7)]
    for i, (cr, cg, cb) in enumerate(colors):
        row = 200 + i * 350
        col = 400 + i * 300
        if row + patch_size < HDR_HEIGHT and col + patch_size < HDR_WIDTH:
            patch_lum = base[row:row+patch_size, col:col+patch_size]
            img[row:row+patch_size, col:col+patch_size, 0] = patch_lum * cr
            img[row:row+patch_size, col:col+patch_size, 1] = patch_lum * cg
            img[row:row+patch_size, col:col+patch_size, 2] = patch_lum * cb

    noise = rng.normal(0.0, 0.001, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return _make_scene(img, seed=seed)


def generate_scene_high_key(seed: int = 43) -> SceneTuple:
    """Scene 2: High-key - bright sky, specular highlights up to 4000 nits."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HDR_HEIGHT, HDR_WIDTH, 3), dtype=np.float64)

    x = np.linspace(0.0, 1.0, HDR_WIDTH)
    y = np.linspace(0.0, 1.0, HDR_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    sky_nits = 2000.0 + 2000.0 * (1.0 - yy) * (0.5 + 0.5 * np.cos(xx * np.pi * 0.3))
    sky = sky_nits / 10000.0

    img[..., 0] = sky * 0.82
    img[..., 1] = sky * 0.90
    img[..., 2] = sky * 1.20

    for _ in range(8):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.6)
        sx = rng.uniform(0.05, 0.15)
        sy = rng.uniform(0.02, 0.06)
        peak = rng.uniform(3000.0, 4000.0) / 10000.0
        cloud = peak * np.exp(-((xx - cx)**2 / (2*sx**2) + (yy - cy)**2 / (2*sy**2)))
        img[..., 0] += cloud * 1.0
        img[..., 1] += cloud * 1.0
        img[..., 2] += cloud * 0.95

    ground_mask = yy > 0.7
    ground_nits = (400.0 + 300.0 * xx[ground_mask]) / 10000.0
    img[ground_mask, 0] = ground_nits * 1.1
    img[ground_mask, 1] = ground_nits * 1.0
    img[ground_mask, 2] = ground_nits * 0.8

    noise = rng.normal(0.0, 0.002, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return _make_scene(img, seed=seed)


def generate_scene_low_key(seed: int = 44) -> SceneTuple:
    """Scene 3: Low-key - dark scene, few bright points."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HDR_HEIGHT, HDR_WIDTH, 3), dtype=np.float64)

    x = np.linspace(0.0, 1.0, HDR_WIDTH)
    y = np.linspace(0.0, 1.0, HDR_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    base_nits = 5.0 + 75.0 * (0.3 + 0.7 * np.sin(xx * np.pi) * np.sin(yy * np.pi * 0.5))
    base = base_nits / 10000.0

    img[..., 0] = base * 0.95
    img[..., 1] = base * 1.00
    img[..., 2] = base * 1.05

    for _ in range(6):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        peak = rng.uniform(800.0, 2000.0) / 10000.0
        sigma = rng.uniform(0.005, 0.02)
        spot = peak * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        img[..., 0] += spot * 1.1
        img[..., 1] += spot * 1.0
        img[..., 2] += spot * 0.9

    noise = rng.normal(0.0, 0.0005, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return _make_scene(img, seed=seed)


def generate_scene_colorful(seed: int = 45) -> SceneTuple:
    """Scene 4: Colorful - saturated R/G/B regions."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HDR_HEIGHT, HDR_WIDTH, 3), dtype=np.float64)

    x = np.linspace(0.0, 1.0, HDR_WIDTH)
    y = np.linspace(0.0, 1.0, HDR_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    band_h = HDR_HEIGHT // 6

    r1 = slice(0, band_h)
    lum = (200.0 + 1000.0 * xx[r1, :]) / 10000.0
    img[r1, :, 0] = lum * 3.0
    img[r1, :, 1] = lum * 0.15
    img[r1, :, 2] = lum * 0.08

    r2 = slice(band_h, 2 * band_h)
    lum = (300.0 + 1500.0 * xx[r2, :]) / 10000.0
    img[r2, :, 0] = lum * 0.12
    img[r2, :, 1] = lum * 2.5
    img[r2, :, 2] = lum * 0.12

    r3 = slice(2 * band_h, 3 * band_h)
    lum = (100.0 + 500.0 * xx[r3, :]) / 10000.0
    img[r3, :, 0] = lum * 0.08
    img[r3, :, 1] = lum * 0.12
    img[r3, :, 2] = lum * 4.0

    r4 = slice(3 * band_h, 4 * band_h)
    lum = (400.0 + 1000.0 * xx[r4, :]) / 10000.0
    img[r4, :, 0] = lum * 0.1
    img[r4, :, 1] = lum * 1.8
    img[r4, :, 2] = lum * 2.0

    r5 = slice(4 * band_h, 5 * band_h)
    lum = (150.0 + 600.0 * xx[r5, :]) / 10000.0
    img[r5, :, 0] = lum * 2.5
    img[r5, :, 1] = lum * 0.1
    img[r5, :, 2] = lum * 2.5

    r6 = slice(5 * band_h, HDR_HEIGHT)
    lum = (500.0 + 2000.0 * xx[r6, :]) / 10000.0
    img[r6, :, 0] = lum * 1.8
    img[r6, :, 1] = lum * 1.7
    img[r6, :, 2] = lum * 0.12

    noise = rng.normal(0.0, 0.001, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return _make_scene(img, seed=seed)


def generate_scene_mixed(seed: int = 46) -> SceneTuple:
    """Scene 5: Mixed - skin tones, foliage, sky, shadows."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HDR_HEIGHT, HDR_WIDTH, 3), dtype=np.float64)

    x = np.linspace(0.0, 1.0, HDR_WIDTH)
    y = np.linspace(0.0, 1.0, HDR_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    sky_mask = yy < 0.25
    sky_nits = (1500.0 + 1500.0 * (0.25 - yy[sky_mask]) / 0.25) / 10000.0
    img[sky_mask, 0] = sky_nits * 0.72
    img[sky_mask, 1] = sky_nits * 0.88
    img[sky_mask, 2] = sky_nits * 1.30

    fol_mask = (yy >= 0.25) & (yy < 0.45)
    n_fol = int(np.sum(fol_mask))
    fol_nits = (100.0 + 300.0 * rng.random(n_fol)) / 10000.0
    img[fol_mask, 0] = fol_nits * 0.45
    img[fol_mask, 1] = fol_nits * 1.60
    img[fol_mask, 2] = fol_nits * 0.30

    skin_mask = (yy >= 0.45) & (yy < 0.65)
    skin_nits = (80.0 + 120.0 * xx[skin_mask]) / 10000.0
    img[skin_mask, 0] = skin_nits * 1.45
    img[skin_mask, 1] = skin_nits * 1.05
    img[skin_mask, 2] = skin_nits * 0.72

    shadow_mask = yy >= 0.65
    shadow_nits = (10.0 + 50.0 * xx[shadow_mask]) / 10000.0
    spec_dist = (xx[shadow_mask] - 0.5)**2 + (yy[shadow_mask] - 0.82)**2
    spec = (2500.0 / 10000.0) * np.exp(-spec_dist / 0.01)
    combined = shadow_nits + spec
    img[shadow_mask, 0] = combined * 1.05
    img[shadow_mask, 1] = combined * 1.05
    img[shadow_mask, 2] = combined * 0.95

    noise = rng.normal(0.0, 0.001, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return _make_scene(img, seed=seed)


def generate_scene_difficult(seed: int = 47) -> SceneTuple:
    """Scene 6: Difficult - SDR clipping, HDR>1000 nits, near-gamut-boundary."""
    rng = np.random.default_rng(seed)
    img = np.zeros((HDR_HEIGHT, HDR_WIDTH, 3), dtype=np.float64)

    x = np.linspace(0.0, 1.0, HDR_WIDTH)
    y = np.linspace(0.0, 1.0, HDR_HEIGHT)
    xx, yy = np.meshgrid(x, y)

    base = (2.0 + 28.0 * yy) / 10000.0
    img[..., 0] = base
    img[..., 1] = base
    img[..., 2] = base

    highlights = [
        (0.2, 0.3, 8000.0, 0.008),
        (0.7, 0.5, 7000.0, 0.012),
        (0.5, 0.8, 6000.0, 0.006),
        (0.85, 0.2, 7500.0, 0.010),
    ]
    for cx, cy, peak, sigma in highlights:
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        spot = (peak / 10000.0) * np.exp(-dist_sq / (2 * sigma))
        img[..., 0] += spot * 1.0
        img[..., 1] += spot * 1.0
        img[..., 2] += spot * 0.95

    r_mask = (xx > 0.05) & (xx < 0.20) & (yy > 0.10) & (yy < 0.25)
    img[r_mask, 0] = 3000.0 / 10000.0
    img[r_mask, 1] = 50.0 / 10000.0
    img[r_mask, 2] = 30.0 / 10000.0

    g_mask = (xx > 0.75) & (xx < 0.90) & (yy > 0.10) & (yy < 0.25)
    img[g_mask, 0] = 40.0 / 10000.0
    img[g_mask, 1] = 2000.0 / 10000.0
    img[g_mask, 2] = 40.0 / 10000.0

    b_mask = (xx > 0.40) & (xx < 0.60) & (yy > 0.05) & (yy < 0.15)
    img[b_mask, 0] = 20.0 / 10000.0
    img[b_mask, 1] = 20.0 / 10000.0
    img[b_mask, 2] = 1500.0 / 10000.0

    o_mask = (xx > 0.30) & (xx < 0.50) & (yy > 0.70) & (yy < 0.85)
    img[o_mask, 0] = 2500.0 / 10000.0
    img[o_mask, 1] = 1200.0 / 10000.0
    img[o_mask, 2] = 50.0 / 10000.0

    noise = rng.normal(0.0, 0.0003, img.shape)
    img = np.clip(img + noise, 0.0, 1.0)
    return _make_scene(img, seed=seed)


def generate_all_scenes() -> list:
    """Generate all 6 test scenes.

    Returns:
        List of (name, (hdr_full, sdr_openmatte, hdr_region_slice)) tuples.
    """
    return [
        ("neutral", generate_scene_neutral()),
        ("high_key", generate_scene_high_key()),
        ("low_key", generate_scene_low_key()),
        ("colorful", generate_scene_colorful()),
        ("mixed", generate_scene_mixed()),
        ("difficult", generate_scene_difficult()),
    ]
