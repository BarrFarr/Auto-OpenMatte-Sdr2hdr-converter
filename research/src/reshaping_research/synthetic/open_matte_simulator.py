"""Open matte simulator: create wider SDR with HDR as center crop.

Simulates the open matte scenario where SDR is a wider frame (e.g., 16:9)
and HDR is a narrower center crop (e.g., 21:9 / 2.39:1).
The overlap region has both HDR and SDR representations available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .sdr_simulator import simulate_sdr


@dataclass
class OpenMattePair:
    """An open matte test pair with all associated data.

    Attributes:
        hdr_center: The HDR center crop in linear BT.2020 nits, shape (hdr_h, W, 3).
        sdr_wide: The full SDR wide frame in gamma BT.709, shape (sdr_h, W, 3).
        overlap_mask: Boolean mask indicating overlap region in sdr_wide coordinates.
        hdr_offset_top: Row offset where HDR starts within the SDR frame.
        hdr_offset_bottom: Row offset where HDR ends within the SDR frame.
        seam_top_row: Row index of the top seam in SDR frame coordinates.
        seam_bottom_row: Row index of the bottom seam in SDR frame coordinates.
    """

    hdr_center: NDArray[np.floating]
    sdr_wide: NDArray[np.floating]
    overlap_mask: NDArray[np.bool_]
    hdr_offset_top: int
    hdr_offset_bottom: int
    seam_top_row: int
    seam_bottom_row: int


def create_open_matte_pair(
    hdr_scene: NDArray[np.floating],
    extension_rows: int = 48,
    tone_map_knee: float = 1000.0,
    tone_map_max: float = 4000.0,
    seed: int = 42,
) -> OpenMattePair:
    """Create an open matte pair from an HDR scene.

    The HDR scene becomes the center crop. Extension rows are added above
    and below to create the wider SDR frame. The extension content is
    generated as a smooth continuation of the HDR edge content.

    Args:
        hdr_scene: HDR scene in linear BT.2020 nits, shape (H, W, 3).
        extension_rows: Number of rows to extend above and below.
        tone_map_knee: Knee point for SDR tone mapping.
        tone_map_max: Max luminance for SDR tone mapping.
        seed: Random seed for extension generation.

    Returns:
        OpenMattePair with HDR center, SDR wide frame, and overlap info.
    """
    rng = np.random.default_rng(seed)
    hdr_h, W, _ = hdr_scene.shape
    sdr_h = hdr_h + 2 * extension_rows

    # Create wider HDR scene (with extensions) for SDR conversion
    # Extensions are smooth continuations of the edge content
    top_extension = _generate_extension(
        hdr_scene[0:4, :, :], extension_rows, "top", rng
    )
    bottom_extension = _generate_extension(
        hdr_scene[-4:, :, :], extension_rows, "bottom", rng
    )

    # Full HDR wide frame (for SDR conversion)
    hdr_wide = np.concatenate([top_extension, hdr_scene, bottom_extension], axis=0)

    # Convert entire wide frame to SDR
    sdr_wide = simulate_sdr(
        hdr_wide, tone_map_knee=tone_map_knee, tone_map_max=tone_map_max
    )

    # Overlap mask: rows where HDR content exists
    overlap_mask = np.zeros((sdr_h, W), dtype=bool)
    overlap_mask[extension_rows : extension_rows + hdr_h, :] = True

    return OpenMattePair(
        hdr_center=hdr_scene,
        sdr_wide=sdr_wide,
        overlap_mask=overlap_mask,
        hdr_offset_top=extension_rows,
        hdr_offset_bottom=extension_rows + hdr_h,
        seam_top_row=extension_rows,
        seam_bottom_row=extension_rows + hdr_h,
    )


def _generate_extension(
    edge_rows: NDArray[np.floating],
    num_rows: int,
    direction: str,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Generate smooth extension content from edge rows.

    Creates a gradual fade/continuation of the edge content to simulate
    what might appear beyond the HDR crop boundary.

    Args:
        edge_rows: Edge rows of the HDR content, shape (n_ref, W, 3).
        num_rows: Number of extension rows to generate.
        direction: "top" or "bottom" - determines fade direction.
        rng: Random number generator.

    Returns:
        Extension content, shape (num_rows, W, 3), in linear BT.2020 nits.
    """
    _, W, C = edge_rows.shape

    # Use the average of edge rows as the base
    if direction == "top":
        base = edge_rows[0:1, :, :]  # First row
    else:
        base = edge_rows[-1:, :, :]  # Last row

    # Create a fade factor: closer to edge = closer to base content
    fade = np.linspace(0.3, 1.0, num_rows, dtype=np.float64)
    if direction == "top":
        fade = fade  # Fades from dim (top) to full (bottom, near HDR)
    else:
        fade = fade[::-1]  # Fades from full (top, near HDR) to dim (bottom)

    # Tile base row and apply fade
    extension = np.tile(base, (num_rows, 1, 1))
    extension = extension * fade[:, np.newaxis, np.newaxis]

    # Add subtle variation
    noise = rng.normal(0.0, 2.0, (num_rows, W, C))
    extension = np.maximum(extension + noise, 0.0)

    return extension
