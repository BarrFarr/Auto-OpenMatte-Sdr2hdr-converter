"""Transform validation and confidence scoring."""

from __future__ import annotations

import logging

import numpy as np

from auto_openmatte.core.models import ShotTransform

logger = logging.getLogger(__name__)


def validate_transform(transform: ShotTransform) -> list[str]:
    """Validate a shot transform for potential issues.

    Args:
        transform: The transform to validate.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings: list[str] = []

    # Check luminance curve monotonicity
    if transform.luminance_curve and len(transform.luminance_curve) >= 2:
        for i in range(1, len(transform.luminance_curve)):
            if transform.luminance_curve[i][1] < transform.luminance_curve[i - 1][1]:
                warnings.append(
                    f"Shot {transform.shot_id}: Luminance curve is not monotonic "
                    f"at point {i}"
                )
                break

    # Check exposure range
    if transform.exposure < 0.1 or transform.exposure > 20.0:
        warnings.append(
            f"Shot {transform.shot_id}: Extreme exposure value {transform.exposure:.3f}"
        )

    # Check contrast range
    if transform.contrast < 0.3 or transform.contrast > 5.0:
        warnings.append(
            f"Shot {transform.shot_id}: Extreme contrast value {transform.contrast:.3f}"
        )

    # Check saturation range
    if transform.saturation < 0.3 or transform.saturation > 3.0:
        warnings.append(
            f"Shot {transform.shot_id}: Extreme saturation value {transform.saturation:.3f}"
        )

    # Check color matrix
    if transform.color_matrix:
        mat = np.array(transform.color_matrix)
        # Check deviation from identity
        deviation = np.linalg.norm(mat - np.eye(3), 'fro')
        if deviation > 0.5:
            warnings.append(
                f"Shot {transform.shot_id}: Large color matrix deviation ({deviation:.3f})"
            )
        # Check for negative diagonal (would invert colors)
        for i in range(3):
            if mat[i, i] < 0:
                warnings.append(
                    f"Shot {transform.shot_id}: Negative diagonal in color matrix"
                )
                break

    # Check confidence
    if transform.confidence < 0.5:
        warnings.append(
            f"Shot {transform.shot_id}: Low confidence ({transform.confidence:.4f})"
        )

    return warnings


def validate_all_transforms(transforms: list[ShotTransform]) -> list[str]:
    """Validate all shot transforms.

    Args:
        transforms: List of all shot transforms.

    Returns:
        All warning messages across all transforms.
    """
    all_warnings: list[str] = []
    low_confidence_count = 0

    for t in transforms:
        warnings = validate_transform(t)
        all_warnings.extend(warnings)
        if t.confidence < 0.5:
            low_confidence_count += 1

    if low_confidence_count > 0:
        all_warnings.append(
            f"SUMMARY: {low_confidence_count}/{len(transforms)} shots have low confidence"
        )

    if all_warnings:
        logger.warning(f"Transform validation: {len(all_warnings)} warnings")
    else:
        logger.info("Transform validation: all transforms PASS")

    return all_warnings
