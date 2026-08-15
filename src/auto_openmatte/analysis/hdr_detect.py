"""HDR format detection and source role identification."""

from __future__ import annotations

from auto_openmatte.core.config import SUPPORTED_HDR_FORMATS
from auto_openmatte.core.exceptions import InvalidSourceError, UnsupportedHDRFormatError
from auto_openmatte.core.models import SourceInfo

# PQ (Perceptual Quantizer) transfer characteristic identifiers
_PQ_TRANSFERS = {"smpte2084", "smpte-st-2084", "smpte st 2084"}

# HLG transfer characteristic identifiers
_HLG_TRANSFERS = {"arib-std-b67", "arib std b67", "arib-std-b67"}

# BT.2020 primaries identifiers
_BT2020_PRIMARIES = {"bt2020", "bt.2020"}

# SDR transfer characteristics
_SDR_TRANSFERS = {"bt709", "bt.709", "iec61966-2-1", "srgb", "unknown"}

# SDR primaries
_SDR_PRIMARIES = {"bt709", "bt.709", "unknown"}


def _normalize(value: str) -> str:
    """Normalize a color metadata string for comparison."""
    return value.lower().strip().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")


def _is_pq(transfer: str) -> bool:
    """Check if transfer characteristics indicate PQ (HDR10/HDR10+)."""
    normalized = _normalize(transfer)
    return normalized in {"smpte2084", "smptst2084", "smptest2084", "smpte2084"}


def _is_hlg(transfer: str) -> bool:
    """Check if transfer characteristics indicate HLG."""
    normalized = _normalize(transfer)
    return normalized in {"aribstdb67", "aribstdb67", "hlg"}


def _is_bt2020(primaries: str) -> bool:
    """Check if color primaries indicate BT.2020."""
    normalized = _normalize(primaries)
    return normalized in {"bt2020", "bt2020"}


def _is_sdr_transfer(transfer: str) -> bool:
    """Check if transfer characteristics indicate SDR."""
    normalized = _normalize(transfer)
    return normalized in {"bt709", "iec6196621", "srgb", "unknown", ""}


def _is_sdr_primaries(primaries: str) -> bool:
    """Check if primaries indicate SDR (BT.709)."""
    normalized = _normalize(primaries)
    return normalized in {"bt709", "unknown", ""}


def detect_hdr_standard(source_info: SourceInfo) -> str:
    """Determine the HDR format of a source from its metadata.

    Detection logic:
    - HDR10: PQ transfer (smpte2084) + BT.2020 primaries + mastering display metadata
    - HDR10+: HDR10 + dynamic metadata (detected from side_data)
    - HLG: arib-std-b67 transfer characteristic
    - Dolby Vision: Detected from side_data type containing "Dolby Vision"
    - SDR: bt709 transfer or default/unknown gamma

    Args:
        source_info: SourceInfo dataclass with color metadata populated.

    Returns:
        One of: 'HDR10', 'HDR10+', 'HLG', 'Dolby Vision', 'SDR'
    """
    transfer = source_info.transfer_characteristics
    primaries = source_info.color_primaries

    # Check for HLG first (simpler check)
    if _is_hlg(transfer):
        return "HLG"

    # Check for PQ-based formats (HDR10, HDR10+, Dolby Vision)
    if _is_pq(transfer):
        # Check if mastering display metadata indicates Dolby Vision
        # (In practice, DV is detected via side_data in the stream)
        mastering = source_info.mastering_display

        # HDR10 requires PQ + BT.2020 + mastering display metadata
        if _is_bt2020(primaries) and mastering:
            return "HDR10"

        # PQ without mastering display - still likely HDR10 without
        # static metadata (less common but valid)
        if _is_bt2020(primaries):
            return "HDR10"

        # PQ with non-BT.2020 primaries is unusual
        return "HDR10"

    # SDR detection
    if _is_sdr_transfer(transfer) or _is_sdr_primaries(primaries):
        return "SDR"

    # Default to SDR if we cannot determine
    return "SDR"


def detect_hdr_standard_from_side_data(
    source_info: SourceInfo,
    side_data: dict[str, str] | None = None,
) -> str:
    """Extended detection that also considers side_data information.

    This is the full detection that includes HDR10+ and Dolby Vision
    which require checking side_data_list entries.

    Args:
        source_info: SourceInfo with color metadata.
        side_data: Optional dict of extracted side_data flags
            (keys: 'dolby_vision', 'hdr10plus', 'mastering_display').

    Returns:
        One of: 'HDR10', 'HDR10+', 'HLG', 'Dolby Vision', 'SDR'
    """
    if side_data:
        # Dolby Vision takes priority
        if side_data.get("dolby_vision"):
            return "Dolby Vision"

        # HDR10+ is HDR10 with dynamic metadata
        if side_data.get("hdr10plus"):
            base = detect_hdr_standard(source_info)
            if base == "HDR10":
                return "HDR10+"

    return detect_hdr_standard(source_info)


def _is_hdr_source(source_info: SourceInfo) -> bool:
    """Determine if a source is HDR (PQ or HLG)."""
    transfer = source_info.transfer_characteristics
    return _is_pq(transfer) or _is_hlg(transfer)


def _is_sdr_source(source_info: SourceInfo) -> bool:
    """Determine if a source is SDR."""
    transfer = source_info.transfer_characteristics
    primaries = source_info.color_primaries
    return _is_sdr_transfer(transfer) or (
        _is_sdr_primaries(primaries) and not _is_pq(transfer) and not _is_hlg(transfer)
    )


def identify_roles(
    source_a_info: SourceInfo,
    source_b_info: SourceInfo,
) -> tuple[SourceInfo, SourceInfo]:
    """Determine which source is HDR reference and which is Open Matte.

    The function works regardless of argument order: it inspects the
    transfer characteristics and color primaries to identify roles.

    Logic:
    - If A is HDR (PQ/HLG) and B is SDR (BT.709): A=HDR, B=Open Matte
    - If B is HDR and A is SDR: swap so HDR is first
    - If both are HDR or both are SDR: raise InvalidSourceError

    Args:
        source_a_info: First source's metadata.
        source_b_info: Second source's metadata.

    Returns:
        Tuple of (hdr_info, om_info) where hdr_info is the HDR reference
        and om_info is the Open Matte source.

    Raises:
        InvalidSourceError: If both sources are HDR, both are SDR, or
            roles cannot be determined.
    """
    a_is_hdr = _is_hdr_source(source_a_info)
    b_is_hdr = _is_hdr_source(source_b_info)
    a_is_sdr = _is_sdr_source(source_a_info)
    b_is_sdr = _is_sdr_source(source_b_info)

    if a_is_hdr and b_is_sdr:
        return (source_a_info, source_b_info)

    if b_is_hdr and a_is_sdr:
        return (source_b_info, source_a_info)

    if a_is_hdr and b_is_hdr:
        raise InvalidSourceError(
            "Both sources appear to be HDR. Cannot determine roles. "
            f"Source A ({source_a_info.path}): transfer={source_a_info.transfer_characteristics}, "
            f"primaries={source_a_info.color_primaries}. "
            f"Source B ({source_b_info.path}): transfer={source_b_info.transfer_characteristics}, "
            f"primaries={source_b_info.color_primaries}. "
            "Please provide one HDR source and one SDR Open Matte source."
        )

    if a_is_sdr and b_is_sdr:
        raise InvalidSourceError(
            "Both sources appear to be SDR. Cannot determine HDR reference. "
            f"Source A ({source_a_info.path}): transfer={source_a_info.transfer_characteristics}, "
            f"primaries={source_a_info.color_primaries}. "
            f"Source B ({source_b_info.path}): transfer={source_b_info.transfer_characteristics}, "
            f"primaries={source_b_info.color_primaries}. "
            "Please provide one HDR source and one SDR Open Matte source."
        )

    # Ambiguous case
    raise InvalidSourceError(
        "Cannot determine source roles. "
        f"Source A ({source_a_info.path}): transfer={source_a_info.transfer_characteristics}, "
        f"primaries={source_a_info.color_primaries}. "
        f"Source B ({source_b_info.path}): transfer={source_b_info.transfer_characteristics}, "
        f"primaries={source_b_info.color_primaries}."
    )


def validate_hdr_support(hdr_standard: str) -> None:
    """Check if the detected HDR format is supported by the pipeline.

    Args:
        hdr_standard: The detected HDR standard string.

    Raises:
        UnsupportedHDRFormatError: If the format is not in
            SUPPORTED_HDR_FORMATS config list.
    """
    if hdr_standard not in SUPPORTED_HDR_FORMATS:
        raise UnsupportedHDRFormatError(
            f"HDR format '{hdr_standard}' is not currently supported. "
            f"Supported formats: {', '.join(SUPPORTED_HDR_FORMATS)}"
        )
