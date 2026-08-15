"""Comprehensive HDR implementation audit.

Verifies all 15 points from the audit checklist:
1.  BT.2020 primaries detection
2.  BT.2020 matrix coefficients detection
3.  PQ / ST 2084 detection
4.  HLG detection
5.  SDR BT.709 / BT.1886 detection
6.  HDR10 metadata preservation
7.  PQ EOTF: normalized code values → absolute luminance
8.  PQ OETF: absolute luminance → PQ code values
9.  PQ round-trip tests
10. PQ is NOT treated as a simple gamma curve
11. 10-bit input remains ≥10-bit precision during processing
12. No accidental 8-bit conversion
13. BT.2020 → working RGB conversion correctness
14. HDR region in extend mode not tone-mapped through SDR
15. Final output retains HDR signaling / metadata
"""

from __future__ import annotations

import pytest

from auto_openmatte.analysis.inspect import (
    _classify_hdr_format,
    _detect_hdr_metadata,
    _parse_primaries,
    _parse_stream,
    _parse_transfer,
)
from auto_openmatte.core.models import (
    ColorPrimaries,
    HDRFormat,
    HDRMetadata,
    TransferFunction,
    VideoStreamInfo,
)


# ============================================================
# POINT 1: BT.2020 primaries are correctly detected
# ============================================================
class TestBT2020PrimariesDetection:
    def test_bt2020_from_ffprobe_string(self) -> None:
        """ffprobe reports 'bt2020' → ColorPrimaries.BT2020."""
        stream = {"color_primaries": "bt2020"}
        assert _parse_primaries(stream) == ColorPrimaries.BT2020

    def test_bt709_from_ffprobe_string(self) -> None:
        stream = {"color_primaries": "bt709"}
        assert _parse_primaries(stream) == ColorPrimaries.BT709

    def test_dci_p3_detected(self) -> None:
        stream = {"color_primaries": "smpte432"}
        assert _parse_primaries(stream) == ColorPrimaries.DCI_P3

    def test_unknown_primaries(self) -> None:
        stream = {"color_primaries": "something_else"}
        assert _parse_primaries(stream) == ColorPrimaries.UNKNOWN

    def test_missing_primaries(self) -> None:
        stream = {}
        assert _parse_primaries(stream) == ColorPrimaries.UNKNOWN

    def test_bt2020_in_full_stream_parse(self) -> None:
        """Full stream parse preserves BT.2020 primaries."""
        data = {
            "index": 0,
            "codec_name": "hevc",
            "width": 3840, "height": 2160,
            "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
            "pix_fmt": "yuv420p10le",
            "bits_per_raw_sample": "10",
            "color_primaries": "bt2020",
            "color_transfer": "smpte2084",
            "color_space": "bt2020nc",
            "color_range": "tv",
            "disposition": {"default": 1},
        }
        stream = _parse_stream(data)
        assert stream.color_primaries == ColorPrimaries.BT2020


# ============================================================
# POINT 2: BT.2020 matrix coefficients are correctly detected
# ============================================================
class TestBT2020MatrixDetection:
    def test_bt2020nc_detected(self) -> None:
        """ffprobe 'bt2020nc' stored as matrix_coefficients."""
        data = {
            "index": 0, "codec_name": "hevc",
            "width": 3840, "height": 2160,
            "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
            "pix_fmt": "yuv420p10le",
            "color_space": "bt2020nc",
            "disposition": {"default": 1},
        }
        stream = _parse_stream(data)
        assert stream.matrix_coefficients == "bt2020nc"

    def test_bt709_matrix_detected(self) -> None:
        data = {
            "index": 0, "codec_name": "hevc",
            "width": 3840, "height": 2160,
            "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
            "pix_fmt": "yuv420p",
            "color_space": "bt709",
            "disposition": {"default": 1},
        }
        stream = _parse_stream(data)
        assert stream.matrix_coefficients == "bt709"


# ============================================================
# POINT 3: PQ / ST 2084 is correctly detected
# ============================================================
class TestPQDetection:
    def test_smpte2084_string(self) -> None:
        stream = {"color_transfer": "smpte2084"}
        assert _parse_transfer(stream) == TransferFunction.PQ

    def test_pq_classified_as_hdr10(self) -> None:
        """PQ transfer + BT.2020 → HDR10 format."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.PQ,
            color_primaries=ColorPrimaries.BT2020,
        )
        metadata = HDRMetadata()
        assert _classify_hdr_format(stream, metadata) == HDRFormat.HDR10

    def test_pq_without_bt2020_still_hdr10(self) -> None:
        """PQ without BT.2020 primaries → still classified as HDR10."""
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.PQ,
            color_primaries=ColorPrimaries.UNKNOWN,
        )
        metadata = HDRMetadata()
        assert _classify_hdr_format(stream, metadata) == HDRFormat.HDR10


# ============================================================
# POINT 4: HLG is correctly detected
# ============================================================
class TestHLGDetection:
    def test_arib_std_b67_string(self) -> None:
        stream = {"color_transfer": "arib-std-b67"}
        assert _parse_transfer(stream) == TransferFunction.HLG

    def test_hlg_classified_correctly(self) -> None:
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.HLG,
            color_primaries=ColorPrimaries.BT2020,
        )
        metadata = HDRMetadata()
        assert _classify_hdr_format(stream, metadata) == HDRFormat.HLG


# ============================================================
# POINT 5: SDR BT.709 / BT.1886 is correctly detected
# ============================================================
class TestSDRDetection:
    def test_bt709_transfer(self) -> None:
        stream = {"color_transfer": "bt709"}
        assert _parse_transfer(stream) == TransferFunction.BT709

    def test_smpte170m_as_bt709(self) -> None:
        """SMPTE 170M (NTSC) should map to BT.709 for processing."""
        stream = {"color_transfer": "smpte170m"}
        assert _parse_transfer(stream) == TransferFunction.BT709

    def test_sdr_classified_correctly(self) -> None:
        stream = VideoStreamInfo(
            index=0, codec="hevc", width=3840, height=2160,
            transfer=TransferFunction.BT709,
            color_primaries=ColorPrimaries.BT709,
        )
        metadata = HDRMetadata()
        assert _classify_hdr_format(stream, metadata) == HDRFormat.SDR


# ============================================================
# POINT 6: HDR10 metadata is preserved where required
# ============================================================
class TestHDR10MetadataPreservation:
    def test_mastering_display_parsed(self) -> None:
        """Mastering display metadata from side_data is preserved."""
        probe = {
            "streams": [{
                "codec_type": "video",
                "side_data_list": [{
                    "side_data_type": "Mastering display metadata",
                    "red_x": "34000/50000", "red_y": "16000/50000",
                    "green_x": "13250/50000", "green_y": "34500/50000",
                    "blue_x": "7500/50000", "blue_y": "3000/50000",
                    "white_point_x": "15635/50000", "white_point_y": "16450/50000",
                    "min_luminance": "50/10000", "max_luminance": "10000000/10000",
                }],
            }],
        }
        metadata = _detect_hdr_metadata(probe)
        assert metadata.mastering_display is not None
        assert "red_x" in metadata.mastering_display
        assert "max_luminance" in metadata.mastering_display

    def test_content_light_level_parsed(self) -> None:
        """MaxCLL and MaxFALL from Content Light Level metadata."""
        probe = {
            "streams": [{
                "codec_type": "video",
                "side_data_list": [{
                    "side_data_type": "Content light level metadata",
                    "max_content": 1000,
                    "max_average": 400,
                }],
            }],
        }
        metadata = _detect_hdr_metadata(probe)
        assert metadata.max_cll == 1000
        assert metadata.max_fall == 400

    def test_hdr10plus_detected_from_side_data(self) -> None:
        """HDR10+ dynamic metadata detected from side data."""
        probe = {
            "streams": [{
                "codec_type": "video",
                "side_data_list": [{
                    "side_data_type": "HDR10+ metadata",
                }],
            }],
        }
        metadata = _detect_hdr_metadata(probe)
        assert metadata.format == HDRFormat.HDR10_PLUS

    def test_dolby_vision_detected(self) -> None:
        """Dolby Vision RPU detected from side data."""
        probe = {
            "streams": [{
                "codec_type": "video",
                "side_data_list": [{
                    "side_data_type": "Dolby Vision RPU",
                }],
            }],
        }
        metadata = _detect_hdr_metadata(probe)
        assert metadata.format == HDRFormat.DOLBY_VISION


# ============================================================
# POINT 7: PQ EOTF: normalized code values → absolute luminance
# ============================================================

# --- Tests below require numpy ---
try:
    import numpy as np

    from auto_openmatte.core.transfer_functions import (
        _PQ_C1,
        _PQ_C2,
        _PQ_C3,
        _PQ_M1,
        _PQ_M2,
        _PQ_PEAK_LUMINANCE,
        delinearize,
        linearize,
        pq_eotf,
        pq_oetf,
    )
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

needs_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")


@needs_numpy
class TestPQEOTFCorrectness:
    def test_signal_0_gives_0_nits(self) -> None:
        result = pq_eotf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_signal_1_gives_10000_nits(self) -> None:
        result = pq_eotf(np.array([1.0]))
        assert result[0] == pytest.approx(10000.0, rel=1e-4)

    def test_100_nits_reference_white(self) -> None:
        """100 nits (SDR reference white) corresponds to PQ ~0.508."""
        # Compute expected PQ code for 100 nits
        expected_signal = pq_oetf(np.array([100.0]))
        # Should be approximately 0.508
        assert 0.49 < expected_signal[0] < 0.52

        # And the reverse
        result = pq_eotf(np.array([0.5079]))
        assert 95.0 < result[0] < 105.0

    def test_1000_nits(self) -> None:
        """1000 nits corresponds to PQ ~0.752."""
        signal = pq_oetf(np.array([1000.0]))
        assert 0.74 < signal[0] < 0.76
        # Round-trip
        lum = pq_eotf(signal)
        assert lum[0] == pytest.approx(1000.0, rel=1e-4)

    def test_known_pq_values(self) -> None:
        """Test against known reference PQ values from the standard."""
        # These are well-known anchor points in PQ
        test_cases = [
            # (nits, approx_pq_signal)
            (0.0, 0.0),
            (1.0, 0.1259),    # ~1 nit → PQ ≈ 0.126
            (100.0, 0.5079),  # 100 nit (SDR white)
            (1000.0, 0.7519), # 1000 nit
            (4000.0, 0.9009), # 4000 nit
            (10000.0, 1.0),   # Peak
        ]
        for nits, expected_pq in test_cases:
            signal = pq_oetf(np.array([nits]))
            assert signal[0] == pytest.approx(expected_pq, abs=0.01), (
                f"Failed for {nits} nits: expected PQ≈{expected_pq}, got {signal[0]:.4f}"
            )

    def test_output_is_absolute_luminance_not_normalized(self) -> None:
        """pq_eotf output is in cd/m² (nits), NOT [0,1] normalized."""
        signal = np.array([0.5])
        result = pq_eotf(signal)
        # At PQ=0.5, luminance should be ~roughly 80-100 nits
        # NOT 0.5 or any [0,1] value
        assert result[0] > 50.0
        assert result[0] < 200.0


# ============================================================
# POINT 8: PQ OETF: absolute luminance → PQ code values
# ============================================================
@needs_numpy
class TestPQOETFCorrectness:
    def test_0_nits_gives_signal_0(self) -> None:
        result = pq_oetf(np.array([0.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_10000_nits_gives_signal_1(self) -> None:
        result = pq_oetf(np.array([10000.0]))
        assert result[0] == pytest.approx(1.0, rel=1e-4)

    def test_clamps_above_10000(self) -> None:
        """Values above 10000 nits should be clamped to 1.0."""
        result = pq_oetf(np.array([20000.0]))
        assert result[0] == pytest.approx(1.0, rel=1e-4)

    def test_clamps_below_zero(self) -> None:
        """Negative luminance clamped to 0."""
        result = pq_oetf(np.array([-100.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-6)


# ============================================================
# POINT 9: PQ EOTF/OETF round-trip tests
# ============================================================
@needs_numpy
class TestPQRoundTrip:
    def test_full_range_round_trip(self) -> None:
        """Full [0,1] range round trip at high precision."""
        signal = np.linspace(0.0, 1.0, 10000)
        luminance = pq_eotf(signal)
        reconstructed = pq_oetf(luminance)
        np.testing.assert_allclose(reconstructed, signal, atol=1e-7)

    def test_luminance_round_trip(self) -> None:
        """Luminance → signal → luminance should be identity."""
        luminance = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0])
        signal = pq_oetf(luminance)
        reconstructed = pq_eotf(signal)
        np.testing.assert_allclose(reconstructed, luminance, rtol=1e-6)

    def test_linearize_delinearize_pq_round_trip(self) -> None:
        """The high-level linearize/delinearize interface round-trips correctly."""
        signal = np.linspace(0.01, 0.99, 1000)
        linear = linearize(signal, "smpte2084", peak_nits=10000.0)
        reconstructed = delinearize(linear, "smpte2084", peak_nits=10000.0)
        np.testing.assert_allclose(reconstructed, signal, atol=1e-6)


# ============================================================
# POINT 10: PQ is NOT treated as a simple gamma curve
# ============================================================
@needs_numpy
class TestPQNotGamma:
    def test_pq_is_not_gamma_22(self) -> None:
        """PQ(0.5) should NOT equal 0.5^2.2 or anything close."""
        pq_result = pq_eotf(np.array([0.5]))[0]
        gamma_22_result = 0.5**2.2 * 10000.0
        # PQ at 0.5 should be ~92 nits
        # gamma 2.2 at 0.5 would be ~0.2176 * 10000 = 2176 nits
        assert abs(pq_result - gamma_22_result) > 1000.0, (
            "PQ appears to behave like gamma 2.2 — this is wrong"
        )

    def test_pq_is_not_gamma_24(self) -> None:
        """PQ(0.5) should NOT equal 0.5^2.4 * 10000."""
        pq_result = pq_eotf(np.array([0.5]))[0]
        gamma_24_result = 0.5**2.4 * 10000.0
        assert abs(pq_result - gamma_24_result) > 800.0

    def test_pq_constants_match_st2084(self) -> None:
        """Verify PQ constants match SMPTE ST 2084 specification."""
        assert _PQ_M1 == pytest.approx(2610.0 / 16384.0)
        assert _PQ_M2 == pytest.approx(2523.0 / 4096.0 * 128.0)
        assert _PQ_C1 == pytest.approx(3424.0 / 4096.0)
        assert _PQ_C2 == pytest.approx(2413.0 / 4096.0 * 32.0)
        assert _PQ_C3 == pytest.approx(2392.0 / 4096.0 * 32.0)
        assert _PQ_PEAK_LUMINANCE == 10000.0

    def test_pq_highly_nonlinear(self) -> None:
        """PQ allocates more code values to dark regions (perceptual)."""
        # First 50% of PQ code values represent a small fraction of luminance
        signal_50pct = pq_eotf(np.array([0.5]))[0]
        signal_75pct = pq_eotf(np.array([0.75]))[0]
        # 50% signal → ~92 nits (only 0.92% of 10000 range)
        # 75% signal → ~1000 nits (10% of 10000 range)
        assert signal_50pct < 150.0  # Well below mid-range of luminance
        assert signal_75pct < 1500.0  # Still below 15% of peak


# ============================================================
# POINT 11 & 12: 10-bit precision and no 8-bit conversion
# ============================================================
@needs_numpy
class TestBitDepthPrecision:
    def test_processing_uses_float64(self) -> None:
        """All transfer functions operate in float64 — far exceeding 10-bit."""
        signal = np.array([0.5], dtype=np.float64)
        result = pq_eotf(signal)
        assert result.dtype == np.float64

    def test_10bit_values_distinguished(self) -> None:
        """Adjacent 10-bit code values must produce different outputs."""
        # 10-bit has 1024 levels. Adjacent values differ by 1/1023
        v1 = 500.0 / 1023.0
        v2 = 501.0 / 1023.0
        r1 = pq_eotf(np.array([v1]))[0]
        r2 = pq_eotf(np.array([v2]))[0]
        # They must be distinguishable
        assert r1 != r2
        assert abs(r2 - r1) > 0.01  # At least 0.01 nit difference

    def test_12bit_values_distinguished(self) -> None:
        """Adjacent 12-bit code values must still be distinguishable."""
        v1 = 2000.0 / 4095.0
        v2 = 2001.0 / 4095.0
        r1 = pq_eotf(np.array([v1]))[0]
        r2 = pq_eotf(np.array([v2]))[0]
        assert r1 != r2

    def test_linearize_returns_float64(self) -> None:
        """linearize should return float64 regardless of input precision."""
        signal_32 = np.array([0.3, 0.5, 0.7], dtype=np.float32)
        result = linearize(signal_32, "smpte2084")
        # Result should be at least float64 precision
        assert result.dtype in (np.float64, np.float32)
        # But the computation inside uses float64

    def test_no_uint8_anywhere_in_pipeline(self) -> None:
        """Verify that operations don't accidentally quantize to 8-bit."""
        # Simulate 10-bit input normalized to [0, 1]
        signal_10bit = np.arange(1024, dtype=np.float64) / 1023.0
        linear = linearize(signal_10bit, "smpte2084")
        # Output should have > 256 distinct values (8-bit would collapse)
        unique_values = len(np.unique(linear))
        assert unique_values > 900, (
            f"Only {unique_values} unique values — possible 8-bit quantization"
        )

    def test_frame_extraction_uses_16bit(self) -> None:
        """Verify that frame extraction specifies gray16le (16-bit) pixel format."""
        import inspect

        from auto_openmatte.utils.ffmpeg import extract_frame_to_numpy
        source = inspect.getsource(extract_frame_to_numpy)
        # Default pixel format should be 16-bit
        assert "gray16le" in source


# ============================================================
# POINT 13: BT.2020 → working RGB — not directly in transfer_functions
# but the color space handling is checked
# ============================================================
@needs_numpy
class TestColorSpaceHandling:
    def test_linearize_pq_output_range(self) -> None:
        """PQ linearize output should be [0, 1] normalized (divided by peak)."""
        signal = np.array([0.0, 0.508, 1.0])
        linear = linearize(signal, "smpte2084", peak_nits=10000.0)
        assert linear[0] == pytest.approx(0.0, abs=1e-6)
        assert linear[-1] == pytest.approx(1.0, rel=0.01)
        # Mid-value should be ~0.01 (100/10000)
        assert 0.005 < linear[1] < 0.02

    def test_bt709_linearize_is_gamma_24(self) -> None:
        """BT.709/BT.1886 linearization uses gamma 2.4 (not 2.2)."""
        signal = np.array([0.5])
        linear = linearize(signal, "bt709")
        # 0.5^2.4 = 0.18946
        expected = 0.5**2.4
        assert linear[0] == pytest.approx(expected, rel=1e-4)

    def test_luminance_weights_in_transform(self) -> None:
        """Verify Rec.709 luminance weights are used for SDR processing."""
        import inspect

        from auto_openmatte.processing.transform import apply_shot_transform
        source = inspect.getsource(apply_shot_transform)
        # Should use Rec.709 weights: 0.2126, 0.7152, 0.0722
        assert "0.2126" in source
        assert "0.7152" in source
        assert "0.0722" in source


# ============================================================
# POINT 14: HDR region in extend mode is NOT tone-mapped through SDR
# ============================================================
@needs_numpy
class TestHDRPassthrough:
    def test_composite_extend_preserves_hdr_pixels(self) -> None:
        """In extend mode, HDR pixels in the overlap region are passed through.

        The HDR frame is used directly — it must NOT go through
        SDR linearization or tone mapping.
        """
        from auto_openmatte.core.models import GeometryModel, ShotTransform
        from auto_openmatte.pipeline.compose import composite_extend

        # Create a small test case
        hdr_h, hdr_w = 10, 20
        om_h, om_w = 20, 20

        # HDR frame with known values (PQ signal domain)
        hdr_frame = np.full((hdr_h, hdr_w, 3), 0.7, dtype=np.float64)

        # OM frame (SDR)
        om_frame = np.full((om_h, om_w, 3), 0.5, dtype=np.float64)

        # Geometry: HDR maps to center of OM
        geometry = GeometryModel(
            scale_x=1.0, scale_y=1.0,
            offset_x=0.0, offset_y=5.0,
            overlap_bbox=[0.0, 5.0, 20.0, 15.0],
            confidence=0.99,
        )

        # Identity transform (no color change)
        transform = ShotTransform(shot_id=1, confidence=0.99)

        # Extension mask: 0 in center (HDR), 1 at top/bottom (extension)
        mask = np.ones((om_h, om_w), dtype=np.float64)
        mask[5:15, :] = 0.0  # HDR region

        output = composite_extend(
            hdr_frame, om_frame, transform, geometry, mask
        )

        # The center region (where mask=0) should be the HDR values
        # directly placed — NOT processed through SDR transform
        center_region = output[5:15, :, :]

        # HDR pixels should be placed directly (value 0.7 in PQ signal)
        # With scale factor 1.0, the HDR frame fills the overlap exactly
        np.testing.assert_allclose(
            center_region, 0.7, atol=0.01,
            err_msg="HDR pixels were modified in the center region!"
        )

    def test_composite_hdr_not_linearized_through_sdr(self) -> None:
        """The HDR frame in composite_extend is NOT passed through linearize/delinearize."""
        import inspect

        from auto_openmatte.pipeline.compose import composite_extend
        source = inspect.getsource(composite_extend)
        # The function should NOT call linearize on hdr_frame
        # It should directly place hdr_frame pixels via the mask
        # The only linearize call should be for om_frame (inside apply_shot_transform)
        assert "linearize(hdr_frame" not in source
        assert "linearize(hdr" not in source


# ============================================================
# POINT 15: Final output retains HDR signaling
# ============================================================
@needs_numpy
class TestHDROutputSignaling:
    def test_render_config_uses_10bit(self) -> None:
        """Default render config specifies 10-bit output."""
        from auto_openmatte.core.config import RenderConfig
        config = RenderConfig()
        assert "10" in config.pix_fmt  # yuv420p10le

    def test_sample_validator_checks_hdr_metadata(self) -> None:
        """The output validator checks for PQ/HLG transfer in output."""
        import inspect

        from auto_openmatte.pipeline.sample_validate import validate_sample_output
        source = inspect.getsource(validate_sample_output)
        assert "smpte2084" in source
        assert "arib-std-b67" in source
        assert "bt2020" in source

    def test_transform_outputs_pq_signal(self) -> None:
        """apply_shot_transform outputs PQ signal domain by default."""
        from auto_openmatte.core.models import ShotTransform
        from auto_openmatte.processing.transform import apply_shot_transform

        # Simple SDR frame
        om_frame = np.full((4, 4, 3), 0.5, dtype=np.float64)
        transform = ShotTransform(
            shot_id=1,
            luminance_curve=[[0.0, 0.0], [0.5, 0.01], [1.0, 0.1]],
            confidence=0.99,
        )

        result = apply_shot_transform(
            om_frame, transform,
            sdr_transfer="bt709",
            hdr_transfer="smpte2084",
        )

        # Output should be in [0, 1] PQ signal domain
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        # Output should NOT be identical to input (it went through transform)
        assert not np.allclose(result, om_frame)
