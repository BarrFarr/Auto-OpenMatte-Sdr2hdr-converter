# Scene-Based SDR-to-HDR Reshaping: Research Report

**Generated:** 2026-08-16 23:40:14
**Total runtime:** 114.0s
**Image dimensions:** HDR 1920x1600, SDR OM 1920x2160
**Overlap region:** SDR rows 280:1880

## A. Project Audit: What Existing Code Does Right

The production codebase (`src/auto_openmatte/`) already implements:

1. **PQ EOTF/OETF** - correct ST 2084 transfer functions with LUT acceleration
2. **BT.709/BT.2020 color matrices** - proper 3x3 gamut conversion
3. **Luminance extraction** - BT.2020 weighted luminance computation
4. **Frame geometry** - correct identification of open matte overlap regions
5. **Pipeline structure** - modular processing chain with proper data flow

These components are technically sound and reusable in the improved pipeline.

## B. Conceptual Problems

The current approach treats SDR-to-HDR as an **inverse tone mapping** problem,
applying a fixed expansion curve without reference data. The fundamental issues:

1. **No ground truth** - without HDR reference in the overlap region, any expansion
   is guesswork. The same SDR value could map to vastly different HDR values
   depending on the original scene content.
2. **Scene-independent parameters** - a single curve cannot handle the diversity of
   real content (dark scenes vs bright scenes, saturated vs neutral).
3. **Chroma handling** - simple luminance scaling distorts color relationships.
   Saturated highlights require different treatment than neutral midtones.
4. **Temporal coherence** - frame-by-frame processing without parameter continuity
   causes flicker and instability.

The correct framing: **reference-guided reshaping** using the overlap region where
both SDR and HDR data exist to estimate scene-specific transform parameters.

## C. Research Background

### Dolby Backward Reshaping (US Patent 9,613,407)
- Polynomial mapping with per-scene coefficients stored in metadata
- 3-channel piecewise polynomial, typically degree 1-3 per piece
- Designed for reconstruction from single-layer (SDR) encoding

### CDF-Based Transfer (Histogram Specification)
- Match cumulative distribution functions between SDR and HDR luminance
- Non-parametric: stores full transfer LUT (~64-256 points)
- Guaranteed monotonic, preserves relative ordering

### Multi-Modal Regression (MMR)
- Multiple regression channels for different luminance ranges
- 3x3 matrix per luminance segment for color correction
- Dolby Vision Profile 7 uses 3-pivot MMR

### Piecewise Techniques
- Knot-based splines with monotonicity constraints
- Sigmoid/logistic shoulder functions for highlight rolloff
- Separate shadow lift for dark region reconstruction

## D. Proposed Architecture

```
SDR Open Matte (BT.709 gamma)
    |
    v
[1] Gamma decode (power 2.4)
    |
    v
[2] BT.709 -> BT.2020 linear (3x3 matrix)
    |
    +---> Extract overlap region (rows 280:1880)
    |         |
    |         v
    |     [3] Fit model params (SDR_linear vs HDR_linear)
    |         |
    v         v
[4] Apply fitted model to FULL SDR Open Matte
    |
    v
[5] Apply PQ OETF -> HDR10 output (BT.2020 + PQ)
```

## E. Candidate Models

| Model | Name | Parameters | Description |
|-------|------|-----------|-------------|
| A | Linear Gain | 2 | Y'=a*Y, C'=b*C |
| B | Piecewise Linear | 10 | 8 luma knots + chroma |
| C | Monotonic Polynomial | 8 | Degree 4 luma + degree 2 sat |
| D | CDF Matching | 65 | 64-pt histogram transfer LUT |
| E | CDF + Regularized | 12 | Smooth 12-pt optimized curve |
| F | Luma + Chroma Matrix | 18 | Degree 4 poly + 3x3 matrix |
| G | Hybrid | 29 | Percentile + shoulder/shadow + matrix + hue |

## F. Synthetic Test Results

### RMSE (nits) by Model and Scene

| Model | colorful | difficult | high_key | low_key | mixed | neutral | Avg |
|---|---|---|---|---|---|---|---|
| A: Linear Gain | 488.4 | 1158.5 | 1057.5 | 16.5 | 323.6 | 93.9 | 523.1 |
| B: Piecewise Linear | 185.6 | 385.9 | 365.4 | 46.3 | 67.0 | 24.2 | 179.1 |
| C: Monotonic Polynomial | 169.1 | 378.7 | 764.3 | 0.0 | 4.4 | 0.0 | 219.4 |
| D: CDF Matching | 181.0 | 964.9 | 1918.8 | 0.1 | 3.2 | 0.0 | 511.3 |
| E: CDF + Regularized | 170.0 | 389.7 | 591.0 | 3.2 | 6.1 | 2.7 | 193.8 |
| F: Luma + Chroma Matrix | 170.4 | 378.7 | 762.9 | 0.0 | 4.4 | 0.0 | 219.4 |
| G: Hybrid | 172.6 | 352.9 | 380.1 | 35.0 | 12.7 | 17.7 | 161.8 |

### MAE (nits) by Model and Scene

| Model | colorful | difficult | high_key | low_key | mixed | neutral | Avg |
|---|---|---|---|---|---|---|---|
| A: Linear Gain | 252.4 | 540.3 | 572.2 | 1.3 | 165.9 | 71.0 | 267.2 |
| B: Piecewise Linear | 115.5 | 178.4 | 185.3 | 5.7 | 42.6 | 15.1 | 90.4 |
| C: Monotonic Polynomial | 100.3 | 176.1 | 569.1 | 0.0 | 3.2 | 0.0 | 141.5 |
| D: CDF Matching | 97.8 | 334.1 | 853.9 | 0.0 | 1.6 | 0.0 | 214.6 |
| E: CDF + Regularized | 99.9 | 162.9 | 302.9 | 1.7 | 4.1 | 1.5 | 95.5 |
| F: Luma + Chroma Matrix | 99.9 | 175.9 | 567.0 | 0.0 | 3.2 | 0.0 | 141.0 |
| G: Hybrid | 104.4 | 142.2 | 147.7 | 5.2 | 8.8 | 9.6 | 69.7 |

### Delta E 2000 by Model and Scene

| Model | colorful | difficult | high_key | low_key | mixed | neutral | Avg |
|---|---|---|---|---|---|---|---|
| A: Linear Gain | 5.69 | 4.58 | 4.11 | 0.03 | 2.53 | 1.08 | 3.00 |
| B: Piecewise Linear | 5.59 | 2.08 | 1.84 | 0.18 | 1.33 | 0.28 | 1.88 |
| C: Monotonic Polynomial | 3.95 | 1.83 | 7.93 | 0.00 | 0.47 | 0.00 | 2.36 |
| D: CDF Matching | 5.52 | 2.08 | 3.79 | 0.01 | 1.16 | 0.08 | 2.11 |
| E: CDF + Regularized | 5.10 | 1.39 | 1.54 | 0.10 | 0.47 | 0.02 | 1.44 |
| F: Luma + Chroma Matrix | 5.37 | 2.11 | 8.47 | 0.01 | 1.30 | 0.09 | 2.89 |
| G: Hybrid | 4.02 | 1.44 | 1.22 | 0.15 | 0.61 | 0.18 | 1.27 |

### Seam Error (nits) by Model and Scene

| Model | colorful | difficult | high_key | low_key | mixed | neutral | Avg |
|---|---|---|---|---|---|---|---|
| A: Linear Gain | 2.4 | 1.0 | 17.6 | 0.0 | 10.3 | 3.0 | 5.7 |
| B: Piecewise Linear | 3.0 | 1.3 | 120.1 | 0.0 | 0.2 | 4.5 | 21.5 |
| C: Monotonic Polynomial | 3.1 | 1.7 | 175.8 | 0.0 | 27.6 | 4.1 | 35.4 |
| D: CDF Matching | 3.3 | 1.1 | 159.0 | 0.0 | 10.7 | 4.1 | 29.7 |
| E: CDF + Regularized | 3.1 | 1.2 | 107.6 | 0.0 | 8.1 | 4.1 | 20.7 |
| F: Luma + Chroma Matrix | 3.1 | 1.7 | 175.8 | 0.0 | 27.6 | 4.1 | 35.4 |
| G: Hybrid | 3.3 | 1.0 | 165.3 | 0.0 | 18.9 | 4.2 | 32.1 |

## G. Stability Results

Coefficient of variation across 5 refits with additive Gaussian noise (sigma=0.005):

| Model | CV Score | Rating |
|-------|----------|--------|
| A: Linear Gain | 0.0002 | Excellent |
| B: Piecewise Linear | 0.0002 | Excellent |
| C: Monotonic Polynomial | 0.7446 | Poor |
| D: CDF Matching | 0.0040 | Excellent |
| E: CDF + Regularized | 0.0045 | Excellent |
| F: Luma + Chroma Matrix | 0.3473 | Poor |
| G: Hybrid | 0.0007 | Excellent |

Lower CV = more temporally stable parameters (less flicker risk).

## H. Parameter Reduction Curve

Model E (CDF + Regularized) tested with varying control point counts:

| Control Points | RMSE (nits) |
|---------------|-------------|
| 4 | 27.0 |
| 6 | 10.3 |
| 8 | 5.7 |
| 12 | 2.7 |
| 16 | 1.6 |
| 24 | 0.9 |
| 32 | 0.7 |

Diminishing returns visible above 12 control points for typical content.

## I. Recommendation

**Best overall RMSE:** G: Hybrid (161.8 nits avg)
**Best overall Delta E:** G: Hybrid (1.27 avg)

### Selection Criteria

The recommended model balances:
1. Reconstruction accuracy (low RMSE and Delta E)
2. Parameter efficiency (fewer params = easier metadata, faster fitting)
3. Temporal stability (low CV = no flicker)
4. Seam quality (smooth transitions at overlap boundaries)

### Final Recommendation

**Recommended model: G: Hybrid**

Rationale:
- Average RMSE: 161.8 nits
- Average Delta E: 1.27
- Stability CV: 0.0007
- Parameter count: 29

This model provides the best trade-off between accuracy, stability, and simplicity
for production integration. The overlap region provides sufficient constraint data
to estimate scene-specific parameters on a per-shot or per-frame basis.

### Integration Path

1. Extract overlap region from frame geometry (already implemented)
2. Linearize SDR overlap (gamma 2.4 decode + BT.709->BT.2020)
3. Fit model parameters from overlap (SDR vs HDR luminance + chroma)
4. Apply fitted model to full SDR open matte extent
5. Encode output as PQ (ST 2084) in BT.2020 container

---

*Report generated by research/run_comparison.py*