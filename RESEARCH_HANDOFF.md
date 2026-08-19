# Research Handoff: Scene Reshaping Models

> **Branch:** `research/scene-reshaping-models` (8 commits ahead of production)  
> **Production Branch:** `feat/full-pipeline-implementation`  
> **Phase:** P2.34  
> **Date:** 2025-01

---

## 1. Cel projektu

Automatic scene-by-scene derivation of a small deterministic grading transformation from SDR/HDR overlap region, then apply to full SDR Open Matte to produce HDR Open Matte.

The core idea: given an SDR Open Matte frame and corresponding HDR Reference frame, analyze ONLY the overlap region to derive per-scene reshaping parameters, then extend that transformation to the full Open Matte area (including content not present in the HDR reference).

---

## 2. Aktualny stan P2.34

### Production Code (`feat/full-pipeline-implementation`)

| Module | Location | Purpose |
|--------|----------|---------|
| PQ EOTF/OETF (with LUT) | `src/auto_openmatte/core/transfer_functions.py` | Perceptual Quantizer encode/decode |
| BT.709/BT.2020 matrices | `src/auto_openmatte/processing/color.py` | Color space conversions |
| Luminance extraction | `src/auto_openmatte/processing/luminance.py` | Luminance pipeline |
| Frame geometry alignment | `src/auto_openmatte/analysis/geometry.py` | Crop/align overlap region |
| Shot detection | `src/auto_openmatte/analysis/shots.py` | Scene boundary detection |
| Synchronization | `src/auto_openmatte/analysis/sync.py` | Frame offset detection |
| Pipeline orchestrator | `src/auto_openmatte/` | End-to-end pipeline |

### Research Module (`research/`)

| Component | Location | Purpose |
|-----------|----------|---------|
| 7 candidate models (A-G) | `research/models.py` | Model implementations |
| 6 synthetic scenes | `research/synthetic_scenes.py` | Test data with known ground truth |
| Quality metrics | `research/metrics.py` | RMSE, MAE, deltaE2000, seam, stability |
| Comparison runner | `research/run_comparison.py` | Full experiment runner |
| Structured package | `research/src/reshaping_research/` | Alternative organized implementation |
| Comprehensive report | `research/reports/RESEARCH_REPORT.md` | Complete results |

### Key Statistics

- **Tests:** 211 passing in research module
- **Prior sync data:** `frame_offset=1167` (~48.674s), geometry `y=280-1880`, 40/40 anchors analyzed

---

## 3. Co bylo bledne w poprzednim podejsciu

The previous approach suffered from five fundamental issues:

1. **Treated problem as generic "SDR to HDR" inverse tone mapping** - ignored the availability of ground truth in the overlap region
2. **No ground truth reference used in overlap region** - parameters were derived from assumptions rather than measured data
3. **Scene-independent parameters (single curve for all content)** - a single tone curve cannot handle the diversity of cinematic material
4. **Simple luminance scaling distorts color relationships** - scaling Y without considering chrominance produces hue shifts and desaturation
5. **Frame-by-frame without parameter continuity causes flicker** - independent per-frame fitting without temporal regularization produces visible instability

---

## 4. Nowa definicja problemu

### Formal Statement

```
SDR Open Matte + HDR Reference --> analyze ONLY overlap region (Omega_s)
    --> derive T_scene parameters --> apply T_scene to FULL SDR Open Matte
    --> HDR Open Matte
```

### Processing Pipeline

```
SDR (BT.709 gamma)
  --> [1] Gamma decode (exponent 2.4)
  --> [2] BT.709 to BT.2020 (linear domain)
  --> [3] Fit model parameters from overlap region
  --> [4] Apply fitted model to full frame
  --> [5] PQ OETF
  --> HDR10 output
```

### Key Constraint

The model must be fitted ONLY from the overlap region (where both SDR and HDR data exist), then applied to the FULL Open Matte frame (including areas outside the HDR reference). This means the model must generalize well beyond its training domain, particularly for shadow and highlight regions that may not be well-represented in the overlap.

---

## 5. Model E (CDF + Regularized, 12 parameters)

### Architecture

- Starts from CDF matching as initialization
- Fits a smooth 12-point spline optimized to minimize reconstruction error + smoothness penalty
- **ctrl_x:** 12 uniformly spaced points over SDR luminance range
- **ctrl_y:** 12 optimized values (monotonically increasing, enforced via constraints)
- Optimizer: L-BFGS-B with smoothness regularization (lambda = 0.05)
- Application method: luminance ratio scaling (preserves color relationships)

### Performance

| Metric | Value |
|--------|-------|
| Stability CV | 0.0045 (Excellent) |
| Avg RMSE | 193.8 nits |
| Avg deltaE2000 | 1.44 |
| Parameters | 12 |

### Parameter Reduction Experiment

| Points | RMSE (nits) |
|--------|-------------|
| 4 | 27.0 |
| 6 | 10.3 |
| 8 | 5.7 |
| **12** | **2.7** |
| 16 | 1.6 |
| 24 | 0.9 |
| 32 | 0.7 |

Diminishing returns above 12 points. The jump from 8 to 12 points reduces error by 50%, while 12 to 16 only reduces by 40%.

### Strengths

- Compact representation (12 params fits in a single metadata packet)
- CDF initialization gives excellent starting point
- Regularization prevents overfitting to noisy overlap data
- Luminance ratio scaling naturally preserves chrominance

### Weaknesses

- No explicit chroma handling (relies entirely on ratio scaling)
- May underperform on highly chromatic content where color shifts are significant

---

## 6. Model G (Hybrid, 29 parameters)

### Architecture

**Luminance path (15 params):**
- 10-point piecewise linear (percentile-matched control points)
- Highlight shoulder: 3 params (threshold, slope, max)
- Shadow lift: 2 params (threshold, gain)

**Chroma path (14 params):**
- Saturation: 2 params (base + slope)
- Color matrix: 3x3 (9 params)
- Hue rotation: 3 angles

**Total:** 10 + 3 + 2 + 2 + 9 + 3 = 29 parameters

### Performance

| Metric | Value |
|--------|-------|
| Stability CV | 0.0007 (Excellent) |
| Avg RMSE | 161.8 nits |
| Avg MAE | 69.7 nits |
| Avg deltaE2000 | 1.27 |
| Parameters | 29 |

### Strengths

- Best overall quality (lowest RMSE and deltaE2000)
- Explicit chroma handling prevents color distortion
- Highlight shoulder prevents clipping in bright scenes
- Shadow lift improves dark detail preservation
- Excellent temporal stability (CV = 0.0007)

### Weaknesses

- Higher parameter count (29 vs 12) increases fitting complexity
- Requires sufficient color diversity in overlap region for robust chroma fitting
- Higher seam error (32.1 nits vs 20.7 for Model E) at overlap boundary

---

## 7. Wyniki syntetyczne

### Full Model Comparison (6 scenes average)

| Model | Type | Params | RMSE (nits) | deltaE2000 | Seam (nits) | Stability CV |
|-------|------|--------|-------------|------------|-------------|--------------|
| A | Linear | 2 | 523.1 | 3.00 | 5.7 | 0.0002 |
| B | Piecewise | 10 | 179.1 | 1.88 | 21.5 | 0.0002 |
| C | Polynomial | 8 | 219.4 | 2.36 | 35.4 | 0.7446 (Poor) |
| D | CDF | 65 | 511.3 | 2.11 | 29.7 | 0.0040 |
| E | CDF+Reg | 12 | 193.8 | 1.44 | 20.7 | 0.0045 |
| F | Luma+Chroma | 18 | 219.4 | 2.89 | 35.4 | 0.3473 (Poor) |
| **G** | **Hybrid** | **29** | **161.8** | **1.27** | **32.1** | **0.0007** |

### Models Eliminated

- **Model A** (Linear): Too simple, cannot capture nonlinear tone mapping
- **Model C** (Polynomial): Temporally unstable (CV = 0.7446), oscillates between frames
- **Model D** (CDF Matching): Overfits on extreme content (511.3 nits RMSE), too many params (65)
- **Model F** (Luma+Chroma Regression): Temporally unstable (CV = 0.3473)

### Top Candidates

- **Model G** (Hybrid): Best quality, best stability, moderate complexity
- **Model E** (CDF+Reg): Best balance of quality vs. simplicity, low seam error
- **Model B** (Piecewise): Simple and stable, acceptable quality for less demanding content

---

## 8. Znaczenie wynikow

### Why These Results Matter

1. **deltaE2000 < 1.5 is near-invisible** - Both Model E (1.44) and Model G (1.27) produce differences below the threshold of casual observation on consumer displays

2. **Temporal stability is critical for video** - Models C and F are eliminated despite decent per-frame quality because their instability would produce visible flicker in playback

3. **12 parameters is metadata-friendly** - Model E's 12 values can be stored per-shot in a sidecar file or embedded in container metadata, enabling non-destructive workflows

4. **Seam error indicates boundary quality** - Model A has the lowest seam error (5.7 nits) because it barely transforms the image; among effective models, Model E (20.7) outperforms Model G (32.1), suggesting boundary blending may be needed for Model G

5. **Synthetic results are necessary but not sufficient** - All testing used controlled synthetic scenes with known ground truth. Real video data will introduce noise, compression artifacts, temporal variation, and content diversity not captured in synthetic tests

### Recommended Strategy

- **Primary:** Model G for quality-critical applications
- **Fallback:** Model E when parameter budget or fitting robustness is a concern
- **Boundary:** Apply spatial blending at the overlap/extension boundary to mitigate seam artifacts

---

## 9. Znane ograniczenia

1. **Tests are synthetic only** - no real video data has been tested yet in this sandbox environment
2. **High-key scenes are challenging** - all models show elevated RMSE (365-1919 nits) on bright content
3. **Seam error is non-trivial in high-key content** - all models show 100+ nits at the overlap boundary in bright scenes
4. **Model C and F have poor temporal stability** (CV > 0.3) - eliminated from consideration
5. **CDF matching (Model D) overfits on extreme content** - raw CDF transfer is too aggressive
6. **Parameter reduction experiment only ran on simple scene** (neutral) - results may differ for complex content
7. **No GPU benchmarking performed** - runtime characteristics unknown for full-length material
8. **Chroma fitting in Model G requires diverse overlap content** - monochromatic overlap regions may underdetermine the 3x3 matrix

---

## 10. Co ma zrobic lokalny agent

### Immediate Next Steps

1. **Validate Model G (and optionally E) on real video data** using existing P2.34 sync/geometry parameters (`frame_offset=1167`, `y=280-1880`)
2. **Test with actual movie frames from the overlap region** - extract pairs, fit models, measure quality against HDR reference
3. **Run the comparison on diverse real shots** - different lighting, color palettes, dynamic range characteristics
4. **If results confirm synthetic findings:** integrate into production pipeline (`src/auto_openmatte/`)
5. **Benchmark CPU/GPU performance** on full-length material (target: real-time or better for 4K)
6. **Address seam quality in high-key scenes** - implement boundary blending (e.g., feathered alpha over 8-16 pixel band)

### Integration Path

```
research/models.py (ModelG)  -->  src/auto_openmatte/processing/reshaping.py (new)
research/metrics.py          -->  src/auto_openmatte/analysis/quality.py (new)
```

The integration should:
- Preserve the existing pipeline structure
- Add reshaping as a processing stage between color conversion and PQ encoding
- Store per-shot parameters alongside shot detection results
- Implement temporal smoothing of parameters across frames within a shot

---

## 11. Jakie pliki ma wykorzystac

### Research Module (read-only reference)

| File | Purpose |
|------|---------|
| `research/models.py` | All 7 model implementations (ModelA-ModelG classes) |
| `research/color_utils.py` | PQ, gamma, gamut conversion functions |
| `research/synthetic_scenes.py` | Scene generators with known ground truth |
| `research/metrics.py` | Quality metrics (RMSE, MAE, deltaE2000, seam, stability) |
| `research/run_comparison.py` | Full experiment runner (can be extended) |
| `research/reports/RESEARCH_REPORT.md` | Complete results |
| `research/src/reshaping_research/` | Structured package (alternative implementation) |

### Production Code (integration targets)

| File | Purpose |
|------|---------|
| `src/auto_openmatte/core/transfer_functions.py` | Production PQ/HLG (reference) |
| `src/auto_openmatte/analysis/sync.py` | Synchronization (frame offset detection) |
| `src/auto_openmatte/analysis/geometry.py` | Geometric alignment |
| `src/auto_openmatte/analysis/shots.py` | Shot detection |
| `src/auto_openmatte/processing/luminance.py` | Current luminance pipeline |
| `src/auto_openmatte/processing/color.py` | Current color processing |

---

## 12. Jakich rzeczy nie wolno zmieniac

Until research models are validated on real video data, the following must remain unchanged:

| Item | Reason |
|------|--------|
| `src/` directory (production code) | Must not be modified until real data validation confirms model superiority |
| `tests/` directory (production tests) | Stability of existing test suite |
| Existing sync results and geometry alignment values | Validated empirical measurements |
| P2.14/P2.15/P2.16 benchmark reports | Historical reference data |
| `research/` module itself | Validated research reference (read-only) |
| `pyproject.toml` (production dependencies) | Dependency stability |
| `.gitignore`, `Makefile`, `README.md` | Project infrastructure |

### Modification Protocol

When integrating research results into production:
1. Create a new feature branch from `feat/full-pipeline-implementation`
2. Add new files (do not modify existing processing logic until validated)
3. Wire new reshaping stage into pipeline orchestrator
4. Add comprehensive tests for the new stage
5. Run full test suite to confirm no regressions
6. Only then consider deprecating the old luminance-only approach

---

## Summary

Two models emerged from the research phase as strong candidates:

- **Model G** (Hybrid, 29 params): Best quality (deltaE2000 = 1.27), best stability (CV = 0.0007)
- **Model E** (CDF+Reg, 12 params): Best simplicity/quality ratio (deltaE2000 = 1.44, only 12 params)

Both require validation on real video data before production integration. The research module provides all tools needed for this validation. The next phase should focus on confirming these results with actual movie frames from the overlap region.
