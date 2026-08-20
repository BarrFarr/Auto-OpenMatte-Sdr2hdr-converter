# Sampling Audit — Frozen MMR-1

**Status:** `COMPLETE`

This is an audit of the current frozen MMR-1 observation contract, followed only when that contract passes by a reduced paired-sample measurement. It does not modify MMR-1, production, source media, synchronization, or geometry.

## Actual current correspondence implementation

For Matrix 08, each representative pair is fixed by `SDR frame = HDR frame − 19`. SDR RGB48LE is decoded at 1920×1080, converted from BT.1886 linear BT.709 to linear BT.2020, then cropped to `[0,140,1920,940]`. HDR RGB48LE is decoded at 3840×1600 and resized once with `cv2.INTER_LINEAR` to the same 1920×800 grid before PQ EOTF. Therefore every fit observation is a same-time, same-common-coordinate pair: SDR `(x,y+140)` and resized-HDR `(x,y)`.

The frozen MMR implementation obtains `Ct_base`/`Cp_base` from H0 ratio-scaled linear BT.2020 at a selected common-grid coordinate, obtains HDR target chroma from that identical coordinate, then fits its fixed six-feature/two-output ridge regression. No separate SDR/HDR random samplers exist. There is no fit-image downsample below 1920×800; the 320-pixel proxy in Phase 6 is only local cut detection. MMR itself has no luminance/chroma/hue/residual/motion correspondence rejection. Its clip and inverse-negative checks are whole-candidate numerical guards after fitting/application.

## Audit result

- Pair-contract audit: `PASS`
- Five exact pairs obey `SDR = HDR − 19`: `True`
- Common grid for every pair: `1920×800`: `True`
- Diagnostic table: `20` coordinates × 5 frames = `100` paired observations: `G:\Auto-OpenMatte-Sdr2hdr-converter — kopia\research\physical_validation\sampling_audit\results\matrix08_coordinate_audit.csv`

The table records the requested frame IDs, common coordinates, SDR source coordinates, HDR mapped coordinates, HDR continuous source-center coordinates, and raw/resized RGB values. Independent validation found exactly 100 data rows, 20 unique coordinates, 5 labels, and 0 rows that violate the fixed frame or coordinate equations.

**Direct audit conclusions:**

- **Every audited representative pair is correctly synchronized under the inherited fixed-offset contract:** yes — all five use `SDR = HDR − 19`. This is code-level confirmation of the locked contract, not a new PTS/optical synchronization measurement.
- **Every audited observation uses the same spatial common-grid coordinate:** yes — SDR `(x,y+140)` is paired with resized HDR `(x,y)`; no independent SDR/HDR point selection occurs.
- **The common fitting grid is full SDR overlap resolution:** yes — `1920×800`, derived from the native `1920×1080` SDR frame and `[0,140,1920,940]` crop.
- **Hidden analytical downsampling:** none. The inherited 320-pixel image is used only for Phase-6 cut detection, never for the H0 or MMR fit image.

The table proves the implemented coordinate contract; it does **not** establish optical-flow residuals, occlusion masks, per-pair sync confidence beyond the inherited locked offset, or truth outside the HDR overlap.

## Efficiency result

The reference fits Model-E luminance from every full-resolution common-grid pixel in four training frames and fits frozen MMR-1 from 200,000 deterministic paired rows per frame. Reduced candidates use the declared paired subset for both H0 and MMR fitting, then are scored by transformed-image difference from the reference across all five frames, including the 90% temporal holdout.

| Stage | Frames | Pairs/frame | Total fit pairs | Holdout | Median fit s | Worst fit s | Pass |
|---|---:|---:|---:|---|---:|---:|---|
| spatial | 4 | 1000 | 4000 | preserved | 3.4063 | 3.4269 | False |
| spatial | 4 | 2500 | 10000 | preserved | 3.4861 | 3.5377 | False |
| spatial | 4 | 5000 | 20000 | preserved | 3.0976 | 3.0983 | True |
| spatial | 4 | 10000 | 40000 | preserved | 2.7776 | 2.8438 | False |
| spatial | 4 | 25000 | 100000 | preserved | 3.1110 | 3.3118 | True |
| spatial | 4 | 50000 | 200000 | preserved | 3.6773 | 3.9416 | True |
| frames | 1 | 5000 | 5000 | preserved | 0.7083 | 0.7097 | False |
| frames | 2 | 5000 | 10000 | preserved | 1.4020 | 1.4459 | False |
| frames | 3 | 5000 | 15000 | preserved | 2.2693 | 2.3145 | False |
| frames | 4 | 5000 | 20000 | preserved | 3.0555 | 3.0709 | True |
| frames | 5 | 5000 | 25000 | consumed (diagnostic) | 3.5092 | 3.5567 | False |

### Recommended audit-supported configuration

- Representative frames: `4` (`10%, 25%, 50%, 75%`).
- Correctly paired fit samples per frame: `5000`.
- Total fit samples per shot: `20000`.
- Fit time: median `3.0555 s`, worst `3.0709 s`.
- Fit-only projected analysis time: 100/500/1000 shots = `305.5/1527.7/3055.5 s` median; `307.1/1535.4/3070.9 s` worst.

### What the selected configuration passed — and what it did not gate

The selected B4 run is a deterministic output-equivalence result, not an absolute HDR-quality proof. Across all five evaluated frames it stayed below the predeclared transformed-image limits: worst mean ΔE2000 versus the high-information reference was `0.046290` (limit `0.050000`) and worst P95 was `0.125557` (limit `0.150000`); all transformed outputs were finite and within the frozen numerical bounds. Its reference-relative luminance MAE ranged `0.391020–0.615146` nits, absolute ICtCp chroma error `0.00044398–0.00076074`, and non-neutral hue error `0.070301–0.196722°`.

The saved diagnostics also show candidate/reference luma seam ratios of `1.009225–1.171770` at the top boundary and `0.981720–1.000341` at the bottom boundary. They are reported evidence, but **not a programmatic acceptance gate**: the implemented pass decision gates only transformed-image ΔE2000 and output validity. No numerical seam-degradation or fitted-parameter-distance threshold was predeclared and applied. Consequently this is an **audit-supported computational recommendation for Matrix 08**, not a claim that seam continuity, parameter closeness, or physical correspondence has been fully accepted across materials or shots.

## Cross-scene status

`EXCLUDED_BY_EXISTING_STATUS`: no additional Matrix or BR2049 anchor is currently valid under the retained Phase-6 bounded-shot/geometry evidence. Matrix 03 is `ANCHOR_NEAR_CUT`, Matrix 12 is `LOCAL_SHOT_UNRESOLVED`, and BR2049 01 is `ANCHOR_NEAR_CUT` with conditional 0.9155 geometry. This audit did not search for new anchors or modify those contracts.

## Stop

STOP. The audit does not authorize MMR-2, a new model family, production changes, full-video processing, or a full-shot render.
