# Matrix-08 v3 — bounded intensity-conditioned chroma residual

**Status:** full GPU PoC complete. The saturation-only ablation is the new bounded **PoC default**, pending visual HDR10 review; it is not yet promoted as a production-wide rule.

## Purpose

The frozen low-frequency chroma field reduced seam-band chroma disagreement but left a visible colour/appearance cutoff in difficult material. The next useful degree of freedom was not a finer spatial field: it was whether the chroma correction should differ systematically between dark and bright material.

This experiment stays within the existing model family:

- v0.1 luminance path unchanged: 48 px boundary gain, base sigma 16 px, detail exponent 0.85;
- HDR centre copied directly from the master;
- existing 12-real-control ICtCp spatial chroma field unchanged;
- no temporal state, AI, LUT, MMR, segmentation, or regenerated HDR centre.

## Added model: two bounded, global slopes

Let `I` be the predicted ICtCp intensity smoothed at sigma 16 px and let `t = clip((I - centre) / normalization, -1, 1)`. The existing complex chroma field `z(x,y)` receives only this residual:

```text
saturation residual = exp(a · t)
hue residual        = exp(i · b · t)
z_final             = z(x,y) · saturation residual · hue residual
```

The fit adds at most two shot-global real controls:

- `a`: log-saturation slope, bounded so `t=±1` remains within `[0.80, 1.25]`;
- `b`: hue slope, bounded to `±5°` at `t=±1`.

The final complex factor is still clipped to the original field’s `[0.50, 2.00]` scale and `±30°` hue bounds. Chroma detail and ICtCp intensity remain untouched.

## Split and held-out gate

For the complete 186-frame Matrix-08 shot, frames with `index mod 5 != 0` formed the spatial/intensity training set (148 frames); the other 38 frames were held out. The gain pass still used all 186 frames, because it is the frozen v0.1 luminance component rather than a chroma-model selection degree of freedom.

A non-baseline ablation becomes the default only when it:

1. lowers held-out low-pass seam-band chroma disagreement by at least 0.2%; and
2. raises held-out low-pass boundary hue step by no more than 0.02°.

| Held-out model | Chroma disagreement | Change vs spatial-only | Boundary hue step | Default status |
|---|---:|---:|---:|---|
| Spatial-only (frozen) | 0.01270496 | — | 12.6508° | baseline |
| Intensity saturation | 0.01240557 | **−2.36%** | 12.6508° | **accepted** |
| Intensity hue | 0.01272997 | +0.20% | 12.8089° | rejected |
| Intensity joint | 0.01230515 | **−3.15%** | 12.8089° | review only |

Fitted global controls were:

- intensity centre `0.273783`, normalization `0.165706`;
- saturation slope `a = +0.170933`, so the residual factor ranges from `0.8429×` to `1.1864×` across the normalized intensity range;
- hue slope tried to fit `+5.247°`, hit the `+5.0°` bound, and failed the hue gate.

## Full GPU renders

Both complete runs reused the 8.0566 GB float32 cache and took about 28.5 seconds end-to-end:

| Artifact set | Left | Right | Meaning |
|---|---|---|---|
| `tools/openmatte_hdr/out/chroma_field_intensity_default/` | `v02_spatial_chroma_hdr10.mkv` | `v04_intensity_chroma_hdr10.mkv` | frozen field vs **accepted saturation default** |
| `tools/openmatte_hdr/out/chroma_field_intensity/` | `v02_spatial_chroma_hdr10.mkv` | `v04_intensity_chroma_hdr10.mkv` | frozen field vs non-default joint review candidate |

For the accepted saturation render, full-render sampled diagnostics changed as follows:

| Quantity | Frozen spatial field | Accepted saturation default |
|---|---:|---:|
| Low-pass seam-band chroma disagreement | 0.01271595 | 0.01230928 (**−3.20%**) |
| Top instantaneous hue step | 0.7020° | 0.7549° |
| Bottom instantaneous hue step | 0.7505° | 0.7581° |
| Negative RGB before clip | 0 | 0 |
| Above-peak RGB before clip | 0 | 0 |

The visual comparison remains the decision criterion. The static 1:1 crop panels show a deliberately subtle change rather than proof that the colour cutoff is solved. Review the HDR10 videos on an HDR-capable path, especially rice paper, walls, skin, dark clothing, and saturated elements.

## Validation

All three videos in the accepted-default artifact set were independently probed after render:

- 186 frames each;
- HEVC `yuv420p10le`;
- `bt2020nc`, `smpte2084`, `bt2020`, limited (`tv`) range;
- 1920×1080 individual variants and 3840×1080 side-by-side variant.

The source frame hashes match the frozen checkpoint’s first and last Matrix-08 source frames. Full machine-readable parameters and metrics are in [`matrix08_intensity_iteration.json`](matrix08_intensity_iteration.json).

## Decision

Keep **intensity saturation only** as the next bounded PoC default. Do **not** enable the intensity-dependent hue slope: its lower residual does not clear the preregistered hue-continuity guardrail.

Before adding any more flexibility, review the accepted HDR10 pair. If it still does not visually remove the cutoff, the next valid question is not another unconstrained field; it is whether a second independent shot/anchor reproduces the same intensity-saturation direction.
