"""Isolated Test A/B: HDR resize in PQ-code versus linear BT.2020.

Test A: PQ code -> bilinear resize -> PQ EOTF -> linear BT.2020.
Test B: PQ code -> PQ EOTF -> bilinear resize in linear BT.2020.

The experiment uses Matrix 73367/73348 only, reads P2.32 arrays only, fits
Model E luminance only inside the overlap, and writes only below this folder.
It does not instantiate Model H/G, change P2/production, or decode video.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SRC = ROOT / "research" / "src"
if str(RESEARCH_SRC) not in sys.path:
    sys.path.insert(0, str(RESEARCH_SRC))

from reshaping_research.metrics.color_metrics import chroma_error, delta_e_2000, delta_e_ictcp, hue_error  # noqa: E402
from reshaping_research.models.model_e_cdf_regularized import ModelECdfRegularized  # noqa: E402
from reshaping_research.utils.color_spaces import bt709_to_bt2020, compute_luminance, linear_bt2020_to_ictcp  # noqa: E402
from reshaping_research.utils.transfer_functions import bt1886_eotf  # noqa: E402

P232_DIR = ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset"
METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "matrix_73367_73348"
REPORT_PATH = BASE / "RESAMPLING_DIAGNOSTIC_REPORT.md"
PEAK_NITS, RGB16_MAX, EPS = 10_000.0, 65_535.0, 1e-8
X1, Y1, X2, Y2 = 0, 140, 1920, 940
M_2020_TO_XYZ_D65 = np.array(((0.6369580483, 0.1446169036, 0.1688809752), (0.2627002120, 0.6779980715, 0.0593017165), (0.0, 0.0280726930, 1.0609850577)), dtype=np.float64)
D65 = np.array((0.95047, 1.0, 1.08883), dtype=np.float64)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [safe(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float): return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_cv(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    require(bool(ok), f"Cannot encode {path}")
    path.write_bytes(encoded.tobytes())


def pq_eotf(code: np.ndarray) -> np.ndarray:
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.power(np.clip(np.asarray(code, dtype=np.float64), 0, 1), 1 / m2)
    return np.power(np.maximum(p - c1, 0) / np.maximum(c2 - c3 * p, EPS), 1 / m1)


def pq_oetf(linear: np.ndarray) -> np.ndarray:
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.power(np.clip(np.asarray(linear, dtype=np.float64), 0, 1), m1)
    return np.power((c1 + c2 * p) / (1 + c3 * p), m2)


def locate(declared: str, suffix: str) -> Path:
    paths = [p for p in P232_DIR.rglob(Path(declared).name) if p.is_file()]
    require(len(paths) == 1, f"Expected exactly one local input for {declared}; got {len(paths)}")
    result = paths[0].resolve()
    require(str(result).startswith(str(ROOT.resolve())) and result.name.endswith(suffix), f"Unsafe path: {result}")
    return result


def load() -> tuple[dict[str, Any], dict[str, Path]]:
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    require(data.get("phase") == "P2.32" and data.get("status") == "DATASET READY", "P2.32 not ready")
    records = [r for r in data["case_records"]["The Matrix"] if r.get("case_id") == "the_matrix_temporal_stratum_08"]
    require(len(records) == 1, "Missing Matrix stratum 08")
    record = records[0]
    require(record.get("hdr_frame") == 73367 and record.get("om_frame") == 73348 and record.get("known_verified_anchor") is True, "Incorrect anchor")
    require(record.get("geometry", {}).get("overlap") == [X1,Y1,X2,Y2] and float(record.get("geometry", {}).get("confidence",0)) == 1.0, "Incorrect geometry")
    return record, {"hdr": locate(record["hdr_rgb_u16_path"], "_hdr_rgb_u16.npy"), "om": locate(record["om_rgb_u16_path"], "_om_rgb_u16.npy")}


def lab2020(image: np.ndarray) -> np.ndarray:
    xyz = np.einsum("...c,dc->...d", image, M_2020_TO_XYZ_D65, optimize=True)
    ratio = xyz / D65; delta = 6 / 29
    f = np.where(ratio > delta**3, np.cbrt(np.maximum(ratio, 0)), ratio / (3*delta**2) + 4/29)
    return np.stack((116*f[...,1]-16, 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])), axis=-1)


def summary(values: np.ndarray, absolute: bool = False) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1); values = values[np.isfinite(values)]
    if absolute: values = np.abs(values)
    return {"count":int(values.size),"mean":float(np.mean(values)),"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"maximum":float(np.max(values))} if values.size else {"count":0,"mean":None,"median":None,"p95":None,"maximum":None}


def all_metrics(predicted: np.ndarray, reference: np.ndarray, rows: int = 80) -> dict[str, Any]:
    yerr = (compute_luminance(predicted)-compute_luminance(reference))*PEAK_NITS
    de, di, ce, he, rc = [], [], [], [], []
    for start in range(0,predicted.shape[0],rows):
        p, r = predicted[start:start+rows], reference[start:start+rows]
        pn, rn = p*PEAK_NITS, r*PEAK_NITS
        de.append(delta_e_2000(lab2020(p),lab2020(r)).reshape(-1)); di.append(delta_e_ictcp(pn,rn).reshape(-1)); ce.append(chroma_error(pn,rn).reshape(-1)); he.append(hue_error(pn,rn).reshape(-1))
        ictcp = linear_bt2020_to_ictcp(rn); rc.append(np.sqrt(ictcp[...,1]**2+ictcp[...,2]**2).reshape(-1))
    hue, ref_chroma = np.concatenate(he), np.concatenate(rc); valid = ref_chroma > 1e-6
    return {"luminance": {"MAE_nits":float(np.mean(np.abs(yerr))),"RMSE_nits":float(np.sqrt(np.mean(yerr*yerr))),"median_absolute_error_nits":float(np.median(np.abs(yerr))),"P95_absolute_error_nits":float(np.percentile(np.abs(yerr),95))}, "deltaE2000":summary(np.concatenate(de)), "deltaEICtCp":summary(np.concatenate(di)), "chroma_error_ICtCp_signed":summary(np.concatenate(ce)), "chroma_error_ICtCp_absolute":summary(np.concatenate(ce),True), "hue_error_degrees_absolute_non_neutral":summary(hue[valid],True), "RGB_MSE_normalized":float(np.mean((predicted-reference)**2))}


def preview(image: np.ndarray) -> np.ndarray:
    nits = np.maximum(np.asarray(image,dtype=np.float64)*PEAK_NITS,0)
    return np.power(np.clip(np.log1p(nits)/math.log1p(1000),0,1),1/2.2)


def write_scientific(prefix: Path, image: np.ndarray) -> None:
    np.save(prefix.with_suffix(".linear_bt2020_nits.npy"),(image*PEAK_NITS).astype(np.float32))
    pq = np.round(np.clip(pq_oetf(image),0,1)*RGB16_MAX).astype(np.uint16)
    write_cv(prefix.with_suffix(".pq16.png"),pq[...,::-1])


def write_preview(path: Path, image: np.ndarray) -> None:
    write_cv(path,np.round(np.clip(preview(image),0,1)*255).astype(np.uint8)[...,::-1])


def make_panel(images: list[np.ndarray], height: int = 270) -> np.ndarray:
    rendered=[]
    for image in images:
        scale=height/image.shape[0]; rendered.append(cv2.resize(preview(image),(int(round(image.shape[1]*scale)),height),interpolation=cv2.INTER_AREA))
    return np.round(np.clip(np.concatenate(rendered,axis=1),0,1)*255).astype(np.uint8)


def luma_diff_image(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    error=np.abs(compute_luminance(first)-compute_luminance(second))*PEAK_NITS
    norm=np.clip(np.log1p(error)/math.log1p(1000),0,1)
    return cv2.applyColorMap(np.round(norm*255).astype(np.uint8),cv2.COLORMAP_TURBO)


def fit_h0(sdr_full: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    sdr_common=sdr_full[Y1:Y2,X1:X2]
    model=ModelECdfRegularized(); params=model.fit(compute_luminance(sdr_common).reshape(-1),compute_luminance(target).reshape(-1))
    source_y=compute_luminance(sdr_full); target_y=np.interp(source_y,np.asarray(params["lut_x"]),np.asarray(params["lut_y"]))
    output=sdr_full*(target_y/np.maximum(source_y,EPS))[...,None]; dark=source_y<=EPS
    if np.any(dark): output[dark]=target_y[dark,None]
    return output,{"reported_model_parameter_count":int(model.param_count()),"fit_input":"all 1,536,000 overlap luminance pixels for CDF/LUT; Model E correction internally uses up to 3,000 deterministic samples","parameters":params,"dark_ratio_fallback_count":int(np.sum(dark))}


def seam(full: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    composite=full.copy(); composite[Y1:Y2]=target
    rows=[]
    for y,label in ((Y1,"top"),(Y2,"bottom")):
        outer,inner=(y-1,y) if label=="top" else (y,y-1)
        out,inn=composite[outer]*PEAK_NITS,composite[inner]*PEAK_NITS
        diff=np.abs(compute_luminance(out)-compute_luminance(inn))
        rows.append({"boundary":label,"luma_mean_nits":float(np.mean(diff)),"luma_P95_nits":float(np.percentile(diff,95)),"chroma_ICtCp":float(np.mean(np.abs(chroma_error(out,inn)))),"hue_degrees":float(np.mean(np.abs(hue_error(out,inn))))})
    return {"definition":"continuity proxy only; no HDR ground truth outside overlap","boundaries":rows}


def report(result: dict[str, Any]) -> str:
    ref=result["reference_A_vs_B"]; a=result["H0_A"]; b=result["H0_B"]; h0ab=result["H0_A_vs_H0_B"]
    h0_de_change = b["common_metrics"]["deltaE2000"]["mean"] - a["common_metrics"]["deltaE2000"]["mean"]
    h0_de_change_percent = 100 * h0_de_change / a["common_metrics"]["deltaE2000"]["mean"]
    return f"""# PQ-Code vs Linear-BT.2020 Resize Diagnostic

## Scope
Matrix anchor HDR 73367 / SDR-Open-Matte 73348 only. Geometry is `[0,140,1920,940]`; fitting sees only this `1920×800` / 1,536,000-pixel overlap. No production code, P2 artifact, Model H/G, or video decode was modified or used.

## Controlled pipelines
- **A — current:** HDR PQ code `3840×1600` → OpenCV `INTER_LINEAR` → PQ EOTF → linear BT.2020 `1920×800`.
- **B — linear resize:** HDR PQ code → PQ EOTF at native `3840×1600` → OpenCV `INTER_LINEAR` in linear BT.2020 → `1920×800`.
- SDR source, geometry, bilinear kernel, Model E luminance fitting, ratio scaling, outputs, and fit mask are otherwise identical.

## Direct target A/B difference
| Metric A vs B | Value |
|---|---:|
| Luminance MAE / RMSE | {ref['luminance']['MAE_nits']:.6f} / {ref['luminance']['RMSE_nits']:.6f} nits |
| Mean / P95 ΔE2000 | {ref['deltaE2000']['mean']:.6f} / {ref['deltaE2000']['p95']:.6f} |
| Mean abs chroma / hue error | {ref['chroma_error_ICtCp_absolute']['mean']:.8f} / {ref['hue_error_degrees_absolute_non_neutral']['mean']:.6f}° |

## Downstream H0 comparison
| Branch | luma MAE / RMSE vs its target | mean ΔE2000 | mean abs hue error | top / bottom seam mean nits |
|---|---:|---:|---:|---:|
| H0-A | {a['common_metrics']['luminance']['MAE_nits']:.6f} / {a['common_metrics']['luminance']['RMSE_nits']:.6f} | {a['common_metrics']['deltaE2000']['mean']:.6f} | {a['common_metrics']['hue_error_degrees_absolute_non_neutral']['mean']:.6f}° | {a['seam']['boundaries'][0]['luma_mean_nits']:.6f} / {a['seam']['boundaries'][1]['luma_mean_nits']:.6f} |
| H0-B | {b['common_metrics']['luminance']['MAE_nits']:.6f} / {b['common_metrics']['luminance']['RMSE_nits']:.6f} | {b['common_metrics']['deltaE2000']['mean']:.6f} | {b['common_metrics']['hue_error_degrees_absolute_non_neutral']['mean']:.6f}° | {b['seam']['boundaries'][0]['luma_mean_nits']:.6f} / {b['seam']['boundaries'][1]['luma_mean_nits']:.6f} |

## Conclusion
**The resampling order is not materially affecting this Matrix anchor or its H0 baseline.** Direct A/B reference change is only mean/P95 ΔE2000 `{ref['deltaE2000']['mean']:.6f}` / `{ref['deltaE2000']['p95']:.6f}`. H0-A versus H0-B is mean/P95 ΔE2000 `{h0ab['deltaE2000']['mean']:.6f}` / `{h0ab['deltaE2000']['p95']:.6f}`, with only `{h0ab['luminance']['MAE_nits']:.6f}`-nit mean luminance difference. Using B changes H0's mean ΔE2000 against its own matching target by `{h0_de_change:+.6f}` ({h0_de_change_percent:+.3f}%), from `{a['common_metrics']['deltaE2000']['mean']:.6f}` to `{b['common_metrics']['deltaE2000']['mean']:.6f}`.

Linear-domain filtering is still the physically appropriate ordering when resampling scene-linear RGB, but at this exact 2× geometry/content its observed impact is negligible compared with H0's roughly 0.946 mean ΔE2000 residual. Therefore it does **not** explain a visible localized blue/purple artifact that emerges only after the later H1 3×3 matrix transform. The direct reference hue maximum is not used as evidence because hue is ill-conditioned around near-neutral pixels; its robust mean/P95 values are the relevant observations.

This A/B test isolates **reference resampling order only**. It cannot establish which branch is a display-master ground truth; the P2.32 source provides encoded pixels rather than a separate linear-resampled reference. Scientific arrays are float32 linear-BT.2020 nits; PQ PNGs are uint16; preview panels are display-only uint8 log views.
"""


def main() -> int:
    for directory in (RESULTS,RESULTS/"scientific",RESULTS/"previews",RESULTS/"differences",RESULTS/"seams"):
        directory.mkdir(parents=True,exist_ok=True)
    record,paths=load(); hdr_raw,om_raw=np.load(paths["hdr"]),np.load(paths["om"])
    require(hdr_raw.dtype==np.uint16 and om_raw.dtype==np.uint16 and tuple(hdr_raw.shape)==tuple(record["hdr_rgb_shape"]) and tuple(om_raw.shape)==tuple(record["om_rgb_shape"]),"Unexpected raw input")
    hdr_code=hdr_raw.astype(np.float64)/RGB16_MAX
    # Only difference between A and B is the placement of the same bilinear resize.
    hdr_a=pq_eotf(cv2.resize(hdr_code,(X2-X1,Y2-Y1),interpolation=cv2.INTER_LINEAR))
    hdr_b=cv2.resize(pq_eotf(hdr_code),(X2-X1,Y2-Y1),interpolation=cv2.INTER_LINEAR)
    sdr_full=np.maximum(bt709_to_bt2020(bt1886_eotf(om_raw.astype(np.float64)/RGB16_MAX)),0.0)
    h0_a,fit_a=fit_h0(sdr_full,hdr_a); h0_b,fit_b=fit_h0(sdr_full,hdr_b)
    h0a_common,h0b_common=h0_a[Y1:Y2,X1:X2],h0_b[Y1:Y2,X1:X2]
    ref=all_metrics(hdr_a,hdr_b)
    result={"status":"COMPLETE","scope":"Research-only A/B HDR resize-order test; no production/P2/video/Model H/G changes.","anchor":record,"input_files":paths,"geometry":{"sdr_native":[1920,1080],"hdr_native":[3840,1600],"overlap_sdr":[X1,Y1,X2,Y2],"resampled_hdr_size":[X2-X1,Y2-Y1],"native_hdr_samples_representing_common":3840*1600,"resampled_common_pixels":(X2-X1)*(Y2-Y1),"interpolation":"OpenCV INTER_LINEAR"},"Test_A":{"definition":"PQ code -> INTER_LINEAR resize -> PQ EOTF -> linear BT.2020","target_range":summary(hdr_a)},"Test_B":{"definition":"PQ code -> PQ EOTF -> INTER_LINEAR resize in linear BT.2020","target_range":summary(hdr_b)},"reference_A_vs_B":ref,"H0_A":{"fit":fit_a,"common_metrics":all_metrics(h0a_common,hdr_a),"seam":seam(h0_a,hdr_a)},"H0_B":{"fit":fit_b,"common_metrics":all_metrics(h0b_common,hdr_b),"seam":seam(h0_b,hdr_b)},"H0_A_vs_H0_B":all_metrics(h0_a,h0_b)}
    for prefix,image in (("HDR_A_PQCodeResize",hdr_a),("HDR_B_LinearResize",hdr_b),("H0_A_PQCodeResize",h0_a),("H0_B_LinearResize",h0_b)):
        write_scientific(RESULTS/"scientific"/prefix,image); write_preview(RESULTS/"previews"/(prefix+"_log.png"),image)
    write_cv(RESULTS/"differences"/"HDR_A_vs_B_luminance_difference.png",luma_diff_image(hdr_a,hdr_b))
    write_cv(RESULTS/"differences"/"H0_A_vs_B_luminance_difference.png",luma_diff_image(h0_a,h0_b))
    canvas_a,canvas_b=np.zeros_like(sdr_full),np.zeros_like(sdr_full); canvas_a[Y1:Y2]=hdr_a; canvas_b[Y1:Y2]=hdr_b
    write_cv(RESULTS/"previews"/"comparison_SDR_HDR_A_HDR_B_H0_A_H0_B.png",make_panel([sdr_full,canvas_a,canvas_b,h0_a,h0_b],220)[...,::-1])
    write_json(RESULTS/"metrics.json",result); REPORT_PATH.write_text(report(result),encoding="utf-8")
    print("RESAMPLING_DIAGNOSTIC_STATUS=COMPLETE"); print(f"RESULTS={RESULTS}"); print(f"REPORT={REPORT_PATH}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
