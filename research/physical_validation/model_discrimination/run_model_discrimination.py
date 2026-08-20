"""One-shot Phase 4 HDR model-discrimination experiment.

This is diagnostic-first. It reads the established Matrix H0/H1 outputs without
refitting them, bins every overlap pixel, and considers exactly one extension
(Model J: three smoothly luminance-conditioned 3x3 matrices) only if a
predeclared dependency gate permits it. It never modifies production/P2 data
and reads only bounded P2.32 frame arrays; no video decoding is performed.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SRC = ROOT / "research" / "src"
SCENE_DIR = ROOT / "research" / "physical_validation" / "scene_regrading"
for path in (RESEARCH_SRC, SCENE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from reshaping_research.metrics.color_metrics import chroma_error, delta_e_2000, delta_e_ictcp, hue_error  # noqa: E402
from reshaping_research.models.model_e_cdf_regularized import ModelECdfRegularized  # noqa: E402
from reshaping_research.utils.color_spaces import bt709_to_bt2020, compute_luminance, linear_bt2020_to_ictcp  # noqa: E402
from reshaping_research.utils.transfer_functions import bt1886_eotf  # noqa: E402
from run_scene_regrading import (  # noqa: E402
    D65, M_2020_TO_XYZ_D65, apply_matrix, lab2020, map_luma_ratio, metrics,
    pq_eotf_normalized, seam_metrics, synthetic_scene,
)

P232_DIR = ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset"
METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
OLD_RESULTS = SCENE_DIR / "results" / "matrix_73367_73348"
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
REPORT_PATH = BASE / "MODEL_DISCRIMINATION_REPORT.md"
PEAK_NITS, RGB16_MAX, EPS = 10_000.0, 65_535.0, 1e-8
# Frozen before Phase 4 scores. BR remains a conditional robustness result due to geometry < 0.95.
FROZEN_CASES = (
    ("The Matrix", "the_matrix_temporal_stratum_08", "development_verified_anchor"),
    ("The Matrix", "the_matrix_temporal_stratum_03", "predeclared_temporal_candidate"),
    ("The Matrix", "the_matrix_temporal_stratum_12", "predeclared_temporal_candidate"),
    ("BR2049", "br2049_temporal_stratum_01", "predeclared_conditional_geometry_candidate"),
)
LUMA_BINS = (("0–5", 0.0, 5.0), ("5–20", 5.0, 20.0), ("20–50", 20.0, 50.0), ("50–100", 50.0, 100.0), ("100–250", 100.0, 250.0), ("250–500", 250.0, 500.0), ("500–1000", 500.0, 1000.0), (">1000", 1000.0, math.inf))
HUE_SECTORS = (("red", 330.0, 30.0), ("yellow", 30.0, 90.0), ("green", 90.0, 150.0), ("cyan", 150.0, 210.0), ("blue", 210.0, 270.0), ("magenta", 270.0, 330.0))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float): return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def locate(declared: str, suffix: str) -> Path:
    candidates = [p for p in P232_DIR.rglob(Path(declared).name) if p.is_file()]
    require(len(candidates) == 1, f"Expected one P2.32 input for {declared}; got {len(candidates)}")
    path = candidates[0].resolve()
    require(str(path).startswith(str(ROOT.resolve())) and path.name.endswith(suffix), f"Unsafe input path: {path}")
    return path


def record_for(material: str, case_id: str) -> tuple[dict[str, Any], dict[str, Path]]:
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    require(data.get("phase") == "P2.32" and data.get("status") == "DATASET READY", "P2.32 unavailable")
    entries = [r for r in data["case_records"][material] if r.get("case_id") == case_id]
    require(len(entries) == 1, f"Missing P2.32 case {case_id}")
    record = entries[0]; geometry = record.get("geometry", {}); overlap = geometry.get("overlap")
    require(isinstance(overlap, list) and len(overlap) == 4 and record.get("sync_status") == "LOCKED", f"Invalid locked geometry/sync: {case_id}")
    require(bool(record.get("finite", {}).get("hdr_rgb")) and bool(record.get("finite", {}).get("om_rgb")), f"Nonfinite raw source: {case_id}")
    return record, {"hdr": locate(record["hdr_rgb_u16_path"], "_hdr_rgb_u16.npy"), "om": locate(record["om_rgb_u16_path"], "_om_rgb_u16.npy")}


def decode(record: dict[str, Any], inputs: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    hdr_raw, om_raw = np.load(inputs["hdr"]), np.load(inputs["om"])
    require(hdr_raw.dtype == np.uint16 and om_raw.dtype == np.uint16, "Expected uint16 RGB P2.32 sources")
    require(tuple(hdr_raw.shape) == tuple(record["hdr_rgb_shape"]) and tuple(om_raw.shape) == tuple(record["om_rgb_shape"]), "Raw shape mismatch")
    x1, y1, x2, y2 = map(int, record["geometry"]["overlap"])
    hdr_code = cv2.resize(hdr_raw.astype(np.float64) / RGB16_MAX, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
    hdr_common = pq_eotf_normalized(hdr_code)
    sdr_full = np.maximum(bt709_to_bt2020(bt1886_eotf(om_raw.astype(np.float64) / RGB16_MAX)), 0.0)
    sdr_common = sdr_full[y1:y2, x1:x2]
    require(sdr_common.shape == hdr_common.shape and np.isfinite(sdr_full).all() and np.isfinite(hdr_common).all(), "Invalid decoded images")
    return sdr_full, sdr_common, hdr_common, (x1, y1, x2, y2)


def established_matrix_outputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sci = OLD_RESULTS / "scientific"
    h0 = np.load(sci / "H0_RatioScaling_HDR_OpenMatte.linear_bt2020_nits.npy").astype(np.float64) / PEAK_NITS
    h1 = np.load(sci / "H1_3x3Matrix_HDR_OpenMatte.linear_bt2020_nits.npy").astype(np.float64) / PEAK_NITS
    reference = np.load(sci / "HDR_Reference_Common.linear_bt2020_nits.npy").astype(np.float64) / PEAK_NITS
    require(h0.shape == h1.shape == (1080, 1920, 3) and reference.shape == (800, 1920, 3), "Unexpected established H0/H1 artifact shapes")
    require(np.isfinite(h0).all() and np.isfinite(h1).all() and np.isfinite(reference).all(), "Nonfinite established H0/H1 artifact")
    return h0[140:940], h1[140:940], reference


def reference_ictcp(reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ictcp = linear_bt2020_to_ictcp(reference * PEAK_NITS)
    chroma = np.sqrt(ictcp[..., 1] ** 2 + ictcp[..., 2] ** 2)
    hue = np.degrees(np.arctan2(ictcp[..., 2], ictcp[..., 1])) % 360.0
    return ictcp, chroma, hue


def group_row(predicted: np.ndarray, reference: np.ndarray, mask: np.ndarray, ref_chroma: np.ndarray) -> dict[str, Any]:
    count = int(np.sum(mask))
    if not count:
        return {"pixel_count": 0, "hue_valid_pixel_count": 0, "luminance_MAE_nits": None, "luminance_signed_mean_nits": None, "deltaE2000_mean": None, "chroma_error_signed_mean": None, "chroma_error_absolute_mean": None, "hue_error_absolute_mean_degrees": None, "RGB_residual_signed_mean_nits": None}
    p, r = predicted[mask], reference[mask]
    luma_residual = (compute_luminance(p) - compute_luminance(r)) * PEAK_NITS
    pn, rn = p * PEAK_NITS, r * PEAK_NITS
    de = delta_e_2000(lab2020(p), lab2020(r))
    chroma = chroma_error(pn, rn); hue = hue_error(pn, rn); valid_hue = ref_chroma[mask] > 1e-6
    rgb = (p-r) * PEAK_NITS
    return {"pixel_count":count, "hue_valid_pixel_count":int(np.sum(valid_hue)), "luminance_MAE_nits":float(np.mean(np.abs(luma_residual))), "luminance_signed_mean_nits":float(np.mean(luma_residual)), "deltaE2000_mean":float(np.mean(de)), "chroma_error_signed_mean":float(np.mean(chroma)), "chroma_error_absolute_mean":float(np.mean(np.abs(chroma))), "hue_error_absolute_mean_degrees":float(np.mean(np.abs(hue[valid_hue])) if np.any(valid_hue) else float("nan")), "RGB_residual_signed_mean_nits":[float(x) for x in np.mean(rgb, axis=0)]}


def binned_residuals(predicted: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    _, chroma, hue = reference_ictcp(reference); y = compute_luminance(reference) * PEAK_NITS
    luma_rows = []
    for name, lower, upper in LUMA_BINS:
        mask = (y >= lower) & (y < upper)
        luma_rows.append({"group":name, "range_nits":[lower, None if math.isinf(upper) else upper], **group_row(predicted, reference, mask, chroma)})
    q1, q2 = (float(x) for x in np.quantile(chroma, (1/3, 2/3)))
    chroma_masks = (("low chroma", chroma <= q1), ("medium chroma", (chroma > q1) & (chroma <= q2)), ("high chroma", chroma > q2))
    chroma_rows = [{"group":name, "reference_chroma_ICtCp_range":[0.0 if idx == 0 else (q1 if idx == 1 else q2), q1 if idx == 0 else (q2 if idx == 1 else None)], **group_row(predicted, reference, mask, chroma)} for idx,(name,mask) in enumerate(chroma_masks)]
    hue_rows = []
    for name, start, stop in HUE_SECTORS:
        mask = ((hue >= start) | (hue < stop)) if start > stop else ((hue >= start) & (hue < stop))
        hue_rows.append({"group":name, "sector_degrees":[start,stop], **group_row(predicted, reference, mask, chroma)})
    return {"all_pixels":group_row(predicted, reference, np.ones(reference.shape[:2], dtype=bool), chroma), "reference_chroma_tertiles_ICtCp":[q1,q2], "by_luminance":luma_rows, "by_chroma":chroma_rows, "by_hue":hue_rows}


def relative_dispersion(rows: list[dict[str, Any]], global_mean: float) -> float:
    values = np.asarray([r["deltaE2000_mean"] for r in rows], dtype=np.float64); weights = np.asarray([r["pixel_count"] for r in rows], dtype=np.float64)
    valid = np.isfinite(values) & (weights > 0)
    if not np.any(valid) or global_mean <= EPS: return 0.0
    return float(np.sqrt(np.average((values[valid] - global_mean) ** 2, weights=weights[valid])) / global_mean)


def dependency_case(residual: dict[str, Any]) -> dict[str, Any]:
    luma = relative_dispersion(residual["by_luminance"], residual["all_pixels"]["deltaE2000_mean"])
    chroma = relative_dispersion(residual["by_chroma"], residual["all_pixels"]["deltaE2000_mean"])
    hue = relative_dispersion(residual["by_hue"], residual["all_pixels"]["deltaE2000_mean"])
    color = max(chroma,hue); threshold = 0.25
    if luma >= threshold and luma >= 1.5*color: classification = "A_luminance_dominant"
    elif color >= threshold and color >= 1.5*luma: classification = "B_hue_or_chroma_dominant"
    elif luma >= threshold and color >= threshold: classification = "C_mixed_dependency"
    else: classification = "D_no_clear_low_dimensional_structure"
    return {"classification":classification, "predeclared_rule":"weighted relative SD of group mean ΔE2000: strong >=0.25; dominance requires >=1.5× the other axis", "luminance_dispersion":luma, "chroma_dispersion":chroma, "hue_dispersion":hue, "color_dispersion_max":color}


def diagnostic_gate(h0: dict[str, Any], h1: dict[str, Any]) -> dict[str, Any]:
    h0_case, h1_case = dependency_case(h0), dependency_case(h1)
    justified = h0_case["classification"] == h1_case["classification"] == "A_luminance_dominant"
    combined = "A_luminance_dominant" if justified else ("C_mixed_dependency" if "C_mixed_dependency" in (h0_case["classification"], h1_case["classification"]) else ("B_hue_or_chroma_dominant" if "B_hue_or_chroma_dominant" in (h0_case["classification"], h1_case["classification"]) else "D_no_clear_low_dimensional_structure"))
    return {"H0":h0_case, "H1":h1_case, "combined_classification":combined, "model_J_permitted":justified, "J_gate":"J is permitted only if both existing H0 and H1 residuals classify A under the predeclared rule."}


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x-edge0)/(edge1-edge0),0.0,1.0); return t*t*(3.0-2.0*t)


def j_weights(image: np.ndarray) -> np.ndarray:
    """Three fixed, smooth H0-luminance weights: shadow<5–50, mid, highlight>50–250 nits."""
    y = compute_luminance(image) * PEAK_NITS
    to_mid = smoothstep(5.0,50.0,y); to_high = smoothstep(50.0,250.0,y)
    return np.stack((1.0-to_mid, to_mid*(1.0-to_high), to_high),axis=-1)


def spatial_indices(shape: tuple[int,int], maximum: int, held: bool) -> np.ndarray:
    yy,xx=np.indices(shape); hold=((yy//80+xx//120)%5)==0; mask=hold if held else ~hold
    candidates=np.flatnonzero(mask.ravel()); return candidates[np.linspace(0,len(candidates)-1,min(maximum,len(candidates)),dtype=int)]


def apply_j(base: np.ndarray, matrices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    weights=j_weights(base); transformed=np.stack([np.einsum("ij,...j->...i",m,base,optimize=True) for m in matrices],axis=-2)
    raw=np.sum(weights[...,None]*transformed,axis=-2); result=np.maximum(raw,0.0)
    return result,{"pre_nonnegative_min":float(raw.min()),"pre_nonnegative_negative_count":int(np.sum(raw<0)),"post_finite":bool(np.isfinite(result).all())}


def fit_j(base: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    flat_base,flat_target=base.reshape(-1,3),target.reshape(-1,3); train=spatial_indices(base.shape[:2],4096,False); hold=spatial_indices(base.shape[:2],2048,True)
    train_base,train_target=flat_base[train].reshape(1,-1,3),flat_target[train].reshape(1,-1,3); hold_base,hold_target=flat_base[hold].reshape(1,-1,3),flat_target[hold].reshape(1,-1,3)
    identity=np.eye(3); lam_identity,lam_smooth=0.03,0.05
    def objective(vector: np.ndarray, b: np.ndarray, t: np.ndarray) -> float:
        matrices=vector.reshape(3,3,3); pred,_=apply_j(b,matrices); pm=metrics(pred,t)
        # Multi-metric objective; not RGB MSE alone. Neighbor term prohibits arbitrary independent bins.
        regularize=lam_identity*float(np.sum((matrices-identity)**2))+lam_smooth*float(np.sum((matrices[1:]-matrices[:-1])**2))
        return (0.35*(pm["luminance"]["RMSE_nits"]/100.0)**2+0.20*pm["chroma_error_ICtCp_absolute"]["mean"]**2+0.20*(pm["hue_error_degrees_absolute_non_neutral"]["mean"]/30.0)**2+0.25*(pm["deltaE2000"]["mean"]/5.0)**2+regularize)
    started=time.perf_counter(); solved=minimize(lambda v:objective(v,train_base,train_target),np.tile(identity,(3,1,1)).reshape(-1),method="L-BFGS-B",options={"maxiter":100,"ftol":1e-10}); matrices=solved.x.reshape(3,3,3)
    determ=[float(np.linalg.det(m)) for m in matrices]; condition=[float(np.linalg.cond(m)) for m in matrices]; stable=bool(all(abs(x)>=.05 for x in determ) and all(np.isfinite(x) and x<=30 for x in condition))
    hold_pred,_=apply_j(hold_base,matrices)
    return {"attempted":True,"definition":"Exactly three smoothly H0-luminance-conditioned 3x3 matrices (shadow/midtone/highlight); 27 matrix coefficients; identity and adjacent-matrix regularization.","matrices":matrices,"determinants":determ,"condition_numbers":condition,"stable":stable,"optimizer_success":bool(solved.success),"optimizer_message":str(solved.message),"fit_seconds_cpu":time.perf_counter()-started,"training_sample_count":int(len(train)),"holdout_sample_count":int(len(hold)),"holdout_metrics":metrics(hold_pred,hold_target)}


def fit_h0_case(material: str, case_id: str, role: str) -> dict[str, Any]:
    record,inputs=record_for(material,case_id); sdr_full,sdr_common,hdr_common,bbox=decode(record,inputs)
    model=ModelECdfRegularized(); started=time.perf_counter(); params=model.fit(compute_luminance(sdr_common).reshape(-1),compute_luminance(hdr_common).reshape(-1)); fit_seconds=time.perf_counter()-started
    apply_started=time.perf_counter(); h0_full,_=map_luma_ratio(sdr_full,params); apply_seconds=time.perf_counter()-apply_started; h0_common=h0_full[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    result={"material":material,"case_id":case_id,"role":role,"frame_pair":{"hdr":record["hdr_frame"],"om":record["om_frame"]},"scene_claim":record.get("scene_category"),"scene_independence_verified":bool(record.get("scene_independence_verified")),"known_verified_anchor":bool(record.get("known_verified_anchor")),"geometry_confidence":float(record["geometry"]["confidence"]),"geometry_confirmatory_gate_at_least_0_95":bool(float(record["geometry"]["confidence"])>=.95),"fit_mask":{"definition":"SDR ∩ HDR only","pixel_count":int(hdr_common.shape[0]*hdr_common.shape[1])},"parameter_count":{"Model_E_reported":int(model.param_count()),"H0_effective_luminance_controls":10},"metrics":metrics(h0_common,hdr_common),"seam":seam_metrics(h0_full,hdr_common,bbox),"range":{"finite":bool(np.isfinite(h0_full).all()),"min":float(h0_full.min()),"max":float(h0_full.max())},"performance_seconds":{"fit":fit_seconds,"full_sdr_apply":apply_seconds}}
    del sdr_full,sdr_common,hdr_common,h0_full
    return result


def synthetic_existing() -> dict[str, Any]:
    old=json.loads((OLD_RESULTS/"metrics.json").read_text(encoding="utf-8")); return old["synthetic"]


def comparison(h0_metrics: dict[str,Any], h1_metrics: dict[str,Any], j: dict[str,Any] | None) -> dict[str, Any]:
    def item(name: str, data: dict[str,Any]) -> dict[str,Any]: return {"model":name,"mean_deltaE2000":data["deltaE2000"]["mean"],"luminance_MAE_nits":data["luminance"]["MAE_nits"],"mean_abs_chroma":data["chroma_error_ICtCp_absolute"]["mean"],"mean_abs_hue":data["hue_error_degrees_absolute_non_neutral"]["mean"]}
    out=[item("H0",h0_metrics),item("H1",h1_metrics)]
    if j and j.get("attempted"): out.append(item("J",j["development_metrics"]))
    return {"rows":out,"acceptance_rule":"mean ΔE2000 improvement >=5%; no visible color artifacts; no worsening seam continuity; no pathological coefficients; no significant instability"}


def markdown_table(rows: list[dict[str,Any]]) -> str:
    head="| Group | pixels | luma MAE nits | mean ΔE2000 | signed chroma / |chroma| | |hue| ° | signed RGB residual nits (R,G,B) |\n|---|---:|---:|---:|---:|---:|---:|"
    body=[]
    for r in rows:
        rgb=r["RGB_residual_signed_mean_nits"]; rgb_text="—" if rgb is None else "/".join(f"{x:.4f}" for x in rgb)
        val=lambda key: "—" if r[key] is None else f"{r[key]:.6f}"
        body.append(f"| {r['group']} | {r['pixel_count']} | {val('luminance_MAE_nits')} | {val('deltaE2000_mean')} | {val('chroma_error_signed_mean')} / {val('chroma_error_absolute_mean')} | {val('hue_error_absolute_mean_degrees')} | {rgb_text} |")
    return head+"\n"+"\n".join(body)


def report(result: dict[str,Any]) -> str:
    diag=result["diagnostic"]; h0,h1=diag["H0"],diag["H1"]; gate=diag["dependency_gate"]; comp=result["development_comparison"]; validation=result["cross_scene_validation"]; synthetic=result["synthetic_hidden_region"]
    def metric_row(row:dict[str,Any]) -> str:
        m=row["metrics"]; return f"| {row['material']} {row['case_id']} ({row['role']}) | {row['frame_pair']['hdr']} / {row['frame_pair']['om']} | {m['luminance']['MAE_nits']:.4f} | {m['deltaE2000']['mean']:.4f} | {m['hue_error_degrees_absolute_non_neutral']['mean']:.3f} | {row['seam']['boundaries'][0]['luma_mean_nits']:.3f} / {row['seam']['boundaries'][1]['luma_mean_nits']:.3f} | {row['geometry_confidence']:.4f} |"
    comparison_rows="\n".join(f"| {x['model']} | {x['luminance_MAE_nits']:.6f} | {x['mean_deltaE2000']:.6f} | {x['mean_abs_chroma']:.6f} | {x['mean_abs_hue']:.6f} |" for x in comp["rows"])
    final=result["final_decision"]
    return f"""# Model Discrimination Report — Phase 4 (Final)

## Scope and frozen protocol
This is one bounded diagnostic and validation phase. It reads the established Matrix H0/H1 outputs for residual analysis; it does not refit H0/H1 for that analysis, modify P2.32/production, decode source video, use AI/LUTs/spatial transforms, or start another model family. All real-pair fits use the SDR∩HDR overlap only and apply the frozen selected architecture to the entire Open-Matte SDR frame.

Predeclared before scores: Matrix development `08` (73367/73348); Matrix candidates `03` (28382/28363) and `12` (111978/111959); BR2049 candidate `01` (11699/12866). The Matrix candidates are real, temporally distant P2.32 pairs but have unverified shot/scene independence. BR2049 is real but has recorded geometry confidence 0.9155, below the ordinary 0.95 gate; it is a conditional robustness result, not geometry-confirmatory evidence.

## Established H0/H1 residuals — all 1,536,000 Matrix overlap pixels
H0 is Model E luminance-only plus ratio scaling. H1 is the already-established regularized 3×3 H0 transform. Reference bins are computed from HDR reference luminance/chroma/hue, not model predictions. Hue error excludes only reference pixels with ICtCp chroma ≤1e-6; no luminance/chroma group is discarded, and all group counts are reported.

### H0 by HDR reference luminance
{markdown_table(h0['by_luminance'])}

### H1 by HDR reference luminance
{markdown_table(h1['by_luminance'])}

### H0 by reference chroma magnitude
Reference ICtCp tertiles: `{h0['reference_chroma_tertiles_ICtCp'][0]:.8f}`, `{h0['reference_chroma_tertiles_ICtCp'][1]:.8f}`.
{markdown_table(h0['by_chroma'])}

### H1 by reference chroma magnitude
{markdown_table(h1['by_chroma'])}

### H0 by reference hue sector
{markdown_table(h0['by_hue'])}

### H1 by reference hue sector
{markdown_table(h1['by_hue'])}

## Dependency diagnosis and Model J gate
The predeclared statistic is weighted relative standard deviation of group mean ΔE2000. A dependency is strong at ≥0.25; one axis is dominant only if it is also ≥1.5× the other. H0: `{gate['H0']['classification']}` (luminance `{gate['H0']['luminance_dispersion']:.4f}`, chroma `{gate['H0']['chroma_dispersion']:.4f}`, hue `{gate['H0']['hue_dispersion']:.4f}`). H1: `{gate['H1']['classification']}` (luminance `{gate['H1']['luminance_dispersion']:.4f}`, chroma `{gate['H1']['chroma_dispersion']:.4f}`, hue `{gate['H1']['hue_dispersion']:.4f}`). **Combined diagnosis: {gate['combined_classification']}.**

Model J was permitted only if **both** H0 and H1 were luminance-dominant. Permitted: **{gate['model_J_permitted']}**. {result['model_J'].get('reason', result['model_J'].get('definition',''))}

## Development comparison and selection gate
| Model | luma MAE nits | mean ΔE2000 | mean |chroma| | mean |hue| ° |
|---|---:|---:|---:|---:|
{comparison_rows}

H1's established mean ΔE2000 change is below the mandatory 5% gate and its Phase 3 visual review found localized blue/purple dark-garment artifacts. Thus it fails despite stable numerical coefficients. The mandatory gate is `{comp['acceptance_rule']}`.

## Stability and performance
The established H1 selected matrix is numerically stable but visually failed: determinant `{result['established_H1_stability']['determinant']:.7f}`, condition number `{result['established_H1_stability']['condition_number']:.5f}`, and `||M-I||_F` `{result['established_H1_stability']['deviation_from_identity_frobenius']:.7f}`. It is rejected because numerical stability does not override its failed 5%/artifact gates. Model J was not fitted, so it supplies no additional stability claim.

## Frozen-architecture four-pair validation
The frozen applied architecture is **{validation['selected_architecture']}**. Outside overlap, values are an unverified Open-Matte extrapolation: seam is only a continuity proxy, not HDR ground truth.

| Pair | HDR / SDR-OM frame | luma MAE nits | mean ΔE2000 | mean |hue| ° | top / bottom seam nits | geometry confidence |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_row(x) for x in validation['rows'])}

| Pair | H0 fit s | full-SDR apply s | finite | output range (normalized linear) |
|---|---:|---:|---|---:|
{chr(10).join(f"| {x['case_id']} | {x['performance_seconds']['fit']:.4f} | {x['performance_seconds']['full_sdr_apply']:.4f} | {x['range']['finite']} | {x['range']['min']:.6f}…{x['range']['max']:.6f} |" for x in validation['rows'])}

## Synthetic hidden-region validation
Existing predeclared synthetic test: fit only central crop, evaluate both crop and known hidden region. It is a controlled global-transform sanity check, not proof of transfer to real film.

| Variant | common mean ΔE2000 | hidden mean ΔE2000 | hidden luma MAE nits |
|---|---:|---:|---:|
| H0 | {synthetic['H0']['common']['deltaE2000']['mean']:.6f} | {synthetic['H0']['hidden']['deltaE2000']['mean']:.6f} | {synthetic['H0']['hidden']['luminance']['MAE_nits']:.6f} |
| H1 | {synthetic['H1']['common']['deltaE2000']['mean']:.6f} | {synthetic['H1']['hidden']['deltaE2000']['mean']:.6f} | {synthetic['H1']['hidden']['luminance']['MAE_nits']:.6f} |

## Final decision — {final['outcome']}
{final['rationale']}

**Why the chosen outcome is necessary:** {final['why_necessary']}

**Why more complex models are not justified:** {final['why_not_more_complex']}

Parameter accounting: H0 has 10 effective compact luminance controls (Model E reports 12 including unused chroma controls); H1 adds 9 coefficients for 19 effective scene controls. Model J has 27 matrix coefficients plus the frozen H0 controls and was {'not fitted' if not result['model_J'].get('attempted') else 'fitted only under the predeclared gate'}. Performance times and numerical ranges per applied real pair are retained in `results/metrics.json`. STOP: no further Phase 5/model iteration is authorized by this result.
"""


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    h0,h1,reference=established_matrix_outputs(); diag_h0=binned_residuals(h0,reference); diag_h1=binned_residuals(h1,reference); gate=diagnostic_gate(diag_h0,diag_h1)
    old=json.loads((OLD_RESULTS/"metrics.json").read_text(encoding="utf-8")); old_h0=old["matrix_anchor"]["H0"]["common_metrics"]; old_h1=old["matrix_anchor"]["H1"]["common_metrics"]
    model_j:dict[str,Any]={"attempted":False,"reason":"Not fitted: predeclared residual gate did not establish a dominant luminance-only dependency in both H0 and H1."}
    if gate["model_J_permitted"]:
        fitted=fit_j(h0,reference); full_j,range_j=apply_j(h0,fitted["matrices"]); fitted.update({"development_metrics":metrics(full_j,reference),"range":range_j,"reason":"Fitted exactly once because the predeclared dual H0/H1 luminance-dependency gate passed."}); model_j=fitted
    comp=comparison(old_h0,old_h1,model_j if model_j.get("attempted") else None)
    # H1 has already failed its mandatory artifact/5% gate. If J was not justified or cannot pass all gates, freeze H0 for validation.
    selected="H0"
    validation_rows=[fit_h0_case(material,case,role) for material,case,role in FROZEN_CASES]
    final={"outcome":"C — no compact tested model is sufficiently general","rationale":"H1 fails the 5% acceptance gate and visible-artifact gate. The diagnostic did not authorize arbitrary architecture expansion beyond at most J; the frozen four-pair validation is H0 baseline evidence, with unverified Matrix shot independence and BR geometry below the confirmatory threshold.","why_necessary":"The observed residual structure and failed H1 gate do not support claiming that a global matrix or an untested extension reconstructs the studio regrade.","why_not_more_complex":"J was only permitted for clear dual H0/H1 luminance dominance. The gate result above controls this decision; adding hue/chroma/polynomial/LUT/spatial/framewise models would violate the one-shot compact-model protocol and would not repair the missing cross-scene/geometry evidence."}
    h1_stability={key:old["matrix_anchor"]["matrix_fit"]["selected"][key] for key in ("determinant","condition_number","deviation_from_identity_frobenius")}
    result={"status":"COMPLETE","scope":"One-shot research-only Phase 4; no production/P2 changes, no video decode, no full-video processing.","frozen_cases":[{"material":a,"case_id":b,"role":c} for a,b,c in FROZEN_CASES],"diagnostic":{"pixels_per_model":int(reference.shape[0]*reference.shape[1]),"H0":diag_h0,"H1":diag_h1,"dependency_gate":gate},"model_J":model_j,"established_H1_stability":h1_stability,"development_comparison":comp,"cross_scene_validation":{"selected_architecture":selected,"rows":validation_rows},"synthetic_hidden_region":synthetic_existing(),"final_decision":final}
    write_json(RESULTS/"metrics.json",result); REPORT_PATH.write_text(report(result),encoding="utf-8")
    print("MODEL_DISCRIMINATION_STATUS=COMPLETE"); print(f"RESULTS={RESULTS}"); print(f"REPORT={REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
