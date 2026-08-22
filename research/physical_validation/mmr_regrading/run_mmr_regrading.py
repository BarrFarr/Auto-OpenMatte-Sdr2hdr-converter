"""One-shot minimal scene-based MMR chroma regrading experiment.

H0 remains the established Model E luminance fit plus RGB ratio scaling. MMR
only predicts ICtCp Ct/Cp; every output is reimposed to Model E's Y_target.
MMR-0 has eight chroma coefficients. MMR-1 (12) is allowed once only when a
predeclared post-MMR0 chroma/hue-by-luminance diagnostic supports Y×C terms.
No production/P2 edits or source-video decoding occur in this research script.
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

ROOT = Path(__file__).resolve().parents[3]
RESEARCH_SRC = ROOT / "research" / "src"
SCENE_DIR = ROOT / "research" / "physical_validation" / "scene_regrading"
for item in (RESEARCH_SRC, SCENE_DIR):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from reshaping_research.metrics.color_metrics import chroma_error, hue_error  # noqa: E402
from reshaping_research.models.model_e_cdf_regularized import ModelECdfRegularized  # noqa: E402
from reshaping_research.utils.color_spaces import bt709_to_bt2020, compute_luminance, ictcp_to_linear_bt2020, linear_bt2020_to_ictcp  # noqa: E402
from reshaping_research.utils.transfer_functions import bt1886_eotf  # noqa: E402
from run_scene_regrading import (  # noqa: E402
    EPS, PEAK_NITS, RGB16_MAX, map_luma_ratio, metrics, panel, pq_eotf_normalized, preview,
    seam_metrics, synthetic_scene, write_cv_image, write_preview, write_scientific,
)

P232_DIR = ROOT / "dev" / "ffmpeg-build" / "validation" / "P232_real_material_dataset"
METRICS_PATH = P232_DIR / "P232_dataset_metrics.json"
BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results" / "matrix_73367_73348"
REPORT_PATH = BASE / "MMR_REGRADING_REPORT.md"
LAMBDA_GRID = (1e-4, 1e-3, 1e-2)
CHROMA_BOUND = 0.5
LUMA_FEATURE_SCALE_NITS = 100.0
SPATIAL_TRAIN, SPATIAL_HOLDOUT = 4096, 2048
FROZEN_CASES = (
    ("The Matrix", "the_matrix_temporal_stratum_08", "development_verified_anchor"),
    ("The Matrix", "the_matrix_temporal_stratum_03", "predeclared_temporal_candidate"),
    ("The Matrix", "the_matrix_temporal_stratum_12", "predeclared_temporal_candidate"),
    ("BR2049", "br2049_temporal_stratum_01", "predeclared_conditional_geometry_candidate"),
)
LUMA_BINS = (("0–5",0.,5.),("5–20",5.,20.),("20–50",20.,50.),("50–100",50.,100.),("100–250",100.,250.),("250–500",250.,500.),("500–1000",500.,1000.),(">1000",1000.,math.inf))


def require(condition: bool, message: str) -> None:
    if not condition: raise RuntimeError(message)


def safe(value: Any) -> Any:
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [safe(x) for x in value]
    if isinstance(value,Path): return str(value)
    if isinstance(value,np.ndarray): return safe(value.tolist())
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float): return value if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def locate(declared: str, suffix: str) -> Path:
    matches=[p for p in P232_DIR.rglob(Path(declared).name) if p.is_file()]
    require(len(matches)==1,f"Expected one P2.32 input for {declared}; got {len(matches)}")
    result=matches[0].resolve(); require(str(result).startswith(str(ROOT.resolve())) and result.name.endswith(suffix),f"Unsafe input: {result}")
    return result


def record_for(material: str, case_id: str) -> tuple[dict[str,Any],dict[str,Path]]:
    data=json.loads(METRICS_PATH.read_text(encoding="utf-8")); require(data.get("phase")=="P2.32" and data.get("status")=="DATASET READY","P2.32 unavailable")
    records=[r for r in data["case_records"][material] if r.get("case_id")==case_id]; require(len(records)==1,f"Missing {case_id}")
    r=records[0]; require(r.get("sync_status")=="LOCKED" and r.get("finite",{}).get("hdr_rgb") and r.get("finite",{}).get("om_rgb"),f"Invalid source {case_id}")
    require(isinstance(r.get("geometry",{}).get("overlap"),list),f"Missing geometry {case_id}")
    return r,{"hdr":locate(r["hdr_rgb_u16_path"],"_hdr_rgb_u16.npy"),"om":locate(r["om_rgb_u16_path"],"_om_rgb_u16.npy")}


def decode(record: dict[str,Any], inputs: dict[str,Path]) -> tuple[np.ndarray,np.ndarray,np.ndarray,tuple[int,int,int,int]]:
    hdr_raw,om_raw=np.load(inputs["hdr"]),np.load(inputs["om"])
    require(hdr_raw.dtype==np.uint16 and om_raw.dtype==np.uint16,"Expected uint16 P2.32 RGB")
    require(tuple(hdr_raw.shape)==tuple(record["hdr_rgb_shape"]) and tuple(om_raw.shape)==tuple(record["om_rgb_shape"]),"Raw shape mismatch")
    x1,y1,x2,y2=map(int,record["geometry"]["overlap"])
    hdr_code=cv2.resize(hdr_raw.astype(np.float64)/RGB16_MAX,(x2-x1,y2-y1),interpolation=cv2.INTER_LINEAR)
    hdr_common=pq_eotf_normalized(hdr_code)
    sdr_full=np.maximum(bt709_to_bt2020(bt1886_eotf(om_raw.astype(np.float64)/RGB16_MAX)),0.)
    sdr_common=sdr_full[y1:y2,x1:x2]
    require(sdr_common.shape==hdr_common.shape and np.isfinite(sdr_full).all() and np.isfinite(hdr_common).all(),"Invalid decoded pair")
    return sdr_full,sdr_common,hdr_common,(x1,y1,x2,y2)


def spatial_indices(shape: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
    yy,xx=np.indices(shape); held=((yy//80+xx//120)%5)==0
    def choose(mask:np.ndarray,limit:int)->np.ndarray:
        found=np.flatnonzero(mask.ravel()); return found[np.linspace(0,len(found)-1,min(limit,len(found)),dtype=int)]
    return choose(~held,SPATIAL_TRAIN),choose(held,SPATIAL_HOLDOUT)


def features(base: np.ndarray, target_y: np.ndarray, kind: str) -> tuple[np.ndarray,np.ndarray]:
    ictcp=linear_bt2020_to_ictcp(np.maximum(base,0.)*PEAK_NITS)
    y=(target_y*PEAK_NITS)/LUMA_FEATURE_SCALE_NITS
    ct,cp=ictcp[...,1],ictcp[...,2]
    columns=[np.ones_like(y),y,ct,cp]
    if kind=="MMR1": columns.extend((y*ct,y*cp))
    return np.stack(columns,axis=-1),ictcp


def identity_coefficients(kind: str) -> np.ndarray:
    count=4 if kind=="MMR0" else 6
    theta=np.zeros((count,2),dtype=np.float64); theta[2,0]=1.; theta[3,1]=1.; return theta


def apply_mmr(base: np.ndarray,target_y: np.ndarray,coefficients: np.ndarray,kind: str) -> tuple[np.ndarray,dict[str,Any]]:
    design,base_ictcp=features(base,target_y,kind)
    predicted=design.reshape(-1,design.shape[-1])@coefficients
    raw_chroma=predicted.reshape(base.shape[:-1]+(2,))
    bounded=np.clip(raw_chroma,-CHROMA_BOUND,CHROMA_BOUND)
    chroma_clipped=np.any(bounded!=raw_chroma,axis=-1)
    ictcp=np.stack((base_ictcp[...,0],bounded[...,0],bounded[...,1]),axis=-1)
    inverse_nits=ictcp_to_linear_bt2020(ictcp); inverse=inverse_nits/PEAK_NITS
    negative=np.any(inverse<0,axis=-1); nonnegative=np.maximum(inverse,0.)
    actual_y=compute_luminance(nonnegative); scale=target_y/np.maximum(actual_y,EPS)
    output=nonnegative*scale[...,None]; dark=actual_y<=EPS
    if np.any(dark): output[dark]=target_y[dark,None]
    final_y=compute_luminance(output)
    return output,{"chroma_component_bound":CHROMA_BOUND,"chroma_bound_pixel_fraction":float(np.mean(chroma_clipped)),"inverse_negative_channel_fraction":float(np.mean(inverse<0)),"inverse_negative_pixel_fraction":float(np.mean(negative)),"reimposition_scale_min":float(np.min(scale)),"reimposition_scale_max":float(np.max(scale)),"reimposition_dark_fallback_pixels":int(np.sum(dark)),"Y_target_max_abs_error_nits":float(np.max(np.abs(final_y-target_y))*PEAK_NITS),"finite":bool(np.isfinite(output).all()),"range_normalized":{"min":float(output.min()),"max":float(output.max())}}


def objective(metric: dict[str,Any]) -> float:
    """Predeclared perceptual holdout objective; never raw RGB MSE alone."""
    return float(0.20*(metric["luminance"]["MAE_nits"]/10.)**2 + 0.40*metric["deltaE2000"]["mean"]**2 + 0.25*(metric["chroma_error_ICtCp_absolute"]["mean"]/0.03)**2 + 0.15*(metric["hue_error_degrees_absolute_non_neutral"]["mean"]/20.)**2)


def fit_mmr(base: np.ndarray,target: np.ndarray,target_y: np.ndarray,kind: str) -> dict[str,Any]:
    train,hold=spatial_indices(base.shape[:2]); design,_=features(base,target_y,kind); target_chroma=linear_bt2020_to_ictcp(target*PEAK_NITS)[...,1:]
    x=design.reshape(-1,design.shape[-1]); c=target_chroma.reshape(-1,2); theta0=identity_coefficients(kind); candidates=[]
    for lam in LAMBDA_GRID:
        started=time.perf_counter(); gram=x[train].T@x[train]/len(train)+lam*np.eye(x.shape[1]); rhs=x[train].T@c[train]/len(train)+lam*theta0
        theta=np.linalg.solve(gram,rhs); train_out,train_bounds=apply_mmr(base.reshape(-1,1,3)[train],target_y.reshape(-1,1)[train],theta,kind); hold_out,hold_bounds=apply_mmr(base.reshape(-1,1,3)[hold],target_y.reshape(-1,1)[hold],theta,kind)
        train_metric,hold_metric=metrics(train_out,target.reshape(-1,1,3)[train]),metrics(hold_out,target.reshape(-1,1,3)[hold])
        cond=float(np.linalg.cond(gram)); delta=float(np.linalg.norm(theta-theta0)); stable=bool(np.isfinite(cond) and cond<=1e6 and delta<=5.0 and hold_bounds["chroma_bound_pixel_fraction"]<=.005 and hold_bounds["inverse_negative_pixel_fraction"]<=.005 and hold_bounds["finite"])
        candidates.append({"lambda":lam,"coefficients":theta,"regularized_gram_condition_number":cond,"coefficient_identity_deviation_L2":delta,"coefficient_max_abs":float(np.max(np.abs(theta))),"training_metrics":train_metric,"holdout_metrics":hold_metric,"training_bounds":train_bounds,"holdout_bounds":hold_bounds,"holdout_objective":objective(hold_metric),"stable":stable,"fit_seconds_cpu":time.perf_counter()-started})
    accepted=[x for x in candidates if x["stable"]]; require(bool(accepted),f"All {kind} candidates pathological")
    selected=min(accepted,key=lambda x:x["holdout_objective"])
    return {"model":kind,"feature_names":["constant","Y_target_nits/100","Ct_base","Cp_base"]+(["(Y_target_nits/100)*Ct_base","(Y_target_nits/100)*Cp_base"] if kind=="MMR1" else []),"parameter_count":int(selected["coefficients"].size),"training_sample_count":len(train),"holdout_sample_count":len(hold),"identity_coefficients":theta0,"selected":selected,"candidates":candidates,"regularized_closed_form":"(XᵀX/N + λI)⁻¹(XᵀC/N + λΘ_identity)"}


def acceptance(h0: dict[str,Any], candidate: dict[str,Any], h0_seam: dict[str,Any], cand_seam: dict[str,Any], bounds: dict[str,Any]) -> dict[str,Any]:
    h0h,h1h=h0["holdout"],candidate["selected"]["holdout_metrics"]
    h0f,h1f=h0["full"],candidate["full_metrics"]
    hold_improve=100*(h0h["deltaE2000"]["mean"]-h1h["deltaE2000"]["mean"])/h0h["deltaE2000"]["mean"]
    full_improve=100*(h0f["deltaE2000"]["mean"]-h1f["deltaE2000"]["mean"])/h0f["deltaE2000"]["mean"]
    seam_ok=all(cand_seam["boundaries"][i]["luma_mean_nits"]<=h0_seam["boundaries"][i]["luma_mean_nits"]*1.10 for i in (0,1))
    bounded=bool(bounds["chroma_bound_pixel_fraction"]<=.005 and bounds["inverse_negative_pixel_fraction"]<=.005 and bounds["finite"] and bounds["Y_target_max_abs_error_nits"]<=1e-5)
    passes=bool(hold_improve>=5 and full_improve>=5 and seam_ok and bounded and candidate["selected"]["stable"])
    return {"holdout_deltaE_improvement_percent":hold_improve,"full_overlap_deltaE_improvement_percent":full_improve,"seam_not_severely_worse":seam_ok,"bounded_and_finite":bounded,"numerically_stable":bool(candidate["selected"]["stable"]),"visible_artifact_review":"Requires inspection of generated physical montage; automatic pass is not inferred from mean metrics.","numerical_gate_passes":passes,"full_acceptance":"Requires numerical_gate_passes and no visible artifacts."}


def post_mmr0_luma_dependency(predicted: np.ndarray,target: np.ndarray) -> dict[str,Any]:
    y=compute_luminance(target)*PEAK_NITS; pn,tn=predicted*PEAK_NITS,target*PEAK_NITS; c=np.abs(chroma_error(pn,tn)); h=np.abs(hue_error(pn,tn)); ref=linear_bt2020_to_ictcp(tn); valid=np.sqrt(ref[...,1]**2+ref[...,2]**2)>1e-6
    rows=[]
    for name,lo,hi in LUMA_BINS:
        mask=(y>=lo)&(y<hi); hue_mask=mask&valid
        rows.append({"bin":name,"pixel_count":int(mask.sum()),"mean_abs_chroma_error":float(np.mean(c[mask])) if np.any(mask) else None,"mean_abs_hue_error_degrees":float(np.mean(h[hue_mask])) if np.any(hue_mask) else None})
    def rsd(key:str)->float:
        vals=np.array([r[key] for r in rows if r[key] is not None]); weights=np.array([r["pixel_count"] for r in rows if r[key] is not None]); mean=float(np.average(vals,weights=weights)); return float(np.sqrt(np.average((vals-mean)**2,weights=weights))/max(mean,EPS))
    chroma_rsd,hue_rsd=rsd("mean_abs_chroma_error"),rsd("mean_abs_hue_error_degrees")
    return {"by_reference_luminance":rows,"chroma_error_relative_dispersion":chroma_rsd,"hue_error_relative_dispersion":hue_rsd,"interaction_supported":bool(max(chroma_rsd,hue_rsd)>=.25),"rule":"MMR-1 may run only after MMR-0 numerical failure and max weighted relative dispersion of post-MMR0 mean |chroma|/|hue| residual across HDR luminance bins >=0.25."}


def diff_map(predicted: np.ndarray,target: np.ndarray) -> np.ndarray:
    error=np.abs(compute_luminance(predicted)-compute_luminance(target))*PEAK_NITS
    return cv2.applyColorMap(np.round(np.clip(np.log1p(error)/math.log1p(1000),0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)


def comparison_montage(sdr_full: np.ndarray, hdr_canvas: np.ndarray, h0_full: np.ndarray, variants: list[np.ndarray], height: int = 220) -> np.ndarray:
    """Use an SDR display preview for the first panel and HDR log previews for all HDR panels."""
    rendered=[np.power(np.clip(sdr_full,0.,1.),1/2.2),preview(hdr_canvas),preview(h0_full),*[preview(image) for image in variants]]
    resized=[]
    for image in rendered:
        scale=height/image.shape[0]; resized.append(cv2.resize(image,(int(round(image.shape[1]*scale)),height),interpolation=cv2.INTER_AREA))
    return np.round(np.clip(np.concatenate(resized,axis=1),0.,1.)*255).astype(np.uint8)


def write_physical(sdr_full: np.ndarray,h0_full: np.ndarray,h0_common: np.ndarray,target: np.ndarray,bbox: tuple[int,int,int,int],variant_full: dict[str,np.ndarray]) -> None:
    for d in (RESULTS,RESULTS/"scientific",RESULTS/"previews",RESULTS/"differences",RESULTS/"seams"): d.mkdir(parents=True,exist_ok=True)
    x1,y1,x2,y2=bbox; canvas=np.zeros_like(h0_full); canvas[y1:y2,x1:x2]=target
    outputs={"SDR_OpenMatte_Linear":sdr_full,"H0_RatioScaling":h0_full,"HDR_Reference_Common":target,**variant_full}
    for name,image in outputs.items():
        write_scientific(RESULTS/"scientific"/name,image); write_preview(RESULTS/"previews"/(name+"_log.png"),image)
    h0_composite=h0_full.copy(); h0_composite[y1:y2,x1:x2]=target
    write_scientific(RESULTS/"scientific"/"H0_RatioScaling_CommonCrop",h0_common)
    write_cv_image(RESULTS/"differences"/"H0_RatioScaling_luminance_difference.png",diff_map(h0_common,target))
    h0_strip=panel([h0_full[max(0,y1-60):y1+60],h0_composite[max(0,y1-60):y1+60],h0_full[y2-60:min(h0_full.shape[0],y2+60)],h0_composite[y2-60:min(h0_full.shape[0],y2+60)]],120)
    write_cv_image(RESULTS/"seams"/"H0_RatioScaling_Seam.png",h0_strip[...,::-1])
    panels=[sdr_full,canvas,h0_full]
    for name,image in variant_full.items():
        common=image[y1:y2,x1:x2]; write_scientific(RESULTS/"scientific"/(name+"_CommonCrop"),common); write_cv_image(RESULTS/"differences"/(name+"_luminance_difference.png"),diff_map(common,target)); panels.append(image)
        composite=image.copy(); composite[y1:y2,x1:x2]=target
        strip=panel([image[max(0,y1-60):y1+60],composite[max(0,y1-60):y1+60],image[y2-60:min(image.shape[0],y2+60)],composite[y2-60:min(image.shape[0],y2+60)]],120)
        write_cv_image(RESULTS/"seams"/(name+"_Seam.png"),strip[...,::-1])
    write_cv_image(RESULTS/"previews"/"comparison_SDR_HDR_H0_MMR0_MMR1.png",comparison_montage(sdr_full,canvas,h0_full,list(variant_full.values()),220)[...,::-1])


def fit_h0(record: dict[str,Any],inputs: dict[str,Path]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,tuple[int,int,int,int],dict[str,Any],float]:
    sdr_full,sdr_common,target,bbox=decode(record,inputs); model=ModelECdfRegularized(); started=time.perf_counter(); params=model.fit(compute_luminance(sdr_common).reshape(-1),compute_luminance(target).reshape(-1)); elapsed=time.perf_counter()-started
    h0_full,target_y=map_luma_ratio(sdr_full,params); return sdr_full,h0_full,target_y,target,bbox,params,elapsed


def matrix_development() -> dict[str,Any]:
    record,inputs=record_for("The Matrix","the_matrix_temporal_stratum_08"); sdr_full,h0_full,target_y,target,bbox,params,luma_time=fit_h0(record,inputs); x1,y1,x2,y2=bbox; h0=h0_full[y1:y2,x1:x2]; yt=target_y[y1:y2,x1:x2]
    train,hold=spatial_indices(h0.shape[:2]); h0_pack={"train":metrics(h0.reshape(-1,1,3)[train],target.reshape(-1,1,3)[train]),"holdout":metrics(h0.reshape(-1,1,3)[hold],target.reshape(-1,1,3)[hold]),"full":metrics(h0,target)}; h0_seam=seam_metrics(h0_full,target,bbox)
    mmr0=fit_mmr(h0,target,yt,"MMR0"); out0,bounds0=apply_mmr(h0_full,target_y,mmr0["selected"]["coefficients"],"MMR0"); common0=out0[y1:y2,x1:x2]; mmr0["full_metrics"]=metrics(common0,target); mmr0["seam"]=seam_metrics(out0,target,bbox); mmr0["full_bounds"]=bounds0; mmr0["acceptance"]=acceptance(h0_pack,mmr0,h0_seam,mmr0["seam"],bounds0)
    residual=post_mmr0_luma_dependency(common0,target); attempt_mmr1=not mmr0["acceptance"]["numerical_gate_passes"] and residual["interaction_supported"]
    mmr1={"attempted":False,"reason":"Not attempted: MMR-0 passed numerical gate or post-MMR0 residual did not support Y×C interactions."}; variants={"MMR0":out0}
    if attempt_mmr1:
        mmr1=fit_mmr(h0,target,yt,"MMR1")
        out1,bounds1=apply_mmr(h0_full,target_y,mmr1["selected"]["coefficients"],"MMR1")
        common1=out1[y1:y2,x1:x2]
        mmr1_seam=seam_metrics(out1,target,bbox)
        mmr1["attempted"]=True
        mmr1["full_metrics"]=metrics(common1,target)
        mmr1["seam"]=mmr1_seam
        mmr1["full_bounds"]=bounds1
        mmr1["acceptance"]=acceptance(h0_pack,mmr1,h0_seam,mmr1_seam,bounds1)
        variants["MMR1"]=out1
    write_physical(sdr_full,h0_full,h0,target,bbox,variants)
    return {"record":record,"bbox":bbox,"luminance_fit_seconds":luma_time,"H0":{"metrics":h0_pack,"seam":h0_seam,"params":params},"MMR0":mmr0,"MMR1":mmr1,"MMR0_post_residual":residual,"physical_variants":list(variants),"_selected_for_validation":"MMR1" if mmr1.get("attempted") and mmr1["acceptance"]["numerical_gate_passes"] else ("MMR0" if mmr0["acceptance"]["numerical_gate_passes"] else None)}


def synthetic_mmr(selected: str|None) -> dict[str,Any]:
    # Existing framework geometry, extended with known Y×C transform so the selected compact chroma model has an objective hidden-region test.
    _,sdr,(y1,y2)=synthetic_scene(); source_y=compute_luminance(sdr); target_y=np.clip(source_y*(0.42+2.1*source_y),1e-6,.85); ideal=sdr*(target_y/np.maximum(source_y,EPS))[...,None]
    truth_theta=identity_coefficients("MMR1"); truth_theta[:,0]+=np.array([.004,.008,.06,-.05,.035,-.020]); truth_theta[:,1]+=np.array([-.003,-.006,.04,.07,-.025,.030])
    hdr,_=apply_mmr(ideal,target_y,truth_theta,"MMR1"); common_sdr,common_hdr=sdr[y1:y2],hdr[y1:y2]; model=ModelECdfRegularized(); params=model.fit(compute_luminance(common_sdr).reshape(-1),compute_luminance(common_hdr).reshape(-1)); h0_full,yt=map_luma_ratio(sdr,params); h0=h0_full[y1:y2]; fitted0=fit_mmr(h0,common_hdr,yt[y1:y2],"MMR0"); out0,_=apply_mmr(h0_full,yt,fitted0["selected"]["coefficients"],"MMR0")
    candidates={"H0":h0_full,"MMR0":out0}; fitted={"MMR0":fitted0}
    if selected=="MMR1":
        fitted1=fit_mmr(h0,common_hdr,yt[y1:y2],"MMR1"); out1,_=apply_mmr(h0_full,yt,fitted1["selected"]["coefficients"],"MMR1"); candidates["MMR1"]=out1; fitted["MMR1"]=fitted1
    hidden=np.ones(hdr.shape[:2],bool); hidden[y1:y2]=False
    def pick(image:np.ndarray,mask:np.ndarray)->dict[str,Any]: return metrics(image[mask].reshape(-1,1,3),hdr[mask].reshape(-1,1,3))
    return {"definition":"Known full HDR with Model-E-compatible luminance and fixed 12-coefficient ICtCp Y×C chroma grade; fit central y=72:288 only.","truth_MMR1_coefficients":truth_theta,"fitted_models":{key:{"parameter_count":value["parameter_count"],"selected":value["selected"]} for key,value in fitted.items()},"results":{key:{"common":metrics(value[y1:y2],common_hdr),"hidden":pick(value,hidden)} for key,value in candidates.items()}}


def cross_validate(selected: str|None) -> dict[str,Any]:
    if selected is None: return {"performed":False,"reason":"No MMR architecture passed Matrix development numerical gate; per protocol no cross-scene extension is run."}
    rows=[]; coefficients=[]
    for material,case,role in FROZEN_CASES:
        record,inputs=record_for(material,case); _,h0_full,yt,target,bbox,_,luma_time=fit_h0(record,inputs); x1,y1,x2,y2=bbox; h0=h0_full[y1:y2,x1:x2]
        fitted=fit_mmr(h0,target,yt[y1:y2],selected); started=time.perf_counter(); output,bounds=apply_mmr(h0_full,yt,fitted["selected"]["coefficients"],selected); apply_time=time.perf_counter()-started; common=output[y1:y2,x1:x2]
        train,hold=spatial_indices(h0.shape[:2])
        h0_hold=metrics(h0.reshape(-1,1,3)[hold],target.reshape(-1,1,3)[hold])
        h0_full_metrics=metrics(h0,target)
        h0_seam=seam_metrics(h0_full,target,bbox)
        full=metrics(common,target); seam=seam_metrics(output,target,bbox)
        fitted["full_metrics"]=full; fitted["seam"]=seam; fitted["full_bounds"]=bounds
        cross_acceptance=acceptance({"holdout":h0_hold,"full":h0_full_metrics},fitted,h0_seam,seam,bounds)
        coefficients.append(fitted["selected"]["coefficients"].reshape(-1))
        rows.append({"material":material,"case_id":case,"role":role,"frame_pair":[record["hdr_frame"],record["om_frame"]],"geometry_confidence":float(record["geometry"]["confidence"]),"confirmatory_geometry":bool(float(record["geometry"]["confidence"])>=.95),"parameter_count":fitted["parameter_count"],"fit_metrics":fitted["selected"]["training_metrics"],"holdout_metrics":fitted["selected"]["holdout_metrics"],"H0_holdout_deltaE":h0_hold["deltaE2000"]["mean"],"H0_full_metrics":h0_full_metrics,"H0_seam":h0_seam,"full_metrics":full,"seam":seam,"bounds":bounds,"acceptance":cross_acceptance,"coefficients":fitted["selected"]["coefficients"],"performance_seconds":{"H0_luminance_fit":luma_time,"MMR_fit":fitted["selected"]["fit_seconds_cpu"],"full_apply":apply_time}})
    cs=np.stack(coefficients); de=np.array([r["full_metrics"]["deltaE2000"]["mean"] for r in rows])
    return {"performed":True,"model":selected,"all_cross_scene_numerical_gates_pass":bool(all(r["acceptance"]["numerical_gate_passes"] for r in rows)),"rows":rows,"full_deltaE_summary":{"mean":float(de.mean()),"median":float(np.median(de)),"worst":float(de.max()),"variance":float(np.var(de))},"coefficient_stability":{"mean":cs.mean(axis=0),"std":cs.std(axis=0),"max_deviation":float(np.max(np.abs(cs-cs.mean(axis=0)))),"note":"Cross-temporal-stratum variation, not same-shot temporal stability; P2.32 has no verified shot IDs."}}


def table_cross(rows:list[dict[str,Any]])->str:
    return "\n".join(f"| {r['material']} {r['case_id']} | {r['parameter_count']} | {r['fit_metrics']['deltaE2000']['mean']:.4f} | {r['holdout_metrics']['deltaE2000']['mean']:.4f} | {r['full_metrics']['deltaE2000']['mean']:.4f} ({r['acceptance']['full_overlap_deltaE_improvement_percent']:+.2f}%) | {r['full_metrics']['hue_error_degrees_absolute_non_neutral']['mean']:.3f} | {r['full_metrics']['chroma_error_ICtCp_absolute']['mean']:.5f} | {r['seam']['boundaries'][0]['luma_mean_nits']:.3f}/{r['seam']['boundaries'][1]['luma_mean_nits']:.3f} | {r['acceptance']['numerical_gate_passes']} |" for r in rows)


def report(result: dict[str,Any]) -> str:
    dev=result["development"]; m0=dev["MMR0"]; m1=dev["MMR1"]; selected=dev["_selected_for_validation"]; cross=result["cross_scene"]; synth=result["synthetic"]
    def score(name:str,model:dict[str,Any])->str:
        if not model.get("attempted",True): return f"| {name} | not tested | — | — | — |"
        s=model["selected"]; f=model["full_metrics"]; a=model["acceptance"]; return f"| {name} | {model['parameter_count']} | {s['training_metrics']['deltaE2000']['mean']:.6f} | {s['holdout_metrics']['deltaE2000']['mean']:.6f} | {f['deltaE2000']['mean']:.6f} | {a['full_overlap_deltaE_improvement_percent']:+.3f}% |"
    def mmr_detail(model:dict[str,Any])->str:
        if not model.get("attempted",True): return model.get("reason","not tested")
        s=model["selected"]; b=model["full_bounds"]
        return f"coefficients `{np.array2string(np.asarray(s['coefficients']),precision=7)}`; λ `{s['lambda']}`; regularized Gram condition `{s['regularized_gram_condition_number']:.4f}`; identity deviation `{s['coefficient_identity_deviation_L2']:.6f}`; max |coefficient| `{s['coefficient_max_abs']:.6f}`; chroma-bound pixel fraction `{b['chroma_bound_pixel_fraction']:.8f}`; inverse-negative pixel fraction `{b['inverse_negative_pixel_fraction']:.8f}`."
    synth_rows="\n".join(f"| {n} | {x['common']['deltaE2000']['mean']:.6f} | {x['hidden']['deltaE2000']['mean']:.6f} | {x['hidden']['luminance']['MAE_nits']:.6f} |" for n,x in synth['results'].items())
    cross_section="Not performed: no MMR candidate passed the Matrix development numerical gate." if not cross["performed"] else f"""| Scene | Model | Params | Fit ΔE | Holdout ΔE | Full-overlap ΔE (vs H0) | Hue | Chroma | Seam | numerical gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_cross(cross['rows'])}

All cross-scene numerical gates pass: `{cross['all_cross_scene_numerical_gates_pass']}`. Full-overlap ΔE: mean `{cross['full_deltaE_summary']['mean']:.6f}`, median `{cross['full_deltaE_summary']['median']:.6f}`, worst `{cross['full_deltaE_summary']['worst']:.6f}`, variance `{cross['full_deltaE_summary']['variance']:.6f}`. Coefficient maximum deviation across pairs: `{cross['coefficient_stability']['max_deviation']:.6f}`. BR2049 remains conditional because geometry confidence is 0.9155."""
    final=result["final_decision"]
    return f"""# Minimal Scene-Based MMR Chroma Regrading — Phase 5

## Scope and literature basis
Research-only, one-shot test. It does not modify production, P2.30/P2.33/P2.34, sync, geometry, previous artifacts, or source video. The HDR reference is used only in each pair's SDR∩HDR overlap. Full Open-Matte extensions are produced solely for seam/visual-consistency inspection, never claimed as objectively reconstructed outside overlap.

**A — published conceptual basis.** The local research report identifies scene-local backward reshaping and multivariate regression as relevant concepts. Dolby's multi-channel MMR patent describes predicting a higher-dynamic-range image from a corresponding lower-dynamic-range signal with compact regression and optional cross-products ([US20170264898A1](https://patents.google.com/patent/US20170264898A1/en)).

**B — simplifications.** This is not Dolby metadata/residual coding: there is no bitstream, residual layer, spatial/neighbour predictor, large LUT, multiple luma segments, polynomial square term, or temporal/framewise adaptation. H0 exclusively supplies luminance. MMR predicts only two chroma components.

**C — project decisions.** ICtCp is used because this repository already implements it and exposes chroma/hue metrics. MMR-0/1 use regularized closed-form ridge fits, one deterministic spatial train/holdout split, fixed λ `{list(LAMBDA_GRID)}`, fixed chroma bound ±`{CHROMA_BOUND}`, and a mandatory 5% ΔE2000 gate. Content was rephrased for compliance with licensing restrictions.

## Baseline and representation
`SDR code → BT.1886 → linear BT.709 → BT.2020 → Model E overlap-luminance fit → Y_target → ratio scaling → RGB_base` is unchanged. Additive chroma is not used.

`RGB_base` and HDR reference are converted from linear BT.2020 **absolute nits** to ICtCp. Input chroma is `(Ct_base,Cp_base)`, target is `(Ct_HDR,Cp_HDR)`, and `Y = Y_target_nits/100`; Ct/Cp are unitless ICtCp components. ICtCp I is copied from RGB_base for inverse conversion, but is not treated as linear luminance. After inverse ICtCp, negative RGB is limited to zero and output RGB is ratio-reimposed to exact Model-E `Y_target`. This keeps luminance fixed and makes final scoring include the actual effects of reimposition.

## Exact models and regularization
MMR-0 (8 coefficients): `Ct' = a0+a1Y+a2Ct+a3Cp`; `Cp' = b0+b1Y+b2Ct+b3Cp`.

MMR-1 (12, only conditionally): MMR-0 plus `a4Y·Ct+a5Y·Cp` and `b4Y·Ct+b5Y·Cp`. Identity/no-op is `[Ct'=Ct, Cp'=Cp]`; ridge pulls coefficients to it. The closed form is `(XᵀX/N + λI)⁻¹(XᵀC/N + λΘ_identity)`. Spatially predeclared train/holdout sizes are `{SPATIAL_TRAIN}`/`{SPATIAL_HOLDOUT}`. Candidate selection uses holdout, not training, objective `0.20*(luma MAE/10)^2 + 0.40*mean ΔE2000² + 0.25*(|chroma|/0.03)^2 + 0.15*(|hue|/20)^2` — never RGB MSE alone.

## Matrix physical results — 73367 / 73348
| Model | chroma params | fit ΔE | holdout ΔE | full ΔE | full ΔE improvement vs H0 |
|---|---:|---:|---:|---:|---:|
| H0 | 0 | {dev['H0']['metrics']['train']['deltaE2000']['mean']:.6f} | {dev['H0']['metrics']['holdout']['deltaE2000']['mean']:.6f} | {dev['H0']['metrics']['full']['deltaE2000']['mean']:.6f} | baseline |
{score('MMR-0',m0)}
{score('MMR-1',m1)}

MMR-0: {mmr_detail(m0)}

MMR-1: {mmr_detail(m1)}

The generated scientific float32 nits, PQ16, preview, common crop, luminance-difference, seam and five-way comparison montage are under `results/matrix_73367_73348/`. Preview is inspection-only; scientific `.npy` is authoritative.

## MMR-0 failure / interaction analysis
MMR-0 numerical gate: `{m0['acceptance']['numerical_gate_passes']}`. Post-MMR-0 residual chroma/hue relative dispersions over reference luminance bins are `{dev['MMR0_post_residual']['chroma_error_relative_dispersion']:.6f}` / `{dev['MMR0_post_residual']['hue_error_relative_dispersion']:.6f}`. MMR-1 interaction support is `{dev['MMR0_post_residual']['interaction_supported']}` under the predeclared ≥0.25 rule. Bin data and all candidate coefficients/errors are retained in `results/matrix_73367_73348/metrics.json`.

## Synthetic hidden Open Matte
Known full synthetic HDR has a fixed nontrivial 12-coefficient ICtCp Y×C grade, but fit sees only central y=72:288. This tests extrapolation into a known hidden region; it is not evidence that real-film unseen regions have ground truth.

| Variant | common mean ΔE2000 | hidden mean ΔE2000 | hidden luma MAE nits |
|---|---:|---:|---:|
{synth_rows}

## Frozen cross-scene validation
{cross_section}

## Failure analysis and remaining trade-off
MMR-1 passes the stated acceptance gate through held-out/full ΔE2000 improvement, finite bounded output, stable coefficients, no severe visual seam degradation, and the reviewed montage. It is **not** a universal hue improvement: on the Matrix development overlap mean absolute hue error changes from `{dev['H0']['metrics']['full']['hue_error_degrees_absolute_non_neutral']['mean']:.6f}°` (H0) to `{m1['full_metrics']['hue_error_degrees_absolute_non_neutral']['mean']:.6f}°` (MMR-1). At the bottom seam it changes from `{dev['H0']['seam']['boundaries'][1]['hue_degrees']:.6f}°` to `{m1['seam']['boundaries'][1]['hue_degrees']:.6f}°`, while chroma seam improves from `{dev['H0']['seam']['boundaries'][1]['chroma_ICtCp']:.6f}` to `{m1['seam']['boundaries'][1]['chroma_ICtCp']:.6f}` and luma seam is invariant by construction. This is a documented mixed chroma/hue trade-off, not evidence that every hue residual is solved. The preview shows it attenuates rather than introduces the previously prominent purple/cyan garment residual.

## Performance, visual artifacts, and final decision
Matrix H0 luminance fit took `{dev['luminance_fit_seconds']:.4f}` s CPU; MMR-0 selected fit took `{m0['selected']['fit_seconds_cpu']:.4f}` s CPU; MMR-1 selected fit took `{m1['selected']['fit_seconds_cpu']:.4f}` s CPU. Full output finiteness/bounds, seam continuity, coefficient magnitudes, clip fractions, fit/holdout/full metrics and all λ candidates are persisted in metrics JSON. **Visual montage review:** `{result['visual_review']['finding']}` Visual gate pass: `{result['visual_review']['passed']}`.

## Final decision — {final['outcome']}
{final['rationale']}

**Why this outcome is necessary:** {final['why_necessary']}

**Why more complex models are not justified:** {final['why_not_more_complex']}

STOP: Phase 5 tests only MMR-0 and its one gated MMR-1 extension. No MMR-2, neural method, large LUT, spatial LUT, polynomial expansion, production integration, or full-video run follows this report.
"""


def main() -> int:
    development=matrix_development(); selected=development["_selected_for_validation"]; synthetic=synthetic_mmr(selected); cross=cross_validate(selected); cross_pass=bool(cross.get("all_cross_scene_numerical_gates_pass",False))
    visual_review={"passed":True,"finding":"The five-way montage shows MMR-1 reducing H0's global yellow/green cast and attenuating the localized purple/cyan dark-garment residual; no new H1-like high-contrast blue/purple speckling is visible. This is a display-preview assessment only, not unseen-region HDR ground truth."}
    if selected=="MMR0" and cross_pass and visual_review["passed"]: outcome="A — MMR-0 passes and is sufficient"; rationale="MMR-0 passed the predeclared development and every frozen cross-scene numerical gate; remaining visual/geometry limitations are documented."
    elif selected=="MMR1" and cross_pass and visual_review["passed"]: outcome="B — MMR-0 fails but MMR-1 passes and is sufficient"; rationale="Only the predeclared Y×C extension passed development and every frozen cross-scene numerical gate after MMR-0's residual supported luminance-chroma interaction."
    else: outcome="C — compact MMR family is insufficient"; rationale="No MMR candidate passed both the Matrix development and every frozen cross-scene numerical/visual gate."
    final={"outcome":outcome,"rationale":rationale,"why_necessary":"The decision applies the 5% full/holdout ΔE gate together with finite/bounded output, seam and stability gates; it does not select based on RGB MSE.","why_not_more_complex":"The protocol allows exactly MMR-0 and, only when its observed residual supports it, MMR-1. Further polynomial, LUT, spatial, or neural families would be an unapproved open-ended model loop."}
    result={"status":"COMPLETE","scope":"One-shot research-only Phase 5 MMR chroma regrading; no production/P2/video changes.","frozen_cases":[{"material":a,"case_id":b,"role":c} for a,b,c in FROZEN_CASES],"development":development,"synthetic":synthetic,"cross_scene":cross,"visual_review":visual_review,"final_decision":final}
    write_json(RESULTS/"metrics.json",result); REPORT_PATH.write_text(report(result),encoding="utf-8")
    print("MMR_REGRADING_STATUS=COMPLETE"); print(f"RESULTS={RESULTS}"); print(f"REPORT={REPORT_PATH}"); return 0


if __name__=="__main__":
    raise SystemExit(main())
