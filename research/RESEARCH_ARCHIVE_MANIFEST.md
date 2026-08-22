# Research archive manifest

**Archive purpose:** preserve the completed research-phase code, reports, metrics, audit records, manifests, reproducibility protocols, and selected small diagnostics without changing algorithms or uploading the generated large binary datasets.

**Archive date:** 2026-08-20  
**Branch:** `feat/full-pipeline-implementation`  
**Base commit before archival commit:** `3e695e8874197dbd85cc0f592538829cfa5f9d1c`  
**Remote:** `https://github.com/BarrFarr/Auto-OpenMatte-Sdr2hdr-converter.git`

## Included scope

The archival commit stages the following explicit scopes; it does not stage the repository wholesale:

- Root `P2.17` through `P2.34` research reports and small result JSON files.
- Root `P2.22_gpu_nvenc_probe.py`, `P2.23_native_cuda_probe.py`, `P2.26_cuda_compile_probe.cu`, the `bench_p*.py` research harnesses, the root `test_*.py` harnesses, `tests/`, and `src/auto_openmatte/processing/transform_backend.py`.
- All research source, report, metric, manifest, audit, and protocol files under `research/` with extensions `.py`, `.md`, `.json`, `.csv`, and `.toml`. This includes `research/src/reshaping_research/` model code and all physical-validation phase reports/protocols.
- Six small P2.22/P2.23 10-frame probe outputs under `P2.22_outputs/` and `P2.23_outputs/`.
- Selected diagnostic PNGs listed below. These are contact sheets/comparison panels, not full render sequences.
- Small, explicitly selected validation records under `dev/ffmpeg-build/validation/` for P230–P234: source harnesses, metrics, manifests, CSV audits, shot curves/boundaries, recovered results, streaming smoke results, and the non-rerun P234 full-shot summary.

### Selected diagnostic PNGs

The 16 selected PNGs total **11,380,251 bytes**:

```text
research/physical_validation/color_diagnostics/results/matrix_73367_73348/previews/comparison_T1_HDR_T2_T3_T4_T5.png
research/physical_validation/final_scene_validation/results/shots/br2049_temporal_stratum_03/contact_sheet_representative_frames.png
research/physical_validation/final_scene_validation/results/shots/the_matrix_temporal_stratum_02/contact_sheet_representative_frames.png
research/physical_validation/final_scene_validation/results/shots/the_matrix_temporal_stratum_08/contact_sheet_representative_frames.png
research/physical_validation/mmr_regrading/results/matrix_73367_73348/previews/comparison_SDR_HDR_H0_MMR0_MMR1.png
research/physical_validation/resampling_diagnostics/results/matrix_73367_73348/previews/comparison_SDR_HDR_A_HDR_B_H0_A_H0_B.png
research/physical_validation/results/matrix_73367_73348/previews/comparison_full_A_B_C_D.png
research/physical_validation/results/matrix_73367_73348/previews/synthetic_hidden_region_comparison.png
research/physical_validation/scene_level_validation/results/phase6_heldout_contact_sheet.png
research/physical_validation/scene_level_validation/results/the_matrix_temporal_stratum_08/previews/10pct_comparison_SDR_HDR_H0_shared_MMR1.png
research/physical_validation/scene_level_validation/results/the_matrix_temporal_stratum_08/previews/25pct_comparison_SDR_HDR_H0_shared_MMR1.png
research/physical_validation/scene_level_validation/results/the_matrix_temporal_stratum_08/previews/50pct_comparison_SDR_HDR_H0_shared_MMR1.png
research/physical_validation/scene_level_validation/results/the_matrix_temporal_stratum_08/previews/75pct_comparison_SDR_HDR_H0_shared_MMR1.png
research/physical_validation/scene_level_validation/results/the_matrix_temporal_stratum_08/previews/90pct_comparison_SDR_HDR_H0_shared_MMR1.png
research/physical_validation/scene_regrading/results/matrix_73367_73348/previews/H1_ComparisonPanel.png
research/physical_validation/scene_regrading/results/matrix_73367_73348/synthetic/comparison.png
```

## Explicit validation records staged from `dev/`

The following records are retained; large sibling arrays/renders are not:

```text
dev/ffmpeg-build/validation/P230_fit_model_experiment.py
dev/ffmpeg-build/validation/P230_fit_models/P230_fit_model_metrics.csv
dev/ffmpeg-build/validation/P230_fit_models/P230_fit_model_metrics.json
dev/ffmpeg-build/validation/P231_model_b_generalization.py
dev/ffmpeg-build/validation/P231_model_b_generalization/P231_frame_selection_manifest.json
dev/ffmpeg-build/validation/P231_model_b_generalization/P231_model_b_metrics.json
dev/ffmpeg-build/validation/P231_model_b_generalization/P231_model_b_per_frame.csv
dev/ffmpeg-build/validation/P232_real_material_dataset.py
dev/ffmpeg-build/validation/P232_real_material_dataset/P232_dataset_manifest.json
dev/ffmpeg-build/validation/P232_real_material_dataset/P232_dataset_metrics.json
dev/ffmpeg-build/validation/P232_real_material_dataset/P232_luminance_histograms.csv
dev/ffmpeg-build/validation/P233_shot_adaptive_tone_mapping.py
dev/ffmpeg-build/validation/P233_tonal_model_experiment.py
dev/ffmpeg-build/validation/P233_shot_adaptive/P233_per_frame_results.csv
dev/ffmpeg-build/validation/P233_shot_adaptive/P233_shot_adaptive_metrics.json
dev/ffmpeg-build/validation/P233_shot_adaptive/shot_curves/*.json
dev/ffmpeg-build/validation/P234*.py
dev/ffmpeg-build/validation/P234_cuda_parity*/parity_result.json
dev/ffmpeg-build/validation/P234_empty_cuda_targeted/*.json
dev/ffmpeg-build/validation/P234_shot_level/shot_boundaries/*.json
dev/ffmpeg-build/validation/P234_shot_level/shot_curves/*.json
dev/ffmpeg-build/validation/P234_shot_level_full/**/*.json
dev/ffmpeg-build/validation/P234_shot_level_full/P234_shot_results.csv
dev/ffmpeg-build/validation/P234_shot_level_recovered/*.{json,csv}
dev/ffmpeg-build/validation/P234_streaming_smoke/**/*.json
```

The P234 `shot_level_full` directory is the small 5,023,246-byte summary set. The `shot_level_full_rerun` directory is intentionally excluded in its entirety.

## Measured excluded artifacts

All measurements below were taken before staging. Sizes are byte counts, not estimates.

### Research tree

The complete `research/` tree measured **3,450,660,913 bytes**:

| Type | Files | Bytes | Treatment |
|---|---:|---:|---|
| `.mkv` | 2 | 1,673,728,273 | Excluded; full FFV1 renders |
| `.npy` | 45 | 1,057,412,736 | Excluded; frame/reference arrays |
| `.png` | 216 | 710,397,381 | 16 small diagnostics above included; remaining 200 files / 699,017,130 bytes excluded |
| `.pyc` | 32 | 780,193 | Excluded cache/bytecode |
| `.json` | 21 | 7,608,069 | Included as research metrics/manifests/protocols |
| `.py` | 49 | 635,545 | Included as research code |
| `.md` | 11 | 76,667 | Included as reports |
| `.csv` | 1 | 21,447 | Included as audit/provenance |
| `.toml` | 1 | 602 | Included as project metadata |

Full research renders excluded:

| Path | Size | SHA-256 | Description |
|---|---:|---|---|
| `research/physical_validation/final_scene_validation/results/renders/the_matrix_temporal_stratum_08/MMR1_generated_HDR_OpenMatte_full_shot_ffv1.mkv` | 991,827,805 | `C99A8D107E48FED235934C577FBB046D40013B0B2D0628E17C0DC4765EB37B91` | Matrix temporal stratum 08 full-shot FFV1 render |
| `research/physical_validation/final_scene_validation/results/renders/the_matrix_temporal_stratum_02/MMR1_generated_HDR_OpenMatte_full_shot_ffv1.mkv` | 681,900,468 | `2526F5224B6623931AC3EF8AE0AB83C5F0A6FB04EEB8D9F9A40CA9B3D6003EDF` | Matrix temporal stratum 02 full-shot FFV1 render |

### Development/validation tree

The complete `dev/` tree measured **18,546,381,250 bytes**. No broad `dev/` pathspec is used. Important measured groups omitted from Git include:

| Type/group | Files | Bytes | Treatment |
|---|---:|---:|---|
| `.npy` arrays | 193 | 7,237,656,704 | Excluded; reference/CUDA/real-material arrays |
| `.dat` model/OOF buffers | 12 | 3,686,400,000 | Excluded; P233 tonal-model buffers |
| `.exe` binaries/installers | 11 | 3,574,800,888 | Excluded; CUDA/FFmpeg tooling and installers |
| `.npz` arrays | 273 | 2,393,570,983 | Excluded; P234 shot-level samples and datasets |
| `.rgb48le` frames | 12 | 373,248,000 | Excluded; raw frame buffers |
| `.png` diagnostics/renders | 41 | 157,723,928 | Excluded from `dev/`; selected research PNGs are listed above |
| `.json` overall | 415 | 339,239,656 | Only explicitly selected small validation JSON is included |
| FFmpeg source/build/install/download trees | measured within total | measured within total | Excluded in full |

Notable excluded validation datasets:

- `dev/ffmpeg-build/validation/P231_model_b_generalization/`: 15 files, 282,110,987 bytes; only its small manifest/metrics/CSV records are retained, while visual/data artifacts are excluded.
- `dev/ffmpeg-build/validation/P232_real_material_dataset/`: 163 files, 3,947,827,267 bytes; dataset arrays and generated frames are excluded, while the script, manifest, metrics, and luminance histogram CSV are retained.
- `dev/ffmpeg-build/validation/P233_tonal_models/`: 12 files, 3,686,400,000 bytes; all tonal-model buffers are excluded.
- `dev/ffmpeg-build/validation/P233_shot_adaptive/`: 129 files, 464,179,255 bytes; only the text metrics/CSV/shot-curve records are retained.
- `dev/ffmpeg-build/validation/P234_shot_level/`: 433 files, 1,653,226,693 bytes; only JSON shot boundaries and shot curves are retained; `.npz` samples and other generated data are excluded.
- `dev/ffmpeg-build/validation/P234_shot_level_full_rerun/`: 43 JSON files, 321,961,681 bytes; excluded in full, including the large rerun summary files below.

### Large P234 rerun records documented but not committed

| Path | Size | SHA-256 | Description |
|---|---:|---|---|
| `dev/ffmpeg-build/validation/P234_shot_level_full_rerun/P234_shot_level_metrics.json` | 116,317,477 | `F179FE6A3B891F109C4AD69F9B2B0BA7A635BCFAC648F75F145E4609324BDAD7` | Full rerun shot-level metrics |
| `dev/ffmpeg-build/validation/P234_shot_level_full_rerun/P234_shot_manifest.json` | 109,567,557 | `917652EB204B3911DB3E8BF13437408F5CB3589753F21056D4B91576943B252D` | Full rerun shot manifest |
| `dev/ffmpeg-build/validation/P234_shot_level_analysis.py` | 72,340 | `136ABA708F2EDF4E393EC18B692CE8279EA63316040469682538D44E28D3FB94` | Analysis script retained separately and staged |

### Workspace caches and temporary output

| Path | Files | Bytes | Treatment |
|---|---:|---:|---|
| `.venv/` | 10,921 | 1,964,614,321 | Excluded environment |
| `test_output/` | 30 | 739,957,904 | Excluded temporary/generated output |
| `.pytest_cache/` | 5 | 27,962 | Excluded cache |
| `.ruff_cache/` | 9 | 7,130 | Excluded cache |

## Small P2.22/P2.23 diagnostic probes retained

These are six short probe files, not full renders:

| Path | Size | SHA-256 |
|---|---:|---|
| `P2.22_outputs/P2.22_variant_A_cpu_nvenc_10f.mkv` | 170,589 | `4FD6EBB1204506EAC8EAE4397D8DE88F300A3AFD4F6FAEEC71A44F86A293EF72` |
| `P2.22_outputs/P2.22_variant_B_gpu_download_nvenc_10f.mkv` | 170,589 | `91BA721E273C76102C72C56FBB8A78D2DF1277C19D7B062C37250DE8AD9EFF` |
| `P2.22_outputs/P2.22_variant_C_host_hwupload_probe.mkv` | 12,341 | `1AC1BA9ED38D8EB55A3229132E95596A7A3B6545FAC9F8F312303FAD8D2CC42C` |
| `P2.23_outputs/P2.23_variant_1_nvdec_nvenc_10f.mkv` | 9,391 | `E83F8FE9473170DDE4418A66E607EBB68AD3939DBA135049B0E3C201E595C5CA` |
| `P2.23_outputs/P2.23_variant_2_nvdec_scale_cuda_nvenc_10f.mkv` | 5,188 | `D959EC43DB41B1CE60F99709B7A808841E6B6642C5627AECF8136D4E10C99835` |
| `P2.23_outputs/P2.23_variant_3_nvdec_colorspace_cuda_p010_10f.mkv` | 9,646 | `5B35C01B373C7FBD04BCE7D28A30B44E313C01A7251E53801307087F86997F56` |

## Reproduction pointers

- Phase reports and frozen protocols are stored beside their runners under `research/physical_validation/`; use the corresponding `*_FROZEN_PROTOCOL.json` or `*_PROTOCOL.json` as the experiment contract.
- Phase 6: `research/physical_validation/scene_level_validation/run_scene_level_validation.py` and `PHASE6_FROZEN_PROTOCOL.json`.
- Phase 7: `research/physical_validation/fast_scene_fit_benchmark/run_fast_scene_fit_benchmark.py`, `sampling_audit/run_sampling_audit.py`, and their protocol/metric files.
- Phase 8: `research/physical_validation/final_scene_validation/run_final_scene_validation.py`, `PHASE8_FROZEN_PROTOCOL.json`, and `FINAL_SCENE_VALIDATION_REPORT.md`.
- Scene/model investigations: the runners under `scene_regrading/`, `model_discrimination/`, `mmr_regrading/`, `resampling_diagnostics/`, and `color_diagnostics/`, with their adjacent reports and metrics.
- P230–P234 validation harnesses and retained summaries are under `dev/ffmpeg-build/validation/` as listed above. The omitted arrays/renders are referenced by their original paths and measured sizes so they can be restored from the working research environment when needed.
- For an artifact hash, use PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <path>`; the large artifacts and key rerun records above have already been hashed.

No new tests were run for this archive-only operation. No algorithm, P2.30, P2.33, or P2.34 artifact was modified or deleted.
