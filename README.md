# MOF Descriptor Adequacy Benchmark

**When Are Simple Descriptors Enough in MOF Adsorption Machine Learning?**

This repository contains the code, processed outputs, manifests, and figure assets for a controlled descriptor-adequacy benchmark of metal–organic framework (MOF) gas adsorption machine learning.

The central idea of the project is that descriptor adequacy is **not** a global property of a descriptor family. A descriptor family that is sufficient for one adsorption target may fail for another. In this benchmark, descriptor adequacy is treated as a property of a **descriptor–target–deployment triple**:

```text
descriptor family × adsorption target × validation/deployment regime
```

The benchmark asks:

1. When are simple geometric descriptors enough?
2. When does chemistry-aware RAC information add transferable value?
3. When does neither descriptor family recover reliable rankings under topology shift?
4. How do target construction and screening objectives change the interpretation of regression accuracy?

The study uses ARC-MOF v7 as a common data foundation and evaluates fixed tabular descriptor families under both repeated random splits and balanced CrystalNets topology-grouped splits. The topology-grouped protocol is used as the primary deployment-like validation setting because it tests whether models transfer to MOF topologies absent from training.

---

## Repository status

This repository contains the cleaned analysis/code release.

It includes:

```text
data_manifest/    Dataset, split, and run provenance files
results/          Processed benchmark outputs used for tables, figures, and SI analyses
src/              Dataset preparation, training, analysis, and figure-generation scripts
figures/main/     Final manuscript figure assets
```

It intentionally does **not** include:

```text
raw ARC-MOF source files
predictions.csv.gz
model checkpoint .pkl files
large structural archives
manuscript .tex files
reviewer comments
temporary inspection files
```

Large artifacts such as `predictions.csv.gz` should be supplied externally through the associated data deposit or local storage.

---

## Scientific scope

The benchmark covers:

- **255,939 MOFs** in a strict common cohort.
- **27 active machine-learning targets**.
- **Five ARC-MOF gas-process families**:
  - methane storage PSA;
  - post-combustion CO₂/N₂ VSA;
  - pre-combustion CO₂/H₂ PSA;
  - natural-gas purification CO₂/CH₄;
  - landfill-gas CO₂/CH₄ VPSA.
- **Three trained target layers**:
  - Layer A: primitive uptake;
  - Layer B: working capacity;
  - Layer C: direct log-selectivity.
- **Layer D post-hoc selectivity**, reconstructed analytically during Stage 3.
- **Six descriptor families**:
  - `geometry_only`;
  - `enriched_interpretable`;
  - `topology_only`;
  - `geometry_plus_topology`;
  - `rac_only`;
  - `geo_plus_rac`.
- **Three tabular model classes**:
  - Ridge regression;
  - Random Forest;
  - LightGBM.
- **Two validation regimes**:
  - repeated random 80/20 splits;
  - balanced CrystalNets topology-grouped folds.

The main outputs quantify:

1. random-versus-topology-OOD topology-transfer gaps;
2. target-by-target descriptor adequacy under topology shift;
3. engineered-geometry and Geo+RAC descriptor gains;
4. working-capacity and log-selectivity target-construction effects;
5. top-K screening recall and champion misidentification;
6. practical-equivalence-with-cost descriptor recommendations.

---

## High-level pipeline

The workflow has three stages.

```text
Stage 1
    Build strict common cohort and layered target table.

Stage 2
    Train controlled tabular models across all target × descriptor × model × split combinations.

Stage 3
    Analyze saved metrics and predictions:
        - topology-transfer gaps
        - descriptor adequacy
        - descriptor gains
        - recommendation table
        - target-construction audits
        - top-K screening diagnostics
        - final figures
```

No models are retrained or retuned in Stage 3. Stage 3 only re-slices and summarizes outputs generated during Stage 2.

---

## Directory layout

```text
.
├── data_manifest/
│   ├── column_manifest.json
│   ├── dataset_audit.json
│   ├── merge_report.json
│   ├── raw_file_md5_manifest.csv
│   ├── run_config.json
│   ├── split_audit.csv
│   ├── split_audit_warnings.csv
│   ├── step2_final_report.json
│   ├── target_coverage_before_strict.csv
│   ├── target_coverage_strict.csv
│   └── splits/
│       ├── balanced_group_fold_0.json
│       ├── balanced_group_fold_1.json
│       ├── balanced_group_fold_2.json
│       ├── balanced_group_fold_3.json
│       ├── random_seed_42.json
│       ├── random_seed_43.json
│       ├── random_seed_44.json
│       └── random_seed_45.json
│
├── figures/
│   └── main/
│       ├── fig1_topology_transfer_gap.pdf
│       ├── fig2_descriptor_adequacy_atlas.pdf
│       ├── fig3_descriptor_gain_map.pdf
│       ├── fig4_structural_audit.pdf
│       ├── fig4_target_construction.pdf
│       └── fig5_screening_yield.pdf
│
├── results/
│   ├── step2_training/
│   ├── step3A_descriptor_analysis/
│   └── step3B_target_screening_analysis/
│
└── src/
    ├── figure_generation/
    ├── step1_prepare_dataset/
    ├── step2_train_models/
    ├── step3A_descriptor_analysis/
    └── step3B_target_screening_analysis/
```

---

## Data manifests

The `data_manifest/` folder stores provenance for the strict cohort, input files, split definitions, and Step 2 run.

### Important files

```text
column_manifest.json
dataset_audit.json
merge_report.json
raw_file_md5_manifest.csv
run_config.json
split_audit.csv
split_audit_warnings.csv
step2_final_report.json
target_coverage_before_strict.csv
target_coverage_strict.csv
```

### Split definitions

```text
data_manifest/splits/
```

contains exact fold definitions for:

```text
balanced_group_fold_0.json
balanced_group_fold_1.json
balanced_group_fold_2.json
balanced_group_fold_3.json
random_seed_42.json
random_seed_43.json
random_seed_44.json
random_seed_45.json
```

The balanced topology-grouped folds hold out CrystalNets topology groups so that no topology appears in both train and test for the same fold. The random splits are retained as in-distribution baselines.

---

## Results

### `results/step2_training/`

This folder contains the primary model-training outputs from Step 2.

```text
raw_metrics.csv
tuned_params.csv
optuna_trials.csv
feature_importances_raw.csv
```

#### `raw_metrics.csv`

Per-fold metrics for every trained combination:

```text
target × descriptor_family × model × split_type × split_name
```

Typical metrics include:

```text
R²
MAE
RMSE
Spearman correlation
Pearson correlation
bias
```

This file is the main source for Step 3A descriptor-adequacy analysis.

#### `tuned_params.csv`

Accepted hyperparameters chosen by the Optuna tuning protocol.

#### `optuna_trials.csv`

Trial-level Optuna records.

#### `feature_importances_raw.csv`

Raw feature-importance records. This file is comparatively large but remains below GitHub's hard 100 MB per-file limit.

---

### `results/step3A_descriptor_analysis/`

This folder contains descriptor-adequacy, topology-transfer, descriptor-gain, and recommendation outputs derived from `raw_metrics.csv`.

```text
atlas_long_lgbm_ood.csv
atlas_pivot_lgbm_ood.csv
descriptor_gain_map_lgbm_ood.csv
recommendation_table_lgbm_ood.csv
synthesizability_gap_per_combo.csv
synthesizability_gap_summary_by_layer_process.csv
```

#### `synthesizability_gap_per_combo.csv`

Despite the historical filename, this table stores the random-versus-topology-OOD performance gap for target/model/descriptor combinations. It is used to quantify topology-transfer optimism.

#### `synthesizability_gap_summary_by_layer_process.csv`

Process/layer-level summaries of the topology-transfer gap.

#### `atlas_long_lgbm_ood.csv`

Long-format LightGBM topology-OOD descriptor-adequacy atlas.

#### `atlas_pivot_lgbm_ood.csv`

Pivoted version of the atlas used for the descriptor-adequacy heatmap.

#### `descriptor_gain_map_lgbm_ood.csv`

Pairwise descriptor-gain results, including:

```text
Enriched − Geo
Geo+RAC − Enriched
```

This table supports the analysis of engineered-geometry effects and RAC chemistry premiums.

#### `recommendation_table_lgbm_ood.csv`

Practical-equivalence-with-cost descriptor recommendation table. For each process/layer group, the rule selects the cheapest descriptor family whose topology-OOD LightGBM performance is practically equivalent to the best family.

---

### `results/step3B_target_screening_analysis/`

This folder contains target-construction and screening-yield outputs derived from `predictions.csv.gz`.

```text
q1_direct_vs_posthoc_lnS_per_fold.csv
q1_direct_vs_posthoc_lnS_summary.csv
q2_working_capacity_components_per_fold.csv
q2_working_capacity_components_summary.csv
q3_topK_per_fold.csv
q3_topK_summary.csv
topK_all27_per_fold.csv
topK_all27_summary.csv
S14_topK_all27.tex
```

#### `q1_direct_vs_posthoc_lnS_*`

Direct Layer C log-selectivity regression versus post-hoc log-selectivity reconstructed from constituent Layer A uptake predictions.

#### `q2_working_capacity_components_*`

Working-capacity component audit. This compares Layer B working-capacity performance to the primitive adsorption-state and desorption-state uptake targets from which working capacity is constructed.

#### `q3_topK_*`

Representative top-K screening-yield diagnostics.

#### `topK_all27_*`

All-27-target top-K recall and champion-misidentification analysis.

#### `S14_topK_all27.tex`

LaTeX table fragment for the all-target top-K SI table.

---

## Figures

The current figure assets are stored in:

```text
figures/main/
```

Current files:

```text
fig1_topology_transfer_gap.pdf
fig2_descriptor_adequacy_atlas.pdf
fig3_descriptor_gain_map.pdf
fig4_structural_audit.pdf
fig4_target_construction.pdf
fig5_screening_yield.pdf
```

The filenames preserve the history of the analysis scripts. In the latest manuscript version, the structural audit was inserted into the main text, so the manuscript figure numbering differs from some historical script filenames.

### Manuscript-level figure mapping

```text
Manuscript Figure 1 -> figures/main/fig1_topology_transfer_gap.pdf
Manuscript Figure 2 -> figures/main/fig2_descriptor_adequacy_atlas.pdf
Manuscript Figure 3 -> figures/main/fig3_descriptor_gain_map.pdf
Manuscript Figure 4 -> figures/main/fig4_structural_audit.pdf
Manuscript Figure 5 -> figures/main/fig4_target_construction.pdf
Manuscript Figure 6 -> figures/main/fig5_screening_yield.pdf
```

The duplicate `fig4_...` filenames are intentionally retained to avoid breaking existing scripts and Overleaf references.

---

## Source code

### `src/step1_prepare_dataset/`

```text
step1_data_merge.py
```

Stage 1 constructs the strict common cohort.

Main responsibilities:

- normalize MOF identifiers;
- merge ARC-MOF adsorption targets;
- merge geometric descriptors;
- merge RAC descriptors;
- merge CrystalNets topology labels;
- enforce complete target and descriptor coverage;
- write manifests and target-coverage reports.

Expected raw ARC-MOF inputs include:

```text
geometric_properties.csv
RACs.csv
all_topology_lists.csv
methane.csv
post_comb_vsa-CO2.csv
post_comb_vsa-N2.csv
pre_comb_4040-CO2.csv
pre_comb_4040-H2.csv
methane_purification-CO2.csv
methane_purification-CH4.csv
landfill-CO2.csv
landfill-CH4.csv
overall_process.csv
```

These raw input files are not stored in this repository.

Typical Stage 1 outputs include:

```text
revision_data.csv
column_manifest.json
merge_report.json
raw_file_md5_manifest.csv
target_coverage_before_strict.csv
target_coverage_strict.csv
```

---

### `src/step2_train_models/`

```text
step2_train_pipeline.py
```

Stage 2 trains the full benchmark.

Main responsibilities:

- load the strict common cohort;
- construct descriptor-family matrices;
- apply fold-specific preprocessing;
- tune hyperparameters with Optuna;
- train Ridge, Random Forest, and LightGBM models;
- evaluate random and topology-grouped splits;
- write metrics, hyperparameters, trial records, feature importances, predictions, and split/global audit files.

Main outputs:

```text
raw_metrics.csv
tuned_params.csv
optuna_trials.csv
feature_importances_raw.csv
predictions.csv.gz
```

The large file `predictions.csv.gz` is required for Step 3B but is not stored in GitHub.

---

### `src/step3A_descriptor_analysis/`

```text
step3_stageA.py
step3_stageA1.py
```

Stage 3A uses only Step 2 metrics. It does not require `predictions.csv.gz`.

#### `step3_stageA.py`

Broader evaluation wrapper. It creates auxiliary summaries such as:

```text
completion_audit.csv
failed_model_runs.csv
random_vs_group_gap.csv
summary_primary_ood.csv
summary_by_target.csv
summary_by_process.csv
feature_importance_summary.csv
table_main_ood_results.csv
table_descriptor_recommendation.csv
```

#### `step3_stageA1.py`

Main manuscript-facing Stage 3A script. It generates:

```text
synthesizability_gap_per_combo.csv
synthesizability_gap_summary_by_layer_process.csv
atlas_long_lgbm_ood.csv
atlas_pivot_lgbm_ood.csv
descriptor_gain_map_lgbm_ood.csv
recommendation_table_lgbm_ood.csv
```

Run:

```bash
python src/step3A_descriptor_analysis/step3_stageA1.py
```

---

### `src/step3B_target_screening_analysis/`

```text
step3_stageB.py
session2_topK_all27.py
```

Stage 3B requires prediction-level data.

The required large file is:

```text
predictions.csv.gz
```

Provide it either at the repository root or via environment variable:

```powershell
$env:PREDICTIONS_CSV_GZ = "path\to\predictions.csv.gz"
```

#### `step3_stageB.py`

Computes:

```text
q1_direct_vs_posthoc_lnS_per_fold.csv
q1_direct_vs_posthoc_lnS_summary.csv
q2_working_capacity_components_per_fold.csv
q2_working_capacity_components_summary.csv
q3_topK_per_fold.csv
q3_topK_summary.csv
```

#### `session2_topK_all27.py`

Computes all-target top-K screening diagnostics:

```text
topK_all27_per_fold.csv
topK_all27_summary.csv
S14_topK_all27.tex
```

Run:

```bash
python src/step3B_target_screening_analysis/step3_stageB.py
python src/step3B_target_screening_analysis/session2_topK_all27.py
```

---

### `src/figure_generation/`

Figure-generation scripts:

```text
make_figures_base_v9.py
make_figures_patch_fig1_fig4_v12.py
```

#### `make_figures_base_v9.py`

Generates:

```text
fig2_descriptor_adequacy_atlas.pdf
fig3_descriptor_gain_map.pdf
fig5_screening_yield.pdf
```

Reads from:

```text
results/step3A_descriptor_analysis/
results/step3B_target_screening_analysis/
```

Writes to:

```text
figures/main/
```

#### `make_figures_patch_fig1_fig4_v12.py`

Generates patched versions of:

```text
fig1_topology_transfer_gap.pdf
fig4_target_construction.pdf
```

Reads from:

```text
results/step3A_descriptor_analysis/synthesizability_gap_per_combo.csv
results/step3A_descriptor_analysis/recommendation_table_lgbm_ood.csv
results/step3B_target_screening_analysis/q1_direct_vs_posthoc_lnS_per_fold.csv
results/step3B_target_screening_analysis/q2_working_capacity_components_per_fold.csv
```

Writes to:

```text
figures/main/
```

Run:

```bash
python src/figure_generation/make_figures_base_v9.py
python src/figure_generation/make_figures_patch_fig1_fig4_v12.py
```

---

## External inputs

### Required for Step 1

Raw ARC-MOF source tables. These are not redistributed here.

### Required for Step 3B

```text
predictions.csv.gz
```

This file is too large for GitHub. Supply it externally.

Example:

```powershell
$env:PREDICTIONS_CSV_GZ = "E:\AMI\Step 2\benchmark_v9_outputs\predictions\predictions.csv.gz"
```

---

## Quick start

### Install requirements

```bash
pip install -r requirements.txt
```

### Reproduce Stage 3A outputs

```bash
python src/step3A_descriptor_analysis/step3_stageA1.py
```

### Reproduce Stage 3B outputs

Requires `predictions.csv.gz`:

```powershell
$env:PREDICTIONS_CSV_GZ = "path\to\predictions.csv.gz"
python src/step3B_target_screening_analysis/step3_stageB.py
python src/step3B_target_screening_analysis/session2_topK_all27.py
```

### Reproduce figures

```bash
python src/figure_generation/make_figures_base_v9.py
python src/figure_generation/make_figures_patch_fig1_fig4_v12.py
```

---

## Validation record

The repository was tested on the benchmark machine with the full `predictions.csv.gz` file available.

The following scripts were rerun successfully:

```text
src/step3A_descriptor_analysis/step3_stageA.py
src/step3A_descriptor_analysis/step3_stageA1.py
src/step3B_target_screening_analysis/step3_stageB.py
src/step3B_target_screening_analysis/session2_topK_all27.py
src/figure_generation/make_figures_base_v9.py
src/figure_generation/make_figures_patch_fig1_fig4_v12.py
```

The regeneration audit reported:

```text
script_run_report.csv: all PASS
comparison_report.csv: 20 MATCH
comparison_failures.csv: empty
```

This validates the analysis chain:

```text
Step 2 metrics + predictions.csv.gz
    -> Step 3A / Step 3B analyses
    -> processed result CSVs
    -> final analysis figures
```

---

## Structural-audit note

The manuscript includes a structure-level audit figure. This figure is a post-hoc visualization of selected topology-OOD prediction cases. CIFs are used only for visualization after model evaluation.

The structural CIFs and renderings are not used for:

```text
model training
descriptor construction
hyperparameter optimization
descriptor-family recommendation
performance evaluation
```

The current structural-audit figure asset is:

```text
figures/main/fig4_structural_audit.pdf
```

If structural-audit panels are updated externally, the corresponding SI table and figure caption should be checked to ensure that MOF IDs, ranks, topology labels, and structural metadata match the final rendered figure.

---

## What is not included

This GitHub repository does not include:

```text
predictions.csv.gz
raw ARC-MOF source tables
model checkpoint files
large CIF archives
Overleaf manuscript source
reviewer reports or private correspondence
```

These should be handled through Zenodo, ARC-MOF's original data release, or private local storage as appropriate.

---

## Citation

If you use this repository, the processed outputs, or the figures in your work, please cite the preprint, the project data deposit, the underlying ARC-MOF dataset, and the CrystalNets topology-identification tool used for the topology-grouped splits.

```bibtex
@misc{alimardani2026mof_descriptor_adequacy_preprint,
  title        = {When Are Simple Descriptors Enough in MOF Adsorption Machine Learning?},
  author       = {Alimardani, Hosein and Abaei, Shayan and Asgari, Mehrdad},
  year         = {2026},
  month        = may,
  howpublished = {ChemRxiv},
  doi          = {10.26434/chemrxiv-2026-15002010-v3},
  note         = {Preprint, version 3}
}

@dataset{alimardani2026mof_descriptor_adequacy_data,
  title     = {Data and processed outputs for "When Are Simple Descriptors Enough in MOF Adsorption Machine Learning?"},
  author    = {Alimardani, Hosein and Abaei, Shayan and Asgari, Mehrdad},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15055758}
}

@dataset{burner2025arcmof_v7,
  title     = {ab initio REPEAT Charge MOF Database (ARC-MOF)},
  author    = {Burner, Jake and Luo, Jun and White, Andrew and Mirmiran, Adam and Kwon, Ohmin and Boyd, Peter G. and Maley, Stephen and Gibaldi, Marco and Simrod, Scott and Woo, Tom K.},
  year      = {2025},
  month     = aug,
  version   = {v7},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.16802743},
  url       = {https://zenodo.org/records/16802743}
}

@article{zoubritzky2022crystalnets,
  title   = {CrystalNets.jl: Identification of Crystal Topologies},
  author  = {Zoubritzky, Lionel and Coudert, Fran\c{c}ois-Xavier},
  journal = {SciPost Chemistry},
  year    = {2022},
  doi     = {10.21468/SciPostChem.1.2.005}
}
```

### How to cite this repository

Alimardani, H., Abaei, S., Asgari, M. *When Are Simple Descriptors Enough in MOF Adsorption Machine Learning?* ChemRxiv preprint, 2026. doi:10.26434/chemrxiv-2026-15002010-v3

Code: https://github.com/<your-org>/<repo-name>

---

## License

This repository is released under the MIT License unless otherwise specified.
