# Figure 4 structural-audit CIF files

This folder contains the four ARC-MOF CIF files used to render the final Figure 4 structural-audit panels in the manuscript:

**When Are Simple Descriptors Enough in MOF Adsorption Machine Learning?**

These files are included only to make the post-hoc visualization reproducible. They were **not** used for model training, descriptor construction, hyperparameter optimization, descriptor-family recommendation, or performance evaluation.

## Files

| Panel | Case | CIF file | Target | Rank summary |
|---|---|---|---|---|
| A | Geometry-saturated success | `Panel_A_rank_10_geometry_saturated_DB7-ddmof_20014_repeat.cif` | `CH4_65bar` | True #3; Geo #34; Geo+RAC #9 |
| B | Chemistry-premium recovery | `Panel_B_rank_05_chemistry_premium_DB10-dia_sym_4_mc_si__L_5_repeat.cif` | `precomb_CO2_16bar` | True #3; Geo #37758; Geo+RAC #10 |
| C | Descriptor-limited miss | `Panel_C_rank_06_descriptor_limited_DB5-hypotheticalMOF_5040363_1_1_1_15_16_5_repeat.cif` | `postcomb_CO2_0p015bar` | True #22; Geo #1606; Geo+RAC #1169 |
| D | Screening-instability false champion | `Panel_D_rank_01_screening_instability_DB12-PAQJUM_freeONLY_repeat.cif` | `postcomb_log_selectivity` | True #15956; Geo #1411; Geo+RAC #1 |

The accompanying file `structural_audit_final_cases_metadata.csv` provides the panel labels, MOF IDs, CIF filenames, target IDs, fold-level ranks, model values, Crystalnet labels, and structural metadata used in the main text and Supporting Information.

## Scope note

The rendered structures are diagnostic audit examples selected from topology-OOD prediction outputs. They should not be interpreted as experimentally validated sorbents, adsorption-site assignments, mechanistic explanations, or synthetic/stability claims.

## Relation to manuscript files

- Main-text figure: `figures/main/fig4_structural_audit.pdf`
- Source data and CIFs: this folder
- SI section: structural-audit case selection / Table S10
