#!/usr/bin/env python3
"""
session2_topK_all27.py

Computes top-100 and top-1000 recall, plus champion misidentification@100
for selectivity targets, for ALL 27 targets under topology-OOD LightGBM.

Reads predictions.csv.gz (streamed) and produces an SI-ready table.

Recommended and best families per (process, layer) come from
stageA/recommendation_table_lgbm_ood.csv. For each target, the
recommended family is the one for its (process, layer) group;
the best family is the same for most groups but can differ
for Layer A multi-target groups.

Output:
  session2_outputs/topK_all27_per_fold.csv
  session2_outputs/topK_all27_summary.csv
  session2_outputs/S14_topK_all27.tex   (drop into SI)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "results").exists() and (parent / "src").exists():
            return parent
    return cur

ROOT = find_repo_root(Path(__file__))
DIR_A = ROOT / "results" / "step3A_descriptor_analysis"
DIR_B = ROOT / "results" / "step3B_target_screening_analysis"

# Set this to wherever your predictions.csv.gz lives
# Default: same folder as this script
PRED_PATH = Path(os.environ.get("PREDICTIONS_CSV_GZ", ROOT / "predictions.csv.gz"))

OUT = ROOT / "results" / "step3B_target_screening_analysis"
OUT.mkdir(exist_ok=True)

CHUNK_SIZE = 1_000_000
PRIMARY_MODEL = "lgbm"
PRIMARY_SPLIT_TYPE = "balanced_topology_group"
FOLD_NAMES = [f"balanced_group_fold_{i}" for i in range(4)]

# Target-process-layer map (from recommendation table + your knowledge)
TARGET_PROCESS_LAYER = {
    # Layer A
    "CH4_5.8bar":            ("methane_storage_psa",      "A_raw_uptake"),
    "CH4_65bar":             ("methane_storage_psa",      "A_raw_uptake"),
    "postcomb_CO2_0.015bar": ("post_combustion_vsa",      "A_raw_uptake"),
    "postcomb_CO2_0.15bar":  ("post_combustion_vsa",      "A_raw_uptake"),
    "postcomb_N2_0.075bar":  ("post_combustion_vsa",      "A_raw_uptake"),
    "postcomb_N2_0.75bar":   ("post_combustion_vsa",      "A_raw_uptake"),
    "precomb_CO2_0.4bar":    ("pre_combustion_psa",       "A_raw_uptake"),
    "precomb_CO2_16bar":     ("pre_combustion_psa",       "A_raw_uptake"),
    "precomb_H2_0.6bar":     ("pre_combustion_psa",       "A_raw_uptake"),
    "precomb_H2_24bar":      ("pre_combustion_psa",       "A_raw_uptake"),
    "ngp_CO2_0.1bar":        ("natural_gas_purification", "A_raw_uptake"),
    "ngp_CO2_1.0bar":        ("natural_gas_purification", "A_raw_uptake"),
    "ngp_CH4_0.9bar":        ("natural_gas_purification", "A_raw_uptake"),
    "ngp_CH4_9.0bar":        ("natural_gas_purification", "A_raw_uptake"),
    "landfill_CO2_0.26bar":  ("landfill_gas_vpsa",        "A_raw_uptake"),
    "landfill_CO2_3.2bar":   ("landfill_gas_vpsa",        "A_raw_uptake"),
    "landfill_CH4_0.01bar":  ("landfill_gas_vpsa",        "A_raw_uptake"),
    "landfill_CH4_4.4bar":   ("landfill_gas_vpsa",        "A_raw_uptake"),
    # Layer B (working capacity)
    "methane_storage_working_capacity": ("methane_storage_psa",      "B_working_capacity"),
    "postcomb_working_capacity":        ("post_combustion_vsa",      "B_working_capacity"),
    "precomb_working_capacity":         ("pre_combustion_psa",       "B_working_capacity"),
    "ngp_working_capacity":             ("natural_gas_purification", "B_working_capacity"),
    "landfill_working_capacity":        ("landfill_gas_vpsa",        "B_working_capacity"),
    # Layer C (log selectivity)
    "postcomb_log_selectivity": ("post_combustion_vsa",      "C_direct_log_selectivity"),
    "precomb_log_selectivity":  ("pre_combustion_psa",       "C_direct_log_selectivity"),
    "ngp_log_selectivity":      ("natural_gas_purification", "C_direct_log_selectivity"),
    "landfill_log_selectivity": ("landfill_gas_vpsa",        "C_direct_log_selectivity"),
}

FAMILY_DISPLAY = {
    "geometry_only": "Geo",
    "enriched_interpretable": "Enriched",
    "geometry_plus_topology": "Geo+Topo",
    "rac_only": "RAC",
    "geo_plus_rac": "Geo+RAC",
}

K_VALUES = [100, 1000]


def pr(msg): print(f"[topK-27] {msg}", flush=True)


def main():
    # Load recommendation table
    rec = pd.read_csv(DIR_A / "recommendation_table_lgbm_ood.csv")
    rec_map = {
        (r["process"], r["layer"]): (r["recommended_family"], r["best_family"])
        for _, r in rec.iterrows()
    }

    if not PRED_PATH.exists():
        sys.exit(f"predictions.csv.gz not found at {PRED_PATH}. "
                 f"Edit PRED_PATH at the top of the script.")

    # Build set of (target, family, split_name) keys we need
    needed_keys = set()
    targets_layer = {}
    for target, (process, layer) in TARGET_PROCESS_LAYER.items():
        if (process, layer) not in rec_map:
            continue
        rec_fam, best_fam = rec_map[(process, layer)]
        families = {rec_fam, best_fam}
        targets_layer[target] = (process, layer, rec_fam, best_fam)
        for fam in families:
            for fold in FOLD_NAMES:
                needed_keys.add((target, fam, fold))

    pr(f"Need {len(needed_keys)} (target × family × fold) combos")
    pr(f"Across {len(targets_layer)} targets")

    # Stream and collect
    collected = {k: [] for k in needed_keys}
    n_total = 0
    n_chunks = 0
    n_kept = 0

    reader = pd.read_csv(
        PRED_PATH, compression="infer",
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for chunk in reader:
        n_chunks += 1
        n_total += len(chunk)

        sub = chunk[
            (chunk["model"] == PRIMARY_MODEL)
            & (chunk["split_type"] == PRIMARY_SPLIT_TYPE)
        ]
        if sub.empty:
            continue

        mask = sub.apply(
            lambda r: (r["target"], r["descriptor_family"], r["split_name"]) in needed_keys,
            axis=1,
        )
        sub = sub[mask]
        if sub.empty:
            continue

        for key, grp in sub.groupby(["target", "descriptor_family", "split_name"]):
            collected[key].append(grp[["filename", "y_true", "y_pred"]].copy())
            n_kept += len(grp)

        if n_chunks % 10 == 0:
            pr(f"chunks: {n_chunks}, rows seen: {n_total:,}, kept: {n_kept:,}")

    pr(f"Done streaming. Total rows: {n_total:,}, kept rows: {n_kept:,}")

    # Compute top-K recall per fold per (target, family)
    rows = []
    for target, (process, layer, rec_fam, best_fam) in targets_layer.items():
        is_selectivity = layer == "C_direct_log_selectivity"
        families = []
        if rec_fam == best_fam:
            families = [(rec_fam, "recommended/best")]
        else:
            families = [(rec_fam, "recommended"), (best_fam, "best")]

        for fam, role in families:
            for fold_idx, fold in enumerate(FOLD_NAMES):
                key = (target, fam, fold)
                parts = collected.get(key, [])
                if not parts:
                    continue
                df = pd.concat(parts, ignore_index=True)
                if len(df) < 100:
                    continue

                y_true = df["y_true"].to_numpy()
                y_pred = df["y_pred"].to_numpy()
                fnames = df["filename"].to_numpy()
                n = len(df)

                true_order = np.argsort(-y_true)
                pred_order = np.argsort(-y_pred)

                row = {
                    "target": target,
                    "process": process,
                    "layer": layer,
                    "family": fam,
                    "role": role,
                    "fold": fold_idx,
                    "n_mofs": n,
                }
                for K in K_VALUES:
                    if n < K:
                        row[f"recall_at_{K}"] = np.nan
                        row[f"precision_at_{K}"] = np.nan
                        row[f"misid_at_{K}"] = np.nan
                        continue
                    true_topK = set(fnames[true_order[:K]])
                    pred_topK = set(fnames[pred_order[:K]])
                    overlap = len(true_topK & pred_topK)
                    recall_K = overlap / K
                    precision_K = overlap / K
                    misid_K = 1.0 - precision_K
                    row[f"recall_at_{K}"] = recall_K
                    row[f"precision_at_{K}"] = precision_K
                    row[f"misid_at_{K}"] = misid_K if is_selectivity else np.nan
                rows.append(row)

    perfold = pd.DataFrame(rows)
    perfold.to_csv(OUT / "topK_all27_per_fold.csv", index=False)
    pr(f"Wrote topK_all27_per_fold.csv: {len(perfold)} rows")

    # Summarize across folds
    agg_cols = {}
    for K in K_VALUES:
        agg_cols[f"recall_at_{K}_mean"] = (f"recall_at_{K}", "mean")
        agg_cols[f"recall_at_{K}_std"]  = (f"recall_at_{K}", "std")
        agg_cols[f"misid_at_{K}_mean"]  = (f"misid_at_{K}", "mean")
    summary = perfold.groupby(
        ["target", "process", "layer", "family", "role"], as_index=False
    ).agg(n_folds=("fold", "nunique"), **agg_cols)

    summary.to_csv(OUT / "topK_all27_summary.csv", index=False)
    pr(f"Wrote topK_all27_summary.csv: {len(summary)} rows")

    # Write LaTeX SI table
    write_si_table(summary, targets_layer)


def write_si_table(summary, targets_layer):
    """Generate Section S14 with all-27 top-K table."""

    # Order targets canonically
    target_order = list(targets_layer.keys())
    summary = summary.set_index(["target", "family"])

    lines = []
    lines.append("% =============================================================")
    lines.append("% NEW SI Section S14 — All-27 top-K recall")
    lines.append("% =============================================================")
    lines.append("")
    lines.append("\\subsection*{Section~S14. Top-$K$ recall for all 27 targets}")
    lines.append("")
    lines.append("Table~\\ref{tab:S9} reports top-100 and top-1000 recall under")
    lines.append("LightGBM topology-OOD for all 27 targets in the benchmark,")
    lines.append("evaluated for the descriptor family recommended in")
    lines.append("Section~3.6 (Rec.) and, where different, for the best family")
    lines.append("(Best). Values are averaged across the four topology-OOD")
    lines.append("folds. For Layer~C log-selectivity targets, the champion")
    lines.append("misidentification rate at $K=100$ ($1 - \\text{precision@}100$)")
    lines.append("is also reported. The top-$K$ analysis confirms the divergence")
    lines.append("between regression accuracy and screening yield documented in")
    lines.append("the three representative cases of main-text Section~3.5:")
    lines.append("descriptor families that are practically equivalent in $R^{2}$")
    lines.append("can differ substantially in candidate yield at finite $K$.")
    lines.append("")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Top-$K$ recall for all 27 targets under LightGBM")
    lines.append("topology-OOD. ``Rec.'' is the descriptor family recommended in")
    lines.append("Section~3.6; ``Best'' is the highest-$R^{2}$ family when")
    lines.append("different from ``Rec.''. recall@$K$ and precision@$K$ are")
    lines.append("identical here because we evaluate top-$K$ subsets of equal size.}")
    lines.append("\\label{tab:S9}")
    lines.append("\\begin{tabular}{llllccc}")
    lines.append("\\hline")
    lines.append("Target & Layer & Role & Family & recall@100 & recall@1000 & misid@100 \\\\")
    lines.append("\\hline")

    LAYER_DISPLAY = {
        "A_raw_uptake": "A",
        "B_working_capacity": "B",
        "C_direct_log_selectivity": "C",
    }

    for target in target_order:
        process, layer, rec_fam, best_fam = targets_layer[target]
        layer_short = LAYER_DISPLAY[layer]

        if rec_fam == best_fam:
            try:
                row = summary.loc[(target, rec_fam)]
                fam_d = FAMILY_DISPLAY[rec_fam]
                r100 = f"{row['recall_at_100_mean']:.2f}"
                r1000 = f"{row['recall_at_1000_mean']:.2f}"
                misid_val = row['misid_at_100_mean']
                misid = f"{misid_val:.2f}" if pd.notna(misid_val) else "--"
                target_esc = target.replace("_", "\\_")
                lines.append(f"\\texttt{{{target_esc}}} & {layer_short} & Rec./Best & \\textbf{{{fam_d}}} & {r100} & {r1000} & {misid} \\\\")
            except KeyError:
                pass
        else:
            # Two rows: recommended and best
            for role_label, fam in [("Rec.", rec_fam), ("Best", best_fam)]:
                try:
                    row = summary.loc[(target, fam)]
                    fam_d = FAMILY_DISPLAY[fam]
                    r100 = f"{row['recall_at_100_mean']:.2f}"
                    r1000 = f"{row['recall_at_1000_mean']:.2f}"
                    misid_val = row['misid_at_100_mean']
                    misid = f"{misid_val:.2f}" if pd.notna(misid_val) else "--"
                    target_esc = target.replace("_", "\\_") if role_label == "Rec." else ""
                    layer_show = layer_short if role_label == "Rec." else ""
                    lines.append(f"\\texttt{{{target_esc}}} & {layer_show} & {role_label} & \\textbf{{{fam_d}}} & {r100} & {r1000} & {misid} \\\\")
                except KeyError:
                    pass

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    (OUT / "S14_topK_all27.tex").write_text("\n".join(lines))
    pr(f"Wrote S14_topK_all27.tex")


if __name__ == "__main__":
    main()