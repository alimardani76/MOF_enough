#!/usr/bin/env python3
"""
step3_stageA1.py
================

Stage A1 of Step 3 for V9 manuscript.

Reads ONLY raw_metrics.csv and produces the data tables for:
  §3.1 Synthesizability gap
  §3.2 Descriptor adequacy atlas (topology-OOD)
  §3.3 Descriptor gain tables
  §3.6 Practical-equivalence recommendation table

No predictions. No feature importances. No GB-scale I/O.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "results").exists() and (parent / "src").exists():
            return parent
    return cur

ROOT = find_repo_root(Path(__file__))

INPUT_CSV = ROOT / "results" / "step2_training" / "raw_metrics.csv"
OUT = ROOT / "results" / "step3A_descriptor_analysis"

PRIMARY_MODEL = "lgbm"

# practical-equivalence rule from Methods
PRACTICAL_EQUIVALENCE_R2 = 0.02

# cost-tier ordering used in §3.6 recommendation rule
COST_TIERS = {
    "geometry_only": 1,
    "enriched_interpretable": 2,
    "geometry_plus_topology": 3,
    "rac_only": 4,
    "geo_plus_rac": 5,
    "topology_only": 99,  # diagnostic, excluded from recommendations
}

# diagnostic family excluded from recommendations
DIAGNOSTIC_FAMILIES = {"topology_only"}

# Layer label canonical order
LAYER_ORDER = ["A_raw_uptake", "B_working_capacity", "C_direct_log_selectivity"]

# Display ordering for processes
PROCESS_ORDER = [
    "methane_storage_psa",
    "post_combustion_vsa",
    "pre_combustion_psa",
    "natural_gas_purification",
    "landfill_gas_vpsa",
]

# Family display ordering
FAMILY_ORDER = [
    "geometry_only",
    "enriched_interpretable",
    "geometry_plus_topology",
    "rac_only",
    "geo_plus_rac",
    "topology_only",
]

# Required raw_metrics columns
REQUIRED_COLS = [
    "target",
    "layer",
    "process",
    "split_type",
    "split_name",
    "descriptor_family",
    "model",
    "r2",
    "mae",
    "rmse",
    "spearman",
    "pearson",
]

# Split-type canonical values used in raw_metrics.csv
SPLIT_RANDOM = "random"
SPLIT_OOD = "topology_group"  # adjust if your raw_metrics uses a different label


# =============================================================================
# UTILITIES
# =============================================================================

def pr(msg: str) -> None:
    print(f"[stage-A1] {msg}", flush=True)


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def detect_split_label(df: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the actual label used in raw_metrics.csv for topology-OOD splits."""
    present = set(df["split_type"].dropna().unique())
    for cand in candidates:
        if cand in present:
            return cand
    raise RuntimeError(
        f"None of the candidate split labels {candidates} found in raw_metrics.csv. "
        f"Present labels: {sorted(present)}"
    )


def aggregate_folds(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean, std, sem, n."""

    agg = {}
    for m in metric_cols:
        if m in df.columns:
            agg[m] = ["mean", "std", "count"]

    grouped = df.groupby(group_cols, dropna=False).agg(agg)
    grouped.columns = [f"{m}_{stat}" for m, stat in grouped.columns]
    grouped = grouped.reset_index()

    for m in metric_cols:
        if m in df.columns and f"{m}_count" in grouped.columns:
            grouped[f"{m}_sem"] = grouped.apply(
                lambda r: (r[f"{m}_std"] / math.sqrt(r[f"{m}_count"]))
                if r[f"{m}_count"] and r[f"{m}_count"] > 1 and pd.notna(r[f"{m}_std"])
                else np.nan,
                axis=1,
            )

    return grouped


# =============================================================================
# LOAD
# =============================================================================

def load_raw_metrics() -> tuple[pd.DataFrame, str]:
    if not INPUT_CSV.exists():
        sys.exit(f"Cannot find {INPUT_CSV}")

    pr(f"Loading {INPUT_CSV.name}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"raw_metrics.csv missing columns: {missing}")

    df = df[df["status"].isna() | (df["status"] == "ok")].copy()

    # detect the topology-OOD label actually used in the file
    ood_label = detect_split_label(
        df,
        candidates=[
            "topology_group",
            "balanced_topology_group",
            "topology_ood",
            "group_topology",
        ],
    )

    if ood_label != SPLIT_OOD:
        pr(f"Detected OOD split label: {ood_label!r} (overriding SPLIT_OOD)")

    pr(f"raw_metrics rows: {len(df):,}")
    pr(f"unique targets: {df['target'].nunique()}")
    pr(f"unique families: {df['descriptor_family'].nunique()}")
    pr(f"unique models: {df['model'].nunique()}")
    pr(f"unique splits: {df['split_name'].nunique()}")
    pr(f"unique split types: {sorted(df['split_type'].unique())}")

    return df, ood_label


# =============================================================================
# §3.1 SYNTHESIZABILITY GAP
# =============================================================================

def build_synthesizability_gap(df: pd.DataFrame, ood_label: str) -> pd.DataFrame:
    """
    Random R² vs topology-OOD R² for every (target × family × model),
    plus the absolute gap.
    """

    keys = ["target", "layer", "process", "descriptor_family", "model"]

    rand = aggregate_folds(
        df[df["split_type"] == SPLIT_RANDOM],
        group_cols=keys,
        metric_cols=["r2", "spearman"],
    ).rename(
        columns={
            "r2_mean": "r2_random_mean",
            "r2_std": "r2_random_std",
            "r2_sem": "r2_random_sem",
            "r2_count": "r2_random_n",
            "spearman_mean": "spearman_random_mean",
            "spearman_std": "spearman_random_std",
            "spearman_sem": "spearman_random_sem",
            "spearman_count": "spearman_random_n",
        }
    )

    ood = aggregate_folds(
        df[df["split_type"] == ood_label],
        group_cols=keys,
        metric_cols=["r2", "spearman"],
    ).rename(
        columns={
            "r2_mean": "r2_ood_mean",
            "r2_std": "r2_ood_std",
            "r2_sem": "r2_ood_sem",
            "r2_count": "r2_ood_n",
            "spearman_mean": "spearman_ood_mean",
            "spearman_std": "spearman_ood_std",
            "spearman_sem": "spearman_ood_sem",
            "spearman_count": "spearman_ood_n",
        }
    )

    out = pd.merge(rand, ood, on=keys, how="inner")
    out["r2_gap"] = out["r2_random_mean"] - out["r2_ood_mean"]

    out = out.sort_values(
        ["layer", "process", "descriptor_family", "model", "target"]
    ).reset_index(drop=True)

    return out


def aggregate_gap_by_layer_process(per_row_gap: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table for §3.1: per (layer × process × model × family).
    """
    keys = ["layer", "process", "model", "descriptor_family"]
    agg = per_row_gap.groupby(keys, dropna=False).agg(
        r2_random_mean=("r2_random_mean", "mean"),
        r2_random_std=("r2_random_mean", "std"),
        r2_ood_mean=("r2_ood_mean", "mean"),
        r2_ood_std=("r2_ood_mean", "std"),
        r2_gap_mean=("r2_gap", "mean"),
        r2_gap_std=("r2_gap", "std"),
        n_targets=("target", "nunique"),
    ).reset_index()
    return agg


# =============================================================================
# §3.2 DESCRIPTOR ADEQUACY ATLAS (topology-OOD, LightGBM)
# =============================================================================

def build_descriptor_adequacy_atlas(
    df: pd.DataFrame,
    ood_label: str,
) -> pd.DataFrame:
    """
    Topology-OOD R² heatmap data for primary model (LightGBM).
    Rows: target × layer × process
    Cols: descriptor_family
    """
    sub = df[
        (df["split_type"] == ood_label)
        & (df["model"] == PRIMARY_MODEL)
    ].copy()

    agg = aggregate_folds(
        sub,
        group_cols=["target", "layer", "process", "descriptor_family"],
        metric_cols=["r2", "spearman", "mae", "rmse"],
    )

    long_keep = [
        "target",
        "layer",
        "process",
        "descriptor_family",
        "r2_mean",
        "r2_std",
        "r2_sem",
        "r2_count",
        "spearman_mean",
        "spearman_std",
        "mae_mean",
        "mae_std",
        "rmse_mean",
        "rmse_std",
    ]
    agg = agg[[c for c in long_keep if c in agg.columns]]

    pivot = agg.pivot_table(
        index=["layer", "process", "target"],
        columns="descriptor_family",
        values="r2_mean",
    ).reset_index()

    # enforce family ordering
    family_cols = [c for c in FAMILY_ORDER if c in pivot.columns]
    pivot = pivot[["layer", "process", "target"] + family_cols]

    # enforce layer ordering then process ordering
    pivot["layer_order"] = pivot["layer"].apply(
        lambda x: LAYER_ORDER.index(x) if x in LAYER_ORDER else 99
    )
    pivot["process_order"] = pivot["process"].apply(
        lambda x: PROCESS_ORDER.index(x) if x in PROCESS_ORDER else 99
    )
    pivot = pivot.sort_values(
        ["layer_order", "process_order", "target"]
    ).drop(columns=["layer_order", "process_order"])

    return agg, pivot


# =============================================================================
# §3.3 DESCRIPTOR GAIN MAP
# =============================================================================

GAIN_COMPARISONS = [
    ("enriched_interpretable", "geometry_only"),
    ("geometry_plus_topology", "enriched_interpretable"),
    ("rac_only", "geometry_only"),
    ("geo_plus_rac", "enriched_interpretable"),
]


def build_descriptor_gain_map(atlas_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    For each (target, comparison_pair), compute Δ R² under topology-OOD.
    """
    records = []
    for plus, base in GAIN_COMPARISONS:
        if plus not in atlas_pivot.columns or base not in atlas_pivot.columns:
            continue
        for _, row in atlas_pivot.iterrows():
            delta = row[plus] - row[base]
            records.append({
                "layer": row["layer"],
                "process": row["process"],
                "target": row["target"],
                "comparison": f"{plus} - {base}",
                "plus_family": plus,
                "base_family": base,
                "r2_plus": row[plus],
                "r2_base": row[base],
                "delta_r2": delta,
            })

    out = pd.DataFrame(records)
    return out


# =============================================================================
# §3.6 PRACTICAL-EQUIVALENCE RECOMMENDATION TABLE
# =============================================================================

def recommend_family(group_metrics: pd.DataFrame) -> dict:
    """
    For a single (process × layer) group under topology-OOD and primary model:
    return recommended family using max(0.02, SE_best).
    """
    g = group_metrics[
        ~group_metrics["descriptor_family"].isin(DIAGNOSTIC_FAMILIES)
    ].copy()

    if g.empty:
        return {}

    g["cost_tier"] = g["descriptor_family"].map(COST_TIERS).fillna(99).astype(int)

    best_idx = g["r2_mean"].idxmax()
    best_r2 = g.loc[best_idx, "r2_mean"]
    best_std = g.loc[best_idx, "r2_std"]
    best_n = g.loc[best_idx, "r2_count"]
    best_family = g.loc[best_idx, "descriptor_family"]

    best_se = float("nan")
    if best_n and pd.notna(best_std) and best_n > 1:
        best_se = best_std / math.sqrt(best_n)

    margin = max(
        PRACTICAL_EQUIVALENCE_R2,
        best_se if (pd.notna(best_se) and best_se > 0) else 0.0,
    )

    equivalent = g[g["r2_mean"] >= best_r2 - margin].copy()
    equivalent = equivalent.sort_values(
        ["cost_tier", "r2_mean"], ascending=[True, False]
    )
    chosen = equivalent.iloc[0]

    return {
        "best_family": best_family,
        "best_r2_mean": float(best_r2),
        "best_r2_se": float(best_se) if pd.notna(best_se) else None,
        "equivalence_margin_used": float(margin),
        "recommended_family": chosen["descriptor_family"],
        "recommended_r2_mean": float(chosen["r2_mean"]),
        "recommended_r2_se": (
            float(chosen["r2_std"] / math.sqrt(chosen["r2_count"]))
            if pd.notna(chosen["r2_std"]) and chosen["r2_count"] > 1
            else None
        ),
        "equivalent_set": "|".join(equivalent["descriptor_family"].tolist()),
    }


def build_recommendation_table(atlas_long: pd.DataFrame) -> pd.DataFrame:
    """
    Per (process × layer) recommendation under topology-OOD using primary model.
    atlas_long already filtered to LGBM and OOD via build_descriptor_adequacy_atlas.
    """
    rows = []
    for (process, layer), g in atlas_long.groupby(["process", "layer"], dropna=False):
        # we want one recommendation per (process × layer), so aggregate across targets
        # by averaging per family across targets within this group
        family_agg = g.groupby("descriptor_family", as_index=False).agg(
            r2_mean=("r2_mean", "mean"),
            r2_std=("r2_mean", "std"),
            r2_count=("target", "nunique"),
        )
        rec = recommend_family(family_agg)
        rec.update({"process": process, "layer": layer, "n_targets": int(g["target"].nunique())})
        rows.append(rec)

    out = pd.DataFrame(rows)
    out["layer_order"] = out["layer"].apply(
        lambda x: LAYER_ORDER.index(x) if x in LAYER_ORDER else 99
    )
    out["process_order"] = out["process"].apply(
        lambda x: PROCESS_ORDER.index(x) if x in PROCESS_ORDER else 99
    )
    out = out.sort_values(["layer_order", "process_order"]).drop(
        columns=["layer_order", "process_order"]
    )
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    pr("=" * 70)
    pr("STAGE A1 — Results §3.1, §3.2, §3.3, §3.6 from raw_metrics.csv")
    pr("=" * 70)

    mkdir(OUT)

    df, ood_label = load_raw_metrics()

    # ---- §3.1 ----
    pr("Building §3.1 synthesizability gap")
    gap_rows = build_synthesizability_gap(df, ood_label)
    gap_rows.to_csv(OUT / "synthesizability_gap_per_combo.csv", index=False)
    pr(f"  wrote {OUT/'synthesizability_gap_per_combo.csv'}")

    gap_summary = aggregate_gap_by_layer_process(gap_rows)
    gap_summary.to_csv(OUT / "synthesizability_gap_summary_by_layer_process.csv", index=False)
    pr(f"  wrote {OUT/'synthesizability_gap_summary_by_layer_process.csv'}")

    # ---- §3.2 ----
    pr("Building §3.2 descriptor adequacy atlas (topology-OOD, LightGBM)")
    atlas_long, atlas_pivot = build_descriptor_adequacy_atlas(df, ood_label)
    atlas_long.to_csv(OUT / "atlas_long_lgbm_ood.csv", index=False)
    atlas_pivot.to_csv(OUT / "atlas_pivot_lgbm_ood.csv", index=False)
    pr(f"  wrote {OUT/'atlas_long_lgbm_ood.csv'}")
    pr(f"  wrote {OUT/'atlas_pivot_lgbm_ood.csv'}")

    # ---- §3.3 ----
    pr("Building §3.3 descriptor gain map")
    gain = build_descriptor_gain_map(atlas_pivot)
    gain.to_csv(OUT / "descriptor_gain_map_lgbm_ood.csv", index=False)
    pr(f"  wrote {OUT/'descriptor_gain_map_lgbm_ood.csv'}")

    # ---- §3.6 ----
    pr("Building §3.6 practical-equivalence recommendation table")
    rec = build_recommendation_table(atlas_long)
    rec.to_csv(OUT / "recommendation_table_lgbm_ood.csv", index=False)
    pr(f"  wrote {OUT/'recommendation_table_lgbm_ood.csv'}")

    # ---- README ----
    readme = OUT / "README.txt"
    readme.write_text(
        "Stage A1 outputs for V9 Results §3.1, §3.2, §3.3, §3.6.\n\n"
        "Files:\n"
        " - synthesizability_gap_per_combo.csv\n"
        "     Per (target × family × model) random vs topology-OOD R² and gap.\n"
        " - synthesizability_gap_summary_by_layer_process.csv\n"
        "     Aggregated gap per (layer × process × family × model).\n"
        " - atlas_long_lgbm_ood.csv\n"
        "     Per-target topology-OOD metrics under LightGBM.\n"
        " - atlas_pivot_lgbm_ood.csv\n"
        "     Long → wide pivot used for §3.2 heatmap.\n"
        " - descriptor_gain_map_lgbm_ood.csv\n"
        "     Δ R² per descriptor comparison under topology-OOD.\n"
        " - recommendation_table_lgbm_ood.csv\n"
        "     Practical-equivalence cost-aware recommendations per process × layer.\n",
        encoding="utf-8",
    )
    pr(f"  wrote {readme}")

    pr("=" * 70)
    pr("STAGE A1 COMPLETE")
    pr(f"All outputs in: {OUT}")
    pr("=" * 70)


if __name__ == "__main__":
    main()