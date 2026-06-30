#!/usr/bin/env python3
"""
step3_stageA.py
===============

Step 3 Stage A — light analysis from Step 2 metrics, splits, and manifest.

Does NOT read predictions.csv.gz.
Streams feature_importances_raw.csv to be safe on large files.

Inputs (auto-detected relative to this script):
    benchmark_v9_outputs/
        _global/
            run_config.json
            dataset_audit.json
            split_audit.csv
            split_audit_warnings.csv
            step2_final_report.json
        metrics/
            raw_metrics.csv
            optuna_trials.csv
            tuned_params.csv
            feature_importances_raw.csv
        splits/
            random_seed_*.json
            balanced_group_fold_*.json
    column_manifest.json

Outputs (created next to this script):
    step3_evaluation_outputs/
        evaluation/
        tables/
        figures/
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

BENCH_DIR = ROOT / "benchmark_v9_outputs"
GLOBAL_DIR = BENCH_DIR / "_global"
METRIC_DIR = BENCH_DIR / "metrics"
SPLIT_DIR = BENCH_DIR / "splits"
MANIFEST_PATH = ROOT / "data_manifest" / "column_manifest.json"

RAW_METRICS_PATH = METRIC_DIR / "raw_metrics.csv"
OPTUNA_TRIALS_PATH = METRIC_DIR / "optuna_trials.csv"
TUNED_PARAMS_PATH = METRIC_DIR / "tuned_params.csv"
FEATURE_IMPORTANCE_PATH = METRIC_DIR / "feature_importances_raw.csv"

OUT_ROOT = ROOT / "step3_evaluation_outputs"
EVAL_DIR = OUT_ROOT / "evaluation"
TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"

# Locked V9 Stage A defaults
PRIMARY_OOD_SPLIT_TYPE = "balanced_topology_group"
PRACTICAL_EQUIVALENCE_R2 = 0.02

COST_TIERS = {
    "geometry_only": 1,
    "enriched_interpretable": 2,
    "geometry_plus_topology": 3,
    "rac_only": 4,
    "geo_plus_rac": 5,
}
DIAGNOSTIC_ONLY_FAMILIES = ["topology_only"]
TOP_N_FEATURES = 30

LAYER_ORDER = [
    "A_raw_uptake",
    "B_working_capacity",
    "C_direct_log_selectivity",
]

PROCESS_ORDER = [
    "methane_storage_psa",
    "post_combustion_vsa",
    "pre_combustion_psa",
    "natural_gas_purification",
    "landfill_gas_vpsa",
]

EXPECTED_MODELS = ["ridge", "rf", "lgbm"]
EXPECTED_SPLIT_TYPES = ["random", "balanced_topology_group"]

FAMILY_DISPLAY = {
    "geometry_only": "GO",
    "enriched_interpretable": "EI",
    "topology_only": "TO",
    "geometry_plus_topology": "GPT",
    "rac_only": "RO",
    "geo_plus_rac": "GPR",
}

MODEL_DISPLAY = {
    "ridge": "Ridge",
    "rf": "RF",
    "lgbm": "LGBM",
}

PROCESS_DISPLAY = {
    "methane_storage_psa": "CH4 storage",
    "post_combustion_vsa": "Post-combustion",
    "pre_combustion_psa": "Pre-combustion",
    "natural_gas_purification": "NGP",
    "landfill_gas_vpsa": "Landfill",
}

LAYER_DISPLAY = {
    "A_raw_uptake": "A: uptake",
    "B_working_capacity": "B: working capacity",
    "C_direct_log_selectivity": "C: log selectivity",
}

warnings.filterwarnings("ignore")


# =============================================================================
# UTILITIES
# =============================================================================

def pr(msg: str) -> None:
    print(f"[step3-A] {msg}", flush=True)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_sem(std: float, n: int) -> float:
    if std is None or pd.isna(std) or n is None or pd.isna(n) or n <= 1:
        return float("nan")
    return float(std) / math.sqrt(float(n))


# =============================================================================
# LOAD MANIFEST + METRICS
# =============================================================================

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing column_manifest.json at {MANIFEST_PATH}")

    manifest = load_json(MANIFEST_PATH)

    # Expected fields
    if "targets" not in manifest:
        raise RuntimeError("Manifest missing 'targets'.")
    if "descriptor_families" not in manifest:
        raise RuntimeError("Manifest missing 'descriptor_families'.")

    return manifest


def load_raw_metrics() -> pd.DataFrame:
    if not RAW_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing raw_metrics.csv at {RAW_METRICS_PATH}")

    df = pd.read_csv(RAW_METRICS_PATH, low_memory=False)

    # Normalize some columns we expect
    if "descriptor_family" not in df.columns:
        raise RuntimeError("raw_metrics.csv missing descriptor_family column.")
    if "model" not in df.columns:
        raise RuntimeError("raw_metrics.csv missing model column.")
    if "target" not in df.columns:
        raise RuntimeError("raw_metrics.csv missing target column.")
    if "split_type" not in df.columns:
        raise RuntimeError("raw_metrics.csv missing split_type column.")
    if "split_name" not in df.columns:
        raise RuntimeError("raw_metrics.csv missing split_name column.")

    # Numeric coercion
    for c in ["r2", "mae", "rmse", "spearman", "pearson", "bias", "elapsed_s",
              "n_train", "n_test", "n_numeric", "n_categorical"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =============================================================================
# COMPLETION AUDIT
# =============================================================================

def completion_audit(raw_metrics: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = list(manifest["targets"])
    families = list(manifest["descriptor_families"].keys())

    expected_split_names = []
    for p in SPLIT_DIR.glob("random_seed_*.json"):
        expected_split_names.append(p.stem)
    for p in SPLIT_DIR.glob("balanced_group_fold_*.json"):
        expected_split_names.append(p.stem)
    expected_split_names = sorted(expected_split_names)

    rows = []
    for split_name in expected_split_names:
        if split_name.startswith("random_seed_"):
            split_type = "random"
        else:
            split_type = "balanced_topology_group"

        for target in targets:
            for family in families:
                for model in EXPECTED_MODELS:
                    sub = raw_metrics[
                        (raw_metrics["split_name"] == split_name)
                        & (raw_metrics["target"] == target)
                        & (raw_metrics["descriptor_family"] == family)
                        & (raw_metrics["model"] == model)
                    ]

                    n_logged = int(len(sub))
                    n_ok = int((sub["status"] == "ok").sum()) if "status" in sub.columns else n_logged
                    n_failed = int(n_logged - n_ok)

                    rows.append({
                        "split_name": split_name,
                        "split_type": split_type,
                        "target": target,
                        "descriptor_family": family,
                        "model": model,
                        "n_logged": n_logged,
                        "n_ok": n_ok,
                        "n_failed": n_failed,
                        "is_missing": n_logged == 0,
                    })

    audit = pd.DataFrame(rows)

    failed = raw_metrics.copy()
    if "status" in failed.columns:
        failed = failed[failed["status"] != "ok"]
    else:
        failed = failed.iloc[0:0]

    return audit, failed


# =============================================================================
# CORE SUMMARIES
# =============================================================================

METRIC_COLS = ["r2", "mae", "rmse", "spearman", "pearson", "bias"]


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """
    Aggregates fold-level metrics into mean / std / n / sem.
    Keeps split_type separate by default unless excluded from `by`.
    """
    keep = [c for c in by if c in df.columns]
    grouped = df.groupby(keep, dropna=False)

    out = grouped[METRIC_COLS].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out = out.reset_index()

    # Number of folds = count of fold-level entries that contributed to r2
    out["n_folds"] = out["r2_count"].fillna(0).astype(int)

    for m in METRIC_COLS:
        out[f"{m}_sem"] = out.apply(
            lambda r: safe_sem(r.get(f"{m}_std"), r.get(f"{m}_count")),
            axis=1,
        )

    return out


def summary_by_target(df_ok: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        df_ok,
        by=[
            "target", "layer", "process", "gas", "gas_pair",
            "split_type", "descriptor_family", "model",
        ],
    )


def summary_by_layer(df_ok: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        df_ok,
        by=["layer", "split_type", "descriptor_family", "model"],
    )


def summary_by_process(df_ok: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        df_ok,
        by=["process", "split_type", "descriptor_family", "model"],
    )


def summary_by_split_type(df_ok: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        df_ok,
        by=["split_type", "descriptor_family", "model"],
    )


def summary_by_descriptor_family(df_ok: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        df_ok,
        by=["descriptor_family", "model", "split_type"],
    )


def summary_by_model(df_ok: pd.DataFrame) -> pd.DataFrame:
    return summarize(
        df_ok,
        by=["model", "split_type", "descriptor_family"],
    )


def summary_primary_ood(target_summary: pd.DataFrame) -> pd.DataFrame:
    return target_summary[target_summary["split_type"] == PRIMARY_OOD_SPLIT_TYPE].copy()


# =============================================================================
# RANDOM vs OOD GAP
# =============================================================================

def random_vs_group_gap(target_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "layer", "process", "gas", "gas_pair",
            "descriptor_family", "model"]
    keys = [k for k in keys if k in target_summary.columns]

    r = target_summary[target_summary["split_type"] == "random"][
        keys + ["r2_mean", "r2_std", "r2_sem", "n_folds"]
    ].copy()
    r = r.rename(columns={
        "r2_mean": "r2_random_mean",
        "r2_std": "r2_random_std",
        "r2_sem": "r2_random_sem",
        "n_folds": "n_random_folds",
    })

    g = target_summary[target_summary["split_type"] == PRIMARY_OOD_SPLIT_TYPE][
        keys + ["r2_mean", "r2_std", "r2_sem", "n_folds"]
    ].copy()
    g = g.rename(columns={
        "r2_mean": "r2_group_mean",
        "r2_std": "r2_group_std",
        "r2_sem": "r2_group_sem",
        "n_folds": "n_group_folds",
    })

    out = pd.merge(r, g, on=keys, how="outer")
    out["generalization_gap_r2"] = out["r2_random_mean"] - out["r2_group_mean"]

    return out


# =============================================================================
# DESCRIPTOR GAIN TABLES
# =============================================================================

GAIN_PAIRS = [
    ("enriched_interpretable", "geometry_only"),
    ("rac_only", "geometry_only"),
    ("geo_plus_rac", "geometry_only"),
    ("geo_plus_rac", "enriched_interpretable"),
    ("geo_plus_rac", "rac_only"),
    ("geometry_plus_topology", "enriched_interpretable"),
]


def descriptor_gain_tables(target_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "layer", "process", "gas", "gas_pair", "split_type", "model"]
    keys = [k for k in keys if k in target_summary.columns]

    rows = []
    pivoted = target_summary.pivot_table(
        index=keys,
        columns="descriptor_family",
        values=["r2_mean", "mae_mean", "spearman_mean"],
        aggfunc="first",
    )

    for plus_fam, base_fam in GAIN_PAIRS:
        try:
            r2_plus = pivoted[("r2_mean", plus_fam)]
            r2_base = pivoted[("r2_mean", base_fam)]
        except KeyError:
            continue

        delta_r2 = r2_plus - r2_base
        delta_mae = None
        delta_spearman = None

        try:
            delta_mae = pivoted[("mae_mean", base_fam)] - pivoted[("mae_mean", plus_fam)]
        except KeyError:
            pass
        try:
            delta_spearman = pivoted[("spearman_mean", plus_fam)] - pivoted[("spearman_mean", base_fam)]
        except KeyError:
            pass

        df_pair = delta_r2.reset_index()
        df_pair.columns = list(df_pair.columns[:-1]) + ["delta_r2"]
        df_pair["plus_family"] = plus_fam
        df_pair["base_family"] = base_fam

        if delta_mae is not None:
            df_pair["delta_mae"] = delta_mae.values
        if delta_spearman is not None:
            df_pair["delta_spearman"] = delta_spearman.values

        rows.append(df_pair)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out


# =============================================================================
# RIDGE VS TREE NONLINEARITY
# =============================================================================

def ridge_vs_tree_nonlinearity(target_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["target", "layer", "process", "gas", "gas_pair",
            "split_type", "descriptor_family"]
    keys = [k for k in keys if k in target_summary.columns]

    pivot = target_summary.pivot_table(
        index=keys,
        columns="model",
        values="r2_mean",
        aggfunc="first",
    ).reset_index()

    if "ridge" not in pivot.columns:
        return pd.DataFrame()

    if "lgbm" in pivot.columns:
        pivot["nonlinearity_gain_lgbm"] = pivot["lgbm"] - pivot["ridge"]
    if "rf" in pivot.columns:
        pivot["nonlinearity_gain_rf"] = pivot["rf"] - pivot["ridge"]

    return pivot


# =============================================================================
# FEATURE IMPORTANCE STREAMING
# =============================================================================

def aggregate_feature_importance(path: Path, chunk_size: int = 500_000):
    if not path.exists():
        pr(f"feature_importances_raw.csv not found at {path} (skipping)")
        return pd.DataFrame(), pd.DataFrame()

    cols_to_use = [
        "target", "layer", "process", "descriptor_family", "model",
        "split_type", "split_name", "feature_name", "importance_value",
    ]

    iterator = pd.read_csv(path, chunksize=chunk_size, low_memory=False)

    agg_dict = {}

    n_chunks = 0
    for chunk in iterator:
        n_chunks += 1
        keep = [c for c in cols_to_use if c in chunk.columns]
        chunk = chunk[keep]

        chunk["importance_value"] = pd.to_numeric(chunk["importance_value"], errors="coerce")

        # group key: (target, layer, process, descriptor_family, model, split_type, feature_name)
        keys = ["target", "layer", "process", "descriptor_family", "model", "split_type", "feature_name"]
        keys = [k for k in keys if k in chunk.columns]

        grouped = chunk.groupby(keys, dropna=False)["importance_value"].agg(
            ["sum", "count", lambda s: (s ** 2).sum()]
        ).reset_index()
        grouped.columns = list(keys) + ["sum_x", "n_x", "sum_x2"]

        for r in grouped.itertuples(index=False):
            k = tuple(getattr(r, c) for c in keys)
            d = agg_dict.get(k)
            if d is None:
                agg_dict[k] = {
                    "sum_x": r.sum_x,
                    "n_x": r.n_x,
                    "sum_x2": r.sum_x2,
                }
            else:
                d["sum_x"] += r.sum_x
                d["n_x"] += r.n_x
                d["sum_x2"] += r.sum_x2

        if n_chunks % 5 == 0:
            pr(f"feature importance chunks processed: {n_chunks}")

    pr(f"feature importance chunks total: {n_chunks}")

    # finalize
    rows = []
    for k, d in agg_dict.items():
        mean_v = d["sum_x"] / d["n_x"] if d["n_x"] > 0 else float("nan")
        var_v = (d["sum_x2"] / d["n_x"]) - mean_v ** 2 if d["n_x"] > 0 else float("nan")
        std_v = math.sqrt(var_v) if (var_v is not None and var_v >= 0) else float("nan")

        row = dict(zip(["target", "layer", "process", "descriptor_family", "model", "split_type", "feature_name"], k))
        row["importance_mean"] = mean_v
        row["importance_std"] = std_v
        row["n_folds_obs"] = d["n_x"]
        rows.append(row)

    full = pd.DataFrame(rows)

    if full.empty:
        return pd.DataFrame(), pd.DataFrame()

    # rank per (family × model × split_type)
    full["rank"] = (
        full
        .groupby(["descriptor_family", "model", "split_type"], dropna=False)["importance_mean"]
        .rank(ascending=False, method="min")
    )

    top = full[full["rank"] <= TOP_N_FEATURES].copy()
    top = top.sort_values(
        ["descriptor_family", "model", "split_type", "rank"]
    )

    return full, top


# =============================================================================
# RECOMMENDATION RULE
# =============================================================================

def recommendation_table(process_summary: pd.DataFrame) -> pd.DataFrame:
    df = process_summary[process_summary["split_type"] == PRIMARY_OOD_SPLIT_TYPE].copy()

    df = df[~df["descriptor_family"].isin(DIAGNOSTIC_ONLY_FAMILIES)].copy()
    df["cost_tier"] = df["descriptor_family"].map(COST_TIERS)

    rows = []
    for (process, model), g in df.groupby(["process", "model"], dropna=False):
        g = g.dropna(subset=["r2_mean"])
        if g.empty:
            continue

        best_idx = g["r2_mean"].idxmax()
        best_r2 = g.loc[best_idx, "r2_mean"]
        best_std = g.loc[best_idx, "r2_std"]
        best_n = g.loc[best_idx, "n_folds"]
        best_se = safe_sem(best_std, best_n)

        margin = max(PRACTICAL_EQUIVALENCE_R2, best_se if not math.isnan(best_se) else 0.0)
        cutoff = best_r2 - margin

        equivalent = g[g["r2_mean"] >= cutoff].copy()
        equivalent = equivalent.sort_values(
            ["cost_tier", "r2_mean"],
            ascending=[True, False],
        )

        chosen = equivalent.iloc[0]

        rows.append({
            "process": process,
            "model": model,
            "best_family_absolute": g.loc[best_idx, "descriptor_family"],
            "best_r2_absolute": float(best_r2),
            "recommended_family": chosen["descriptor_family"],
            "recommended_r2": float(chosen["r2_mean"]),
            "recommended_cost_tier": int(chosen["cost_tier"]),
            "equivalence_margin_used": float(margin),
            "fixed_rope_r2": PRACTICAL_EQUIVALENCE_R2,
            "best_family_se": float(best_se) if not math.isnan(best_se) else None,
            "rule": (
                "Cheapest descriptor family within "
                "max(0.02 R2, one-SE observed margin) of best OOD R2; "
                "topology_only excluded from recommendations."
            ),
        })

    return pd.DataFrame(rows)


# =============================================================================
# PROCESS COVERAGE TABLE
# =============================================================================

def process_coverage_table(manifest: dict) -> pd.DataFrame:
    meta = manifest.get("target_metadata", {})

    by_process_layer = {}

    for tname, m in meta.items():
        process = m.get("process")
        layer = m.get("layer")
        if process is None or layer is None:
            continue
        by_process_layer.setdefault((process, layer), []).append(tname)

    rows = []
    for process in PROCESS_ORDER:
        row = {
            "process": process,
            "raw_uptake_targets": "|".join(sorted(by_process_layer.get((process, "A_raw_uptake"), []))),
            "working_capacity_targets": "|".join(sorted(by_process_layer.get((process, "B_working_capacity"), []))),
            "log_selectivity_targets": "|".join(sorted(by_process_layer.get((process, "C_direct_log_selectivity"), []))),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN OOD RESULT TABLE
# =============================================================================

def main_ood_results_table(target_summary: pd.DataFrame) -> pd.DataFrame:
    df = target_summary[target_summary["split_type"] == PRIMARY_OOD_SPLIT_TYPE].copy()
    df = df[[
        "target", "layer", "process", "gas", "gas_pair",
        "descriptor_family", "model",
        "r2_mean", "r2_std", "r2_sem",
        "mae_mean", "mae_std",
        "rmse_mean", "rmse_std",
        "spearman_mean", "spearman_std",
        "n_folds",
    ]].copy()

    return df.sort_values(
        ["process", "layer", "target", "descriptor_family", "model"]
    )


def model_specs_used_table(run_config: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "models": ",".join(run_config.get("models", [])) if isinstance(run_config.get("models", []), list) else run_config.get("models", ""),
        "optuna_trials": run_config.get("optuna_trials"),
        "optuna_n_jobs": run_config.get("optuna_n_jobs"),
        "model_n_jobs": run_config.get("model_n_jobs"),
        "random_split_seeds": ",".join(map(str, run_config.get("random_split_seeds", []))),
        "n_group_folds": run_config.get("n_group_folds"),
        "test_size": run_config.get("test_size"),
        "inner_validation_fraction": run_config.get("inner_validation_fraction"),
        "dtype_float": run_config.get("dtype_float"),
        "sklearnex_enabled": run_config.get("sklearnex_enabled"),
        "sklearn": run_config.get("sklearn"),
        "lightgbm": run_config.get("lightgbm"),
        "python": run_config.get("python"),
        "platform": run_config.get("platform"),
    }])


# =============================================================================
# FIGURES
# =============================================================================

def fig_process_layer_heatmap(process_summary: pd.DataFrame) -> None:
    df = process_summary[process_summary["split_type"] == PRIMARY_OOD_SPLIT_TYPE].copy()
    df = df[df["model"] == "lgbm"].copy()

    if df.empty:
        return

    df["family_label"] = df["descriptor_family"].map(FAMILY_DISPLAY).fillna(df["descriptor_family"])
    df["process_label"] = df["process"].map(PROCESS_DISPLAY).fillna(df["process"])

    families_in_order = [
        FAMILY_DISPLAY[f] for f in COST_TIERS.keys() if f in FAMILY_DISPLAY
    ]
    families_in_order += [FAMILY_DISPLAY["topology_only"]]

    pivot = df.pivot_table(
        index="family_label",
        columns="process_label",
        values="r2_mean",
        aggfunc="first",
    )

    # enforce family order
    pivot = pivot.reindex(index=[f for f in families_in_order if f in pivot.index])
    # enforce process order
    pivot = pivot.reindex(columns=[PROCESS_DISPLAY[p] for p in PROCESS_ORDER if PROCESS_DISPLAY[p] in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)

    ax.set_title("Topology-OOD mean R2 by process × descriptor family (LightGBM)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("R2")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_process_layer_heatmap.png", dpi=200)
    plt.close(fig)


def fig_random_vs_group_gap(gap_df: pd.DataFrame) -> None:
    if gap_df.empty:
        return

    df = gap_df.dropna(subset=["r2_random_mean", "r2_group_mean"]).copy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df["r2_random_mean"], df["r2_group_mean"], s=12, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Random-split mean R2 (IID)")
    ax.set_ylabel("Topology-OOD mean R2")
    ax.set_title("Random vs Topology-OOD R2 across all (target × family × model)")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_random_vs_group_gap.png", dpi=200)
    plt.close(fig)


def fig_ridge_vs_lgbm_nonlinearity(nonl_df: pd.DataFrame) -> None:
    if nonl_df.empty or "nonlinearity_gain_lgbm" not in nonl_df.columns:
        return

    df = nonl_df.dropna(subset=["nonlinearity_gain_lgbm"]).copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["nonlinearity_gain_lgbm"], bins=40, color="#3a72c2", edgecolor="white")
    ax.axvline(0.0, color="black", lw=1)

    ax.set_xlabel("R2 gain (LightGBM − Ridge)")
    ax.set_ylabel("count")
    ax.set_title("Nonlinearity gain of LightGBM over Ridge")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ridge_vs_lgbm_nonlinearity.png", dpi=200)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start = time.time()
    pr("=" * 80)
    pr("STEP 3 STAGE A — light analysis (no predictions, no training)")
    pr("=" * 80)

    if not BENCH_DIR.exists():
        pr(f"ERROR: bundle dir not found at {BENCH_DIR}")
        sys.exit(1)

    mkdir(EVAL_DIR)
    mkdir(TABLE_DIR)
    mkdir(FIG_DIR)

    pr(f"Bundle: {BENCH_DIR}")
    pr(f"Outputs: {OUT_ROOT}")

    # --- manifest + run config ---
    manifest = load_manifest()
    run_config_path = GLOBAL_DIR / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}

    # --- load raw metrics ---
    pr("Loading raw_metrics.csv ...")
    rm = load_raw_metrics()
    pr(f"raw_metrics rows: {len(rm):,}")

    # --- completion audit ---
    pr("Completion audit ...")
    audit_df, failed_df = completion_audit(rm, manifest)
    audit_df.to_csv(EVAL_DIR / "completion_audit.csv", index=False)
    failed_df.to_csv(EVAL_DIR / "failed_model_runs.csv", index=False)

    # --- only successful rows for science ---
    if "status" in rm.columns:
        rm_ok = rm[rm["status"] == "ok"].copy()
    else:
        rm_ok = rm.copy()

    if rm_ok.empty:
        pr("No successful rows found. Stopping.", "ERROR")
        sys.exit(1)

    # --- summaries ---
    pr("Summaries by target / layer / process / split_type / descriptor_family / model ...")
    target_sum = summary_by_target(rm_ok)
    layer_sum = summary_by_layer(rm_ok)
    process_sum = summary_by_process(rm_ok)
    split_sum = summary_by_split_type(rm_ok)
    family_sum = summary_by_descriptor_family(rm_ok)
    model_sum = summary_by_model(rm_ok)

    target_sum.to_csv(EVAL_DIR / "summary_by_target.csv", index=False)
    layer_sum.to_csv(EVAL_DIR / "summary_by_layer.csv", index=False)
    process_sum.to_csv(EVAL_DIR / "summary_by_process.csv", index=False)
    split_sum.to_csv(EVAL_DIR / "summary_by_split_type.csv", index=False)
    family_sum.to_csv(EVAL_DIR / "summary_by_descriptor_family.csv", index=False)
    model_sum.to_csv(EVAL_DIR / "summary_by_model.csv", index=False)

    # --- primary OOD slice ---
    primary_ood = summary_primary_ood(target_sum)
    primary_ood.to_csv(EVAL_DIR / "summary_primary_ood.csv", index=False)

    # --- random vs group gap ---
    pr("Random vs OOD gap ...")
    gap = random_vs_group_gap(target_sum)
    gap.to_csv(EVAL_DIR / "random_vs_group_gap.csv", index=False)

    # --- descriptor gain tables ---
    pr("Descriptor gain tables ...")
    gain = descriptor_gain_tables(target_sum)
    if not gain.empty:
        gain.to_csv(EVAL_DIR / "descriptor_gain_tables.csv", index=False)

    # --- ridge vs tree nonlinearity ---
    pr("Ridge vs tree nonlinearity ...")
    nonl = ridge_vs_tree_nonlinearity(target_sum)
    if not nonl.empty:
        nonl.to_csv(EVAL_DIR / "ridge_vs_tree_nonlinearity.csv", index=False)

    # --- feature importances (streamed) ---
    pr("Aggregating feature importances (streamed) ...")
    fi_full, fi_top = aggregate_feature_importance(FEATURE_IMPORTANCE_PATH)
    if not fi_full.empty:
        fi_full.to_csv(EVAL_DIR / "feature_importance_summary.csv", index=False)
        fi_top.to_csv(EVAL_DIR / "feature_importance_top_per_model_family.csv", index=False)
    else:
        pr("Skipped feature importance outputs (file empty or missing).")

    # --- tables ---
    pr("Building tables ...")
    main_ood_table = main_ood_results_table(target_sum)
    main_ood_table.to_csv(TABLE_DIR / "table_main_ood_results.csv", index=False)

    rec_table = recommendation_table(process_sum)
    rec_table.to_csv(TABLE_DIR / "table_descriptor_recommendation.csv", index=False)

    cov_table = process_coverage_table(manifest)
    cov_table.to_csv(TABLE_DIR / "table_process_coverage.csv", index=False)

    specs_table = model_specs_used_table(run_config)
    specs_table.to_csv(TABLE_DIR / "table_model_specs_used.csv", index=False)

    # --- figures ---
    pr("Figures ...")
    fig_process_layer_heatmap(process_sum)
    fig_random_vs_group_gap(gap)
    fig_ridge_vs_lgbm_nonlinearity(nonl)

    elapsed = time.time() - start

    pr("=" * 80)
    pr(f"STEP 3 STAGE A COMPLETE in {elapsed/60:.2f} min")
    pr(f"Eval folder: {EVAL_DIR}")
    pr(f"Tables folder: {TABLE_DIR}")
    pr(f"Figures folder: {FIG_DIR}")
    pr("=" * 80)


if __name__ == "__main__":
    main()
