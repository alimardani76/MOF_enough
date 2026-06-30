#!/usr/bin/env python3
"""
step3_stageB.py  (patched target IDs)

Stage B of Step 3 for V9 manuscript.

Reads only predictions.csv.gz (streamed) and produces:
  §3.4 Q1 direct vs post-hoc log-selectivity
  §3.4 Q2 working-capacity component audit
  §3.5 Q3 top-K screening yield + champion misidentification
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

INPUT_PRED = Path(os.environ.get("PREDICTIONS_CSV_GZ", ROOT / "predictions.csv.gz"))
OUT = ROOT / "results" / "step3B_target_screening_analysis"

CHUNK_SIZE = 1_000_000

PRIMARY_MODEL = "lgbm"
PRIMARY_SPLIT_TYPE = "balanced_topology_group"

TOP_K_VALUES = [100, 1000]
UPTAKE_FLOOR = 1e-5

# Composition factors (Methods SI Table S5)
COMPOSITION_FACTORS = {
    "post_combustion_vsa":      0.83 / 0.17,
    "pre_combustion_psa":       0.60 / 0.40,
    "natural_gas_purification": 0.90 / 0.10,
    "landfill_gas_vpsa":        4.4 / 3.2,
}

# Layer C ln(S) targets (true ARC-MOF process names from overall_process.csv).
# Names as they appear in predictions.csv.gz:
POSTHOC_PAIRS = {
    "post_combustion_vsa": {
        "direct_target":     "postcomb_log_selectivity",
        "num_uptake_target": "postcomb_CO2_0.15bar",
        "den_uptake_target": "postcomb_N2_0.75bar",
    },
    "pre_combustion_psa": {
        "direct_target":     "precomb_log_selectivity",
        "num_uptake_target": "precomb_CO2_16bar",
        "den_uptake_target": "precomb_H2_24bar",
    },
    "natural_gas_purification": {
        "direct_target":     "ngp_log_selectivity",
        "num_uptake_target": "ngp_CO2_1.0bar",
        "den_uptake_target": "ngp_CH4_9.0bar",
    },
    "landfill_gas_vpsa": {
        "direct_target":     "landfill_log_selectivity",
        "num_uptake_target": "landfill_CO2_3.2bar",
        "den_uptake_target": "landfill_CH4_4.4bar",
    },
}

# Layer B working-capacity targets (also from overall_process.csv).
WC_COMPONENTS = {
    "methane_storage_psa": {
        "wc_target":    "methane_storage_working_capacity",
        "q_ads_target": "CH4_65bar",
        "q_des_target": "CH4_5.8bar",
    },
    "post_combustion_vsa": {
        "wc_target":    "postcomb_working_capacity",
        "q_ads_target": "postcomb_CO2_0.15bar",
        "q_des_target": "postcomb_CO2_0.015bar",
    },
    "pre_combustion_psa": {
        "wc_target":    "precomb_working_capacity",
        "q_ads_target": "precomb_CO2_16bar",
        "q_des_target": "precomb_CO2_0.4bar",
    },
    "natural_gas_purification": {
        "wc_target":    "ngp_working_capacity",
        "q_ads_target": "ngp_CO2_1.0bar",
        "q_des_target": "ngp_CO2_0.1bar",
    },
    "landfill_gas_vpsa": {
        "wc_target":    "landfill_working_capacity",
        "q_ads_target": "landfill_CO2_3.2bar",
        "q_des_target": "landfill_CO2_0.26bar",
    },
}

# §3.6 recommendations from Stage A1 (locked).
RECOMMENDED_FAMILY = {
    ("methane_storage_psa", "A_raw_uptake"):                  "geo_plus_rac",
    ("methane_storage_psa", "B_working_capacity"):            "geometry_only",
    ("post_combustion_vsa", "A_raw_uptake"):                  "geometry_only",
    ("post_combustion_vsa", "B_working_capacity"):            "geo_plus_rac",
    ("post_combustion_vsa", "C_direct_log_selectivity"):      "geo_plus_rac",
    ("pre_combustion_psa",  "A_raw_uptake"):                  "geometry_only",
    ("pre_combustion_psa",  "B_working_capacity"):            "geo_plus_rac",
    ("pre_combustion_psa",  "C_direct_log_selectivity"):      "geo_plus_rac",
    ("natural_gas_purification", "A_raw_uptake"):             "geo_plus_rac",
    ("natural_gas_purification", "B_working_capacity"):       "geo_plus_rac",
    ("natural_gas_purification", "C_direct_log_selectivity"): "geo_plus_rac",
    ("landfill_gas_vpsa",   "A_raw_uptake"):                  "geo_plus_rac",
    ("landfill_gas_vpsa",   "B_working_capacity"):            "geo_plus_rac",
    ("landfill_gas_vpsa",   "C_direct_log_selectivity"):      "geo_plus_rac",
}

# §3.5 representative targets (dot form to match predictions.csv.gz)
TOPK_TARGETS = {
    "CH4_65bar":               ["geometry_only", "geo_plus_rac"],
    "postcomb_CO2_0.015bar":   ["geometry_only", "geo_plus_rac"],
    "postcomb_log_selectivity": ["geometry_only", "geo_plus_rac"],
}

# fold names match Step 2 split JSON layout
FOLD_NAMES = [f"balanced_group_fold_{i}" for i in range(4)]

REQUIRED_COLS = {
    "target", "layer", "process",
    "split_type", "split_name",
    "descriptor_family", "model",
    "filename", "y_true", "y_pred",
}


# =============================================================================
# UTILITIES
# =============================================================================

def pr(msg: str) -> None:
    print(f"[stage-B] {msg}", flush=True)


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def safe_spearman(y_true, y_pred) -> float:
    try:
        rho = spearmanr(y_true, y_pred).statistic
        return float(rho) if np.isfinite(rho) else float("nan")
    except Exception:
        return float("nan")


def safe_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# =============================================================================
# COLLECTION
# =============================================================================

def needed_keys_for_q1() -> set:
    keys = set()
    for process, m in POSTHOC_PAIRS.items():
        fam_C = RECOMMENDED_FAMILY[(process, "C_direct_log_selectivity")]
        fam_A = RECOMMENDED_FAMILY[(process, "A_raw_uptake")]
        for fold_name in FOLD_NAMES:
            keys.add((m["direct_target"],     fam_C, fold_name))
            keys.add((m["num_uptake_target"], fam_A, fold_name))
            keys.add((m["den_uptake_target"], fam_A, fold_name))
    return keys


def needed_keys_for_q2() -> set:
    keys = set()
    for process, m in WC_COMPONENTS.items():
        fam_B = RECOMMENDED_FAMILY[(process, "B_working_capacity")]
        fam_A = RECOMMENDED_FAMILY[(process, "A_raw_uptake")]
        for fold_name in FOLD_NAMES:
            keys.add((m["wc_target"],    fam_B, fold_name))
            keys.add((m["q_ads_target"], fam_A, fold_name))
            keys.add((m["q_des_target"], fam_A, fold_name))
    return keys


def needed_keys_for_q3() -> set:
    keys = set()
    for target, families in TOPK_TARGETS.items():
        pl = _process_layer_for_target(target)
        if pl is None:
            continue
        recommended = RECOMMENDED_FAMILY[pl]
        for fam in list(dict.fromkeys(families + [recommended])):
            for fold_name in FOLD_NAMES:
                keys.add((target, fam, fold_name))
    return keys


def _process_layer_for_target(target: str):
    layer_map = {
        "CH4_65bar":                ("methane_storage_psa", "A_raw_uptake"),
        "postcomb_CO2_0.015bar":    ("post_combustion_vsa", "A_raw_uptake"),
        "postcomb_log_selectivity": ("post_combustion_vsa", "C_direct_log_selectivity"),
    }
    return layer_map.get(target)


def collect_predictions(needed_keys: set) -> dict:
    if not INPUT_PRED.exists():
        sys.exit(f"Cannot find {INPUT_PRED}")
    pr(f"Streaming {INPUT_PRED.name} in chunks of {CHUNK_SIZE:,} rows")

    collected = {k: [] for k in needed_keys}

    n_total = 0
    n_chunks = 0
    n_kept = 0

    reader = pd.read_csv(
        INPUT_PRED,
        compression="infer",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk in reader:
        n_chunks += 1
        n_total += len(chunk)

        missing = REQUIRED_COLS - set(chunk.columns)
        if missing:
            sys.exit(f"predictions.csv.gz missing required columns: {missing}")

        chunk = chunk[
            (chunk["model"] == PRIMARY_MODEL)
            & (chunk["split_type"] == PRIMARY_SPLIT_TYPE)
        ]
        if chunk.empty:
            continue

        mask = chunk.apply(
            lambda r: (r["target"], r["descriptor_family"], r["split_name"]) in needed_keys,
            axis=1,
        )
        chunk = chunk[mask]
        if chunk.empty:
            continue

        for key, grp in chunk.groupby(["target", "descriptor_family", "split_name"]):
            collected[key].append(grp[["filename", "y_true", "y_pred"]].copy())
            n_kept += len(grp)

        if n_chunks % 10 == 0:
            pr(f"  chunks: {n_chunks}, rows seen: {n_total:,}, kept: {n_kept:,}")

    pr(f"Done streaming. Total rows: {n_total:,}, kept rows: {n_kept:,}")

    out = {}
    for k, parts in collected.items():
        out[k] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["filename", "y_true", "y_pred"]
        )
    return out


# =============================================================================
# Q1
# =============================================================================

def compute_q1(predictions: dict) -> pd.DataFrame:
    rows = []
    for process, m in POSTHOC_PAIRS.items():
        fam_C = RECOMMENDED_FAMILY[(process, "C_direct_log_selectivity")]
        fam_A = RECOMMENDED_FAMILY[(process, "A_raw_uptake")]
        cf = COMPOSITION_FACTORS[process]

        for fold_idx, fold_name in enumerate(FOLD_NAMES):
            df_d = predictions.get((m["direct_target"],     fam_C, fold_name))
            df_n = predictions.get((m["num_uptake_target"], fam_A, fold_name))
            df_p = predictions.get((m["den_uptake_target"], fam_A, fold_name))
            if df_d is None or df_n is None or df_p is None:
                continue
            if df_d.empty or df_n.empty or df_p.empty:
                continue

            joined = (
                df_d.merge(df_n, on="filename", suffixes=("_direct", "_num"))
                    .merge(df_p, on="filename")
            )
            joined = joined.rename(columns={"y_true": "y_true_den", "y_pred": "y_pred_den"})
            if joined.empty:
                continue

            y_true_lnS = joined["y_true_direct"].to_numpy()
            y_pred_lnS_direct = joined["y_pred_direct"].to_numpy()

            q_num = np.clip(joined["y_pred_num"].to_numpy(), UPTAKE_FLOOR, None)
            q_den = np.clip(joined["y_pred_den"].to_numpy(), UPTAKE_FLOOR, None)
            y_pred_lnS_posthoc = np.log((q_num / q_den) * cf)

            rows.append({
                "process": process,
                "fold": fold_idx,
                "n_mofs": int(len(joined)),
                "family_C": fam_C,
                "family_A": fam_A,
                "composition_factor": cf,
                "r2_direct_lnS":      safe_r2(y_true_lnS, y_pred_lnS_direct),
                "r2_posthoc_lnS":     safe_r2(y_true_lnS, y_pred_lnS_posthoc),
                "spearman_direct_lnS":  safe_spearman(y_true_lnS, y_pred_lnS_direct),
                "spearman_posthoc_lnS": safe_spearman(y_true_lnS, y_pred_lnS_posthoc),
                "mae_direct_lnS":  safe_mae(y_true_lnS, y_pred_lnS_direct),
                "mae_posthoc_lnS": safe_mae(y_true_lnS, y_pred_lnS_posthoc),
                "delta_r2_direct_minus_posthoc":
                    safe_r2(y_true_lnS, y_pred_lnS_direct)
                    - safe_r2(y_true_lnS, y_pred_lnS_posthoc),
            })
    return pd.DataFrame(rows)


def summarize_q1(per_fold: pd.DataFrame) -> pd.DataFrame:
    if per_fold.empty:
        return pd.DataFrame()
    return per_fold.groupby("process", as_index=False).agg(
        n_folds=("fold", "nunique"),
        r2_direct_mean=("r2_direct_lnS", "mean"),
        r2_direct_std=("r2_direct_lnS", "std"),
        r2_posthoc_mean=("r2_posthoc_lnS", "mean"),
        r2_posthoc_std=("r2_posthoc_lnS", "std"),
        spearman_direct_mean=("spearman_direct_lnS", "mean"),
        spearman_posthoc_mean=("spearman_posthoc_lnS", "mean"),
        mae_direct_mean=("mae_direct_lnS", "mean"),
        mae_posthoc_mean=("mae_posthoc_lnS", "mean"),
        delta_r2_mean=("delta_r2_direct_minus_posthoc", "mean"),
        delta_r2_std=("delta_r2_direct_minus_posthoc", "std"),
    )


# =============================================================================
# Q2
# =============================================================================

def compute_q2(predictions: dict) -> pd.DataFrame:
    rows = []
    for process, m in WC_COMPONENTS.items():
        fam_B = RECOMMENDED_FAMILY[(process, "B_working_capacity")]
        fam_A = RECOMMENDED_FAMILY[(process, "A_raw_uptake")]
        for fold_idx, fold_name in enumerate(FOLD_NAMES):
            df_wc  = predictions.get((m["wc_target"],    fam_B, fold_name))
            df_ads = predictions.get((m["q_ads_target"], fam_A, fold_name))
            df_des = predictions.get((m["q_des_target"], fam_A, fold_name))
            if df_wc is None or df_ads is None or df_des is None:
                continue
            if df_wc.empty or df_ads.empty or df_des.empty:
                continue
            joined = (
                df_wc.merge(df_ads, on="filename", suffixes=("_wc", "_ads"))
                     .merge(df_des, on="filename")
            )
            joined = joined.rename(columns={"y_true": "y_true_des", "y_pred": "y_pred_des"})
            if joined.empty:
                continue
            r2_wc  = safe_r2(joined["y_true_wc"],  joined["y_pred_wc"])
            r2_ads = safe_r2(joined["y_true_ads"], joined["y_pred_ads"])
            r2_des = safe_r2(joined["y_true_des"], joined["y_pred_des"])
            rows.append({
                "process": process,
                "fold": fold_idx,
                "n_mofs": int(len(joined)),
                "family_B": fam_B,
                "family_A": fam_A,
                "wc_target": m["wc_target"],
                "q_ads_target": m["q_ads_target"],
                "q_des_target": m["q_des_target"],
                "r2_wc": r2_wc,
                "r2_q_ads": r2_ads,
                "r2_q_des": r2_des,
                "wc_minus_q_des": r2_wc - r2_des,
                "wc_minus_q_ads": r2_wc - r2_ads,
            })
    return pd.DataFrame(rows)


def summarize_q2(per_fold: pd.DataFrame) -> pd.DataFrame:
    if per_fold.empty:
        return pd.DataFrame()
    g = per_fold.groupby("process", as_index=False).agg(
        n_folds=("fold", "nunique"),
        r2_wc_mean=("r2_wc", "mean"),
        r2_wc_std=("r2_wc", "std"),
        r2_q_ads_mean=("r2_q_ads", "mean"),
        r2_q_ads_std=("r2_q_ads", "std"),
        r2_q_des_mean=("r2_q_des", "mean"),
        r2_q_des_std=("r2_q_des", "std"),
        wc_minus_q_des_mean=("wc_minus_q_des", "mean"),
        wc_minus_q_des_std=("wc_minus_q_des", "std"),
        wc_minus_q_ads_mean=("wc_minus_q_ads", "mean"),
        wc_minus_q_ads_std=("wc_minus_q_ads", "std"),
    )
    g["component_masking_flag"] = (g["wc_minus_q_des_mean"] >= 0.10).astype(int)
    return g


# =============================================================================
# Q3
# =============================================================================

def compute_q3(predictions: dict) -> pd.DataFrame:
    rows = []
    for target, families in TOPK_TARGETS.items():
        pl = _process_layer_for_target(target)
        if pl is None:
            continue
        process, layer = pl
        recommended = RECOMMENDED_FAMILY[pl]
        families_to_use = list(dict.fromkeys(families + [recommended]))
        is_selectivity = (layer == "C_direct_log_selectivity")

        for fam in families_to_use:
            for fold_idx, fold_name in enumerate(FOLD_NAMES):
                df = predictions.get((target, fam, fold_name))
                if df is None or df.empty:
                    continue
                y_true = df["y_true"].to_numpy()
                y_pred = df["y_pred"].to_numpy()
                fnames = df["filename"].to_numpy()
                n = len(df)
                if n < 100:
                    continue
                true_order = np.argsort(-y_true)
                pred_order = np.argsort(-y_pred)

                row = {
                    "target": target,
                    "process": process,
                    "layer": layer,
                    "fold": fold_idx,
                    "family": fam,
                    "is_recommended": int(fam == recommended),
                    "n_mofs": n,
                }
                for K in TOP_K_VALUES:
                    if n < K:
                        continue
                    true_topK = set(fnames[true_order[:K]])
                    pred_topK = set(fnames[pred_order[:K]])
                    overlap = len(true_topK & pred_topK)
                    recall_K = overlap / K
                    precision_K = overlap / K
                    misid_K = 1.0 - precision_K
                    row[f"recall_at_{K}"] = recall_K
                    row[f"precision_at_{K}"] = precision_K
                    row[f"champion_misidentification_at_{K}"] = (
                        misid_K if is_selectivity else float("nan")
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_q3(per_fold: pd.DataFrame) -> pd.DataFrame:
    if per_fold.empty:
        return pd.DataFrame()
    K_cols = [f"recall_at_{K}" for K in TOP_K_VALUES]
    P_cols = [f"precision_at_{K}" for K in TOP_K_VALUES]
    M_cols = [f"champion_misidentification_at_{K}" for K in TOP_K_VALUES]
    agg_kwargs = {}
    for c in K_cols + P_cols + M_cols:
        agg_kwargs[f"{c}_mean"] = (c, "mean")
        agg_kwargs[f"{c}_std"]  = (c, "std")
    return per_fold.groupby(["target", "family", "is_recommended"], as_index=False).agg(
        n_folds=("fold", "nunique"),
        **agg_kwargs,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    pr("=" * 70)
    pr("STAGE B — Results §3.4 and §3.5")
    pr("=" * 70)

    mkdir(OUT)

    keys_q1 = needed_keys_for_q1()
    keys_q2 = needed_keys_for_q2()
    keys_q3 = needed_keys_for_q3()
    all_keys = keys_q1 | keys_q2 | keys_q3
    pr(f"Need {len(all_keys)} combos | Q1={len(keys_q1)} Q2={len(keys_q2)} Q3={len(keys_q3)}")

    predictions = collect_predictions(all_keys)

    # Q1
    pr("Computing Q1")
    q1_per_fold = compute_q1(predictions)
    q1_summary  = summarize_q1(q1_per_fold)
    q1_per_fold.to_csv(OUT / "q1_direct_vs_posthoc_lnS_per_fold.csv", index=False)
    q1_summary.to_csv(OUT / "q1_direct_vs_posthoc_lnS_summary.csv", index=False)
    pr(f"  Q1 per-fold rows: {len(q1_per_fold)}; summary rows: {len(q1_summary)}")

    # Q2
    pr("Computing Q2")
    q2_per_fold = compute_q2(predictions)
    q2_summary  = summarize_q2(q2_per_fold)
    q2_per_fold.to_csv(OUT / "q2_working_capacity_components_per_fold.csv", index=False)
    q2_summary.to_csv(OUT / "q2_working_capacity_components_summary.csv", index=False)
    pr(f"  Q2 per-fold rows: {len(q2_per_fold)}; summary rows: {len(q2_summary)}")

    # Q3
    pr("Computing Q3")
    q3_per_fold = compute_q3(predictions)
    q3_summary  = summarize_q3(q3_per_fold)
    q3_per_fold.to_csv(OUT / "q3_topK_per_fold.csv", index=False)
    q3_summary.to_csv(OUT / "q3_topK_summary.csv", index=False)
    pr(f"  Q3 per-fold rows: {len(q3_per_fold)}; summary rows: {len(q3_summary)}")

    (OUT / "README.txt").write_text(
        "Stage B outputs (patched target IDs).\n"
        "Q1: direct vs post-hoc ln(S) per process under LightGBM topology-OOD.\n"
        "Q2: R²(q_ads), R²(q_des), R²(wc) under recommended Layer B family.\n"
        "Q3: top-100/1000 recall + champion misidentification, three targets.\n",
        encoding="utf-8",
    )

    pr("=" * 70)
    pr("STAGE B COMPLETE")
    pr(f"Outputs in: {OUT}")
    pr("=" * 70)


if __name__ == "__main__":
    main()