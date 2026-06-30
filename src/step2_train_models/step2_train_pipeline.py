#!/usr/bin/env python3
"""
step2_model_training.py
=======================

V9 process-aware ARC-MOF model training (Crash-Resilient Checkpoint Version).

Purpose
-------
Controlled descriptor-adequacy benchmark, not a model leaderboard.
Features robust atomicity, trial-level resumption, and self-healing data outputs.
"""

from __future__ import annotations

# =============================================================================
# OPTIONAL CPU ACCELERATION
# =============================================================================
USE_SKLEARNEX = True
SKLEARNEX_ENABLED = False

if USE_SKLEARNEX:
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        SKLEARNEX_ENABLED = True
        print("[step2] scikit-learn-intelex enabled.")
    except Exception as e:
        print(f"[step2] scikit-learn-intelex unavailable; using stock sklearn. Reason: {e}")

# =============================================================================
# IMPORTS
# =============================================================================

import gzip
import json
import math
import os
import platform
import time
import traceback
import warnings
import pickle
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import NopPruner
from optuna.trial import TrialState

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
except Exception as e:
    raise ImportError("LightGBM is required. Install with: pip install lightgbm") from e

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "revision_data.csv"
MANIFEST_PATH = ROOT / "column_manifest.json"

OUT = ROOT / "benchmark_v9_outputs"
GLOBAL_DIR = OUT / "_global"
SPLIT_DIR = OUT / "splits"
METRIC_DIR = OUT / "metrics"
PRED_DIR = OUT / "predictions"
CHECKPOINT_DIR = OUT / "checkpoints"

RAW_METRICS_PATH = METRIC_DIR / "raw_metrics.csv"
OPTUNA_TRIALS_PATH = METRIC_DIR / "optuna_trials.csv"
TUNED_PARAMS_PATH = METRIC_DIR / "tuned_params.csv"
FEATURE_IMPORTANCE_PATH = METRIC_DIR / "feature_importances_raw.csv"
PREDICTIONS_PATH = PRED_DIR / "predictions.csv.gz"

OPTUNA_DB_PATH = CHECKPOINT_DIR / "optuna_study.db"

SEED = 42
RANDOM_SPLIT_SEEDS = [42, 43, 44, 45]
N_GROUP_FOLDS = 4
TEST_SIZE = 0.20

OPTUNA_TRIALS = 10
OPTUNA_N_JOBS = 1

MODEL_N_JOBS = 11
DTYPE_FLOAT = "float32"

SKIP_COMPLETED = True
FORCE_RERUN = False

INNER_VALIDATION_FRACTION = 0.20
INNER_VALIDATION_SEED = 941

UPTAKE_FLOOR_FOR_STEP3_NOTE_ONLY = 1e-5

MODELS = ["ridge", "rf", "lgbm"]


# =============================================================================
# UTILITIES
# =============================================================================

def pr(msg: str, level="INFO") -> None:
    print(f"[step2] [{level}] {msg}", flush=True)

def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def json_dump(obj, path: Path) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    mkdir(path.parent)
    header = not path.exists()

    if str(path).endswith(".gz"):
        mode = "at"
        with gzip.open(path, mode, encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False, header=header)
    else:
        df.to_csv(path, mode="a", index=False, header=header)

def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(DTYPE_FLOAT)

    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], downcast="integer")

    if "Crystalnet" in df.columns:
        df["Crystalnet"] = df["Crystalnet"].astype("category")

    return df

def get_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out = {}
    try: out["r2"] = float(r2_score(y_true, y_pred))
    except Exception: out["r2"] = np.nan

    try: out["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception: out["mae"] = np.nan

    try: out["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception: out["rmse"] = np.nan

    try: out["spearman"] = safe_float(spearmanr(y_true, y_pred).statistic)
    except Exception: out["spearman"] = np.nan

    try: out["pearson"] = safe_float(pearsonr(y_true, y_pred).statistic)
    except Exception: out["pearson"] = np.nan

    try: out["bias"] = float(np.mean(y_pred - y_true))
    except Exception: out["bias"] = np.nan

    out["n_test"] = int(len(y_true))
    return out

def safe_filename(name: str) -> str:
    return re.sub(r'[^\w\-]', '_', str(name))

def get_combo_id(split_name, target, family, model_name):
    return f"{safe_filename(split_name)}__{safe_filename(target)}__{safe_filename(family)}__{safe_filename(model_name)}"

# =============================================================================
# ROBUST CHECKPOINTING & SELF-HEALING
# =============================================================================

def load_completed_combo_ids() -> set[str]:
    if FORCE_RERUN or not CHECKPOINT_DIR.exists():
        return set()
    return {p.stem for p in CHECKPOINT_DIR.glob("*.pkl")}

def rebuild_master_csvs():
    """ Rebuilds master CSV files from atomic checkpoints on startup to guarantee zero file corruption. """
    if not CHECKPOINT_DIR.exists():
        return
    
    pkl_files = list(CHECKPOINT_DIR.glob("*.pkl"))
    if not pkl_files:
        return

    pr(f"Found {len(pkl_files)} successfully completed checkpoints.")
    pr("Rebuilding master CSVs from atomic checkpoints to ensure a pristine state...")

    # Wipe existing master files if rebuilding from source-of-truth checkpoints
    for p in [RAW_METRICS_PATH, OPTUNA_TRIALS_PATH, TUNED_PARAMS_PATH, FEATURE_IMPORTANCE_PATH, PREDICTIONS_PATH]:
        if p.exists():
            p.unlink()

    # Process in chunks to prevent memory blowups for massive numbers of files
    chunk_size = 50
    for i in range(0, len(pkl_files), chunk_size):
        chunk = pkl_files[i:i+chunk_size]
        metrics_list, trials_list, tuned_list, pred_list, fi_list = [], [], [], [], []

        for pkl in chunk:
            try:
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                metrics_list.append(data.get("metrics", pd.DataFrame()))
                trials_list.append(data.get("trials", pd.DataFrame()))
                tuned_list.append(data.get("tuned", pd.DataFrame()))
                pred_list.append(data.get("pred", pd.DataFrame()))
                fi_list.append(data.get("fi", pd.DataFrame()))
            except Exception as e:
                pr(f"Corrupted checkpoint detected: {pkl.name} - Deleting. Error: {e}", "WARN")
                pkl.unlink()

        if metrics_list:
            append_csv(pd.concat(metrics_list, ignore_index=True), RAW_METRICS_PATH)
        if any(not df.empty for df in trials_list):
            append_csv(pd.concat([df for df in trials_list if not df.empty], ignore_index=True), OPTUNA_TRIALS_PATH)
        if any(not df.empty for df in tuned_list):
            append_csv(pd.concat([df for df in tuned_list if not df.empty], ignore_index=True), TUNED_PARAMS_PATH)
        if any(not df.empty for df in pred_list):
            append_csv(pd.concat([df for df in pred_list if not df.empty], ignore_index=True), PREDICTIONS_PATH)
        if any(not df.empty for df in fi_list):
            append_csv(pd.concat([df for df in fi_list if not df.empty], ignore_index=True), FEATURE_IMPORTANCE_PATH)

    pr("Master CSVs strictly synchronized with checkpoints.")

# =============================================================================
# DATA LOAD
# =============================================================================

def load_inputs():
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Missing {DATA_PATH}")
    if not MANIFEST_PATH.exists(): raise FileNotFoundError(f"Missing {MANIFEST_PATH}")

    pr(f"Loading {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = optimize_dtypes(df)

    pr(f"Loading {MANIFEST_PATH.name}")
    manifest = load_json(MANIFEST_PATH)

    return df, manifest

def audit_dataset(df: pd.DataFrame, manifest: dict) -> dict:
    targets = manifest["targets"]
    families = manifest["descriptor_families"]

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_targets": int(len(targets)),
        "targets": targets,
        "descriptor_families": {
            k: {
                "n_num": len(v.get("num", [])),
                "n_cat": len(v.get("cat", [])),
            }
            for k, v in families.items()
        },
        "has_filename": "filename" in df.columns,
        "has_Crystalnet": "Crystalnet" in df.columns,
        "duplicate_filenames": int(df["filename"].duplicated().sum()) if "filename" in df.columns else None,
        "target_missing_counts": {
            t: int(df[t].isna().sum()) for t in targets if t in df.columns
        },
    }

# =============================================================================
# SPLITS
# =============================================================================

def make_random_splits(df: pd.DataFrame) -> list[dict]:
    idx = np.arange(len(df))
    splits = []
    for seed in RANDOM_SPLIT_SEEDS:
        train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, shuffle=True)
        splits.append({
            "split_type": "random", "split_name": f"random_seed_{seed}", "seed": seed,
            "train": train_idx.tolist(), "test": test_idx.tolist(),
        })
    return splits

def make_balanced_group_folds(df: pd.DataFrame, group_col: str = "Crystalnet", n_splits: int = N_GROUP_FOLDS) -> list[dict]:
    groups = df[group_col].astype(str)
    group_sizes = groups.value_counts().sort_values(ascending=False)

    fold_groups = {i: [] for i in range(n_splits)}
    fold_sizes = {i: 0 for i in range(n_splits)}

    for group, size in group_sizes.items():
        smallest = min(fold_sizes, key=fold_sizes.get)
        fold_groups[smallest].append(group)
        fold_sizes[smallest] += int(size)

    all_idx = np.arange(len(df))
    splits = []
    for fold in range(n_splits):
        test_groups = set(fold_groups[fold])
        test_mask = groups.isin(test_groups).to_numpy()
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]

        splits.append({
            "split_type": "balanced_topology_group", "split_name": f"balanced_group_fold_{fold}",
            "group_col": group_col, "train": train_idx.tolist(), "test": test_idx.tolist(),
            "test_groups": sorted(test_groups), "n_test_groups": len(test_groups),
        })
    return splits

def audit_splits(df: pd.DataFrame, splits: list[dict], group_col: str = "Crystalnet"):
    rows = []
    warnings_rows = []

    all_groups = df[group_col].astype(str)
    dataset_group_counts = all_groups.value_counts()
    ideal_fold_fraction = 1.0 / N_GROUP_FOLDS

    for split in splits:
        train_groups = df.iloc[split["train"]][group_col].astype(str)
        test_groups = df.iloc[split["test"]][group_col].astype(str)
        overlap_groups = set(train_groups.unique()) & set(test_groups.unique())

        rows.append({"split_name": split["split_name"], "split_type": split["split_type"], "n_train": len(split["train"]), "n_test": len(split["test"])})

        if split["split_type"] != "random" and len(overlap_groups) > 0:
            warnings_rows.append({"split_name": split["split_name"], "warning": "GROUP_LEAKAGE", "message": "Groups leak."})

    return pd.DataFrame(rows), pd.DataFrame(warnings_rows)

def save_splits(splits: list[dict]) -> None:
    mkdir(SPLIT_DIR)
    for split in splits:
        json_dump(split, SPLIT_DIR / f"{split['split_name']}.json")

def make_or_load_splits(df: pd.DataFrame) -> list[dict]:
    splits = []
    need_make = False

    expected_names = [f"random_seed_{s}" for s in RANDOM_SPLIT_SEEDS] + [f"balanced_group_fold_{i}" for i in range(N_GROUP_FOLDS)]
    for name in expected_names:
        if not (SPLIT_DIR / f"{name}.json").exists():
            need_make = True; break

    if need_make or FORCE_RERUN:
        splits = make_random_splits(df) + make_balanced_group_folds(df)
        save_splits(splits)
    else:
        for name in expected_names: splits.append(load_json(SPLIT_DIR / f"{name}.json"))

    audit, warnings_df = audit_splits(df, splits)
    audit.to_csv(GLOBAL_DIR / "split_audit.csv", index=False)
    warnings_df.to_csv(GLOBAL_DIR / "split_audit_warnings.csv", index=False)
    return splits

# =============================================================================
# PREPROCESSING
# =============================================================================

def get_family_columns(manifest: dict, family: str):
    spec = manifest["descriptor_families"][family]
    return list(spec.get("num", [])), list(spec.get("cat", []))

def build_sklearn_pipeline(model_name: str, numeric_cols: list[str], categorical_cols: list[str], params: dict):
    transformers = []
    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "ridge": num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    if categorical_cols:
        transformers.append(("cat", get_onehot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)

    if model_name == "ridge": estimator = Ridge(**params)
    elif model_name == "rf": estimator = RandomForestRegressor(random_state=SEED, n_jobs=MODEL_N_JOBS, **params)
    else: raise ValueError(model_name)

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

def prepare_lgbm_data(df: pd.DataFrame, train_idx: list[int], test_idx: list[int], numeric_cols: list[str], categorical_cols: list[str]):
    cols = numeric_cols + categorical_cols
    X_train, X_test = df.iloc[train_idx][cols].copy(), df.iloc[test_idx][cols].copy()

    for c in numeric_cols:
        X_train[c] = pd.to_numeric(X_train[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        med = X_train[c].median()
        X_train[c] = X_train[c].fillna(med).astype(np.float32)
        X_test[c] = X_test[c].fillna(med).astype(np.float32)

    for c in categorical_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype(pd.CategoricalDtype(categories=X_train[c].cat.categories))

    return X_train, X_test

# =============================================================================
# HYPERPARAMETER SPACES
# =============================================================================

def sample_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "ridge": return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if model_name == "rf": return {"n_estimators": trial.suggest_categorical("n_estimators", [100, 200]), "max_depth": trial.suggest_categorical("max_depth", [10, 14, 18]), "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2, 4, 8]), "max_features": trial.suggest_categorical("max_features", [0.5, 0.75, 1.0])}
    if model_name == "lgbm": return {"n_estimators": trial.suggest_categorical("n_estimators", [300, 500, 800]), "learning_rate": trial.suggest_categorical("learning_rate", [0.03, 0.05, 0.08]), "num_leaves": trial.suggest_categorical("num_leaves", [31, 63]), "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10]), "min_child_samples": trial.suggest_categorical("min_child_samples", [50, 100, 200]), "subsample": trial.suggest_categorical("subsample", [0.8, 1.0]), "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.8, 1.0]), "reg_lambda": trial.suggest_categorical("reg_lambda", [0.1, 1.0, 5.0])}
    raise ValueError(model_name)

def default_params(model_name: str) -> dict:
    if model_name == "ridge": return {"alpha": 1.0}
    if model_name == "rf": return {"n_estimators": 100, "max_depth": 18, "min_samples_leaf": 2, "max_features": 0.75}
    if model_name == "lgbm": return {"objective": "regression", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8, "min_child_samples": 100, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": SEED, "n_jobs": MODEL_N_JOBS, "verbosity": -1}

# =============================================================================
# TUNING
# =============================================================================

def tune_model(
    df: pd.DataFrame, target: str, target_meta: dict, family: str, model_name: str,
    numeric_cols: list[str], categorical_cols: list[str], train_idx: list[int], split: dict, combo_id: str
):
    inner_train_pos, inner_val_pos = train_test_split(np.arange(len(train_idx)), test_size=INNER_VALIDATION_FRACTION, random_state=INNER_VALIDATION_SEED, shuffle=True)
    inner_train_idx = [train_idx[i] for i in inner_train_pos]
    inner_val_idx = [train_idx[i] for i in inner_val_pos]

    y_train_inner = df.iloc[inner_train_idx][target].astype(float).to_numpy()
    y_val_inner = df.iloc[inner_val_idx][target].astype(float).to_numpy()

    def objective(trial):
        params = sample_params(trial, model_name)
        try:
            if model_name in ["ridge", "rf"]:
                pipe = build_sklearn_pipeline(model_name, numeric_cols, categorical_cols, params)
                X_train, X_val = df.iloc[inner_train_idx][numeric_cols + categorical_cols], df.iloc[inner_val_idx][numeric_cols + categorical_cols]
                pipe.fit(X_train, y_train_inner)
                pred = pipe.predict(X_val)
            elif model_name == "lgbm":
                X_train, X_val = prepare_lgbm_data(df, inner_train_idx, inner_val_idx, numeric_cols, categorical_cols)
                full_params = default_params("lgbm")
                full_params.update(params)
                model = LGBMRegressor(**full_params)
                model.fit(X_train, y_train_inner, categorical_feature=categorical_cols if categorical_cols else "auto")
                pred = model.predict(X_val)

            return float(r2_score(y_val_inner, pred))
        except Exception:
            trial.set_user_attr("error", traceback.format_exc())
            return -1e12

    try:
        import sqlalchemy
        db_path_str = str(OPTUNA_DB_PATH.absolute()).replace('\\', '/')
        storage = f"sqlite:///{db_path_str}"
    except ImportError:
        storage = None
        pr("SQLAlchemy not installed. Trial-level resuming disabled (falling back to memory storage).", "WARN")

    study = optuna.create_study(
        study_name=combo_id,
        storage=storage,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=NopPruner(),
        load_if_exists=True
    )

    # Only count fully completed trials so interrupted ones don't consume the budget
    valid_states = [TrialState.COMPLETE, TrialState.PRUNED]
    valid_trials = [t for t in study.trials if t.state in valid_states]
    
    trials_run_so_far = len(valid_trials)
    n_trials_to_run = OPTUNA_TRIALS - trials_run_so_far
    
    if trials_run_so_far > 0 and n_trials_to_run > 0:
        pr(f"    -> Resuming Optuna tuning: {trials_run_so_far}/{OPTUNA_TRIALS} valid trials found in DB. Running {n_trials_to_run} more.")
    
    if n_trials_to_run > 0:
        study.optimize(objective, n_trials=n_trials_to_run, n_jobs=OPTUNA_N_JOBS, show_progress_bar=False)

    trial_rows = []
    for tr in study.trials:
        trial_rows.append({
            "target": target, "layer": target_meta.get("layer"), "process": target_meta.get("process"),
            "split_type": split["split_type"], "split_name": split["split_name"], "descriptor_family": family,
            "model": model_name, "trial_number": tr.number, "trial_value": safe_float(tr.value),
            "trial_params": json.dumps(tr.params, default=str), "state": str(tr.state), "error": tr.user_attrs.get("error", ""),
        })

    trials_df = pd.DataFrame(trial_rows)
    if study.best_value <= -1e11:
        best_params, best_score, status = default_params(model_name), None, "failed_all_trials_default_used"
    else:
        best_params, best_score, status = dict(study.best_params), safe_float(study.best_value), "ok"

    if model_name == "lgbm":
        full = default_params("lgbm")
        full.update(best_params)
        best_params = full

    tuned_row = pd.DataFrame([{
        "target": target, "layer": target_meta.get("layer"), "process": target_meta.get("process"),
        "split_type": split["split_type"], "split_name": split["split_name"], "descriptor_family": family,
        "model": model_name, "best_cv_score": best_score, "best_params": json.dumps(best_params, default=str),
        "n_trials": len(study.trials), "status": status,
    }])
    return best_params, trials_df, tuned_row


# =============================================================================
# FIT / PREDICT / IMPORTANCE
# =============================================================================

def get_importances(model, names, type_val, target, meta, family, split, model_name):
    rows = []
    try:
        for name, val in zip(names, model.feature_importances_ if hasattr(model, "feature_importances_") else np.abs(model.coef_)):
            rows.append({
                "target": target, "layer": meta.get("layer"), "process": meta.get("process"),
                "descriptor_family": family, "model": model_name, "split_type": split["split_type"],
                "split_name": split["split_name"], "feature_name": name, "importance_type": type_val, "importance_value": safe_float(val),
            })
    except Exception: pass
    return pd.DataFrame(rows)

def fit_predict_one(
    df: pd.DataFrame, target: str, target_meta: dict, family: str, model_name: str,
    numeric_cols: list[str], categorical_cols: list[str], split: dict, params: dict
):
    train_idx, test_idx = split["train"], split["test"]
    y_train, y_test = df.iloc[train_idx][target].astype(float).to_numpy(), df.iloc[test_idx][target].astype(float).to_numpy()

    t0 = time.time()
    if model_name in ["ridge", "rf"]:
        m_params = dict(params)
        if model_name == "rf":
            m_params.pop("random_state", None)
            m_params.pop("n_jobs", None)
        pipe = build_sklearn_pipeline(model_name, numeric_cols, categorical_cols, m_params)
        X_train, X_test = df.iloc[train_idx][numeric_cols + categorical_cols], df.iloc[test_idx][numeric_cols + categorical_cols]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        fi_df = get_importances(
            pipe.named_steps["model"], list(pipe.named_steps["preprocessor"].get_feature_names_out()),
            "coefficient_abs" if model_name == "ridge" else "mdi", target, target_meta, family, split, model_name
        )
    elif model_name == "lgbm":
        X_train, X_test = prepare_lgbm_data(df, train_idx, test_idx, numeric_cols, categorical_cols)
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, categorical_feature=categorical_cols if categorical_cols else "auto")
        y_pred = model.predict(X_test)
        fi_df = get_importances(model, list(X_train.columns), "split", target, target_meta, family, split, model_name)

    elapsed = time.time() - t0
    metrics = compute_metrics(y_test, y_pred)

    metric_row = {
        "target": target, "layer": target_meta.get("layer"), "process": target_meta.get("process"),
        "gas": target_meta.get("gas"), "gas_pair": target_meta.get("gas_pair"), "unit": target_meta.get("unit"),
        "target_transform": target_meta.get("transform", "none"), "split_type": split["split_type"],
        "split_name": split["split_name"], "descriptor_family": family, "model": model_name,
        "n_train": len(train_idx), "n_test": len(test_idx), "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols), "params": json.dumps(params, default=str),
        "elapsed_s": round(elapsed, 3), "status": "ok", "error": "",
    }
    metric_row.update(metrics)

    pred_df = pd.DataFrame({
        "target": target, "layer": target_meta.get("layer"), "process": target_meta.get("process"),
        "gas": target_meta.get("gas"), "gas_pair": target_meta.get("gas_pair"), "unit": target_meta.get("unit"),
        "target_transform": target_meta.get("transform", "none"), "split_type": split["split_type"],
        "split_name": split["split_name"], "descriptor_family": family, "model": model_name,
        "filename": df.iloc[test_idx]["filename"].values, "Crystalnet": df.iloc[test_idx]["Crystalnet"].astype(str).values,
        "y_true": y_test, "y_pred": y_pred, "residual": y_pred - y_test, "abs_error": np.abs(y_pred - y_test),
    })

    return pd.DataFrame([metric_row]), pred_df, fi_df

# =============================================================================
# RUN CONFIG
# =============================================================================

def write_global_outputs(df: pd.DataFrame, manifest: dict, splits: list[dict]) -> None:
    run_config = {
        "version": "v9_step2_model_training_resilient",
        "purpose": "controlled descriptor adequacy benchmark",
        "models": MODELS, "optuna_trials": OPTUNA_TRIALS, "optuna_n_jobs": OPTUNA_N_JOBS,
        "model_n_jobs": MODEL_N_JOBS, "skip_completed": SKIP_COMPLETED, "force_rerun": FORCE_RERUN,
        "sklearnex_requested": USE_SKLEARNEX, "sklearnex_enabled": SKLEARNEX_ENABLED,
    }
    json_dump(run_config, GLOBAL_DIR / "run_config.json")
    json_dump(audit_dataset(df, manifest), GLOBAL_DIR / "dataset_audit.json")

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    try:
        start = time.time()
        mkdir(OUT); mkdir(GLOBAL_DIR); mkdir(METRIC_DIR); mkdir(PRED_DIR); mkdir(SPLIT_DIR); mkdir(CHECKPOINT_DIR)

        if FORCE_RERUN:
            pr("FORCE_RERUN is active. Clearing out old checkpoints...", "WARN")
            shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
            mkdir(CHECKPOINT_DIR)

        pr("=" * 100)
        pr("STEP 2 V9: PROCESS-AWARE MODEL TRAINING (RESILIENT)")
        pr("=" * 100)

        df, manifest = load_inputs()
        if "filename" not in df.columns or "Crystalnet" not in df.columns:
            raise RuntimeError("revision_data.csv must contain 'filename' and 'Crystalnet'.")

        targets = list(manifest["targets"])
        target_metadata = manifest.get("target_metadata", {})
        families = list(manifest["descriptor_families"].keys())

        splits = make_or_load_splits(df)
        write_global_outputs(df, manifest, splits)

        # 1) Auto-Heal CSVs from strictly verified checkpoint sources.
        rebuild_master_csvs()

        # 2) Extract IDs of completed combinations.
        completed_ids = load_completed_combo_ids()

        total = len(splits) * len(targets) * len(families) * len(MODELS)
        current_step = 0
        skipped_counter = 0

        pr(f"Total Combinations: {total:,} | Optuna Trials: {OPTUNA_TRIALS} per combo")
        
        for split in splits:
            for target in targets:
                meta = target_metadata.get(target, {})
                for family in families:
                    num_cols, cat_cols = get_family_columns(manifest, family)
                    num_cols, cat_cols = [c for c in num_cols if c in df.columns], [c for c in cat_cols if c in df.columns]

                    if not num_cols and not cat_cols:
                        continue

                    for model_name in MODELS:
                        current_step += 1
                        combo_id = get_combo_id(split["split_name"], target, family, model_name)
                        
                        if SKIP_COMPLETED and combo_id in completed_ids:
                            skipped_counter += 1
                            continue
                        
                        pr(f"[{current_step:,}/{total:,}] Processing: {split['split_name']} | {target} | {family} | {model_name}")

                        try:
                            params, trials_df, tuned_df = tune_model(
                                df, target, meta, family, model_name, num_cols, cat_cols, split["train"], split, combo_id
                            )

                            metrics_df, pred_df, fi_df = fit_predict_one(
                                df, target, meta, family, model_name, num_cols, cat_cols, split, params
                            )
                            
                            # --- ATOMIC CHECKPOINT SAVE ---
                            # Guarantees no half-written or corrupted files on crash.
                            data_to_save = {
                                "metrics": metrics_df, "trials": trials_df, "tuned": tuned_df, "pred": pred_df, "fi": fi_df
                            }
                            tmp_pkl = CHECKPOINT_DIR / f"{combo_id}.pkl.tmp"
                            with open(tmp_pkl, "wb") as f:
                                pickle.dump(data_to_save, f)
                            tmp_pkl.replace(CHECKPOINT_DIR / f"{combo_id}.pkl")

                            # --- LIVE APPENDING ---
                            # It is now perfectly safe to append; if interrupted during write, it heals next boot!
                            append_csv(metrics_df, RAW_METRICS_PATH)
                            append_csv(trials_df, OPTUNA_TRIALS_PATH)
                            append_csv(tuned_df, TUNED_PARAMS_PATH)
                            append_csv(pred_df, PREDICTIONS_PATH)
                            append_csv(fi_df, FEATURE_IMPORTANCE_PATH)
                            
                            completed_ids.add(combo_id)

                            m = metrics_df.iloc[0]
                            pr(f"  --> OK R2={m['r2']:.4f} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} elapsed={m['elapsed_s']}s")

                        except Exception as e:
                            err = traceback.format_exc()
                            pr(f"  --> FAILED: {e}", "WARN")

                            fail_row = pd.DataFrame([{
                                "target": target, "layer": meta.get("layer"), "process": meta.get("process"),
                                "split_type": split["split_type"], "split_name": split["split_name"],
                                "descriptor_family": family, "model": model_name, "status": "failed", "error": err
                            }])
                            fail_data = {
                                "metrics": fail_row, "trials": pd.DataFrame(), "tuned": pd.DataFrame(), "pred": pd.DataFrame(), "fi": pd.DataFrame()
                            }

                            # Save the failed state atomically so it isn't infinitely restarted
                            tmp_pkl = CHECKPOINT_DIR / f"{combo_id}.pkl.tmp"
                            with open(tmp_pkl, "wb") as f: pickle.dump(fail_data, f)
                            tmp_pkl.replace(CHECKPOINT_DIR / f"{combo_id}.pkl")
                            
                            append_csv(fail_row, RAW_METRICS_PATH)
                            completed_ids.add(combo_id)

        elapsed = time.time() - start
        json_dump({"status": "complete", "elapsed_h": round(elapsed / 3600, 3)}, GLOBAL_DIR / "step2_final_report.json")
        
        pr("=" * 100)
        pr(f"STEP 2 COMPLETE in {elapsed / 3600:.2f} h")
        pr(f"Skipped existing combinations: {skipped_counter:,}")
        pr("=" * 100)

    except KeyboardInterrupt:
        pr("\n" + "=" * 100, "WARN")
        pr("KEYBOARD INTERRUPT DETECTED (Ctrl+C)", "WARN")
        pr("Gracefully shutting down. All completed combinations are safely saved.", "WARN")
        pr("The interrupted Optuna trials have been preserved in the database.", "WARN")
        pr("Restart the script to resume exactly where you left off.", "WARN")
        pr("=" * 100, "WARN")


if __name__ == "__main__":
    main()