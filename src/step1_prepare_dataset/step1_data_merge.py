#!/usr/bin/env python3
"""
step1_data_merge.py
===================

V9 process-aware ARC-MOF data builder.

Purpose
-------
Build a reviewer-transparent benchmark dataset from ARC-MOF tabular files.

This version:
  - Uses all five ARC-MOF gas-process families:
      1. post-combustion VSA        CO2/N2
      2. pre-combustion PSA         CO2/H2
      3. natural gas purification   CO2/CH4
      4. landfill gas VPSA          CO2/CH4
      5. methane storage PSA        CH4 only

  - Constructs a physically organized target hierarchy:
      Layer A: primitive raw uptake targets, mmol/g
      Layer B: process working capacities, mmol/g_working_capacity
      Layer C: direct process log-selectivity, ln(S)

  - Does NOT train models.
  - Does NOT compute post-hoc selectivity; that belongs in Step 3.
  - Does NOT include RDFs, CIF archives, clusters, or ARC-MOF_Dim as active features.

Required files in same folder:
  geometric_properties.csv
  RACs.csv
  all_topology_lists.csv

  methane.csv
  post_comb_vsa-CO2.csv
  post_comb_vsa-N2.csv
  pre_comb_4040-CO2.csv
  pre_comb_4040-H2.csv
  methane_purification-CO2.csv   or methane_purification_CO2.csv
  methane_purification-CH4.csv
  landfill-CO2.csv
  landfill-CH4.csv
  overall_process.csv

Outputs:
  revision_data.csv
  column_manifest.json
  adsorption_file_inventory.csv
  target_coverage_before_strict.csv
  target_coverage_strict.csv
  merge_report.json
  raw_file_md5_manifest.csv
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

ROOT = Path(__file__).resolve().parent
ID = "filename"
EPS = 1e-8

RAC_VARIANCE_THRESHOLD = 0.01

# If True, rows must have all active targets, known topology, and RACs.
# This gives a common cohort for all descriptor-family comparisons.
STRICT_COMMON_COHORT = True


# =============================================================================
# FILE RESOLUTION
# =============================================================================

def resolve_file(*names: str) -> Path:
    """
    Return the first existing file among candidate names.
    If none exist, return the first candidate path so require_files() reports it.
    """
    for name in names:
        path = ROOT / name
        if path.exists():
            return path
    return ROOT / names[0]


FILES = {
    # descriptors / metadata
    "geometry": ROOT / "geometric_properties.csv",
    "racs": ROOT / "RACs.csv",
    "topology": ROOT / "all_topology_lists.csv",

    # raw uptake files
    "methane": ROOT / "methane.csv",

    "postcomb_co2": ROOT / "post_comb_vsa-CO2.csv",
    "postcomb_n2": ROOT / "post_comb_vsa-N2.csv",

    "precomb_co2": ROOT / "pre_comb_4040-CO2.csv",
    "precomb_h2": ROOT / "pre_comb_4040-H2.csv",

    "ngp_co2": resolve_file(
        "methane_purification-CO2.csv",
        "methane_purification_CO2.csv",
    ),
    "ngp_ch4": ROOT / "methane_purification-CH4.csv",

    "landfill_co2": ROOT / "landfill-CO2.csv",
    "landfill_ch4": ROOT / "landfill-CH4.csv",

    # process-level outputs
    "process": ROOT / "overall_process.csv",
}


ADSORPTION_FILE_KEYS = [
    "methane",
    "postcomb_co2",
    "postcomb_n2",
    "precomb_co2",
    "precomb_h2",
    "ngp_co2",
    "ngp_ch4",
    "landfill_co2",
    "landfill_ch4",
    "process",
]


# =============================================================================
# TARGET NAMES
# =============================================================================

# -------------------------
# Layer A: raw uptake
# -------------------------

T_CH4_5P8 = "CH4_5.8bar"
T_CH4_65 = "CH4_65bar"

T_POST_CO2_0015 = "postcomb_CO2_0.015bar"
T_POST_CO2_015 = "postcomb_CO2_0.15bar"
T_POST_N2_0075 = "postcomb_N2_0.075bar"
T_POST_N2_075 = "postcomb_N2_0.75bar"

T_PRE_CO2_04 = "precomb_CO2_0.4bar"
T_PRE_CO2_16 = "precomb_CO2_16bar"
T_PRE_H2_06 = "precomb_H2_0.6bar"
T_PRE_H2_24 = "precomb_H2_24bar"

T_NGP_CO2_01 = "ngp_CO2_0.1bar"
T_NGP_CO2_1 = "ngp_CO2_1.0bar"
T_NGP_CH4_09 = "ngp_CH4_0.9bar"
T_NGP_CH4_9 = "ngp_CH4_9.0bar"

T_LF_CO2_026 = "landfill_CO2_0.26bar"
T_LF_CO2_32 = "landfill_CO2_3.2bar"
T_LF_CH4_001 = "landfill_CH4_0.01bar"
T_LF_CH4_44 = "landfill_CH4_4.4bar"

LAYER_A_TARGETS = [
    T_CH4_5P8,
    T_CH4_65,

    T_POST_CO2_0015,
    T_POST_CO2_015,
    T_POST_N2_0075,
    T_POST_N2_075,

    T_PRE_CO2_04,
    T_PRE_CO2_16,
    T_PRE_H2_06,
    T_PRE_H2_24,

    T_NGP_CO2_01,
    T_NGP_CO2_1,
    T_NGP_CH4_09,
    T_NGP_CH4_9,

    T_LF_CO2_026,
    T_LF_CO2_32,
    T_LF_CH4_001,
    T_LF_CH4_44,
]


# -------------------------
# Layer B: working capacity
# -------------------------

T_POST_WC = "postcomb_working_capacity"
T_PRE_WC = "precomb_working_capacity"
T_NGP_WC = "ngp_working_capacity"
T_LF_WC = "landfill_working_capacity"
T_CH4_STORAGE_WC = "methane_storage_working_capacity"

LAYER_B_TARGETS = [
    T_POST_WC,
    T_PRE_WC,
    T_NGP_WC,
    T_LF_WC,
    T_CH4_STORAGE_WC,
]


# -------------------------
# Layer C: direct log-selectivity
# -------------------------

T_POST_LOGSEL = "postcomb_log_selectivity"
T_PRE_LOGSEL = "precomb_log_selectivity"
T_NGP_LOGSEL = "ngp_log_selectivity"
T_LF_LOGSEL = "landfill_log_selectivity"

LAYER_C_TARGETS = [
    T_POST_LOGSEL,
    T_PRE_LOGSEL,
    T_NGP_LOGSEL,
    T_LF_LOGSEL,
]


ACTIVE_TARGETS = LAYER_A_TARGETS + LAYER_B_TARGETS + LAYER_C_TARGETS


# =============================================================================
# PROCESS DEFINITIONS
# =============================================================================

PROCESS_ROWS = {
    "post_combustion_vsa": "post-combustion-vsa",
    "pre_combustion_psa": "pre-combustion-40-40",
    "natural_gas_purification": "natural-gas-purification",
    "landfill_gas_vpsa": "landfill-gas-vpsa",
    "methane_storage_psa": "methane-storage-psa",
}


RAW_UPTAKE_SPECS = [
    # methane storage, pure CH4
    {
        "target": T_CH4_5P8,
        "file_key": "methane",
        "process": "methane_storage_psa",
        "gas": "CH4",
        "pressure_bar": 5.8,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_CH4_65,
        "file_key": "methane",
        "process": "methane_storage_psa",
        "gas": "CH4",
        "pressure_bar": 65.0,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },

    # post-combustion VSA
    {
        "target": T_POST_CO2_0015,
        "file_key": "postcomb_co2",
        "process": "post_combustion_vsa",
        "gas": "CO2",
        "pressure_bar": 0.015,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_POST_CO2_015,
        "file_key": "postcomb_co2",
        "process": "post_combustion_vsa",
        "gas": "CO2",
        "pressure_bar": 0.150,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },
    {
        "target": T_POST_N2_0075,
        "file_key": "postcomb_n2",
        "process": "post_combustion_vsa",
        "gas": "N2",
        "pressure_bar": 0.075,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_POST_N2_075,
        "file_key": "postcomb_n2",
        "process": "post_combustion_vsa",
        "gas": "N2",
        "pressure_bar": 0.750,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },

    # pre-combustion PSA
    {
        "target": T_PRE_CO2_04,
        "file_key": "precomb_co2",
        "process": "pre_combustion_psa",
        "gas": "CO2",
        "pressure_bar": 0.4,
        "temperature_K": 313.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_PRE_CO2_16,
        "file_key": "precomb_co2",
        "process": "pre_combustion_psa",
        "gas": "CO2",
        "pressure_bar": 16.0,
        "temperature_K": 313.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },
    {
        "target": T_PRE_H2_06,
        "file_key": "precomb_h2",
        "process": "pre_combustion_psa",
        "gas": "H2",
        "pressure_bar": 0.6,
        "temperature_K": 313.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_PRE_H2_24,
        "file_key": "precomb_h2",
        "process": "pre_combustion_psa",
        "gas": "H2",
        "pressure_bar": 24.0,
        "temperature_K": 313.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },

    # natural gas purification
    {
        "target": T_NGP_CO2_01,
        "file_key": "ngp_co2",
        "process": "natural_gas_purification",
        "gas": "CO2",
        "pressure_bar": 0.1,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_NGP_CO2_1,
        "file_key": "ngp_co2",
        "process": "natural_gas_purification",
        "gas": "CO2",
        "pressure_bar": 1.0,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },
    {
        "target": T_NGP_CH4_09,
        "file_key": "ngp_ch4",
        "process": "natural_gas_purification",
        "gas": "CH4",
        "pressure_bar": 0.9,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_NGP_CH4_9,
        "file_key": "ngp_ch4",
        "process": "natural_gas_purification",
        "gas": "CH4",
        "pressure_bar": 9.0,
        "temperature_K": 298.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },

    # landfill gas VPSA
    {
        "target": T_LF_CO2_026,
        "file_key": "landfill_co2",
        "process": "landfill_gas_vpsa",
        "gas": "CO2",
        "pressure_bar": 0.26,
        "temperature_K": 338.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_LF_CO2_32,
        "file_key": "landfill_co2",
        "process": "landfill_gas_vpsa",
        "gas": "CO2",
        "pressure_bar": 3.20,
        "temperature_K": 338.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },
    {
        "target": T_LF_CH4_001,
        "file_key": "landfill_ch4",
        "process": "landfill_gas_vpsa",
        "gas": "CH4",
        "pressure_bar": 0.01,
        "temperature_K": 338.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "desorption_state",
    },
    {
        "target": T_LF_CH4_44,
        "file_key": "landfill_ch4",
        "process": "landfill_gas_vpsa",
        "gas": "CH4",
        "pressure_bar": 4.40,
        "temperature_K": 338.0,
        "unit": "mmol/g",
        "layer": "A_raw_uptake",
        "role": "adsorption_state",
    },
]


WORKING_CAPACITY_SPECS = [
    {
        "target": T_POST_WC,
        "process": "post_combustion_vsa",
        "process_row": PROCESS_ROWS["post_combustion_vsa"],
        "source_column": "mmol/g_working_capacity",
        "unit": "mmol/g",
        "layer": "B_working_capacity",
        "q_ads_target": T_POST_CO2_015,
        "q_des_target": T_POST_CO2_0015,
    },
    {
        "target": T_PRE_WC,
        "process": "pre_combustion_psa",
        "process_row": PROCESS_ROWS["pre_combustion_psa"],
        "source_column": "mmol/g_working_capacity",
        "unit": "mmol/g",
        "layer": "B_working_capacity",
        "q_ads_target": T_PRE_CO2_16,
        "q_des_target": T_PRE_CO2_04,
    },
    {
        "target": T_NGP_WC,
        "process": "natural_gas_purification",
        "process_row": PROCESS_ROWS["natural_gas_purification"],
        "source_column": "mmol/g_working_capacity",
        "unit": "mmol/g",
        "layer": "B_working_capacity",
        "q_ads_target": T_NGP_CO2_1,
        "q_des_target": T_NGP_CO2_01,
    },
    {
        "target": T_LF_WC,
        "process": "landfill_gas_vpsa",
        "process_row": PROCESS_ROWS["landfill_gas_vpsa"],
        "source_column": "mmol/g_working_capacity",
        "unit": "mmol/g",
        "layer": "B_working_capacity",
        "q_ads_target": T_LF_CO2_32,
        "q_des_target": T_LF_CO2_026,
    },
    {
        "target": T_CH4_STORAGE_WC,
        "process": "methane_storage_psa",
        "process_row": PROCESS_ROWS["methane_storage_psa"],
        "source_column": "mmol/g_working_capacity",
        "unit": "mmol/g",
        "layer": "B_working_capacity",
        "q_ads_target": T_CH4_65,
        "q_des_target": T_CH4_5P8,
    },
]


LOG_SELECTIVITY_SPECS = [
    {
        "target": T_POST_LOGSEL,
        "process": "post_combustion_vsa",
        "process_row": PROCESS_ROWS["post_combustion_vsa"],
        "source_column": "selectivity",
        "unit": "ln(selectivity)",
        "layer": "C_direct_log_selectivity",
        "gas_pair": "CO2/N2",
        "numerator_gas": "CO2",
        "denominator_gas": "N2",
        "feed_y_numerator": 0.17,
        "feed_y_denominator": 0.83,
        "transform": "log",
    },
    {
        "target": T_PRE_LOGSEL,
        "process": "pre_combustion_psa",
        "process_row": PROCESS_ROWS["pre_combustion_psa"],
        "source_column": "selectivity",
        "unit": "ln(selectivity)",
        "layer": "C_direct_log_selectivity",
        "gas_pair": "CO2/H2",
        "numerator_gas": "CO2",
        "denominator_gas": "H2",
        "feed_y_numerator": 0.40,
        "feed_y_denominator": 0.60,
        "transform": "log",
    },
    {
        "target": T_NGP_LOGSEL,
        "process": "natural_gas_purification",
        "process_row": PROCESS_ROWS["natural_gas_purification"],
        "source_column": "selectivity",
        "unit": "ln(selectivity)",
        "layer": "C_direct_log_selectivity",
        "gas_pair": "CO2/CH4",
        "numerator_gas": "CO2",
        "denominator_gas": "CH4",
        "feed_y_numerator": 0.10,
        "feed_y_denominator": 0.90,
        "transform": "log",
    },
    {
        "target": T_LF_LOGSEL,
        "process": "landfill_gas_vpsa",
        "process_row": PROCESS_ROWS["landfill_gas_vpsa"],
        "source_column": "selectivity",
        "unit": "ln(selectivity)",
        "layer": "C_direct_log_selectivity",
        "gas_pair": "CO2/CH4",
        "numerator_gas": "CO2",
        "denominator_gas": "CH4",
        "feed_y_numerator": None,
        "feed_y_denominator": None,
        "composition_source": "dynamic_from_partial_pressures",
        "transform": "log",
    },
]


POSTHOC_SELECTIVITY_DEFINITIONS = {
    "postcomb_posthoc_log_selectivity": {
        "process": "post_combustion_vsa",
        "direct_log_selectivity_target": T_POST_LOGSEL,
        "numerator_uptake_target": T_POST_CO2_015,
        "denominator_uptake_target": T_POST_N2_075,
        "numerator_gas": "CO2",
        "denominator_gas": "N2",
        "feed_y_numerator": 0.17,
        "feed_y_denominator": 0.83,
        "composition_factor": 0.83 / 0.17,
        "composition_source": "SI_fixed_feed_fraction",
        "formula": "ln((q_num/q_den)*(y_den/y_num))",
    },
    "precomb_posthoc_log_selectivity": {
        "process": "pre_combustion_psa",
        "direct_log_selectivity_target": T_PRE_LOGSEL,
        "numerator_uptake_target": T_PRE_CO2_16,
        "denominator_uptake_target": T_PRE_H2_24,
        "numerator_gas": "CO2",
        "denominator_gas": "H2",
        "feed_y_numerator": 0.40,
        "feed_y_denominator": 0.60,
        "composition_factor": 0.60 / 0.40,
        "composition_source": "SI_fixed_feed_fraction",
        "formula": "ln((q_num/q_den)*(y_den/y_num))",
    },
    "ngp_posthoc_log_selectivity": {
        "process": "natural_gas_purification",
        "direct_log_selectivity_target": T_NGP_LOGSEL,
        "numerator_uptake_target": T_NGP_CO2_1,
        "denominator_uptake_target": T_NGP_CH4_9,
        "numerator_gas": "CO2",
        "denominator_gas": "CH4",
        "feed_y_numerator": 0.10,
        "feed_y_denominator": 0.90,
        "composition_factor": 0.90 / 0.10,
        "composition_source": "SI_fixed_feed_fraction",
        "formula": "ln((q_num/q_den)*(y_den/y_num))",
    },
    "landfill_posthoc_log_selectivity": {
        "process": "landfill_gas_vpsa",
        "direct_log_selectivity_target": T_LF_LOGSEL,
        "numerator_uptake_target": T_LF_CO2_32,
        "denominator_uptake_target": T_LF_CH4_44,
        "numerator_gas": "CO2",
        "denominator_gas": "CH4",
        "feed_y_numerator": None,
        "feed_y_denominator": None,
        "composition_factor": 4.40 / 3.20,
        "composition_source": "dynamic_from_raw_partial_pressures_high_pressure_state",
        "formula": "ln((q_num/q_den)*(p_den/p_num))",
    },
}


# =============================================================================
# DESCRIPTOR COLUMNS
# =============================================================================

GEO_CORE = [
    "Density",
    "AVAf",
    "AVA",
    "ASA",
    "Df",
    "Di",
]

BASE_EXTRA = [
    "UC_volume",
    "vASA",
    "NASA",
    "POAVA",
    "Dif",
]

ALL_EXTRA_GEO = [
    "gASA",
    "gNASA",
    "vNASA",
    "AVAg",
    "NAVA",
    "NAVAf",
    "NAVAg",
    "POAVAf",
    "POAVAg",
    "NPOAVA",
    "NPOAVAf",
    "NPOAVAg",
]

ENGINEERED = [
    "lcd_pld_ratio",
    "cavity_window_gap",
    "sa_pv_ratio",
    "vf_density_ratio",
    "log_pld_plus1",
    "log_lcd_plus1",
]

RAC_META = {
    "Unnamed: 0",
    "filename",
    "ARC_MOF",
    "ARC-MOF",
    "DB_num",
    "order_f-lig",
    "bool_f-lig",
    "order_mc",
    "bool_mc",
    "order_func",
    "bool_func",
    "order_lc",
    "bool_lc",
}

METADATA_COLS_NOT_FEATURES = [
    "ARC-MOF",
    "ARC_MOF",
    "DB_num",
    "order_geo",
    "bool_geo",
    "has_topology",
    "has_RAC",
]


# =============================================================================
# UTILITIES
# =============================================================================

def pr(msg: str) -> None:
    print(f"[step1] {msg}", flush=True)


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_md5_manifest() -> None:
    rows = []

    for logical_name, path in FILES.items():
        rows.append({
            "logical_name": logical_name,
            "file": path.name,
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else None,
            "md5": md5sum(path) if path.exists() else "MISSING",
        })

    pd.DataFrame(rows).to_csv(ROOT / "raw_file_md5_manifest.csv", index=False)


def require_files() -> None:
    missing = [path.name for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def clean_filename_value(x):
    if pd.isna(x):
        return x

    s = str(x).strip()
    s = s.replace("\\", "/").split("/")[-1]

    if s.endswith(".cif"):
        s = s[:-4]

    if s.endswith("_repeat"):
        s = s[:-7]

    return s


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    bad = [c for c in df.columns if c.startswith("Unnamed")]
    if bad:
        df = df.drop(columns=bad)
    return df


def read_csv(path: Path) -> pd.DataFrame:
    pr(f"Loading {path.name}")
    return drop_unnamed(pd.read_csv(path, low_memory=False))


def choose_best_id_column(
    df: pd.DataFrame,
    base_ids: set,
    source_name: str,
    candidates: list[str],
) -> tuple[str, int, dict]:
    best_col = None
    best_overlap = -1
    report = {}

    for col in candidates:
        if col not in df.columns:
            continue

        cleaned = df[col].map(clean_filename_value)
        overlap = len(set(cleaned.dropna()) & base_ids)
        report[col] = int(overlap)

        pr(f"{source_name}: candidate key '{col}' overlap = {overlap:,}")

        if overlap > best_overlap:
            best_col = col
            best_overlap = overlap

    if best_col is None:
        raise ValueError(f"{source_name}: no candidate ID columns found from {candidates}")

    if best_overlap == 0:
        raise ValueError(
            f"{source_name}: best key '{best_col}' has ZERO overlap. Report: {report}"
        )

    pr(f"{source_name}: using key '{best_col}' with overlap {best_overlap:,}")
    return best_col, int(best_overlap), report


def normalize_with_best_key(
    df: pd.DataFrame,
    base_ids: set,
    source_name: str,
    candidates: list[str],
) -> tuple[pd.DataFrame, dict]:
    best_col, overlap, report = choose_best_id_column(
        df=df,
        base_ids=base_ids,
        source_name=source_name,
        candidates=candidates,
    )

    out = df.copy()
    out[ID] = out[best_col].map(clean_filename_value)

    key_report = {
        "source": source_name,
        "selected_key": best_col,
        "selected_overlap": int(overlap),
        "candidate_overlaps": report,
    }

    return out, key_report


def assert_unique(df: pd.DataFrame, name: str) -> None:
    dups = int(df[ID].duplicated().sum())
    if dups:
        raise ValueError(f"{name} has {dups:,} duplicated filenames after processing.")


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_log_selectivity(series: pd.Series, target_name: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    bad = x <= 0
    if bad.any():
        pr(f"WARNING: {target_name}: {int(bad.sum()):,} non-positive selectivity values set to NaN before log.")
        x = x.mask(bad, np.nan)
    return np.log(x)


# =============================================================================
# BASE GEOMETRY
# =============================================================================

def load_geometry() -> tuple[pd.DataFrame, dict]:
    geo = read_csv(FILES["geometry"])

    if "filename" not in geo.columns:
        raise ValueError("geometric_properties.csv must contain filename column.")

    geo[ID] = geo["filename"].map(clean_filename_value)
    assert_unique(geo, "geometry")

    numeric_candidates = [
        c for c in geo.columns
        if c not in {ID, "filename", "ARC-MOF", "DB_num", "order_geo", "bool_geo"}
    ]
    geo = to_numeric_safe(geo, numeric_candidates)

    base_ids = set(geo[ID].dropna())

    report = {
        "source": "geometry",
        "n_rows": int(len(geo)),
        "n_unique_ids": int(geo[ID].nunique()),
    }

    return geo, report


# =============================================================================
# TOPOLOGY
# =============================================================================

def load_topology(base_ids: set) -> tuple[pd.DataFrame, dict]:
    topo_raw = read_csv(FILES["topology"])

    topo, key_report = normalize_with_best_key(
        topo_raw,
        base_ids=base_ids,
        source_name="topology",
        candidates=["Name", "filename"],
    )

    if "Crystalnet" not in topo.columns:
        raise ValueError("all_topology_lists.csv must contain Crystalnet column.")

    keep = [ID, "Crystalnet"]
    if "likely topology" in topo.columns:
        keep.append("likely topology")

    topo = topo[keep].dropna(subset=[ID]).copy()

    # Some files can list many structures with the same topology-derived filename.
    # Keep one topology assignment per normalized filename.
    topo = topo.drop_duplicates(subset=[ID], keep="first")

    topo["Crystalnet"] = topo["Crystalnet"].astype(str)
    topo["has_topology"] = True

    report = {
        **key_report,
        "n_rows_after_dedup": int(len(topo)),
        "n_unique_ids": int(topo[ID].nunique()),
        "n_crystalnet": int(topo["Crystalnet"].nunique()),
    }

    return topo, report


# =============================================================================
# ADSORPTION INVENTORY
# =============================================================================

def make_adsorption_file_inventory() -> pd.DataFrame:
    rows = []

    file_metadata = {
        "methane": {
            "process": "methane_storage_psa",
            "gas": "CH4",
            "kind": "raw_uptake",
        },
        "postcomb_co2": {
            "process": "post_combustion_vsa",
            "gas": "CO2",
            "kind": "raw_uptake",
        },
        "postcomb_n2": {
            "process": "post_combustion_vsa",
            "gas": "N2",
            "kind": "raw_uptake",
        },
        "precomb_co2": {
            "process": "pre_combustion_psa",
            "gas": "CO2",
            "kind": "raw_uptake",
        },
        "precomb_h2": {
            "process": "pre_combustion_psa",
            "gas": "H2",
            "kind": "raw_uptake",
        },
        "ngp_co2": {
            "process": "natural_gas_purification",
            "gas": "CO2",
            "kind": "raw_uptake",
        },
        "ngp_ch4": {
            "process": "natural_gas_purification",
            "gas": "CH4",
            "kind": "raw_uptake",
        },
        "landfill_co2": {
            "process": "landfill_gas_vpsa",
            "gas": "CO2",
            "kind": "raw_uptake",
        },
        "landfill_ch4": {
            "process": "landfill_gas_vpsa",
            "gas": "CH4",
            "kind": "raw_uptake",
        },
        "process": {
            "process": "all_processes",
            "gas": None,
            "kind": "process_summary",
        },
    }

    for key in ADSORPTION_FILE_KEYS:
        path = FILES[key]
        meta = file_metadata[key]

        row = {
            "logical_name": key,
            "file": path.name,
            "exists": path.exists(),
            "kind": meta["kind"],
            "process": meta["process"],
            "gas": meta["gas"],
            "n_rows": None,
            "n_unique_mofs": None,
            "temperatures_K": None,
            "pressures_bar": None,
            "process_values": None,
            "columns": None,
        }

        if path.exists():
            df = read_csv(path)
            row["n_rows"] = int(len(df))
            row["columns"] = "|".join(df.columns)

            if "filename" in df.columns:
                clean_ids = df["filename"].map(clean_filename_value)
                row["n_unique_mofs"] = int(clean_ids.nunique())

            if "T/K" in df.columns:
                vals = sorted(pd.to_numeric(df["T/K"], errors="coerce").dropna().unique())
                row["temperatures_K"] = "|".join(str(float(v)) for v in vals)

            if "p/bar" in df.columns:
                vals = sorted(pd.to_numeric(df["p/bar"], errors="coerce").dropna().unique())
                row["pressures_bar"] = "|".join(str(float(v)) for v in vals)

            if "process" in df.columns:
                vals = sorted(df["process"].dropna().astype(str).unique())
                row["process_values"] = "|".join(vals)

        rows.append(row)

    inventory = pd.DataFrame(rows)
    inventory.to_csv(ROOT / "adsorption_file_inventory.csv", index=False)
    pr("Wrote adsorption_file_inventory.csv")

    return inventory


# =============================================================================
# TARGET EXTRACTION
# =============================================================================

def select_pressure_target(
    path: Path,
    base_ids: set,
    pressure: float,
    target_name: str,
    value_col: str = "mmol/g",
    atol: float = 1e-7,
) -> tuple[pd.DataFrame, dict]:
    raw = read_csv(path)

    normalized, key_report = normalize_with_best_key(
        raw,
        base_ids=base_ids,
        source_name=f"{path.name}:{target_name}",
        candidates=["filename", "Name"],
    )

    if "p/bar" not in normalized.columns:
        raise ValueError(f"{path.name} missing p/bar column.")
    if value_col not in normalized.columns:
        raise ValueError(f"{path.name} missing {value_col} column.")

    normalized["p/bar"] = pd.to_numeric(normalized["p/bar"], errors="coerce")
    normalized[value_col] = pd.to_numeric(normalized[value_col], errors="coerce")

    subset = normalized[np.isclose(normalized["p/bar"], pressure, atol=atol)].copy()

    if subset.empty:
        available = sorted(normalized["p/bar"].dropna().unique())
        raise ValueError(
            f"{path.name}:{target_name}: no rows at pressure {pressure}. "
            f"Available pressures: {available}"
        )

    out = (
        subset[[ID, value_col]]
        .groupby(ID, as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: target_name})
    )

    assert_unique(out, target_name)

    report = {
        **key_report,
        "target": target_name,
        "file": path.name,
        "pressure_bar": float(pressure),
        "value_col": value_col,
        "n_rows_selected": int(len(subset)),
        "n_unique_ids": int(out[ID].nunique()),
        "missing_after_groupby": int(out[target_name].isna().sum()),
    }

    return out, report


def select_process_column(
    process_df: pd.DataFrame,
    base_ids: set,
    process_row: str,
    source_column: str,
    target_name: str,
    transform: str = "none",
) -> tuple[pd.DataFrame, dict]:
    if "process" not in process_df.columns:
        raise ValueError("overall_process.csv missing process column.")
    if source_column not in process_df.columns:
        raise ValueError(f"overall_process.csv missing {source_column} column.")

    subset = process_df[process_df["process"].astype(str) == process_row].copy()

    if subset.empty:
        available = sorted(process_df["process"].dropna().astype(str).unique())
        raise ValueError(
            f"overall_process.csv: no rows for process '{process_row}'. "
            f"Available processes: {available}"
        )

    subset[source_column] = pd.to_numeric(subset[source_column], errors="coerce")

    if transform == "log":
        subset[target_name] = safe_log_selectivity(subset[source_column], target_name)
    elif transform == "none":
        subset[target_name] = subset[source_column]
    else:
        raise ValueError(f"Unknown transform: {transform}")

    out = (
        subset[[ID, target_name]]
        .groupby(ID, as_index=False)[target_name]
        .mean()
    )

    assert_unique(out, target_name)

    report = {
        "source": "overall_process",
        "target": target_name,
        "process_row": process_row,
        "source_column": source_column,
        "transform": transform,
        "n_rows_selected": int(len(subset)),
        "n_unique_ids": int(out[ID].nunique()),
        "missing_after_groupby": int(out[target_name].isna().sum()),
    }

    return out, report


def load_targets(base_ids: set) -> tuple[pd.DataFrame, dict]:
    tables = []
    reports = {
        "raw_uptake": [],
        "working_capacity": [],
        "log_selectivity": [],
    }

    # Layer A
    for spec in RAW_UPTAKE_SPECS:
        tab, rep = select_pressure_target(
            path=FILES[spec["file_key"]],
            base_ids=base_ids,
            pressure=spec["pressure_bar"],
            target_name=spec["target"],
            value_col="mmol/g",
        )
        tables.append(tab)
        reports["raw_uptake"].append(rep)

    # Process-level data
    process_raw = read_csv(FILES["process"])

    process_norm, key_report = normalize_with_best_key(
        process_raw,
        base_ids=base_ids,
        source_name="overall_process",
        candidates=["filename", "Name"],
    )

    reports["overall_process_key"] = key_report

    # Layer B
    for spec in WORKING_CAPACITY_SPECS:
        tab, rep = select_process_column(
            process_df=process_norm,
            base_ids=base_ids,
            process_row=spec["process_row"],
            source_column=spec["source_column"],
            target_name=spec["target"],
            transform="none",
        )
        tables.append(tab)
        reports["working_capacity"].append(rep)

    # Layer C
    for spec in LOG_SELECTIVITY_SPECS:
        tab, rep = select_process_column(
            process_df=process_norm,
            base_ids=base_ids,
            process_row=spec["process_row"],
            source_column=spec["source_column"],
            target_name=spec["target"],
            transform="log",
        )
        tables.append(tab)
        reports["log_selectivity"].append(rep)

    # Merge target tables
    targets = None
    for tab in tables:
        if targets is None:
            targets = tab
        else:
            targets = targets.merge(tab, on=ID, how="outer")

    if targets is None:
        raise RuntimeError("No target tables generated.")

    assert_unique(targets, "targets")

    return targets, reports


# =============================================================================
# ENGINEERED FEATURES
# =============================================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in GEO_CORE + BASE_EXTRA + ALL_EXTRA_GEO:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # LCD/PLD ratio: Di / Df
    if "Di" in df.columns and "Df" in df.columns:
        df["lcd_pld_ratio"] = df["Di"] / (df["Df"] + EPS)

    # absolute difference between largest included sphere and pore limiting diameter
    if "Di" in df.columns and "Df" in df.columns:
        df["cavity_window_gap"] = df["Di"] - df["Df"]

    # surface area per accessible volume proxy
    if "ASA" in df.columns and "AVA" in df.columns:
        df["sa_pv_ratio"] = df["ASA"] / (df["AVA"] + EPS)

    # accessible void fraction normalized by density
    if "AVAf" in df.columns and "Density" in df.columns:
        df["vf_density_ratio"] = df["AVAf"] / (df["Density"] + EPS)

    if "Df" in df.columns:
        df["log_pld_plus1"] = np.log1p(df["Df"].clip(lower=0))

    if "Di" in df.columns:
        df["log_lcd_plus1"] = np.log1p(df["Di"].clip(lower=0))

    return df


# =============================================================================
# RACS
# =============================================================================

def load_and_filter_racs(
    base_ids: set,
    base_filenames: pd.Series,
) -> tuple[pd.DataFrame, list[str], dict]:
    rac_raw = read_csv(FILES["racs"])

    rac, key_report = normalize_with_best_key(
        rac_raw,
        base_ids=base_ids,
        source_name="RACs",
        candidates=["filename", "Name"],
    )

    assert_unique(rac, "RACs")

    candidate_cols = [
        c for c in rac.columns
        if c not in RAC_META and c != ID
    ]

    rac = to_numeric_safe(rac, candidate_cols)

    numeric_cols = [
        c for c in candidate_cols
        if c in rac.columns and pd.api.types.is_numeric_dtype(rac[c])
    ]

    rac_numeric = rac[[ID] + numeric_cols].copy()

    missing_before = int(rac_numeric[numeric_cols].isna().sum().sum())

    # Fill with medians for variance filtering and later model imputation.
    medians = rac_numeric[numeric_cols].median(numeric_only=True)
    filled = rac_numeric[numeric_cols].fillna(medians)

    # Drop all-NaN / constant-ish columns safely
    good_numeric_cols = [
        c for c in numeric_cols
        if filled[c].notna().any()
    ]

    filled = filled[good_numeric_cols]

    if filled.shape[1] == 0:
        raise ValueError("No usable RAC numeric columns after cleaning.")

    selector = VarianceThreshold(threshold=RAC_VARIANCE_THRESHOLD)
    selector.fit(filled)

    kept_cols = [c for c, keep in zip(good_numeric_cols, selector.get_support()) if keep]
    dropped_cols = [c for c in good_numeric_cols if c not in kept_cols]

    out = rac_numeric[[ID] + kept_cols].copy()
    out["has_RAC"] = True

    report = {
        **key_report,
        "n_raw_rac_columns": int(len(candidate_cols)),
        "n_numeric_rac_columns": int(len(numeric_cols)),
        "n_kept_rac_columns": int(len(kept_cols)),
        "n_dropped_low_variance": int(len(dropped_cols)),
        "variance_threshold": RAC_VARIANCE_THRESHOLD,
        "missing_values_before_imputation": missing_before,
        "n_unique_ids": int(out[ID].nunique()),
    }

    return out, kept_cols, report


# =============================================================================
# COVERAGE
# =============================================================================

def make_target_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for target in ACTIVE_TARGETS:
        if target not in df.columns:
            rows.append({
                "target": target,
                "layer": target_metadata()[target]["layer"],
                "process": target_metadata()[target].get("process"),
                "n_nonmissing": 0,
                "missing": len(df),
                "coverage": 0.0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
            })
            continue

        x = pd.to_numeric(df[target], errors="coerce")

        rows.append({
            "target": target,
            "layer": target_metadata()[target]["layer"],
            "process": target_metadata()[target].get("process"),
            "n_nonmissing": int(x.notna().sum()),
            "missing": int(x.isna().sum()),
            "coverage": float(x.notna().mean()),
            "mean": float(x.mean()) if x.notna().any() else np.nan,
            "std": float(x.std()) if x.notna().any() else np.nan,
            "min": float(x.min()) if x.notna().any() else np.nan,
            "max": float(x.max()) if x.notna().any() else np.nan,
        })

    return pd.DataFrame(rows)


def write_coverage(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    cov = make_target_coverage(df)
    cov.to_csv(ROOT / filename, index=False)

    pr(f"Target coverage written to {filename}:")
    print(
        cov[["target", "layer", "process", "n_nonmissing", "missing", "coverage"]]
        .to_string(index=False)
    )

    return cov


# =============================================================================
# TARGET METADATA
# =============================================================================

def target_metadata() -> dict:
    meta = {}

    for spec in RAW_UPTAKE_SPECS:
        meta[spec["target"]] = dict(spec)

    for spec in WORKING_CAPACITY_SPECS:
        meta[spec["target"]] = dict(spec)

    for spec in LOG_SELECTIVITY_SPECS:
        meta[spec["target"]] = dict(spec)

    return meta


# =============================================================================
# MANIFEST
# =============================================================================

def build_descriptor_families(df: pd.DataFrame, rac_cols: list[str]) -> dict:
    geo_core = [c for c in GEO_CORE if c in df.columns]
    base_extra = [c for c in BASE_EXTRA if c in df.columns]
    all_extra = [c for c in ALL_EXTRA_GEO if c in df.columns]
    engineered = [c for c in ENGINEERED if c in df.columns]

    enriched_interpretable = list(dict.fromkeys(
        geo_core + base_extra + all_extra + engineered
    ))

    geometry_only = geo_core
    geometry_plus_topology_num = enriched_interpretable
    topology_cat = ["Crystalnet"] if "Crystalnet" in df.columns else []

    families = {
        "geometry_only": {
            "num": geometry_only,
            "cat": [],
            "description": "Core geometric descriptors only.",
        },
        "enriched_interpretable": {
            "num": enriched_interpretable,
            "cat": [],
            "description": "Core geometry plus extended geometry and engineered interpretable descriptors.",
        },
        "topology_only": {
            "num": [],
            "cat": topology_cat,
            "description": "Crystalnet topology label only.",
        },
        "geometry_plus_topology": {
            "num": geometry_plus_topology_num,
            "cat": topology_cat,
            "description": "Enriched interpretable geometry plus Crystalnet topology.",
        },
        "rac_only": {
            "num": rac_cols,
            "cat": [],
            "description": "RAC descriptors after low-variance filtering.",
        },
        "geo_plus_rac": {
            "num": list(dict.fromkeys(enriched_interpretable + rac_cols)),
            "cat": [],
            "description": "Enriched interpretable geometry plus filtered RAC descriptors.",
        },
    }

    return families


def write_column_manifest(
    df: pd.DataFrame,
    families: dict,
    rac_cols: list[str],
    merge_report: dict,
) -> None:
    manifest = {
        "version": "v9_process_aware_step1",
        "description": "Process-aware ARC-MOF benchmark dataset with Layer A/B/C targets.",
        "strict_common_cohort": STRICT_COMMON_COHORT,
        "id_column": ID,
        "targets": ACTIVE_TARGETS,
        "target_layers": {
            "A_raw_uptake": LAYER_A_TARGETS,
            "B_working_capacity": LAYER_B_TARGETS,
            "C_direct_log_selectivity": LAYER_C_TARGETS,
        },
        "target_metadata": target_metadata(),
        "posthoc_selectivity_definitions": POSTHOC_SELECTIVITY_DEFINITIONS,
        "descriptor_families": families,
        "rac_columns": rac_cols,
        "metadata_columns_not_features": [
            c for c in METADATA_COLS_NOT_FEATURES if c in df.columns
        ],
        "process_rows": PROCESS_ROWS,
        "notes": {
            "selectivity_training": "Layer C targets are log-transformed ln(selectivity), not raw selectivity.",
            "posthoc_selectivity": "Layer D selectivity is not generated in Step 1; Step 3 reconstructs it from Layer A predictions.",
            "uptake_unit": "Layer A raw uptake targets use mmol/g.",
            "working_capacity_unit": "Layer B working capacities use mmol/g_working_capacity from overall_process.csv.",
            "categorical_feature": "Crystalnet is retained as categorical topology metadata for Step 2.",
            "topology_ood_warning": "Topology-only features cannot extrapolate to unseen Crystalnet labels under topology-grouped splits.",
        },
        "merge_report_summary": {
            "n_rows_final": int(len(df)),
            "n_cols_final": int(df.shape[1]),
            "n_targets": int(len(ACTIVE_TARGETS)),
            "n_layer_A": int(len(LAYER_A_TARGETS)),
            "n_layer_B": int(len(LAYER_B_TARGETS)),
            "n_layer_C": int(len(LAYER_C_TARGETS)),
        },
        "merge_report_keys": sorted(merge_report.keys()),
    }

    json_dump(manifest, ROOT / "column_manifest.json")
    pr("Wrote column_manifest.json")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start = time.time()

    pr("=" * 100)
    pr("STEP 1 V9: PROCESS-AWARE ARC-MOF DATA MERGE")
    pr("=" * 100)

    require_files()
    write_md5_manifest()

    inventory = make_adsorption_file_inventory()

    merge_report = {
        "files": {
            k: {
                "path": str(v),
                "exists": v.exists(),
                "name": v.name,
            }
            for k, v in FILES.items()
        },
        "adsorption_file_inventory_rows": int(len(inventory)),
    }

    # -------------------------------------------------------------------------
    # Load geometry as base universe
    # -------------------------------------------------------------------------
    geo, geo_report = load_geometry()
    merge_report["geometry"] = geo_report

    base_ids = set(geo[ID].dropna())
    pr(f"Base geometry rows: {len(geo):,}")
    pr(f"Base unique IDs: {len(base_ids):,}")

    # -------------------------------------------------------------------------
    # Add topology
    # -------------------------------------------------------------------------
    topo, topo_report = load_topology(base_ids)
    merge_report["topology"] = topo_report

    df = geo.merge(topo, on=ID, how="left")

    df["has_topology"] = df["Crystalnet"].notna()

    # -------------------------------------------------------------------------
    # Add engineered geometry
    # -------------------------------------------------------------------------
    df = add_engineered_features(df)

    # -------------------------------------------------------------------------
    # Add RACs
    # -------------------------------------------------------------------------
    rac_df, rac_cols, rac_report = load_and_filter_racs(
        base_ids=base_ids,
        base_filenames=geo[ID],
    )
    merge_report["racs"] = rac_report

    df = df.merge(rac_df, on=ID, how="left")
    df["has_RAC"] = df["has_RAC"].fillna(False).astype(bool)

    # -------------------------------------------------------------------------
    # Add targets
    # -------------------------------------------------------------------------
    target_df, target_report = load_targets(base_ids)
    merge_report["targets"] = target_report

    df = df.merge(target_df, on=ID, how="left")

    # Coverage before strict filtering
    write_coverage(df, "target_coverage_before_strict.csv")

    # -------------------------------------------------------------------------
    # Strict cohort filtering
    # -------------------------------------------------------------------------
    n_before = len(df)

    if STRICT_COMMON_COHORT:
        pr("Applying strict common cohort filter.")

        df = df[df["has_topology"]].copy()
        df = df[df["has_RAC"]].copy()

        for target in ACTIVE_TARGETS:
            df = df[df[target].notna()].copy()

    n_after = len(df)

    merge_report["strict_filter"] = {
        "strict_common_cohort": STRICT_COMMON_COHORT,
        "n_before": int(n_before),
        "n_after": int(n_after),
        "n_removed": int(n_before - n_after),
        "fraction_retained": float(n_after / n_before) if n_before else None,
    }

    # Final uniqueness check
    assert_unique(df, "final revision_data")

    # Coverage after strict filtering
    write_coverage(df, "target_coverage_strict.csv")

    # -------------------------------------------------------------------------
    # Descriptor families and manifest
    # -------------------------------------------------------------------------
    families = build_descriptor_families(df, rac_cols)

    merge_report["descriptor_families"] = {
        fam: {
            "n_num": len(spec.get("num", [])),
            "n_cat": len(spec.get("cat", [])),
        }
        for fam, spec in families.items()
    }

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    output_csv = ROOT / "revision_data.csv"
    df.to_csv(output_csv, index=False)
    pr(f"Wrote {output_csv.name}: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    json_dump(merge_report, ROOT / "merge_report.json")
    pr("Wrote merge_report.json")

    write_column_manifest(
        df=df,
        families=families,
        rac_cols=rac_cols,
        merge_report=merge_report,
    )

    elapsed = time.time() - start

    pr("=" * 100)
    pr("STEP 1 COMPLETE")
    pr(f"Elapsed: {elapsed / 60:.2f} min")
    pr(f"Final rows: {len(df):,}")
    pr(f"Final columns: {df.shape[1]:,}")
    pr(f"Active targets: {len(ACTIVE_TARGETS)}")
    pr("=" * 100)


if __name__ == "__main__":
    main()