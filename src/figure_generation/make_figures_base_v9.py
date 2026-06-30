#!/usr/bin/env python3
"""
make_figures_base_v9_phase3.py

Phase 3.2: Figure 3 (descriptor gain map) rebuilt with SE error caps.

Changes vs. v9:
- figure3(): now reads raw_metrics.csv to compute per-target ΔR² and
  propagated SE for (Enriched − Geo) and (Geo+RAC − Enriched). Panel B
  bars get horizontal SE caps; bar color = light blue for positive
  gains, light red for negative gains. Panel A keeps point-estimate
  bars (its magnitudes are too small for caps to be useful).
- figure2(): unchanged.
- figure5(): unchanged.

Figure 3 caption claims ("8 of 27 robust at 1σ", etc.) are now visible
on the chart: any bar whose left cap stays past the 0.10 reference is
robust at 1σ.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


# ============================================================
# paths
# ============================================================
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
OUT = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

F_RAW         = ROOT / "results" / "step2_training" / "raw_metrics.csv"
F_ATLAS_PIVOT = DIR_A / "atlas_pivot_lgbm_ood.csv"
F_GAIN        = DIR_A / "descriptor_gain_map_lgbm_ood.csv"
F_REC         = DIR_A / "recommendation_table_lgbm_ood.csv"
F_Q3          = DIR_B / "q3_topK_summary.csv"
F_Q3_PERFOLD  = DIR_B / "q3_topK_per_fold.csv"


# ============================================================
# style
# ============================================================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Arial Black", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold",
    "font.size": 10,
    "axes.titlesize": 0,
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.25,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

W2 = 7.20

COLOR_RED_SOFT   = "#c0504d"   # legacy (unused in fig3 now)
COLOR_GREEN      = "#3a8a3a"   # legacy (unused in fig3 now)
COLOR_BLUE       = "#1f4e79"
COLOR_LIGHT_BLUE = "#9ecae1"   # POS_COLOR for fig3
COLOR_LIGHT_RED  = "#fcae91"   # NEG_COLOR for fig3


# ============================================================
# constants
# ============================================================
PROCESS_ORDER = [
    "methane_storage_psa",
    "post_combustion_vsa",
    "pre_combustion_psa",
    "natural_gas_purification",
    "landfill_gas_vpsa",
]
PROCESS_DISPLAY = {
    "methane_storage_psa":      r"CH$_4$ storage",
    "post_combustion_vsa":      r"Post-comb. CO$_2$/N$_2$",
    "pre_combustion_psa":       r"Pre-comb. CO$_2$/H$_2$",
    "natural_gas_purification": r"NGP CO$_2$/CH$_4$",
    "landfill_gas_vpsa":        r"Landfill CO$_2$/CH$_4$",
}
LAYER_ORDER = ["A_raw_uptake", "B_working_capacity", "C_direct_log_selectivity"]
LAYER_DISPLAY = {
    "A_raw_uptake":             "uptake (A)",
    "B_working_capacity":       "working cap. (B)",
    "C_direct_log_selectivity": r"log $S$ (C)",
}
FAMILY_ORDER = [
    "geometry_only",
    "enriched_interpretable",
    "geometry_plus_topology",
    "rac_only",
    "geo_plus_rac",
]
FAMILY_DISPLAY = {
    "geometry_only":           "Geo",
    "enriched_interpretable":  "Enriched",
    "geometry_plus_topology":  "Geo+Topo",
    "rac_only":                "RAC",
    "geo_plus_rac":            "Geo+RAC",
    "topology_only":           "Topo",
}


# ============================================================
# helpers
# ============================================================
def panel_label(ax, letter, x=-0.06, y=1.04, size=14):
    ax.text(x, y, letter,
            transform=ax.transAxes,
            fontsize=size, fontweight="bold",
            fontfamily="Arial",
            ha="left", va="bottom")


def short_target(t):
    s = (t or "")
    s = s.replace("methane_storage_working_capacity", r"CH$_4$ wc")
    s = s.replace("postcomb_working_capacity",        "post wc")
    s = s.replace("precomb_working_capacity",         "pre wc")
    s = s.replace("ngp_working_capacity",             "NGP wc")
    s = s.replace("landfill_working_capacity",        "lf wc")
    s = s.replace("postcomb_log_selectivity",         "post lnS")
    s = s.replace("precomb_log_selectivity",          "pre lnS")
    s = s.replace("ngp_log_selectivity",              "NGP lnS")
    s = s.replace("landfill_log_selectivity",         "lf lnS")
    s = s.replace("postcomb_CO2_", r"post CO$_2$ ")
    s = s.replace("postcomb_N2_",  r"post N$_2$ ")
    s = s.replace("precomb_CO2_",  r"pre CO$_2$ ")
    s = s.replace("precomb_H2_",   r"pre H$_2$ ")
    s = s.replace("ngp_CO2_",      r"NGP CO$_2$ ")
    s = s.replace("ngp_CH4_",      r"NGP CH$_4$ ")
    s = s.replace("landfill_CO2_", r"lf CO$_2$ ")
    s = s.replace("landfill_CH4_", r"lf CH$_4$ ")
    s = s.replace("CH4_", r"CH$_4$ ")
    s = s.replace("bar", " bar")
    return s


def get_pressure(t):
    m = re.search(r"_(\d+(?:\.\d+)?)bar", t)
    return float(m.group(1)) if m else 9999.0


def order_atlas(atlas):
    d = atlas.copy()
    d["pidx"] = d["process"].map({p: i for i, p in enumerate(PROCESS_ORDER)})
    d["lidx"] = d["layer"].map({l: i for i, l in enumerate(LAYER_ORDER)})
    d["pbar"] = d["target"].map(get_pressure)
    d = d.sort_values(["pidx", "lidx", "pbar", "target"]).reset_index(drop=True)
    return d


def print_done(name):
    print(f"[ok] wrote {OUT/name}")


# ============================================================
# data wrangling for fig3 SE
# ============================================================
def per_target_r2_se_lgbm_ood():
    """Per (target, descriptor_family) topology-OOD R² mean and SE (LightGBM)."""
    df = pd.read_csv(F_RAW)
    df = df[(df["model"] == "lgbm") &
            (df["status"] == "ok") &
            (df["split_type"] == "balanced_topology_group")].copy()
    g = (df.groupby(["target", "descriptor_family"])["r2"]
           .agg(["mean", "std", "count"])
           .reset_index())
    g["se"] = g["std"] / np.sqrt(g["count"])
    return g.rename(columns={"mean": "r2_mean", "se": "r2_se"})[
        ["target", "descriptor_family", "r2_mean", "r2_se"]]


def descriptor_gain_with_se(plus_fam, base_fam):
    """ΔR² = R²(plus) − R²(base) per target, with propagated SE."""
    per = per_target_r2_se_lgbm_ood()
    plus = per[per["descriptor_family"] == plus_fam].rename(
        columns={"r2_mean": "r2_plus", "r2_se": "se_plus"}).drop(
        columns=["descriptor_family"])
    base = per[per["descriptor_family"] == base_fam].rename(
        columns={"r2_mean": "r2_base", "r2_se": "se_base"}).drop(
        columns=["descriptor_family"])
    m = plus.merge(base, on="target", how="inner")
    m["delta_r2"] = m["r2_plus"] - m["r2_base"]
    m["delta_se"] = np.sqrt(m["se_plus"]**2 + m["se_base"]**2)
    return m[["target", "delta_r2", "delta_se"]]


# ============================================================
# FIGURE 2 — unchanged from v9
# ============================================================
def figure2():
    atlas = pd.read_csv(F_ATLAS_PIVOT)
    atlas = order_atlas(atlas)
    fams = [f for f in FAMILY_ORDER if f in atlas.columns]
    M = atlas[fams].to_numpy()
    labels = [short_target(t) for t in atlas["target"].tolist()]

    rec = pd.read_csv(F_REC)
    rec_map = {(r["process"], r["layer"]): r["recommended_family"]
               for _, r in rec.iterrows()}

    h = 0.40 * len(atlas) + 2.0
    fig = plt.figure(figsize=(W2 + 0.4, h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.05], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    norm = Normalize(vmin=-0.5, vmax=1.0)
    cmap = plt.get_cmap("RdBu")
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

    for i in range(M.shape[0]):
        for j, fam in enumerate(fams):
            v = M[i, j]
            if np.isnan(v):
                continue
            tcolor = "white" if (v > 0.65 or v < -0.1) else "black"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=tcolor)

    for i in range(len(atlas)):
        key = (atlas.iloc[i]["process"], atlas.iloc[i]["layer"])
        rec_fam = rec_map.get(key)
        if rec_fam in fams:
            j = fams.index(rec_fam)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, ec="black", lw=2.6))

    last_proc = None
    for i, p in enumerate(atlas["process"]):
        if last_proc is not None and p != last_proc:
            ax.axhline(i - 0.5, color="black", lw=1.2)
        last_proc = p

    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([FAMILY_DISPLAY[f] for f in fams],
                       fontweight="bold", fontsize=12)
    ax.set_yticks(range(len(atlas)))
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"$R^{2}$ (topology-OOD)",
                   fontweight="bold", fontsize=25)
    cbar.ax.tick_params(labelsize=18)

    fig.savefig(OUT / "fig2_descriptor_adequacy_atlas.pdf")
    plt.close(fig)
    print_done("fig2_descriptor_adequacy_atlas.pdf")


# ============================================================
# FIGURE 3 — Phase 3.2: SE caps on Panel B, light blue / light red bars
# ============================================================
def figure3():
    # ---- compute ΔR² + SE for both panel comparisons ----
    panelA_data = descriptor_gain_with_se(
        plus_fam="enriched_interpretable",
        base_fam="geometry_only",
    )
    panelB_data = descriptor_gain_with_se(
        plus_fam="geo_plus_rac",
        base_fam="enriched_interpretable",
    )


    # ---- drop targets whose SE is so large the cap dominates the plot ----
    # postcomb_N2_0.075bar has SE 0.58 (cap reaches −0.95) and would force
    # a useless wide xlim. It is descriptor-limited and already discussed
    # in §3.3 prose / Fig 3 caption.
    EXCLUDED_FROM_PANELB = ["postcomb_N2_0.075bar"]
    panelB_data = panelB_data[~panelB_data["target"].isin(EXCLUDED_FROM_PANELB)]

    # ---- order targets by Panel B descending ΔR² ----
    panelB_data = panelB_data.sort_values("delta_r2",
                                          ascending=False).reset_index(drop=True)
    target_order = panelB_data["target"].tolist()

    panelA_data = panelA_data.set_index("target").reindex(
        target_order).reset_index()

    n_targets = len(target_order)
    fig_width = (W2 + 1.5) * 1.10
    fig, (axA, axB) = plt.subplots(
        1, 2,
        figsize=(fig_width, 0.30 * n_targets + 2.0),
        sharey=True,
        gridspec_kw={"wspace": 0.05, "width_ratios": [0.65, 1.0]},
    )
    y_positions = np.arange(n_targets)[::-1]

    # ---------- Panel A: Enriched − Geo, point estimates only ----------
    valsA = panelA_data["delta_r2"].to_numpy()
    colorsA = [COLOR_LIGHT_BLUE if v >= 0 else COLOR_LIGHT_RED for v in valsA]
    axA.barh(y_positions, valsA,
             color=colorsA, edgecolor="black", linewidth=0.4, height=0.7)
    axA.axvline(0, color="black", lw=0.8)
    axA.axvline(0.10, color="black", lw=0.7, ls="--", alpha=0.5)
    axA.set_xlim(-0.05, 0.60)
    axA.set_xlabel(r"$\Delta R^{2}$  (Enriched − Geo)",
                   fontweight="bold", fontsize=12)
    axA.tick_params(axis="x", labelsize=17)
    axA.grid(axis="x", ls=":", alpha=0.4)
    panel_label(axA, "A", x=-0.32, y=1.02, size=22)

    # ---------- Panel B: Geo+RAC − Enriched, SE caps ----------
    valsB = panelB_data["delta_r2"].to_numpy()
    sesB  = panelB_data["delta_se"].to_numpy()
    colorsB = [COLOR_LIGHT_BLUE if v >= 0 else COLOR_LIGHT_RED for v in valsB]

    axB.barh(y_positions, valsB,
             color=colorsB, edgecolor="black", linewidth=0.4, height=0.7,
             zorder=2)


    # error caps on bar tips — subtle gray, thin
    axB.errorbar(valsB, y_positions, xerr=sesB,
                 fmt="none", ecolor="#555555",
                 elinewidth=0.7, capsize=3, capthick=0.7,
                 alpha=0.75, zorder=4)


    axB.axvline(0, color="black", lw=0.8)
    axB.axvline(0.10, color="black", lw=0.8, ls="--", alpha=0.6)
    axB.text(0.11, 0.5, r"$\Delta R^{2}=0.10$",
             ha="left", va="bottom", fontsize=10, fontweight="bold",
             rotation=90, color="#333333")

    axB.set_xlim(-0.65, 0.65)    # widened slightly to accommodate caps
    axB.set_xlabel(r"$\Delta R^{2}$  (Geo+RAC − Enriched)",
                   fontweight="bold", fontsize=12)
    axB.tick_params(axis="x", labelsize=17)
    axB.grid(axis="x", ls=":", alpha=0.4)
    panel_label(axB, "B", x=-0.05, y=1.02, size=22)

    axA.set_yticks(y_positions)
    axA.set_yticklabels([short_target(t) for t in target_order],
                        fontweight="bold", fontsize=9)
    axA.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.28)
    fig.savefig(OUT / "fig3_descriptor_gain_map.pdf")
    plt.close(fig)
    print_done("fig3_descriptor_gain_map.pdf")

    # ---------- sanity print ----------
    print("\n[Figure 3 — Panel B ΔR² (Geo+RAC − Enriched) per target]")
    n10 = ((valsB >= 0.10)).sum()
    n10_sig = ((valsB - sesB) >= 0.10).sum()
    n20 = ((valsB >= 0.20)).sum()
    n20_sig = ((valsB - sesB) >= 0.20).sum()
    print(f"  ΔR² >= 0.10:  point {n10}/27   robust@1σ {n10_sig}/27")
    print(f"  ΔR² >= 0.20:  point {n20}/27   robust@1σ {n20_sig}/27")
    print(f"  max ΔR²: {valsB.max():+.3f} on {target_order[int(np.argmax(valsB))]}")
    neg = panelB_data[panelB_data["delta_r2"] < 0]
    print(f"  negative ΔR² targets: {neg['target'].tolist()}")


# ============================================================
# FIGURE 5 — unchanged from v9
# ============================================================
def figure5():
    perfold = pd.read_csv(F_Q3_PERFOLD)
    summary = pd.read_csv(F_Q3)

    targets = [
        ("CH4_65bar",                r"CH$_4$ 65 bar"),
        ("postcomb_CO2_0.015bar",    r"post CO$_2$ 0.015 bar"),
    ]
    families = ["geometry_only", "geo_plus_rac"]

    family_color = {
        "geometry_only": "#525252",
        "geo_plus_rac":  "#1f4e79",
    }
    target_ls = {
        "CH4_65bar":             "-",
        "postcomb_CO2_0.015bar": "--",
    }
    color_map_B = {
        "geometry_only": "#3aa39a",
        "geo_plus_rac":  "#e07b39",
    }

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W2 + 1.0, 6.5),
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})

    for tg, tg_label in targets:
        for fam in families:
            rows = perfold[(perfold["target"] == tg) & (perfold["family"] == fam)]
            if rows.empty:
                continue
            r100 = (rows["recall_at_100_mean"].mean()
                    if "recall_at_100_mean" in rows.columns
                    else rows["recall_at_100"].mean())
            r1000 = (rows["recall_at_1000_mean"].mean()
                     if "recall_at_1000_mean" in rows.columns
                     else rows["recall_at_1000"].mean())
            n_test = (rows["n_mofs"].mean()
                      if "n_mofs" in rows.columns else 60000)
            K_anchors = np.array([1.0, 100.0, 1000.0, n_test])
            r_anchors = np.array([max(r100 * 0.01, 1.0 / n_test),
                                  r100, r1000, 1.0])
            K_grid = np.logspace(0, np.log10(n_test), 400)
            r_grid = np.interp(np.log10(K_grid),
                               np.log10(K_anchors),
                               r_anchors)
            color = family_color[fam]
            ls = target_ls[tg]
            axA.plot(K_grid, r_grid,
                     color=color, lw=2.5, ls=ls,
                     label=f"{FAMILY_DISPLAY[fam]} | {tg_label}")
            axA.scatter([100, 1000], [r100, r1000],
                        s=70, color=color, edgecolor="white",
                        linewidth=1.3, zorder=10)

    K_diag = np.logspace(0, np.log10(60000), 100)
    axA.plot(K_diag, K_diag / 60000, color="grey", lw=1.0, ls="-",
             alpha=0.5, label="random")
    axA.set_xscale("log")
    axA.set_xlim(1, 1e5)
    axA.set_ylim(0, 1.05)
    axA.set_xlabel(r"$K$ (top-$K$ cutoff)", fontweight="bold", fontsize=13)
    axA.set_ylabel("Cumulative recall (topology-OOD)",
                   fontweight="bold", fontsize=13)
    axA.tick_params(axis="x", labelsize=11)
    axA.tick_params(axis="y", labelsize=15)
    axA.grid(True, which="both", ls=":", alpha=0.4)
    axA.axvline(100, color="black", lw=0.6, ls=":", alpha=0.5)
    axA.axvline(1000, color="black", lw=0.6, ls=":", alpha=0.5)
    axA.text(100, 1.02, r"$K=100$", ha="center", fontsize=9,
             fontweight="bold", color="#555555")
    axA.text(1000, 1.02, r"$K=1000$", ha="center", fontsize=9,
             fontweight="bold", color="#555555")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               frameon=False, ncol=2, fontsize=10)
    panel_label(axA, "A", x=-0.14, y=1.04, size=22)

    target_sel = "postcomb_log_selectivity"
    bars, labels = [], []
    for fam in families:
        row = summary[(summary["target"] == target_sel) & (summary["family"] == fam)]
        v = (row["champion_misidentification_at_100_mean"].iloc[0]
             if not row.empty else np.nan)
        bars.append(v)
        labels.append(FAMILY_DISPLAY[fam])

    xb = np.arange(len(bars))
    axB.bar(xb, bars,
            color=[color_map_B["geometry_only"], color_map_B["geo_plus_rac"]],
            edgecolor="black", lw=0.6)
    for i, v in enumerate(bars):
        axB.text(i, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
    axB.set_xticks(xb)
    axB.set_xticklabels(labels, fontweight="bold", fontsize=12)
    axB.set_ylim(0.0, 1.05)
    axB.set_ylabel("Champion misidentification @ $K{=}100$\n(post-comb lnS)",
                   fontweight="bold", fontsize=12)
    axB.tick_params(axis="y", labelsize=15)
    axB.grid(axis="y", ls=":", alpha=0.4)
    panel_label(axB, "B", x=-0.14, y=1.04, size=22)

    fig.subplots_adjust(wspace=0.32, bottom=0.30)
    fig.savefig(OUT / "fig5_screening_yield.pdf")
    plt.close(fig)
    print_done("fig5_screening_yield.pdf")


def main():
    print(f"[run] writing to {OUT}")
    figure2()
    figure3()
    figure5()
    print("[done]")


if __name__ == "__main__":
    main()