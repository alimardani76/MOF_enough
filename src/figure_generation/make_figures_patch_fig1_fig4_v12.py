#!/usr/bin/env python3
"""
polished_v12.py

Patches v11:
  Fig 1:
    - A/B letters 20% smaller (50 → 40)
    - Make top room for A so it doesn't overlap CH4 storage label
      (top margin pushed down)
    - REMOVE error bars (not visibly informative at SE ~ 0.05)

  Fig 4:
    - 10% wider
    - Legend moves to TOP CENTER (above each panel, not below)
    - Keep error bars (they were visible and useful here)
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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

F_GAP_PER = DIR_A / "synthesizability_gap_per_combo.csv"
F_REC     = DIR_A / "recommendation_table_lgbm_ood.csv"
F_Q1_FOLD = DIR_B / "q1_direct_vs_posthoc_lnS_per_fold.csv"
F_Q2_FOLD = DIR_B / "q2_working_capacity_components_per_fold.csv"


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
COLOR_BLUE = "#1f4e79"
COLOR_LIGHT_BLUE = "#9ecae1"

PROCESS_ORDER = [
    "methane_storage_psa",
    "post_combustion_vsa",
    "pre_combustion_psa",
    "natural_gas_purification",
    "landfill_gas_vpsa",
]
PROCESS_DISPLAY = {
    "methane_storage_psa":      "CH$_4$ storage",
    "post_combustion_vsa":      "Post-comb. CO$_2$/N$_2$",
    "pre_combustion_psa":       "Pre-comb. CO$_2$/H$_2$",
    "natural_gas_purification": "NGP CO$_2$/CH$_4$",
    "landfill_gas_vpsa":        "Landfill CO$_2$/CH$_4$",
}
LAYER_ORDER = ["A_raw_uptake", "B_working_capacity", "C_direct_log_selectivity"]
LAYER_DISPLAY = {
    "A_raw_uptake":              "uptake (A)",
    "B_working_capacity":        "working cap. (B)",
    "C_direct_log_selectivity":  "log $S$ (C)",
}


def print_done(name):
    print(f"[ok] wrote {OUT/name}")


# =============================================================================
# FIGURE 1 — no error bars; A/B 20% smaller; A doesn't overlap CH4 storage
# =============================================================================

def figure1():
    gap = pd.read_csv(F_GAP_PER)
    rec = pd.read_csv(F_REC)

    gap = gap[gap["model"] == "lgbm"].copy()
    rec_map = {
        (r["process"], r["layer"]): r["recommended_family"]
        for _, r in rec.iterrows()
        if pd.notna(r.get("recommended_family"))
    }

    keep = []
    for _, r in gap.iterrows():
        k = (r["process"], r["layer"])
        if k in rec_map and r["descriptor_family"] == rec_map[k]:
            keep.append(r)
    rec_gap = pd.DataFrame(keep)

    sub = rec_gap.groupby(["process", "layer"], as_index=False).agg(
        r2_random=("r2_random_mean", "mean"),
        r2_ood=("r2_ood_mean", "mean"),
    )
    sub["r2_gap"] = sub["r2_random"] - sub["r2_ood"]

    perproc_records = []
    for proc in PROCESS_ORDER:
        proc_data = sub[sub["process"] == proc]
        if proc_data.empty:
            continue
        perproc_records.append({
            "process": proc,
            "r2_random": proc_data["r2_random"].mean(),
            "r2_ood": proc_data["r2_ood"].mean(),
            "r2_gap": proc_data["r2_random"].mean() - proc_data["r2_ood"].mean(),
        })
    perproc = pd.DataFrame(perproc_records)

    fig = plt.figure(figsize=(W2 * 1.65, 11.5))
    # top=0.92 leaves room for A letter above the data
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42,
                          left=0.32, right=0.92, top=0.92, bottom=0.07)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])

    # Panel A — NO error bars
    y = np.arange(len(PROCESS_ORDER))[::-1]
    for yi, proc in zip(y, PROCESS_ORDER):
        row = perproc[perproc["process"] == proc]
        if row.empty:
            continue
        row = row.iloc[0]
        rr, ro = row["r2_random"], row["r2_ood"]
        g = rr - ro

        axA.plot([ro, rr], [yi, yi], color="#444444", lw=3.0, zorder=2)
        axA.scatter([rr], [yi], s=180, color=COLOR_BLUE,
                    edgecolor="black", lw=1.0, zorder=3,
                    label="random $R^{2}$" if yi == y[0] else None)
        axA.scatter([ro], [yi], s=180, facecolors="white",
                    edgecolor=COLOR_BLUE, lw=2.5, zorder=3,
                    label="topology-OOD $R^{2}$" if yi == y[0] else None)
        axA.text(max(rr, ro) + 0.03, yi, f"+{g:.2f}",
                 ha="left", va="center", fontsize=14, fontweight="bold")

    axA.set_yticks(y)
    axA.set_yticklabels([PROCESS_DISPLAY[p] for p in PROCESS_ORDER],
                        fontweight="bold", fontsize=18)
    axA.set_xlim(-0.02, 1.20)
    axA.set_xlabel(r"$R^{2}$ (recommended family, mean over layers)",
                   fontweight="bold", fontsize=21)
    axA.tick_params(axis="x", labelsize=20)
    axA.grid(axis="x", ls=":", alpha=0.4)
    axA.legend(loc="lower left", frameon=False, fontsize=17)

    # A letter — 20% smaller (50 → 40), positioned ABOVE the axes
    # so it doesn't overlap CH4 storage label
    fig.text(0.10, 0.95, "A",
             fontsize=40, fontweight="bold",
             fontfamily="Arial", ha="left", va="top")

    # Panel B — heatmap unchanged structure
    pivot = sub.pivot_table(index="process", columns="layer", values="r2_gap")
    pivot = pivot.reindex(index=PROCESS_ORDER, columns=LAYER_ORDER)
    vmax = max(0.5, float(np.nanmax(pivot.values)))
    im = axB.imshow(pivot.values, cmap="Oranges",
                    vmin=0.0, vmax=vmax, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            tcolor = "white" if v > 0.6 * vmax else "black"
            axB.text(j, i, f"{v:.2f}",
                     ha="center", va="center",
                     fontsize=20, fontweight="bold", color=tcolor)

    axB.set_xticks(range(len(LAYER_ORDER)))
    axB.set_xticklabels([LAYER_DISPLAY[l] for l in LAYER_ORDER],
                        fontweight="bold", fontsize=20)
    axB.set_yticks(range(len(PROCESS_ORDER)))
    axB.set_yticklabels([PROCESS_DISPLAY[p] for p in PROCESS_ORDER],
                        fontweight="bold", fontsize=18)
    cbar = plt.colorbar(im, ax=axB, shrink=0.85, pad=0.04)
    cbar.set_label(r"random $R^{2}\;-\;$OOD $R^{2}$",
                   fontweight="bold", fontsize=17)
    cbar.ax.tick_params(labelsize=20)

    # B letter — same x as A, between the two panels
    fig.text(0.10, 0.495, "B",
             fontsize=40, fontweight="bold",
             fontfamily="Arial", ha="left", va="top")

    fig.savefig(OUT / "fig1_topology_transfer_gap.pdf")
    plt.close(fig)
    print_done("fig1_topology_transfer_gap.pdf")


# =============================================================================
# FIGURE 4 — 10% wider; legend at TOP CENTER above each panel; keep err bars
# =============================================================================

def figure4():
    q1 = pd.read_csv(F_Q1_FOLD)
    q2 = pd.read_csv(F_Q2_FOLD)

    q2_summary = q2.groupby("process", as_index=False).agg(
        r2_q_ads_mean=("r2_q_ads", "mean"),
        r2_q_ads_se=("r2_q_ads", lambda x: x.std() / np.sqrt(len(x))),
        r2_q_des_mean=("r2_q_des", "mean"),
        r2_q_des_se=("r2_q_des", lambda x: x.std() / np.sqrt(len(x))),
        r2_wc_mean=("r2_wc", "mean"),
        r2_wc_se=("r2_wc", lambda x: x.std() / np.sqrt(len(x))),
    )
    q2_summary = q2_summary.set_index("process").reindex(PROCESS_ORDER).reset_index()

    q1_summary = q1.groupby("process", as_index=False).agg(
        r2_direct_mean=("r2_direct_lnS", "mean"),
        r2_direct_se=("r2_direct_lnS", lambda x: x.std() / np.sqrt(len(x))),
        r2_posthoc_mean=("r2_posthoc_lnS", "mean"),
        r2_posthoc_se=("r2_posthoc_lnS", lambda x: x.std() / np.sqrt(len(x))),
    )
    binary = [p for p in PROCESS_ORDER if p != "methane_storage_psa"]
    q1_summary = q1_summary.set_index("process").reindex(binary).reset_index()

    # 10% wider: was (W2 + 1.5) * 1.6 = 13.92; now * 1.10 = 15.31
    fig = plt.figure(figsize=((W2 + 1.5) * 1.6 * 1.15, 10.0 * 1.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.45)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])

    # Panel A
    n = len(q2_summary)
    x = np.arange(n)
    w = 0.27
    axA.bar(x - w, q2_summary["r2_q_ads_mean"], w,
            yerr=q2_summary["r2_q_ads_se"], capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label=r"$q_{\mathrm{ads}}$",
            color=COLOR_LIGHT_BLUE, edgecolor="black", lw=0.5)
    axA.bar(x, q2_summary["r2_q_des_mean"], w,
            yerr=q2_summary["r2_q_des_se"], capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label=r"$q_{\mathrm{des}}$",
            color="#fdae6b", edgecolor="black", lw=0.5)
    axA.bar(x + w, q2_summary["r2_wc_mean"], w,
            yerr=q2_summary["r2_wc_se"], capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label="working cap.",
            color="#31a354", edgecolor="black", lw=0.5)

    for i, row in q2_summary.iterrows():
        gap = row["r2_wc_mean"] - row["r2_q_des_mean"]
        if not np.isnan(gap):
            axA.text(i + w, row["r2_wc_mean"] + row["r2_wc_se"] + 0.03,
                     f"+{gap:.2f}", ha="center", va="bottom",
                     fontsize=15, fontweight="bold", color="#1f6c34")

    axA.set_xticks(x)
    axA.set_xticklabels([PROCESS_DISPLAY[p] for p in q2_summary["process"]],
                        fontweight="bold", fontsize=18)
    # Extend ylim for legend at top
    axA.set_ylim(-0.20, 1.45)
    axA.axhline(0, color="black", lw=0.6, alpha=0.5)
    axA.set_ylabel(r"$R^{2}$ (topology-OOD)",
                   fontweight="bold", fontsize=25)
    axA.tick_params(axis="y", labelsize=22)
    # LEGEND TOP CENTER above panel
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
               frameon=False, ncol=3, fontsize=21,
               markerscale=2.0, handlelength=2.5, handleheight=2.5)
    axA.grid(axis="y", ls=":", alpha=0.4)
    fig.text(0.06, 0.965, "A",
             fontsize=44, fontweight="bold",
             fontfamily="Arial", ha="left", va="top")

    # Panel B
    m = len(q1_summary)
    x = np.arange(m)
    w = 0.32
    axB.bar(x - w/2, q1_summary["r2_direct_mean"], w,
            yerr=q1_summary["r2_direct_se"], capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label=r"direct $\ln S$",
            color="#54278f", edgecolor="black", lw=0.5)
    axB.bar(x + w/2, q1_summary["r2_posthoc_mean"], w,
            yerr=q1_summary["r2_posthoc_se"], capsize=4,
            error_kw={"elinewidth": 1.5, "ecolor": "black"},
            label=r"post-hoc $\ln S$",
            color="#bcbddc", edgecolor="black", lw=0.5)

    for i, row in q1_summary.iterrows():
        gap = row["r2_direct_mean"] - row["r2_posthoc_mean"]
        top = max(row["r2_direct_mean"] + row["r2_direct_se"],
                  row["r2_posthoc_mean"] + row["r2_posthoc_se"])
        axB.text(i, top + 0.03,
                 f"+{gap:.2f}", ha="center", va="bottom",
                 fontsize=15, fontweight="bold", color="#3b1059")

    axB.set_xticks(x)
    axB.set_xticklabels([PROCESS_DISPLAY[p] for p in q1_summary["process"]],
                        fontweight="bold", fontsize=18)
    # Extend ylim for legend at top
    axB.set_ylim(-0.05, 1.45)
    axB.set_ylabel(r"$R^{2}$ (topology-OOD)",
                   fontweight="bold", fontsize=25)
    axB.tick_params(axis="y", labelsize=22)
    # LEGEND TOP CENTER above panel
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
               frameon=False, ncol=2, fontsize=21,
               markerscale=2.0, handlelength=2.5, handleheight=2.5)
    axB.grid(axis="y", ls=":", alpha=0.4)
    fig.text(0.06, 0.485, "B",
             fontsize=44, fontweight="bold",
             fontfamily="Arial", ha="left", va="top")

    fig.savefig(OUT / "fig4_target_construction.pdf")
    plt.close(fig)
    print_done("fig4_target_construction.pdf")


def main():
    print(f"[run] writing to {OUT}")
    figure1()
    figure4()
    print("[done]")


if __name__ == "__main__":
    main()