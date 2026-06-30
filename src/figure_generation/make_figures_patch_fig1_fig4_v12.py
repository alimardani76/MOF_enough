#!/usr/bin/env python3
"""
make_figures_patch_fig1_fig4_v13_phase3.py

Phase 3.1: Figure 1 (topology-transfer gap) rebuilt with SE-aware visuals.

Panel A — vertical stack of per-process range bars:
  - Light-blue range bar from OOD R² (left) to random R² (right)
  - Error caps on BOTH ends use the propagated gap SE so they are visibly
    asymmetric and informative (random-side R² SE is ~0 and would not show)
  - Endpoint R² values printed outside the caps
  - All bars same color, no confidence-tier styling here

Panel B — heatmap of the gap (random R² − OOD R²) per process × layer:
  - Blues colormap + colorbar restored
  - Each cell shows "+mean" (large bold) and "±SE" (smaller) on a second line
  - Cells with SE >= 0.10 are hatched (//// in black) to flag low confidence
  - Methane storage Layer C is NaN (single-gas process)

Figure 4 (= manuscript Fig 5, file fig4_target_construction.pdf) is a TODO
for Phase 3.3 — left as a no-op here so this script can ship Figure 1 alone.

Data source: results/step2_training/raw_metrics.csv  (per-fold R² per cell)
Filtering: model == "lgbm", status == "ok"
Aggregation: 4 folds for both random and balanced_topology_group splits,
             per-target SE, then propagated cell- and process-level SE.
"""
from __future__ import annotations
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
F_RAW = ROOT / "results" / "step2_training" / "raw_metrics.csv"
OUT = ROOT / "figures" / "main"
OUT.mkdir(parents=True, exist_ok=True)

ROOT = find_repo_root(Path(__file__))
W2 = 7.20
DIR_A = ROOT / "results" / "step3A_descriptor_analysis"
DIR_B = ROOT / "results" / "step3B_target_screening_analysis"
F_Q1_FOLD = DIR_B / "q1_direct_vs_posthoc_lnS_per_fold.csv"
F_Q2_FOLD = DIR_B / "q2_working_capacity_components_per_fold.csv"
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

BAR_COLOR  = "#9ecae1"     # light blue for Panel A bars
COLOR_GREY = "#666666"


# ============================================================
# manuscript constants
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

# recommended descriptor family per (process, layer) — matches Table 1
RECOMMENDED = {
    ("methane_storage_psa",      "A_raw_uptake"):              "geo_plus_rac",
    ("methane_storage_psa",      "B_working_capacity"):        "geometry_only",
    ("post_combustion_vsa",      "A_raw_uptake"):              "geometry_only",
    ("post_combustion_vsa",      "B_working_capacity"):        "geo_plus_rac",
    ("post_combustion_vsa",      "C_direct_log_selectivity"):  "geo_plus_rac",
    ("pre_combustion_psa",       "A_raw_uptake"):              "geometry_only",
    ("pre_combustion_psa",       "B_working_capacity"):        "geo_plus_rac",
    ("pre_combustion_psa",       "C_direct_log_selectivity"):  "geo_plus_rac",
    ("natural_gas_purification", "A_raw_uptake"):              "geo_plus_rac",
    ("natural_gas_purification", "B_working_capacity"):        "geo_plus_rac",
    ("natural_gas_purification", "C_direct_log_selectivity"):  "geo_plus_rac",
    ("landfill_gas_vpsa",        "A_raw_uptake"):              "geo_plus_rac",
    ("landfill_gas_vpsa",        "B_working_capacity"):        "geo_plus_rac",
    ("landfill_gas_vpsa",        "C_direct_log_selectivity"):  "geo_plus_rac",
}

LOW_CONF_THRESHOLD = 0.10   # SE >= this -> hatched cell in Panel B
BAR_HEIGHT = 0.42           # 20% thicker bars than the previous 0.35


# ============================================================
# data wrangling
# ============================================================
def build_phase3_tables():
    """Return (panelA_df, panelB_df) with means + SE per process and per cell."""
    df = pd.read_csv(F_RAW)
    df = df[(df["model"] == "lgbm") & (df["status"] == "ok")].copy()

    # per (target, family, split_type): mean and SE across folds
    g = (df.groupby(["target", "process", "layer",
                     "descriptor_family", "split_type"])["r2"]
           .agg(["mean", "std", "count"])
           .reset_index())
    g["se"] = g["std"] / np.sqrt(g["count"])

    rand = g[g["split_type"] == "random"].rename(
        columns={"mean": "r2_rand_mean", "se": "r2_rand_se"})
    ood = g[g["split_type"] == "balanced_topology_group"].rename(
        columns={"mean": "r2_ood_mean", "se": "r2_ood_se"})

    keys = ["target", "process", "layer", "descriptor_family"]
    per_target = rand[keys + ["r2_rand_mean", "r2_rand_se"]].merge(
        ood[keys + ["r2_ood_mean", "r2_ood_se"]], on=keys, how="inner")

    per_target["gap_mean"] = per_target["r2_rand_mean"] - per_target["r2_ood_mean"]
    per_target["gap_se"] = np.sqrt(per_target["r2_rand_se"]**2 +
                                   per_target["r2_ood_se"]**2)

    # Panel B: per (process, layer) cell under recommended family
    rows = []
    for (proc, layer), fam in RECOMMENDED.items():
        sub = per_target[(per_target["process"] == proc) &
                         (per_target["layer"] == layer) &
                         (per_target["descriptor_family"] == fam)]
        if sub.empty:
            continue
        n = len(sub)
        rows.append({
            "process": proc,
            "layer": layer,
            "family": fam,
            "n_targets": n,
            "r2_rand_mean": sub["r2_rand_mean"].mean(),
            "r2_rand_se":   np.sqrt((sub["r2_rand_se"]**2).sum()) / n,
            "r2_ood_mean":  sub["r2_ood_mean"].mean(),
            "r2_ood_se":    np.sqrt((sub["r2_ood_se"]**2).sum()) / n,
            "gap_mean":     sub["gap_mean"].mean(),
            "gap_se":       np.sqrt((sub["gap_se"]**2).sum()) / n,
        })
    panelB = pd.DataFrame(rows)

    # Panel A: per-process bar = mean of cell means across layers
    rows = []
    for proc in PROCESS_ORDER:
        sub = panelB[panelB["process"] == proc]
        if sub.empty:
            continue
        n = len(sub)
        rows.append({
            "process": proc,
            "n_layers": n,
            "r2_rand_mean": sub["r2_rand_mean"].mean(),
            "r2_rand_se":   np.sqrt((sub["r2_rand_se"]**2).sum()) / n,
            "r2_ood_mean":  sub["r2_ood_mean"].mean(),
            "r2_ood_se":    np.sqrt((sub["r2_ood_se"]**2).sum()) / n,
            "gap_mean":     sub["gap_mean"].mean(),
            "gap_se":       np.sqrt((sub["gap_se"]**2).sum()) / n,
        })
    panelA = pd.DataFrame(rows)

    return panelA, panelB


# ============================================================
# Figure 1
# ============================================================
def figure1():
    panelA, panelB = build_phase3_tables()

    fig = plt.figure(figsize=(9.2, 8.6))
    gs = fig.add_gridspec(2, 2,
                          height_ratios=[1.0, 1.0],
                          width_ratios=[1.0, 0.035],
                          hspace=0.45, wspace=0.03)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # ---------- Panel A ----------
    n_proc = len(PROCESS_ORDER)
    y = np.arange(n_proc)[::-1]   # methane on top

    for i, proc in enumerate(PROCESS_ORDER):
        r = panelA[panelA["process"] == proc]
        if r.empty:
            continue
        r = r.iloc[0]
        yy = y[i]

        # range bar OOD → random
        axA.barh(yy, width=r["gap_mean"], left=r["r2_ood_mean"],
                 height=BAR_HEIGHT, color=BAR_COLOR,
                 edgecolor="black", linewidth=0.7, zorder=2)

        # SE caps — both ends use propagated gap SE so they are visible
        axA.errorbar(r["r2_ood_mean"], yy, xerr=r["gap_se"],
                     fmt="none", ecolor="black",
                     elinewidth=1.6, capsize=6, capthick=1.6, zorder=4)
        axA.errorbar(r["r2_ood_mean"] + r["gap_mean"], yy, xerr=r["gap_se"],
                     fmt="none", ecolor="black",
                     elinewidth=1.6, capsize=6, capthick=1.6, zorder=4)

        # endpoint R² labels, outside the caps
        axA.text(r["r2_ood_mean"] - r["gap_se"] - 0.012, yy,
                 f"{r['r2_ood_mean']:.2f}",
                 va="center", ha="right",
                 fontsize=9.5, fontweight="bold", color="#333333")
        axA.text(r["r2_ood_mean"] + r["gap_mean"] + r["gap_se"] + 0.012, yy,
                 f"{r['r2_rand_mean']:.2f}",
                 va="center", ha="left",
                 fontsize=9.5, fontweight="bold", color="#333333")

    axA.set_xlim(0, 1.10)
    axA.set_ylim(-0.6, n_proc - 0.4)
    axA.set_yticks(y)
    axA.set_yticklabels([PROCESS_DISPLAY[p] for p in PROCESS_ORDER],
                        fontweight="bold", fontsize=11)
    axA.set_xlabel(r"$R^{2}$  (left: topology-OOD $\;\rightarrow\;$ right: random)",
                   fontweight="bold", fontsize=11)
    axA.tick_params(axis="y", length=0)
    axA.tick_params(axis="x", labelsize=10)
    axA.grid(axis="x", ls=":", alpha=0.35)
    axA.text(-0.22, 1.04, "A", transform=axA.transAxes,
             fontsize=22, fontweight="bold", ha="left", va="bottom")

    # ---------- Panel B ----------
    n_lay = len(LAYER_ORDER)
    M = np.full((n_proc, n_lay), np.nan)
    S = np.full((n_proc, n_lay), np.nan)
    for _, row in panelB.iterrows():
        i = PROCESS_ORDER.index(row["process"])
        j = LAYER_ORDER.index(row["layer"])
        M[i, j] = row["gap_mean"]
        S[i, j] = row["gap_se"]

    norm = Normalize(vmin=0.0, vmax=0.5)
    cmap = plt.get_cmap("Blues")
    im = axB.imshow(M, cmap=cmap, norm=norm, aspect="auto")

    # NaN cells: light grey
    for i in range(n_proc):
        for j in range(n_lay):
            if np.isnan(M[i, j]):
                axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=True, facecolor="#eeeeee",
                                        edgecolor="white", lw=0))

    # cell text + hatching for SE >= LOW_CONF_THRESHOLD
    for i in range(n_proc):
        for j in range(n_lay):
            v, s = M[i, j], S[i, j]
            if np.isnan(v):
                continue
            tcolor = "white" if v > 0.30 else "black"
            axB.text(j, i - 0.12, f"{v:+.2f}",
                     ha="center", va="center",
                     fontsize=12.5, fontweight="bold", color=tcolor)
            axB.text(j, i + 0.22, f"\u00b1{s:.2f}",
                     ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color=tcolor)
            if s >= LOW_CONF_THRESHOLD:
                axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        fill=False, hatch="////",
                                        edgecolor="black", lw=0.0))

    axB.set_xticks(range(n_lay))
    axB.set_xticklabels([LAYER_DISPLAY[l] for l in LAYER_ORDER],
                        fontweight="bold", fontsize=11)
    axB.set_yticks(range(n_proc))
    axB.set_yticklabels([PROCESS_DISPLAY[p] for p in PROCESS_ORDER],
                        fontweight="bold", fontsize=11)
    axB.tick_params(axis="both", length=0)
    axB.set_xlim(-0.5, n_lay - 0.5)
    axB.set_ylim(n_proc - 0.5, -0.5)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"random $R^{2}$ $-$ OOD $R^{2}$",
                   fontweight="bold", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    axB.text(-0.22, 1.04, "B", transform=axB.transAxes,
             fontsize=22, fontweight="bold", ha="left", va="bottom")

    # footnote — normal size, black
    fig.text(0.5, 0.012,
             r"Hatched cells: SE $\geq$ 0.10 (low confidence)",
             ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="black")

    fig.subplots_adjust(left=0.22, right=0.94, top=0.95, bottom=0.07)
    out_pdf = OUT / "fig1_topology_transfer_gap.pdf"
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[ok] wrote {out_pdf}")

    # ---------- sanity print ----------
    print("\n[Panel A — per-process range bars]")
    for _, r in panelA.iterrows():
        print(f"  {r['process']:<28s}  "
              f"OOD {r['r2_ood_mean']:.3f}\u00b1{r['r2_ood_se']:.3f}  "
              f"rand {r['r2_rand_mean']:.3f}\u00b1{r['r2_rand_se']:.3f}  "
              f"gap {r['gap_mean']:+.3f}\u00b1{r['gap_se']:.3f}")
    print("\n[Panel B — heatmap cells]")
    for _, r in panelB.iterrows():
        flag = "  LOW" if r["gap_se"] >= LOW_CONF_THRESHOLD else ""
        print(f"  {r['process']:<28s}  {r['layer']:<28s}  "
              f"{r['gap_mean']:+.3f}\u00b1{r['gap_se']:.3f}{flag}")


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
