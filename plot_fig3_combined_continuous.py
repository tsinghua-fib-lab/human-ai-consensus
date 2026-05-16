"""
plot_fig3_combined_continuous.py
================================
intermediate results. Uses continuous early-diffusion CLR (v2w).


  Col 1: NAdopt Early vs Late (human vs agent)
  Col 2: Final consensus strength
  Col 3: CLR (Conceptual-Lexical Ratio)

Required intermediate files (all pre-computed):
  ./processed_data/intermediate_result/all_contribution_results_v2w.pkl
  processed_data/consensus_dict_remove_ind.pkl

Usage:
  python plot_fig3_combined_continuous.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size":   9,
})

INTER_DIR = "./processed_data/intermediate_result/"
SAVE_DIR  = "./figures/"
os.makedirs(SAVE_DIR, exist_ok=True)


# Extra conditions to compute NAdopt for
# colours
C_HUMAN    = "#4e9ecf"
C_AGENT    = "#e08c3a"
C_NEUTRAL  = "#4e9ecf"
C_HA       = "#5aaa6b"
C_HP       = "#e08c3a"
C_32B      = "#4e9ecf"
C_7B       = "#e08c3a"

CLR_HIGH   = "#67b0a8"   # CLR > 1 (conceptual dominant)
CLR_LOW    = "#c0514e"   # CLR < 1 (lexical dominant)


# Helpers

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def summarize_early_late(mat, roles, role, first_k=20, last_k=20):
    """Return (early_mean, early_sem, late_mean, late_sem) for a role."""
    mat   = np.asarray(mat, dtype=float)
    roles = np.asarray(roles)
    T     = mat.shape[1]
    k1    = min(first_k, T)
    k2    = min(last_k,  T)

    sub = mat[roles == role, :]
    early_ind = np.nanmean(sub[:, np.arange(0, k1)],        axis=1)
    late_ind  = np.nanmean(sub[:, np.arange(T - k2, T)],    axis=1)

    def _ms(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return np.nan, np.nan
        return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

    return _ms(early_ind) + _ms(late_ind)


# Panel plotters

def panel_consensus(ax, consensus_dict, cond_keys, cond_labels, cond_colors,
                    baseline_key=None, title=""):
    """
    Final consensus strength from consensus_dict['all_pairs_consensus_adjusted'][key].
    Each key's value is a DataFrame with columns: round, mean_consensus, sem_consensus.
    """
    metric = "all_pairs_consensus_adjusted"
    x = np.arange(len(cond_keys), dtype=float)

    means, sems = [], []
    for cond in cond_keys:
        try:
            df = consensus_dict[metric][cond]
            final = df[df["round"] == df["round"].max()]
            m  = float(final["mean_consensus"].iloc[0])
            se = float(final["sem_consensus"].iloc[0])
            means.append(m); sems.append(1.96 * se)
        except (KeyError, IndexError):
            means.append(np.nan); sems.append(np.nan)

    for xi, (m, ci, color) in enumerate(zip(means, sems, cond_colors)):
        if np.isfinite(m):
            ax.errorbar(xi, m, yerr=ci,
                        fmt="s", color=color, markersize=6,
                        capsize=5, linewidth=1.4, zorder=4)

    # baseline
    if baseline_key:
        try:
            df = consensus_dict[metric][baseline_key]
            bm = float(df[df["round"] == df["round"].max()]["mean_consensus"].iloc[0])
            ax.axhline(bm, color="grey", lw=1.0, linestyle="--", alpha=0.7)
        except (KeyError, IndexError):
            pass

    # significance brackets (z test using mean SEM)
    for xi in range(len(cond_keys) - 1):
        try:
            df0 = consensus_dict[metric][cond_keys[xi]]
            df1 = consensus_dict[metric][cond_keys[xi+1]]
            f0  = df0[df0["round"] == df0["round"].max()]
            f1  = df1[df1["round"] == df1["round"].max()]
            m0, s0 = float(f0["mean_consensus"].iloc[0]), float(f0["sem_consensus"].iloc[0])
            m1, s1 = float(f1["mean_consensus"].iloc[0]), float(f1["sem_consensus"].iloc[0])
            z  = (m1 - m0) / np.sqrt(s0**2 + s1**2)
            from scipy.special import ndtr
            p  = 2 * (1 - ndtr(abs(z)))
            stars = ("***" if p < 0.001 else "**" if p < 0.01
                     else "*" if p < 0.05 else "")
            if stars:
                ymax = max(m0, m1)
                yr   = ax.get_ylim()
                off  = (yr[1] - yr[0]) * 0.04 if yr[1] > yr[0] else 0.005
                ax.plot([xi, xi+1], [ymax + off, ymax + off],
                        color="black", lw=0.9)
                ax.text((xi + xi + 1) / 2, ymax + off * 1.5, stars,
                        ha="center", va="bottom", fontsize=9)
        except (KeyError, IndexError):
            pass

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Final consensus strength")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.22, axis="y")


def panel_clr(ax, clr_dict, cond_labels, title=""):
    """
    CLR bar chart, coloured by > 1 (teal) or < 1 (red).
    """
    x = np.arange(len(cond_labels), dtype=float)
    for xi, lbl in enumerate(cond_labels):
        if lbl not in clr_dict:
            continue
        clr_val = clr_dict[lbl]["CLR"]
        color   = CLR_HIGH if clr_val >= 1.0 else CLR_LOW
        ax.bar(xi, clr_val, width=0.55, color=color, alpha=0.88)
        ax.text(xi, clr_val + 0.01, f"{clr_val:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(1.0, color="black", lw=1.0, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("CLR")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.22, axis="y")

    # legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=CLR_HIGH, label="CLR > 1 (conceptual dominant)"),
        Patch(facecolor=CLR_LOW,  label="CLR < 1 (lexical dominant)"),
    ], frameon=False, fontsize=7.5, loc="upper left")


# Main figure

def _get_consensus(consensus_dict, cons_key):
    """Return (mean, sem) of final-round consensus."""
    try:
        df = consensus_dict["all_pairs_consensus_adjusted"][cons_key]
        final = df[df["round"] == df["round"].max()]
        return float(final["mean_consensus"].iloc[0]), float(final["sem_consensus"].iloc[0])
    except (KeyError, IndexError):
        return np.nan, np.nan


def _get_clr(clr_dict, label):
    """Return CLR value for a label."""
    try:
        return float(clr_dict[label]["CLR"])
    except KeyError:
        return np.nan


def plot_discrete_outcomes(consensus_dict, all_contrib_results, group_name,
                          panel_adj=None, save_path=None):
    """
    Two panels for ONE experiment group:
      Left  panel: Final consensus strength (box plot from panel_adj)
      Right panel: CLR (bar chart, single value per condition)

    x-axis = discrete condition labels.
    """
    GROUP_DEFS = {
        "agent_trait": {
            "cons_keys":    ["Agents 33%|High Persistence|32B",
                             "Agents 33%|Neutral|32B",
                             "Agents 33%|High Adoption|32B"],
            "clr_labels":   ["High Persist.", "Neutral", "High Adopt."],
            "point_labels": ["High Persistence", "Neutral", "High Adoption"],
        },
        "base_model": {
            "cons_keys":    ["Agents 33%|Neutral|32B",
                             "Agents 33%|Neutral|7B"],
            "clr_labels":   ["32B", "7B"],
            "point_labels": ["32B", "7B"],
        },
    }

    COL_CON = "#4292c6"
    COL_CLR = "#b95a58"
    alpha   = 0.88

    cfg      = GROUP_DEFS[group_name]
    clr_dict = all_contrib_results[group_name]["clr"]
    n_cond   = len(cfg["cons_keys"])

    # collect data
    xs       = np.arange(n_cond, dtype=float)
    ys_clr   = []
    labels   = []
    box_data = []

    for ck, cl, lbl in zip(
            cfg["cons_keys"], cfg["clr_labels"], cfg["point_labels"]):

        clr_val = _get_clr(clr_dict, cl)
        ys_clr.append(clr_val)
        labels.append(lbl)

        # extract individual-level consensus from panel_adj
        if panel_adj is not None:
            parts = ck.split("|")  # e.g. ["Agents 33%", "High Persistence", "32B"]
            agent_ratio, agent_style, agent_basem = parts[0], parts[1], parts[2]
            sub = panel_adj[
                (panel_adj["agent_ratio"] == agent_ratio) &
                (panel_adj["agent_style"] == agent_style) &
                (panel_adj["agent_basem"] == agent_basem)
            ]
            final_round = sub["round"].max()
            y_adj = sub.loc[sub["round"] == final_round, "y_adj"].to_numpy()
            box_data.append(y_adj)
        else:
            box_data.append(np.array([]))

    # plot: 1 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4.2))
    bar_width = 0.55 if n_cond >= 3 else 0.45

    # Left panel: consensus strength (box plot)
    if panel_adj is not None and all(len(d) > 0 for d in box_data):
        bp = ax1.boxplot(
            box_data, positions=xs, widths=bar_width * 0.85,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=COL_CON, alpha=alpha),
            medianprops=dict(color="white", linewidth=1.4),
            whiskerprops=dict(color=COL_CON, linewidth=1.2),
            capprops=dict(color=COL_CON, linewidth=1.2),
        )
    else:
        # fallback: bar chart if panel_adj not available
        ys_con, ys_con_err = [], []
        for ck in cfg["cons_keys"]:
            m, s = _get_consensus(consensus_dict, ck)
            ys_con.append(m)
            ys_con_err.append(1.96 * s if np.isfinite(s) else 0.0)
        ax1.bar(xs, ys_con, width=bar_width, yerr=ys_con_err,
                color=COL_CON, alpha=alpha,
                error_kw={"capsize": 4, "linewidth": 1.1})

    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Final consensus strength", fontsize=13)
    ax1.grid(True, alpha=0.18, axis="y")

    # Right panel: CLR (bar chart)
    ax2.bar(xs, ys_clr, width=bar_width,
            color=COL_CLR, alpha=alpha)
    # only show CLR=1 reference line when any value approaches or exceeds 1
    if any(v >= 0.9 for v in ys_clr if np.isfinite(v)):
        ax2.axhline(1.0, color="black", lw=1.0, linestyle="--", alpha=0.6)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("CLR", fontsize=13)
    ax2.grid(True, alpha=0.18, axis="y")

    xlabel = "Agent Trait" if group_name == "agent_trait" else "Base Model"
    ax1.set_xlabel(xlabel, fontsize=13)
    ax2.set_xlabel(xlabel, fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# MAIN

if __name__ == "__main__":


    # Load contribution results (CLR)
    contrib_path = os.path.join(INTER_DIR, "all_contribution_results_v2w.pkl")
    if not os.path.exists(contrib_path):
        raise FileNotFoundError(
            f"Contribution results not found: {contrib_path}\n"
            "Run run_contribution_analysis_continuous.py first.")
    print("Preparing Fig. 3d-g...")
    all_contrib_results = load_pkl(contrib_path)

    # Load consensus dict
    consensus_path = "processed_data/consensus_dict_remove_ind.pkl"
    if not os.path.exists(consensus_path):
        raise FileNotFoundError(f"Consensus dict not found: {consensus_path}")
    consensus_dict = load_pkl(consensus_path)
    metric_keys = list(consensus_dict.get("all_pairs_consensus_adjusted", {}).keys())

    # Plot

    # Load panel_adj for individual level consensus box plots
    panel_adj_path = "processed_data/panel_adj.pkl"
    if os.path.exists(panel_adj_path):
        panel_adj = pd.read_pickle(panel_adj_path)
    else:
        panel_adj = None

    # Scatter plots: discrete conditions vs outcomes
    for gname in ["agent_trait", "base_model"]:
        plot_discrete_outcomes(
            consensus_dict, all_contrib_results,
            group_name=gname,
            panel_adj=panel_adj,
            save_path=os.path.join(SAVE_DIR, "fig3de.pdf" if gname == "agent_trait" else "fig3fg.pdf"),
        )

