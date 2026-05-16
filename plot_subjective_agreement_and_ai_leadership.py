# -*- coding: utf-8 -*-
"""
plot_subjective_agreement_and_ai_leadership.py
==============================================
Generates Fig. 4a.

Input:  results/merged_questionnaire_common_questions_results_{exp}.csv
Output: figures/fig4a.pdf
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
RESULTS_DIR = "processed_data/results"
FIGS_DIR    = "figures"
os.makedirs(FIGS_DIR, exist_ok=True)

# Experiment metadata
experiment2ratio = {
    'A2': 50.0, 'A3': 33.3, 'A4': 12.5, 'A5': 75.0,
}

# Column names
Q3_COL = "agreement"
Q6_COL = "ai_leadership"

# Load data
frames = []
for exp in experiment2ratio:
    path = os.path.join(RESULTS_DIR,
                        f"merged_questionnaire_common_questions_results_{exp}.csv")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        continue
    df = pd.read_csv(path)
    df["agent_ratio"] = experiment2ratio[exp]
    frames.append(df)

common_df = pd.concat(frames, ignore_index=True)
common_df[Q3_COL] = pd.to_numeric(common_df[Q3_COL], errors="coerce")
common_df[Q6_COL] = pd.to_numeric(common_df[Q6_COL], errors="coerce")

# Style
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42,
})

C_H1 = "#4e9ecf"
C_H2 = "#e08c3a"
C_H3 = "#5aaa6b"
PHASE_BG = [
    (0, 0, "#d6eaf8", "H1"),
    (1, 1, "#fde8d0", "H2"),
    (2, 2, "#d5f0dc", "H3"),
]

STAGES       = ["H1", "H2", "H3"]
STAGE_LABELS = ["H1", "H2", "H3"]
X_TICKS      = np.arange(len(STAGES))

# Map ratio stage
def map_stage(r):
    if r == 12.5:          return "H1"
    if r in [33.3, 50.0]:  return "H2"
    if r == 75.0:          return "H3"
    return np.nan

common_df["stage"] = common_df["agent_ratio"].apply(map_stage)

# Compute summaries by stage
def get_summary(col):
    sub = common_df[common_df["stage"].isin(STAGES)].dropna(subset=[col])
    return (sub.groupby("stage")[col]
               .agg(mean="mean", sem="sem")
               .reindex(STAGES))

s3 = get_summary(Q3_COL)
s6 = get_summary(Q6_COL)

y3 = s3["mean"].values;  e3 = s3["sem"].values
y6 = s6["mean"].values;  e6 = s6["sem"].values

base_df = common_df[common_df["agent_ratio"] == 0.0]
b3 = base_df[Q3_COL].mean()
b6 = base_df[Q6_COL].mean()

# Colors
C_Q3 = "#4292c6"
C_Q6 = "#b95a58"

# Figure: single panel, dual y axis
fig, ax1 = plt.subplots(figsize=(5.5, 4.2))
ax2 = ax1.twinx()

# phase bands
for i_lo, i_hi, color, label in PHASE_BG:
    ax1.axvspan(i_lo - 0.4, i_hi + 0.4, color=color, alpha=0.4, zorder=0)

## baselines

# Q3 agreement (left axis, blue)
ax1.errorbar(X_TICKS, y3, yerr=e3,
             fmt="o-", color=C_Q3, ms=6, lw=1.5,
             capsize=3, elinewidth=1.2, zorder=3,
             label="Agreement with final description")

# Q6 AI leadership (right axis, red, square markers)
ax2.errorbar(X_TICKS, y6, yerr=e6,
             fmt="s--", color=C_Q6, ms=6, lw=1.5,
             capsize=3, elinewidth=1.2, zorder=3,
             label="Perceived AI leadership")

# axis formatting
ax1.set_xticks(X_TICKS)
ax1.set_xticklabels(STAGE_LABELS, fontsize=11)
ax1.set_xlabel("Agent Proportion Regime", fontsize=14)
ax1.set_ylabel("Agreement with final description", color=C_Q3, fontsize=13)
ax2.set_ylabel("Perceived extent of AI leadership", color=C_Q6, fontsize=13)
ax1.tick_params(axis="y", labelcolor=C_Q3)
ax2.tick_params(axis="y", labelcolor=C_Q6)
ax1.set_xlim(-0.6, 2.6)

# align both y-axes to the same limits
all_vals = np.concatenate([y3 - e3, y3 + e3, y6 - e6, y6 + e6])
ymin = np.nanmin(all_vals)
ymax = np.nanmax(all_vals)
pad  = (ymax - ymin) * 0.15
ax1.set_ylim(ymin - pad, ymax + pad)
ax2.set_ylim(ymin - pad, ymax + pad)

ax1.grid(alpha=0.2, axis="y")

## legend
#]

plt.tight_layout()
plt.subplots_adjust(top=0.90)
out = os.path.join(FIGS_DIR, "fig4a.pdf")
plt.savefig(out, dpi=300, bbox_inches="tight")

plt.close()
print(f"Generated figure: {out}")
