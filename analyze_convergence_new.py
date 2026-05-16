import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def shade_block_by_idx(ax, i_left, i_right, label, facecolor, alpha=0.40):
    left = i_left - 0.5
    right = i_right + 0.5
    ax.axvspan(left, right, facecolor=facecolor, alpha=alpha, zorder=0)
    ax.text(
        (left + right) / 2,
        0.98,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
    )


def load_final_baseline(consensus_dict, key):
    df = consensus_dict["all_pairs_consensus_adjusted"][key]
    final_round = df["round"].max()
    return float(df.loc[df["round"] == final_round, "mean_consensus"].iloc[0])


def final_adjusted_values(panel_adj, agent_ratio, agent_style="Neutral", agent_basem="32B"):
    sub = panel_adj[
        (panel_adj["agent_ratio"] == agent_ratio)
        & (panel_adj["agent_style"] == agent_style)
        & (panel_adj["agent_basem"] == agent_basem)
    ]
    final_round = sub["round"].max()
    return sub.loc[sub["round"] == final_round, "y_adj"].to_numpy()


def main():
    os.makedirs("figures", exist_ok=True)

    with open("processed_data/consensus_dict_remove_ind.pkl", "rb") as f:
        consensus_dict = pickle.load(f)
    panel_adj = pd.read_pickle("processed_data/panel_adj.pkl")

    agent_ratio_list = ["Agents 0%", "Agents 12.5%", "Agents 33%", "Agents 50%", "Agents 75%"]
    agent_label_list = ["0%", "12.5%", "33.3%", "50%", "75%"]
    keys = [f"{r}|Neutral|32B" for r in agent_ratio_list]

    box_data = [final_adjusted_values(panel_adj, r) for r in agent_ratio_list[1:]]
    box_labels = agent_label_list[1:]
    baseline = load_final_baseline(consensus_dict, keys[0])

    phase_colors = {
        0: "#5aaa6b",
        1: "#e08c3a",
        2: "#e08c3a",
        3: "#4e9ecf",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    shade_block_by_idx(ax, 0, 0, "H1", facecolor="#d9f2d9")
    shade_block_by_idx(ax, 1, 2, "H2", facecolor="#ffe6cc")
    shade_block_by_idx(ax, 3, 3, "H3", facecolor="#d9e8ff")

    bp = ax.boxplot(
        box_data,
        positions=np.arange(len(box_data)),
        widths=0.45,
        patch_artist=True,
        notch=False,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2.0),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        boxprops=dict(linewidth=1.4),
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(phase_colors[i])
        patch.set_alpha(0.75)

    ax.axhline(baseline, linewidth=1.6, linestyle="--", alpha=0.8, color="grey")
    ax.text(
        0.02,
        baseline - 0.025,
        "Pure-human condition",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=12,
        color="grey",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
    )

    ax.set_ylim(0.46, 0.9)
    ax.set_ylabel("Final Consensus Level")
    ax.set_xlabel("Agent Ratio")
    ax.set_xticks(np.arange(len(box_labels)))
    ax.set_xticklabels(box_labels, fontsize=14)
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    plt.tight_layout()
    out = "figures/fig1b.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved at: {out}")


if __name__ == "__main__":
    main()
