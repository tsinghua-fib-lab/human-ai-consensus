import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def load_clr_results():
    path = "processed_data/intermediate_result/all_contribution_results_v2w.pkl"
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results["agent_ratio"]["clr"]


def main():
    os.makedirs("figures", exist_ok=True)

    labels = ["12.5%", "33.3%", "50%", "75%"]
    clr = load_clr_results()
    x = np.arange(len(labels))
    lexical = [clr[label]["C_lex"] for label in labels]
    conceptual = [clr[label]["C_concept"] for label in labels]
    ratio = [clr[label]["CLR"] for label in labels]

    fig_a, ax_a = plt.subplots(figsize=(6, 5))
    width = 0.35
    bars_lex = ax_a.bar(
        x - width / 2,
        lexical,
        width,
        color="#4292c6",
        label="Lexical-level contribution",
        alpha=0.88,
    )
    bars_concept = ax_a.bar(
        x + width / 2,
        conceptual,
        width,
        color="#b95a58",
        label="Conceptual-level contribution",
        alpha=0.88,
    )
    for bar in list(bars_lex) + list(bars_concept):
        h = bar.get_height()
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=12)
    ax_a.set_xlabel("Agent Ratio", fontsize=15)
    ax_a.set_ylabel("Agent contribution proportion", fontsize=15)
    ax_a.set_ylim(0, 1)
    ax_a.legend(frameon=False, fontsize=12)
    ax_a.grid(axis="y", linestyle="--", alpha=0.3)
    fig_a.tight_layout()
    out_a = "figures/fig2a.pdf"
    fig_a.savefig(out_a, dpi=300, bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(6, 5))
    colors = ["#4292c6" if v < 1.0 else "#b95a58" for v in ratio]
    ax_b.bar(x, ratio, width=0.5, color=colors, alpha=0.88)
    ax_b.axhline(1.0, color="black", linewidth=1.3, linestyle="--")
    for i, v in enumerate(ratio):
        ax_b.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=12)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=12)
    ax_b.set_xlabel("Agent Ratio", fontsize=15)
    ax_b.set_ylabel("Conceptual-Lexical Ratio (CLR)", fontsize=15)
    ax_b.set_ylim(0, max(ratio) * 1.2)
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)
    fig_b.tight_layout()
    out_b = "figures/fig2b.pdf"
    fig_b.savefig(out_b, dpi=300, bbox_inches="tight")
    plt.close(fig_b)

    print(f"Figures saved at: {out_a}, {out_b}")


if __name__ == "__main__":
    main()
