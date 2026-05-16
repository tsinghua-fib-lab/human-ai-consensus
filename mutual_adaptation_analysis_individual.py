"""
Individual-level Mutual Adaptation Analysis

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def compute_mutual_adaptation_metrics_individual(
    agent_ratio: str,
    pack: dict,
    normalize: bool = True
) -> dict:
    """


            H0: Human at t=0, shape (n_human, d)
            A0: Agent at t=0, shape (n_agent, d)
            Hf: Human at final, shape (n_human, d)
            Af: Agent at final, shape (n_agent, d)

    """

    H0 = pack['H0']  # (n_human, d)
    A0 = pack['A0']  # (n_agent, d)
    Hf = pack['Hf']
    Af = pack['Af']

    # Agent
    if A0 is None or Af is None:
        return {
            "agent_ratio": agent_ratio,
            "human_approach_mean": np.nan,
            "human_approach_std": np.nan,
            "human_approach_sem": np.nan,
            "human_approach_median": np.nan,
            "human_approach_iqr": np.nan,
            "human_proportion_approaching": np.nan,
            "agent_approach_mean": np.nan,
            "agent_approach_std": np.nan,
            "agent_approach_sem": np.nan,
            "agent_approach_median": np.nan,
            "agent_approach_iqr": np.nan,
            "agent_proportion_approaching": np.nan,
            "asymmetry_index": np.nan,
            "asymmetry_proportion": np.nan,
            "convergence_index": np.nan,
            "n_human": 0,
            "n_agent": 0,
        }

    n_human = H0.shape[0]
    n_agent = A0.shape[0]

    # individual embeddings
    if normalize:
        H0 = F.normalize(H0, p=2, dim=1)
        A0 = F.normalize(A0, p=2, dim=1)
        Hf = F.normalize(Hf, p=2, dim=1)
        Af = F.normalize(Af, p=2, dim=1)

    H0_center = torch.mean(H0, dim=0)
    A0_center = torch.mean(A0, dim=0)
    Hf_center = torch.mean(Hf, dim=0)
    Af_center = torch.mean(Af, dim=0)

    if normalize:
        H0_center = F.normalize(H0_center.unsqueeze(0), p=2, dim=1).squeeze(0)
        A0_center = F.normalize(A0_center.unsqueeze(0), p=2, dim=1).squeeze(0)
        Hf_center = F.normalize(Hf_center.unsqueeze(0), p=2, dim=1).squeeze(0)
        Af_center = F.normalize(Af_center.unsqueeze(0), p=2, dim=1).squeeze(0)

    # Individual level

    # 1. humanA
    human_approaches = []
    human_lateral_drifts = []
    human_total_movements = []

    for i in range(n_human):
        h0_i = H0[i]
        hf_i = Hf[i]
        delta_h_i = hf_i - h0_i

        # h0_iA0_center
        gap_i = A0_center - h0_i
        gap_norm_i = float(torch.norm(gap_i))

        if gap_norm_i > 1e-9:
            u_i = gap_i / gap_norm_i
            # agent
            approach_i = float(torch.dot(delta_h_i, u_i))
            # gap
            parallel_vec_i = approach_i * u_i
            lateral_vec_i = delta_h_i - parallel_vec_i
            lateral_i = float(torch.norm(lateral_vec_i))
        else:
            approach_i = 0.0
            lateral_i = float(torch.norm(delta_h_i))

        total_movement_i = float(torch.norm(delta_h_i))

        human_approaches.append(approach_i)
        human_lateral_drifts.append(lateral_i)
        human_total_movements.append(total_movement_i)

    # 2. agentH
    agent_approaches = []
    agent_lateral_drifts = []
    agent_total_movements = []

    for j in range(n_agent):
        a0_j = A0[j]
        af_j = Af[j]
        delta_a_j = af_j - a0_j

        # a0_jH0_center
        gap_j = H0_center - a0_j
        gap_norm_j = float(torch.norm(gap_j))

        if gap_norm_j > 1e-9:
            u_j = gap_j / gap_norm_j
            # human
            approach_j = float(torch.dot(delta_a_j, u_j))
            parallel_vec_j = approach_j * u_j
            lateral_vec_j = delta_a_j - parallel_vec_j
            lateral_j = float(torch.norm(lateral_vec_j))
        else:
            approach_j = 0.0
            lateral_j = float(torch.norm(delta_a_j))

        total_movement_j = float(torch.norm(delta_a_j))

        agent_approaches.append(approach_j)
        agent_lateral_drifts.append(lateral_j)
        agent_total_movements.append(total_movement_j)


    # numpy
    human_approaches = np.array(human_approaches)
    agent_approaches = np.array(agent_approaches)
    human_lateral_drifts = np.array(human_lateral_drifts)
    agent_lateral_drifts = np.array(agent_lateral_drifts)
    human_total_movements = np.array(human_total_movements)
    agent_total_movements = np.array(agent_total_movements)

    # Human
    h_mean = np.mean(human_approaches)
    h_std = np.std(human_approaches, ddof=1) if n_human > 1 else 0.0
    h_sem = h_std / np.sqrt(n_human) if n_human > 1 else 0.0
    h_median = np.median(human_approaches)
    h_q25 = np.percentile(human_approaches, 25)
    h_q75 = np.percentile(human_approaches, 75)
    h_iqr = h_q75 - h_q25
    h_proportion = np.mean(human_approaches > 0)

    # Agent
    a_mean = np.mean(agent_approaches)
    a_std = np.std(agent_approaches, ddof=1) if n_agent > 1 else 0.0
    a_sem = a_std / np.sqrt(n_agent) if n_agent > 1 else 0.0
    a_median = np.median(agent_approaches)
    a_q25 = np.percentile(agent_approaches, 25)
    a_q75 = np.percentile(agent_approaches, 75)
    a_iqr = a_q75 - a_q25
    a_proportion = np.mean(agent_approaches > 0)

    denom = abs(h_mean) + abs(a_mean)
    if denom > 1e-9:
        asymmetry_index = (h_mean - a_mean) / denom
    else:
        asymmetry_index = 0.0

    # Bootstrapasymmetry index
    n_bootstrap = 1000
    asymmetry_samples = []

    for _ in range(n_bootstrap):
        # individual approaches
        h_resample = np.random.choice(human_approaches, size=n_human, replace=True)
        a_resample = np.random.choice(agent_approaches, size=n_agent, replace=True)

        # asymmetry
        h_m_boot = np.mean(h_resample)
        a_m_boot = np.mean(a_resample)
        denom_boot = abs(h_m_boot) + abs(a_m_boot)

        if denom_boot > 1e-9:
            asym_boot = (h_m_boot - a_m_boot) / denom_boot
        else:
            asym_boot = 0.0

        asymmetry_samples.append(asym_boot)

    asymmetry_std = np.std(asymmetry_samples, ddof=1)
    asymmetry_sem = asymmetry_std  # Bootstrap SEM
    asymmetry_ci_lower = np.percentile(asymmetry_samples, 2.5)
    asymmetry_ci_upper = np.percentile(asymmetry_samples, 97.5)

    asymmetry_proportion = h_proportion - a_proportion

    # Bootstrapasymmetry proportion
    h_prop_var = h_proportion * (1 - h_proportion) / n_human
    a_prop_var = a_proportion * (1 - a_proportion) / n_agent
    asymmetry_proportion_sem = np.sqrt(h_prop_var + a_prop_var)

    initial_distance = float(torch.norm(A0_center - H0_center))
    final_distance = float(torch.norm(Af_center - Hf_center))
    if initial_distance > 1e-9:
        convergence_index = (initial_distance - final_distance) / initial_distance
    else:
        convergence_index = 0.0

    return {
        "agent_ratio": agent_ratio,

        # Human directional movement statistics
        "human_approach_mean": h_mean,
        "human_approach_std": h_std,
        "human_approach_sem": h_sem,
        "human_approach_median": h_median,
        "human_approach_q25": h_q25,
        "human_approach_q75": h_q75,
        "human_approach_iqr": h_iqr,
        "human_proportion_approaching": h_proportion,
        "human_lateral_mean": np.mean(human_lateral_drifts),
        "human_lateral_std": np.std(human_lateral_drifts, ddof=1) if n_human > 1 else 0.0,
        "human_total_movement_mean": np.mean(human_total_movements),
        "human_total_movement_std": np.std(human_total_movements, ddof=1) if n_human > 1 else 0.0,

        # Agent directional movement statistics
        "agent_approach_mean": a_mean,
        "agent_approach_std": a_std,
        "agent_approach_sem": a_sem,
        "agent_approach_median": a_median,
        "agent_approach_q25": a_q25,
        "agent_approach_q75": a_q75,
        "agent_approach_iqr": a_iqr,
        "agent_proportion_approaching": a_proportion,
        "agent_lateral_mean": np.mean(agent_lateral_drifts),
        "agent_lateral_std": np.std(agent_lateral_drifts, ddof=1) if n_agent > 1 else 0.0,
        "agent_total_movement_mean": np.mean(agent_total_movements),
        "agent_total_movement_std": np.std(agent_total_movements, ddof=1) if n_agent > 1 else 0.0,

        # Asymmetry indices
        "asymmetry_index": asymmetry_index,
        "asymmetry_std": asymmetry_std,
        "asymmetry_sem": asymmetry_sem,
        "asymmetry_ci_lower": asymmetry_ci_lower,
        "asymmetry_ci_upper": asymmetry_ci_upper,
        "asymmetry_proportion": asymmetry_proportion,
        "asymmetry_proportion_sem": asymmetry_proportion_sem,

        # Convergence
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "convergence_index": convergence_index,

        # Sample sizes
        "n_human": n_human,
        "n_agent": n_agent,

        # Raw individual data (for further analysis if needed)
        "human_approaches_raw": human_approaches,
        "agent_approaches_raw": agent_approaches,
        "human_lateral_raw": human_lateral_drifts,
        "agent_lateral_raw": agent_lateral_drifts,
    }


def visualize_mutual_adaptation_individual(df: pd.DataFrame, output_dir: str = "figures", agent_label_list: list = None, error_bar_type: str = "sem"):
    """

    """

    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12

    df_valid = df[df['human_approach_mean'].notna()].copy()

    if len(df_valid) == 0:
        print(" No valid data to visualize")
        return

    # x
    x_labels = df_valid['agent_ratio'].values
    x_pos = np.arange(len(x_labels))

    # error_bar_type
    if error_bar_type == "sem":
        h_err = df_valid['human_approach_sem'].values
        a_err = df_valid['agent_approach_sem'].values
        err_label = "SEM"
    elif error_bar_type == "std":
        h_err = df_valid['human_approach_std'].values
        a_err = df_valid['agent_approach_std'].values
        err_label = "SD"
    elif error_bar_type == "ci95":
        # 95% CI = 1.96 * SEM
        h_err = 1.96 * df_valid['human_approach_sem'].values
        a_err = 1.96 * df_valid['agent_approach_sem'].values
        err_label = "95% CI"
    elif error_bar_type == "iqr":
        # IQRerror barq25q75
        h_err = df_valid['human_approach_iqr'].values / 2
        a_err = df_valid['agent_approach_iqr'].values / 2
        err_label = "IQR/2"
    else:
        raise ValueError(f"Unknown error_bar_type: {error_bar_type}")

    # 1: Directional Movement (point + error bar)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Consistent color scheme with other figures
    COLOR_HUMAN = '#4292c6'
    COLOR_AGENT = '#b95a58'

    h_mean = df_valid['human_approach_mean'].values
    a_mean = df_valid['agent_approach_mean'].values

    # Human points and error bars
    ax.errorbar(x_pos - 0.15, h_mean, yerr=h_err,
                fmt='o', markersize=12, capsize=7, capthick=3,
                color=COLOR_HUMAN, ecolor=COLOR_HUMAN,
                label='Human → Agent', linewidth=2, alpha=0.8)

    # Agent points and error bars
    ax.errorbar(x_pos + 0.15, a_mean, yerr=a_err,
                fmt='^', markersize=12, capsize=7, capthick=3,
                color=COLOR_AGENT, ecolor=COLOR_AGENT,
                label='Agent → Human', linewidth=2, alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Agent Ratio', fontsize=15)
    ax.set_ylabel(f'Directional Movement', fontsize=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels if agent_label_list is None else agent_label_list, fontsize=13)
    ax.legend(frameon=True, fontsize=13, loc='best')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.2, 1)
    # Black border spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1c.pdf", dpi=300, bbox_inches='tight')
    print(f"Figure saved: {output_dir}/fig1c.pdf")
    plt.close()
    return

def print_detailed_metrics_table_individual(df: pd.DataFrame):
    return


if __name__ == "__main__":
    pass
