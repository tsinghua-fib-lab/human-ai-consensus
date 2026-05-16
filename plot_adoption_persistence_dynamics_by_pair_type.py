"""Generate Fig. 3a and Fig. 3b-c from precomputed intermediate results."""

import os
import sys
import time
import pickle
from typing import Any, Dict, List

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# colours
C_AA   = "#b95a58"   # agent / agent-agent
C_HH   = "#4292c6"   # human / human-human
C_HA   = "#5aaa6b"   # human-agent cross
alpha = 0.88

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size":   9,
})

# Shared helpers

def load_dataframe(fname):
    df = pickle.load(open(fname, 'rb'))
    df.pop('rules', None)
    return df

def is_agent(pid): return str(pid).lower().startswith("agent")
def is_human(pid): return str(pid).lower().startswith("human")

def truncate_to_full_rounds(players, answers, N):
    total = len(players)
    if total % N != 0:
        total = (total // N) * N
        players, answers = players[:total], answers[:total]
    return players, answers, len(players) // N

def collect_unique_texts(answers):
    texts = []
    for a0, a1 in answers:
        if str(a0): texts.append(str(a0))
        if str(a1): texts.append(str(a1))
    return list(dict.fromkeys(texts))

def collect_all_ids(players):
    ids = set()
    for pair in players:
        if pair and len(pair) >= 2:
            ids.add(str(pair[0])); ids.add(str(pair[1]))
    return sorted(ids)

def build_round_maps(players, total_rounds, N):
    maps = []
    for r in range(total_rounds):
        m = {}
        for j in range(N):
            g = r * N + j
            id0, id1 = str(players[g][0]), str(players[g][1])
            m[id0] = (g, 0); m[id1] = (g, 1)
        maps.append(m)
    return maps

def embed_expressions(expressions, model=None, tokenizer=None,
                      batch_size=64, normalize=True):
    if hasattr(model, "encode"):
        return np.asarray(model.encode(expressions, show_progress_bar=False, batch_size=batch_size, normalize_embeddings=normalize), dtype=float)
    model_device = next(model.parameters()).device
    def _pool(h, mask):
        m = mask.unsqueeze(-1).expand(h.size()).float()
        return torch.sum(h * m, 1) / torch.clamp(m.sum(1), min=1e-9)
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(expressions), batch_size):
            enc = tokenizer(expressions[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(model_device) for k, v in enc.items()}
            outs.append(_pool(model(**enc).last_hidden_state, enc["attention_mask"]).cpu().numpy())
    X = np.vstack(outs).astype(float)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


# Core computation

def mean_pairwise_sim(vecs: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity for a set of unit vectors."""
    n = len(vecs)
    if n < 2:
        return np.nan
    sim_mat = vecs @ vecs.T          # (n, n)
    # sum off-diagonal
    total = (sim_mat.sum() - np.trace(sim_mat)) / (n * (n - 1))
    return float(total)


def calc_attractor_dynamics(tracker, model=None, tokenizer=None, batch_size=64, N=12) -> Dict[str, Any]:
    """
    For each round r, compute:
      - agent-agent intra-group similarity
      - human-human intra-group similarity
      - human-agent cross-group similarity
        (mean sim between every human and every agent at that round)

    Also returns round-0 values separately for the bar chart.

    Returns
    -------
    dict:
      "aa_series"  : np.ndarray (T,)   agent-agent per round
      "hh_series"  : np.ndarray (T,)   human-human per round
      "ha_series"  : np.ndarray (T,)   human-agent cross per round
      "aa_r0"      : float             round-0 agent-agent
      "hh_r0"      : float             round-0 human-human
      "ha_r0"      : float             round-0 human-agent
      "n_agents"   : int
      "n_humans"   : int
    """
    players = tracker.get("players") or []
    answers = tracker.get("answers") or []
    players, answers, total_rounds = truncate_to_full_rounds(players, answers, N)

    unique_texts = collect_unique_texts(answers)
    if not unique_texts:
        return {}

    X_all = embed_expressions(unique_texts, model=model, tokenizer=tokenizer, batch_size=batch_size)
    txt2idx = {t: i for i, t in enumerate(unique_texts)}

    round_maps = build_round_maps(players, total_rounds, N)
    all_ids    = collect_all_ids(players)
    agent_ids  = [pid for pid in all_ids if is_agent(pid)]
    human_ids  = [pid for pid in all_ids if is_human(pid)]

    aa_series = np.full(total_rounds, np.nan)
    hh_series = np.full(total_rounds, np.nan)
    ha_series = np.full(total_rounds, np.nan)

    for r in range(total_rounds):
        rmap = round_maps[r]

        # collect embeddings for this round
        a_vecs, h_vecs = [], []
        for pid in agent_ids:
            if pid not in rmap: continue
            g, side = rmap[pid]
            idx = txt2idx.get(str(answers[g][side]))
            if idx is not None: a_vecs.append(X_all[idx])
        for pid in human_ids:
            if pid not in rmap: continue
            g, side = rmap[pid]
            idx = txt2idx.get(str(answers[g][side]))
            if idx is not None: h_vecs.append(X_all[idx])

        a_vecs = np.stack(a_vecs) if a_vecs else None
        h_vecs = np.stack(h_vecs) if h_vecs else None

        if a_vecs is not None and len(a_vecs) >= 2:
            aa_series[r] = mean_pairwise_sim(a_vecs)
        if h_vecs is not None and len(h_vecs) >= 2:
            hh_series[r] = mean_pairwise_sim(h_vecs)
        if a_vecs is not None and h_vecs is not None:
            # cross-group: mean sim between every human and every agent
            cross = h_vecs @ a_vecs.T          # (n_h, n_a)
            ha_series[r] = float(np.nanmean(cross))

    return {
        "aa_series": aa_series,
        "hh_series": hh_series,
        "ha_series": ha_series,
        "aa_r0":     float(aa_series[0]) if np.isfinite(aa_series[0]) else np.nan,
        "hh_r0":     float(hh_series[0]) if np.isfinite(hh_series[0]) else np.nan,
        "ha_r0":     float(ha_series[0]) if np.isfinite(ha_series[0]) else np.nan,
        "n_agents":  len(agent_ids),
        "n_humans":  len(human_ids),
        # individual-level similarity for initial (r=0) and final (r=T-1)
        # used for SEM in the initial-final two-point plot
        "_agent_ids": agent_ids,
        "_human_ids": human_ids,
        "_round_maps": round_maps,
        "_answers": answers,
        "_txt2idx": txt2idx,
        "_X_all": X_all,
        "_total_rounds": total_rounds,
    }


# Visualization

def _ind_sim_to_group(pid_list, other_pid_list, rmap, answers, txt2idx, X_all):
    """
    For each individual in pid_list, compute their mean cosine similarity
    to all individuals in other_pid_list at the given round (via rmap).
    Returns np.ndarray of per-individual means.
    """
    # collect other vecs
    other_vecs = []
    for pid in other_pid_list:
        if pid not in rmap: continue
        g, side = rmap[pid]
        idx = txt2idx.get(str(answers[g][side]))
        if idx is not None: other_vecs.append(X_all[idx])
    if not other_vecs:
        return np.full(len(pid_list), np.nan)
    other_mat = np.stack(other_vecs)   # (n_other, dim)

    ind_means = []
    for pid in pid_list:
        if pid not in rmap:
            ind_means.append(np.nan); continue
        g, side = rmap[pid]
        idx = txt2idx.get(str(answers[g][side]))
        if idx is None:
            ind_means.append(np.nan); continue
        e = X_all[idx]
        sims = other_mat @ e           # (n_other,)
        # exclude self if pid is in other_pid_list
        if pid in other_pid_list:
            self_idx = other_pid_list.index(pid)
            sims = np.concatenate([sims[:self_idx], sims[self_idx+1:]])
        ind_means.append(float(np.nanmean(sims)))
    return np.array(ind_means)


def _ms(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return np.nan, np.nan
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))


def _get_initial_final(d):
    """
    Extract per-individual similarities for initial (round 0) and
    final (round T-1) rounds, for all three pair types.
    Returns dict of {key: (initial_ind_arr, final_ind_arr)}.
    """
    agent_ids    = d["_agent_ids"]
    human_ids    = d["_human_ids"]
    round_maps   = d["_round_maps"]
    answers      = d["_answers"]
    txt2idx      = d["_txt2idx"]
    X_all        = d["_X_all"]
    T            = d["_total_rounds"]

    r0_map = round_maps[0]
    rT_map = round_maps[T - 1]

    out = {}
    # agent-agent: each agent vs all other agents
    out["aa"] = (
        _ind_sim_to_group(agent_ids, agent_ids, r0_map, answers, txt2idx, X_all),
        _ind_sim_to_group(agent_ids, agent_ids, rT_map, answers, txt2idx, X_all),
    )
    # human-human: each human vs all other humans
    out["hh"] = (
        _ind_sim_to_group(human_ids, human_ids, r0_map, answers, txt2idx, X_all),
        _ind_sim_to_group(human_ids, human_ids, rT_map, answers, txt2idx, X_all),
    )
    # human-agent: each human vs all agents
    out["ha"] = (
        _ind_sim_to_group(human_ids, agent_ids, r0_map, answers, txt2idx, X_all),
        _ind_sim_to_group(human_ids, agent_ids, rT_map, answers, txt2idx, X_all),
    )
    return out


def print_summary(dynamics_dict, condition_keys, condition_labels):
    return


def plot_baseline_comparison(dynamics_dict, condition_keys, condition_labels, save_path=None):
    """
    Single focused figure:
    Round-1 (pre-interaction) agent-agent vs human-human similarity,
    one grouped bar per condition, with individual-level SEM and
    one-sided t-test (AA > HH) significance annotation.

    This is the first step of the structural attractor argument:
    agents share a common linguistic prior before any interaction occurs.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2)) # (6, 4)

    x     = np.arange(len(condition_keys), dtype=float)
    width = 0.32

    aa_means, aa_sems = [], []
    hh_means, hh_sems = [], []
    pvals = []

    for cond in condition_keys:
        d = dynamics_dict[cond]
        if "_agent_ids" not in d:
            aa_means.append(np.nan); aa_sems.append(np.nan)
            hh_means.append(np.nan); hh_sems.append(np.nan)
            pvals.append(np.nan); continue

        # individual-level similarities at round 0
        aa_ind = _ind_sim_to_group(
            d["_agent_ids"], d["_agent_ids"],
            d["_round_maps"][0], d["_answers"], d["_txt2idx"], d["_X_all"])
        hh_ind = _ind_sim_to_group(
            d["_human_ids"], d["_human_ids"],
            d["_round_maps"][0], d["_answers"], d["_txt2idx"], d["_X_all"])

        m_aa, s_aa = _ms(aa_ind)
        m_hh, s_hh = _ms(hh_ind)
        aa_means.append(m_aa); aa_sems.append(s_aa)
        hh_means.append(m_hh); hh_sems.append(s_hh)

        # one-sided t-test: AA > HH
        aa_v = aa_ind[np.isfinite(aa_ind)]
        hh_v = hh_ind[np.isfinite(hh_ind)]
        if aa_v.size > 1 and hh_v.size > 1:
            pv = float(stats.ttest_ind(aa_v, hh_v, equal_var=False, alternative="greater").pvalue)
        else:
            pv = np.nan
        pvals.append(pv)

    aa_means = np.array(aa_means)
    hh_means = np.array(hh_means)

    ax.bar(x - width/2, aa_means, width, yerr=aa_sems, label="Agent–Agent",
           color=C_AA, alpha=alpha, error_kw={"capsize": 4, "linewidth": 1.1})
    ax.bar(x + width/2, hh_means, width, yerr=hh_sems, label="Human–Human",
           color=C_HH, alpha=alpha, error_kw={"capsize": 4, "linewidth": 1.1})

    # significance brackets
    def _stars(p):
        if not np.isfinite(p): return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "n.s."

    for xi, pv, m_aa, m_hh in zip(x, pvals, aa_means, hh_means):
        s = _stars(pv)
        if s and s != "n.s.":
            top = max(m_aa, m_hh) + 0.018
            ax.text(xi, top + 0.004, s, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=12)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=11)
    ax.set_xlabel("Agent Ratio", fontsize=15)
    ax.set_ylabel("Initial mean cosine similarity", fontsize=15)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(True, alpha=0.22, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_adoption_persistence_collapsed(
    adoption_series_dict, persistence_series_dict,
    condition_keys, first_k=20, last_k=20, save_path=None,
):
    """
    Two-panel figure replacing the coordination-index plot.

    Left  panel: Adoption   (Early vs Late), Human vs Agent
    Right panel: Persistence (Early vs Late), Human vs Agent

    Data **pooled across all conditions** in condition_keys, then
    Significance: Welch t-test, Human vs Agent at each time point.
    Style: grouped box plot (no outliers, white median line).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    x      = np.array([0.0, 1.0])
    xticks = ["Early", "Late"]

    panel_cfgs = [
        ("adoption",    adoption_series_dict,    "Adoption",    axes[0]),
        ("persistence", persistence_series_dict, "Persistence", axes[1]),
    ]

    legend_handles, legend_labels = None, None

    for metric_key, series_dict, ylabel, ax in panel_cfgs:

        # pool individual level window means across conditions
        all_human_early, all_human_late = [], []
        all_agent_early, all_agent_late = [], []

        for cond in condition_keys:
            per   = series_dict[cond]["per_individual"]
            roles = np.asarray(per["roles"])
            mat   = np.asarray(per[metric_key], dtype=float)
            T     = mat.shape[1]
            k1    = min(first_k, T)
            k2    = min(last_k,  T)
            early_idx = np.arange(0, k1)
            late_idx  = np.arange(T - k2, T)

            for who, e_list, l_list in [
                ("human", all_human_early, all_human_late),
                ("agent", all_agent_early, all_agent_late),
            ]:
                mask = (roles == who)
                sub  = mat[mask, :]
                if sub.size == 0:
                    continue
                e_ind = np.nanmean(sub[:, early_idx], axis=1)
                l_ind = np.nanmean(sub[:, late_idx],  axis=1)
                e_list.extend(e_ind[np.isfinite(e_ind)].tolist())
                l_list.extend(l_ind[np.isfinite(l_ind)].tolist())

        h_early = np.array(all_human_early)
        h_late  = np.array(all_human_late)
        a_early = np.array(all_agent_early)
        a_late  = np.array(all_agent_late)

        def _mean_sem(arr):
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            return m, s

        h_m_e, h_s_e = _mean_sem(h_early)
        h_m_l, h_s_l = _mean_sem(h_late)
        a_m_e, a_s_e = _mean_sem(a_early)
        a_m_l, a_s_l = _mean_sem(a_late)

        h_means = [h_m_e, h_m_l]
        h_sems  = [h_s_e, h_s_l]
        a_means = [a_m_e, a_m_l]
        a_sems  = [a_s_e, a_s_l]

        # grouped bar chart: x Human/Agent, bars Early/Late
        C_EARLY = "#a6cde3"   # lighter
        C_LATE  = "#2171b5"   # darker
        x_role  = np.array([0.0, 1.0])   # Human, Agent
        rlabels = ["Human", "Agent"]
        width   = 0.32

        early_vals = [h_m_e, a_m_e]
        early_errs = [h_s_e, a_s_e]
        late_vals  = [h_m_l, a_m_l]
        late_errs  = [h_s_l, a_s_l]

        bar_e = ax.bar(x_role - width/2, early_vals, width, yerr=early_errs,
                       color=C_EARLY, alpha=alpha, label="Early",
                       error_kw={"capsize": 4, "linewidth": 1.1})
        bar_l = ax.bar(x_role + width/2, late_vals,  width, yerr=late_errs,
                       color=C_LATE,  alpha=alpha, label="Late",
                       error_kw={"capsize": 4, "linewidth": 1.1})

        # significance: Early vs Late within each role
        def _stars(p):
            if not np.isfinite(p): return ""
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return "n.s."

        for xi, e_arr, l_arr, e_m, l_m, e_s, l_s in [
            (0.0, h_early, h_late, h_m_e, h_m_l, h_s_e, h_s_l),
            (1.0, a_early, a_late, a_m_e, a_m_l, a_s_e, a_s_l),
        ]:
            pv = float(stats.ttest_ind(e_arr, l_arr, equal_var=False).pvalue)
            s  = _stars(pv)
            if s and s != "n.s.":
                y_top = max(e_m, l_m) + 0.003 # max(e_m + e_s, l_m + l_s) + 0.003
                ax.text(xi, y_top, s, ha="center", va="bottom", fontsize=12)

        # axis styling
        ax.set_xticks([0, 1])
        ax.set_xticklabels(rlabels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.22, axis="y")

        if legend_handles is None:
            legend_handles = [bar_e, bar_l]
            legend_labels  = ["Early (Rounds 1-20)", "Late (Rounds 21-40)"]

    fig.legend(legend_handles, legend_labels,
               loc="lower center", ncol=2, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


# MAIN

if __name__ == "__main__":
    from modelscope import snapshot_download

    SAVE_DIR  = "./figures/"
    INTER_DIR = "./processed_data/intermediate_result/"
    os.makedirs(SAVE_DIR,  exist_ok=True)
    os.makedirs(INTER_DIR, exist_ok=True)

    CACHE_PATH = os.path.join(INTER_DIR, "attractor_dynamics_dict.pkl")


    # data
    fnames           = ["processed_data/A4.pkl",
                        "processed_data/A3.pkl",
                        "processed_data/A2.pkl",
                        "processed_data/A5.pkl"]
    agent_ratio_list = ["Agents 12.5%", "Agents 33%",
                        "Agents 50%",   "Agents 75%"]
    condition_keys   = [f"{r}|Neutral" for r in agent_ratio_list]
    condition_labels = ["12.5%", "33.3%", "50%", "75%"]

    # compute or load
    # Recompute if cache missing new individual-level fields
    def _cache_valid():
        if not os.path.exists(CACHE_PATH): return False
        with open(CACHE_PATH, "rb") as f:
            d = pickle.load(f)
        return all("_agent_ids" in d.get(c, {}) for c in condition_keys)

    if _cache_valid():
        print("Loading cached attractor dynamics ")
        with open(CACHE_PATH, "rb") as f:
            dynamics_dict = pickle.load(f)
    else:
        # Load the embedding model only when the cached attractor dynamics are absent.
        import utils as ut
        import torch
        from transformers import AutoTokenizer, AutoModel

        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
        CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "processed_data/model_cache/all-MiniLM-L6-v2")
        snapshot_download(MODEL_ID, cache_dir=CACHE_DIR)
        model_dir   = ut.find_hf_model_dir(CACHE_DIR)
        tokenizer   = AutoTokenizer.from_pretrained(model_dir)
        embed_model = AutoModel.from_pretrained(model_dir).to(device)
        embed_model.eval()

        print("Computing Fig. 3a intermediates...")
        dynamics_dict = {}
        t0 = time.time()
        for fname, cond in zip(fnames, condition_keys):
            df  = load_dataframe(fname)
            trk = df[0]["tracker"]
            dynamics_dict[cond] = calc_attractor_dynamics(
                trk, model=embed_model, tokenizer=tokenizer, N=12)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(dynamics_dict, f)

    # Focused baseline figure: AA vs HH at round 1
    plot_baseline_comparison(
        dynamics_dict, condition_keys, condition_labels,
        save_path=os.path.join(SAVE_DIR, "fig3a.pdf"),
    )

    # Fig 3b: Adoption & Persistence (collapsed across conditions)
    adopt_pkl = os.path.join(INTER_DIR, "adoption_series_dict_with_individual_results.pkl")
    perst_pkl = os.path.join(INTER_DIR, "persistence_series_dict_with_individual_results.pkl")

    if os.path.exists(adopt_pkl) and os.path.exists(perst_pkl):
        print("Loading adoption/persistence data ")
        with open(adopt_pkl, "rb") as f:
            adoption_series_dict = pickle.load(f)
        with open(perst_pkl, "rb") as f:
            persistence_series_dict = pickle.load(f)

        plot_adoption_persistence_collapsed(
            adoption_series_dict, persistence_series_dict,
            condition_keys, first_k=20, last_k=20,
            save_path=os.path.join(SAVE_DIR, "fig3bc.pdf"),
        )

    else:
        raise FileNotFoundError(
            "Missing required adoption/persistence intermediate files:\n"
            f"{adopt_pkl}\n{perst_pkl}"
        )

