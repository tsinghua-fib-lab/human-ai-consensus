# -*- coding: utf-8 -*-
"""
plot_ai_penalty_with_controls.py
=================================
Fig 4b + 4c: AI penalty regression with additional controls:
  - pair_sim  : cosine similarity between you and peer expressions
                (all-MiniLM-L6-v2, mean pooling + L2 normalize, aligned with
                 calculate_consensus.py)
  - peer_len  : word count of peer expression
  - you_len   : word count of your expression

Model b (global):
  willingness ~ C(ratio_cat) + judged_ai + pair_sim + peer_len + you_len

Model c (stage-specific):
  willingness ~ C(stage) + judged_ai*C(stage) + pair_sim + peer_len + you_len

Output: figures/fig4b.pdf, figures/fig4c.pdf, figures/fig4d.pdf, figures/table_s1.tex
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import statsmodels.formula.api as smf
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = "processed_data/results"
DATA_DIR    = "processed_data"
FIGS_DIR    = "figures"
MODEL_CACHE = os.environ.get("MODEL_CACHE_DIR", "processed_data/model_cache/all-MiniLM-L6-v2")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Experiment metadata ───────────────────────────────────────────────────────
experiment2ratio = {
    'A2': 50.0, 'A3': 33.3, 'A4': 12.5, 'A5': 75.0,
}

def map_stage(r):
    if r == 12.5:          return "H1"
    if r in [33.3, 50.0]:  return "H2"
    if r == 75.0:          return "H3"
    return np.nan

def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

# ── Find HF model dir ─────────────────────────────────────────────────────────
def find_hf_model_dir(cache_dir):
    for root, dirs, files in os.walk(cache_dir):
        if "config.json" in files and "tokenizer_config.json" in files:
            return root
    return cache_dir

# ── Load embedding model (aligned with calculate_consensus.py) ────────────────
print("Preparing Fig. 4b-d...")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = find_hf_model_dir(MODEL_CACHE)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
emb_model = AutoModel.from_pretrained(model_dir).to(device)
emb_model.eval()

@torch.no_grad()
def encode_sentences(texts, batch_size=64):
    """Mean pooling + L2 normalize, aligned with calculate_consensus.py."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        outputs = emb_model(**inputs)
        mask = inputs["attention_mask"]
        emb = (outputs.last_hidden_state * mask.unsqueeze(-1)).sum(1) \
              / mask.sum(1, keepdim=True)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)

def cosine_sim_pairs(texts_a, texts_b):
    """Cosine similarity for each (a, b) pair."""
    emb_a = encode_sentences(texts_a)
    emb_b = encode_sentences(texts_b)
    return (emb_a * emb_b).sum(dim=1).numpy()

# ── Load questionnaire data ───────────────────────────────────────────────────
frames = []
missing_questionnaire_files = []
for exp in experiment2ratio:
    path = os.path.join(RESULTS_DIR,
        f"merged_questionnaire_personalized_questions_results_{exp}.csv")
    if not os.path.exists(path):
        missing_questionnaire_files.append(path)
        continue
    df = pd.read_csv(path)
    df["agent_ratio"] = experiment2ratio[exp]
    df["exp_id"] = exp
    frames.append(df)

if missing_questionnaire_files:
    missing = "\n".join(missing_questionnaire_files)
    raise FileNotFoundError(
        "Missing required personalized questionnaire file(s):\n"
        f"{missing}\n"
        "Fig. 4b-d requires A4, A3, A2, and A5. "
        "Do not silently skip missing conditions, because that changes H2/H3 results."
    )

raw = pd.concat(frames, ignore_index=True)
raw["willingness"] = pd.to_numeric(raw["willingness"], errors="coerce")
raw["agent_ratio"] = pd.to_numeric(raw["agent_ratio"],  errors="coerce")
raw["judged_ai"] = pd.to_numeric(raw["judged_ai"], errors="coerce")

df_all = (raw[raw["agent_ratio"].isin([12.5, 33.3, 50.0, 75.0])]
          .dropna(subset=["willingness", "judged_ai", "you", "peer"])
          .copy()
          .reset_index(drop=True))

# ── Compute text features ─────────────────────────────────────────────────────
df_all["pair_sim"]   = cosine_sim_pairs(
    df_all["you"].astype(str).tolist(),
    df_all["peer"].astype(str).tolist()
)
df_all["peer_len"]   = df_all["peer"].astype(str).str.split().str.len()

# z-score standardize continuous predictors (judged_ai kept as binary)
df_all["pair_sim_z"] = (df_all["pair_sim"] - df_all["pair_sim"].mean()) / df_all["pair_sim"].std()
df_all["peer_len_z"] = (df_all["peer_len"] - df_all["peer_len"].mean()) / df_all["peer_len"].std()

# ── Compute group centroid per round from pkl data ────────────────────────────
experiment2pkl = {
    'A2': os.path.join(DATA_DIR, "A2.pkl"),
    'A3': os.path.join(DATA_DIR, "A3.pkl"),
    'A4': os.path.join(DATA_DIR, "A4.pkl"),
    'A5': os.path.join(DATA_DIR, "A5.pkl"),
}
N_PAIRS = 12

def compute_round_centroids(tracker, n_pairs=12):
    players = tracker['players']
    answers = tracker['answers']
    total_rounds = len(players) // n_pairs
    all_texts = []
    round_text_indices = {}
    for r in range(total_rounds):
        texts_this_round = []
        for j in range(n_pairs):
            g = r * n_pairs + j
            t0 = str(answers[g][0]).strip()
            t1 = str(answers[g][1]).strip()
            if t0: texts_this_round.append(t0)
            if t1: texts_this_round.append(t1)
        start_idx = len(all_texts)
        all_texts.extend(texts_this_round)
        round_text_indices[r] = (start_idx, len(all_texts))
    if not all_texts:
        return {}
    all_embs = encode_sentences(all_texts)
    centroids = {}
    for r, (si, ei) in round_text_indices.items():
        if ei > si:
            centroid = all_embs[si:ei].mean(dim=0)
            centroid = centroid / centroid.norm()
            centroids[r] = centroid.numpy()
    return centroids

centroid_lookup = {}
for exp_id, pkl_path in experiment2pkl.items():
    if not os.path.exists(pkl_path):
        continue
    data = pickle.load(open(pkl_path, 'rb'))
    if 'rules' in data: data.pop('rules')
    tracker = data[0]['tracker']
    centroids = compute_round_centroids(tracker, n_pairs=N_PAIRS)
    for r, vec in centroids.items():
        centroid_lookup[(exp_id, r)] = vec

# ── Compute peer_centroid_sim ─────────────────────────────────────────────────
df_all["round"] = pd.to_numeric(df_all["round"], errors="coerce").astype(int)
peer_texts = df_all["peer"].astype(str).tolist()
peer_embs = encode_sentences(peer_texts)

peer_centroid_sims = np.full(len(df_all), np.nan)
matched, missing = 0, 0
for idx in range(len(df_all)):
    key = (df_all.loc[idx, "exp_id"], df_all.loc[idx, "round"])
    if key in centroid_lookup:
        centroid = torch.tensor(centroid_lookup[key], dtype=torch.float32)
        sim = float(torch.dot(peer_embs[idx], centroid))
        peer_centroid_sims[idx] = sim
        matched += 1
    else:
        missing += 1

df_all["peer_centroid_sim"] = peer_centroid_sims
df_all["peer_centroid_sim_z"] = (df_all["peer_centroid_sim"] - df_all["peer_centroid_sim"].mean()) / df_all["peer_centroid_sim"].std()
df_all = df_all.dropna(subset=["peer_centroid_sim_z"]).reset_index(drop=True)

df_all["ratio_cat"] = df_all["agent_ratio"].astype(str)
df_all["stage"]     = df_all["agent_ratio"].apply(map_stage)
df_all["stage"]     = pd.Categorical(df_all["stage"],
                                      categories=["H1","H2","H3"], ordered=True)

# ── Model 1: global with controls ────────────────────────────────────────────
m_all = smf.ols(
    "willingness ~ C(ratio_cat) + judged_ai + pair_sim_z + peer_len_z + peer_centroid_sim_z",
    data=df_all
).fit(cov_type="HC3")

beta = m_all.params["judged_ai"]
se_b = m_all.bse["judged_ai"]
p_b  = m_all.pvalues["judged_ai"]
ci_b = m_all.conf_int().loc["judged_ai"].tolist()

# ── Models 2a/2b/2c: separate OLS per stage ──────────────────────────────────
# Standardization uses global mean/std (computed above) for comparability

stage_defs = {
    "H1": df_all[df_all["agent_ratio"] == 12.5].copy(),
    "H2": df_all[df_all["agent_ratio"].isin([33.3, 50.0])].copy(),
    "H3": df_all[df_all["agent_ratio"] == 75.0].copy(),
}

b = {}
for stage, sub in stage_defs.items():
    sub["ratio_cat_s"] = sub["agent_ratio"].astype(str)
    if stage == "H2":
        formula = "willingness ~ C(ratio_cat_s) + judged_ai + pair_sim_z + peer_len_z + peer_centroid_sim_z"
    else:
        formula = "willingness ~ judged_ai + pair_sim_z + peer_len_z + peer_centroid_sim_z"
    m = smf.ols(formula, data=sub).fit(cov_type="HC3")
    coef = m.params["judged_ai"]
    se   = m.bse["judged_ai"]
    p    = m.pvalues["judged_ai"]
    b[stage] = (coef, se, p, (coef - 1.96*se, coef + 1.96*se))

# ── Z-tests comparing adjacent stages ────────────────────────────────────────
def z_test_diff(b1, b2):
    diff = b2[0] - b1[0]
    se   = np.sqrt(b1[1]**2 + b2[1]**2)
    z    = diff / se
    p    = float(2 * (1 - stats.norm.cdf(abs(z))))
    return diff, se, p

diff_H2_H1, se_H2_H1, p_H2_H1 = z_test_diff(b["H1"], b["H2"])
diff_H3_H2, se_H3_H2, p_H3_H2 = z_test_diff(b["H2"], b["H3"])


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial", "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42,
})
COLOR_HUMAN = "#4292c6"
COLOR_AGENT = "#b95a58"
C_H1, C_H2, C_H3 = "#4e9ecf", "#e08c3a", "#5aaa6b"
STAGE_COLORS = {"H1": C_H1, "H2": C_H2, "H3": C_H3}
STAGES       = ["H1", "H2", "H3"]
STAGE_LABELS = ["H1", "H2", "H3"]
Y_POS        = [2, 1, 0]

# ── Panel b (new): Δ adoption willingness (judged AI − judged Human) by stage ─
will_diff_data = []
for stage_name in STAGES:
    sub = stage_defs[stage_name]
    h_vals = sub[sub["judged_ai"] == 0]["willingness"].dropna()
    a_vals = sub[sub["judged_ai"] == 1]["willingness"].dropna()
    diff = a_vals.mean() - h_vals.mean()
    se   = np.sqrt(a_vals.sem()**2 + h_vals.sem()**2)
    # Mann-Whitney U test
    if len(h_vals) > 0 and len(a_vals) > 0:
        u, p = stats.mannwhitneyu(a_vals, h_vals, alternative="two-sided")
    else:
        p = np.nan
    will_diff_data.append({"stage": stage_name, "diff": diff, "se": se, "p": p,
                           "n_h": len(h_vals), "n_a": len(a_vals)})

will_diffs = [d["diff"] for d in will_diff_data]
will_ses   = [d["se"]   for d in will_diff_data]
will_ps    = [d["p"]    for d in will_diff_data]

x_pos_b = np.arange(len(STAGES))
bar_colors = [C_H1, C_H2, C_H3]

PHASE_BG = [
    (0, 0, "#d6eaf8"),
    (1, 1, "#fde8d0"),
    (2, 2, "#d5f0dc"),
]

fig_bw, ax_bw = plt.subplots(figsize=(5.5, 4.2))

# Phase background
for i_lo, i_hi, color in PHASE_BG:
    ax_bw.axvspan(i_lo - 0.4, i_hi + 0.4, color=color, alpha=0.4, zorder=0)

ax_bw.axhline(0, color="black", lw=0.9, ls="--", alpha=0.45, zorder=1)

ax_bw.bar(x_pos_b, will_diffs, yerr=will_ses,
          width=0.5, color=bar_colors, edgecolor="white", lw=0.8,
          capsize=5, error_kw=dict(lw=1.5, capthick=1.5),
          zorder=3, alpha=0.85)

ax_bw.set_xticks(x_pos_b)
ax_bw.set_xticklabels(STAGE_LABELS, fontsize=11)
ax_bw.set_xlabel("Agent Proportion Regime", fontsize=14)
ax_bw.set_ylabel("Adoption willingness gap", fontsize=13)
ax_bw.set_xlim(-0.6, 2.6)
ax_bw.grid(alpha=0.2, axis="y")

fig_bw.tight_layout()
out_bw = os.path.join(FIGS_DIR, "fig4b.pdf")
fig_bw.savefig(out_bw, dpi=300, bbox_inches="tight")

plt.close(fig_bw)
print(f"Generated figure: {out_bw}")

# ── Panel c (forest plot): all non-fixed-effect predictors ───────────────────
# Extract coef, CI, p for each predictor
predictors = [
    ("judged_ai",            "Perceived AI\nidentity"),
    ("pair_sim_z",           "Self–peer\nsimilarity"),
    ("peer_len_z",           "Peer description\nlength"),
    ("peer_centroid_sim_z",  "Peer–centroid\nproximity"),
]

def point_color(coef, p):
    if p >= 0.05:   return "#999999"           # n.s. → grey
    if coef > 0:    return COLOR_AGENT          # positive significant → agent red
    return COLOR_HUMAN                          # negative significant → human blue

fig_b, ax_b = plt.subplots(figsize=(5.5, 4.2))
ax_b.axvline(0, color="black", lw=0.9, ls="--", alpha=0.45, zorder=1)

for y_pos, (term, label) in enumerate(reversed(predictors)):
    coef  = m_all.params[term]
    ci_lo = m_all.conf_int().loc[term, 0]
    ci_hi = m_all.conf_int().loc[term, 1]
    p     = m_all.pvalues[term]
    color = point_color(coef, p)

    ax_b.plot([ci_lo, ci_hi], [y_pos, y_pos],
              color=color, lw=2.5, solid_capstyle="round", zorder=2)
    ax_b.plot(coef, y_pos, "D", color=color, ms=7, zorder=4)
    p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax_b.text(ci_hi + 0.02, y_pos, f"{sig_label(p)}  {p_str}",
              va="center", ha="left", fontsize=8, color=color)

n_preds = len(predictors)
ax_b.set_yticks(range(n_preds))
ax_b.set_yticklabels([label for _, label in reversed(predictors)], fontsize=11)
ax_b.set_ylim(-0.6, n_preds - 0.4)

all_ci_b = [m_all.conf_int().loc[t] for t, _ in predictors]
ax_b.set_xlim(min(c[0] for c in all_ci_b) - 0.15,
              max(c[1] for c in all_ci_b) + 0.65)
ax_b.set_xlabel("Effect on adoption willingness (β, 95% CI)", fontsize=12)
fig_b.tight_layout()
out_b = os.path.join(FIGS_DIR, "fig4c.pdf")
fig_b.savefig(out_b, dpi=300, bbox_inches="tight")

plt.close(fig_b)
print(f"Generated figure: {out_b}")

# ── Panel d: peer_centroid_sim by judged identity × stage ─────────────────────
STAGES       = ["H1", "H2", "H3"]
STAGE_LABELS = ["H1", "H2", "H3"]
C_H1, C_H2, C_H3 = "#4e9ecf", "#e08c3a", "#5aaa6b"
STAGE_COLORS_LIST = [C_H1, C_H2, C_H3]
PHASE_BG = [
    (0, 0, "#d6eaf8"),
    (1, 1, "#fde8d0"),
    (2, 2, "#d5f0dc"),
]

C_HUMAN = COLOR_HUMAN   # blue — judged as human
C_AI    = COLOR_AGENT   # red  — judged as AI

# Compute mean ± SEM per stage × judged identity
gap_data = []
for stage_name, sub in stage_defs.items():
    for j, label in [(0, "Judged Human"), (1, "Judged AI")]:
        vals = sub[sub["judged_ai"] == j]["peer_centroid_sim"].dropna()
        gap_data.append({
            "stage": stage_name,
            "judged": label,
            "mean": vals.mean(),
            "sem": vals.sem(),
            "n": len(vals),
        })

gap_df = pd.DataFrame(gap_data)

# Statistical tests (Mann-Whitney U per stage)
for stage_name, sub in stage_defs.items():
    h_vals = sub[sub["judged_ai"] == 0]["peer_centroid_sim"].dropna()
    a_vals = sub[sub["judged_ai"] == 1]["peer_centroid_sim"].dropna()
    if len(h_vals) > 0 and len(a_vals) > 0:
        u, p = stats.mannwhitneyu(a_vals, h_vals, alternative="greater")

# Plot — difference (AI - Human) per stage
fig_c, ax_c = plt.subplots(figsize=(5.5, 4.2))

x_pos = np.arange(len(STAGES))

# Phase background
for i_lo, i_hi, color in PHASE_BG:
    ax_c.axvspan(i_lo - 0.4, i_hi + 0.4, color=color, alpha=0.4, zorder=0)

# Zero line
ax_c.axhline(0, color="black", lw=0.9, ls="--", alpha=0.45, zorder=1)

# Compute difference and propagated SEM per stage
diff_means = []
diff_sems  = []
for stage_name, sub in stage_defs.items():
    h_vals = sub[sub["judged_ai"] == 0]["peer_centroid_sim"].dropna()
    a_vals = sub[sub["judged_ai"] == 1]["peer_centroid_sim"].dropna()
    diff = a_vals.mean() - h_vals.mean()
    se   = np.sqrt(a_vals.sem()**2 + h_vals.sem()**2)
    diff_means.append(diff)
    diff_sems.append(se)

diff_means = np.array(diff_means)
diff_sems  = np.array(diff_sems)

# Bar plot
bar_colors = [C_H1, C_H2, C_H3]
bars = ax_c.bar(x_pos, diff_means, yerr=diff_sems,
                width=0.5, color=bar_colors, edgecolor="white", lw=0.8,
                capsize=5, error_kw=dict(lw=1.5, capthick=1.5),
                zorder=3, alpha=0.85)

ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(STAGE_LABELS, fontsize=11)
ax_c.set_xlabel("Agent Proportion Regime", fontsize=14)
ax_c.set_ylabel("Peer–centroid proximity gap", fontsize=13)
ax_c.set_xlim(-0.6, 2.6)
ax_c.grid(alpha=0.2, axis="y")

fig_c.tight_layout()
out_c = os.path.join(FIGS_DIR, "fig4d.pdf")
fig_c.savefig(out_c, dpi=300, bbox_inches="tight")

plt.close(fig_c)
print(f"Generated figure: {out_c}")

# H1+H2 pooled: is gap negative?
h12 = pd.concat([stage_defs["H1"], stage_defs["H2"]])
h12_human = h12[h12["judged_ai"] == 0]["willingness"].dropna()
h12_ai    = h12[h12["judged_ai"] == 1]["willingness"].dropna()
u_h12, p_h12 = stats.mannwhitneyu(h12_ai, h12_human, alternative="two-sided")
diff_h12 = h12_ai.mean() - h12_human.mean()

# H3 vs H1+H2: does the gap significantly change?
h123 = pd.concat([stage_defs["H1"], stage_defs["H2"], stage_defs["H3"]]).copy()
h123["is_H3"] = (h123["stage"] == "H3").astype(int)
m_will_gap = smf.ols("willingness ~ judged_ai * is_H3", data=h123).fit(cov_type="HC3")
interact_coef = m_will_gap.params["judged_ai:is_H3"]
interact_p    = m_will_gap.pvalues["judged_ai:is_H3"]

# --- Fig 4c: regression coefficients ---
for term, label in predictors:
    coef = m_all.params[term]
    ci   = m_all.conf_int().loc[term]
    p    = m_all.pvalues[term]

# --- Fig 4d: peer-centroid proximity gap H1 vs H3 ---

# Per-stage gap values (already computed)
# H1+H2 vs H3 gap difference: interaction test (judged_ai × is_H3) on peer_centroid_sim
df_h1h2h3 = pd.concat([stage_defs["H1"], stage_defs["H2"], stage_defs["H3"]]).copy()
df_h1h2h3["is_H3"] = (df_h1h2h3["stage"] == "H3").astype(int)
m_gap = smf.ols("peer_centroid_sim ~ judged_ai * is_H3", data=df_h1h2h3).fit(cov_type="HC3")
interaction_coef = m_gap.params["judged_ai:is_H3"]
interaction_p    = m_gap.pvalues["judged_ai:is_H3"]

# H1 vs H3 only
df_h1h3 = pd.concat([stage_defs["H1"], stage_defs["H3"]]).copy()
df_h1h3["is_H3"] = (df_h1h3["stage"] == "H3").astype(int)
m_gap2 = smf.ols("peer_centroid_sim ~ judged_ai * is_H3", data=df_h1h3).fit(cov_type="HC3")
interact2_coef = m_gap2.params["judged_ai:is_H3"]
interact2_p    = m_gap2.pvalues["judged_ai:is_H3"]

# Permutation test: H1 vs H3 gap difference
def compute_gap(sub):
    ai = sub.loc[sub["judged_ai"] == 1, "peer_centroid_sim"].mean()
    hu = sub.loc[sub["judged_ai"] == 0, "peer_centroid_sim"].mean()
    return ai - hu

h1_data = stage_defs["H1"][["judged_ai", "peer_centroid_sim"]].dropna().copy()
h3_data = stage_defs["H3"][["judged_ai", "peer_centroid_sim"]].dropna().copy()
observed_delta = compute_gap(h3_data) - compute_gap(h1_data)

rng = np.random.RandomState(42)
n_perm = 10000
perm_deltas = np.empty(n_perm)
for i in range(n_perm):
    h1_perm = h1_data.copy()
    h3_perm = h3_data.copy()
    h1_perm["judged_ai"] = rng.permutation(h1_perm["judged_ai"].values)
    h3_perm["judged_ai"] = rng.permutation(h3_perm["judged_ai"].values)
    perm_deltas[i] = compute_gap(h3_perm) - compute_gap(h1_perm)

p_perm_two = float(np.mean(np.abs(perm_deltas) >= np.abs(observed_delta)))
p_perm_one = float(np.mean(perm_deltas >= observed_delta))

# Spearman trend: peer_centroid_sim vs stage_ord, separately by judged identity
from scipy.stats import spearmanr
df_trend = pd.concat([stage_defs["H1"], stage_defs["H2"], stage_defs["H3"]]).copy()
stage_ord_map = {"H1": 1, "H2": 2, "H3": 3}
df_trend["stage_ord"] = df_trend["stage"].map(stage_ord_map)
for j, label in [(1, "AI-judged"), (0, "Human-judged")]:
    sub = df_trend[(df_trend["judged_ai"] == j)].dropna(subset=["peer_centroid_sim"])
    rho, p_sp = spearmanr(sub["stage_ord"], sub["peer_centroid_sim"])




 
# ══════════════════════════════════════════════════════════════════════════════
# LaTeX 回归表生成
# ══════════════════════════════════════════════════════════════════════════════
 
def format_p(p):
    if p < 0.001:
        return "$<$0.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.3f}"
 
def format_sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""
 
def model_to_latex_rows(model, label_map=None):
    """
    从 statsmodels OLS result 提取每个系数的行。
 
    Parameters
    ----------
    model : statsmodels RegressionResultsWrapper
    label_map : dict, optional
        {原始变量名: 显示名}, 未映射的变量用原名
 
    Returns
    -------
    list of str (LaTeX table rows)
    """
    if label_map is None:
        label_map = {}
 
    rows = []
    ci = model.conf_int()
    for term in model.params.index:
        display = label_map.get(term, term)
        coef = model.params[term]
        se = model.bse[term]
        p = model.pvalues[term]
        lo, hi = ci.loc[term]
        sig = format_sig(p)
 
        rows.append(
            f"  {display} & {coef:+.3f} & {se:.3f} & "
            f"[{lo:.3f}, {hi:.3f}] & {format_p(p)}{sig} \\\\"
        )
    return rows
 
 
# ── 变量名映射（仅保留感兴趣的预测变量）───────────────────────────────────
LABEL_MAP_GLOBAL = {
    "judged_ai":              "Perceived AI identity",
    "pair_sim_z":             "Self--peer similarity",
    "peer_len_z":             "Peer description length",
    "peer_centroid_sim_z":    "Peer--centroid proximity",
}
 
# 只输出感兴趣的预测变量，跳过 Intercept 和控制变量
DISPLAY_TERMS = list(LABEL_MAP_GLOBAL.keys())
 
 
# ── 生成主模型表格 ────────────────────────────────────────────────────────────
# 只保留 DISPLAY_TERMS 对应的行
all_rows = []
ci = m_all.conf_int()
for term in m_all.params.index:
    if term not in DISPLAY_TERMS:
        continue
    display = LABEL_MAP_GLOBAL[term]
    coef = m_all.params[term]
    se = m_all.bse[term]
    p = m_all.pvalues[term]
    lo, hi = ci.loc[term]
    sig = format_sig(p)
    all_rows.append(
        f"  {display} & {coef:+.3f} & {se:.3f} & "
        f"[{lo:.3f}, {hi:.3f}] & {format_p(p)}{sig} \\\\"
    )
 
n_obs = int(m_all.nobs)
r2 = m_all.rsquared
r2_adj = m_all.rsquared_adj
 
latex_table = r"""
\begin{table}[h]
\centering
\caption{OLS regression predicting adoption willingness. Heteroskedasticity-consistent
standard errors (HC3) are used throughout. Continuous predictors are $z$-score
standardized; perceived AI identity is a binary indicator (1 = judged as AI).}
\label{tab:regression}
\begin{tabular}{l c c c c}
\hline
Predictor & $\beta$ & SE & 95\% CI & $p$ \\
\hline
"""
 
for row in all_rows:
    latex_table += row + "\n"
 
latex_table += r"""\hline
\multicolumn{5}{l}{$N$ = """ + str(n_obs) + r"""; $R^2$ = """ + f"{r2:.3f}" + r"""; Adjusted $R^2$ = """ + f"{r2_adj:.3f}" + r"""} \\
\multicolumn{5}{l}{Agent proportion condition controlled (categorical, ref: 12.5\%).} \\
\hline
\end{tabular}
\end{table}
"""
 
# ── 保存 ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(FIGS_DIR, "table_s1.tex")
with open(out_path, "w") as f:
    f.write(latex_table)
print(f"Generated table: {out_path}")
