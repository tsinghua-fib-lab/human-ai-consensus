"""
plot_fig2_boxplot_v5.py
=======================
Fig 2c：五维 box plot 对比 Human-led (12.5%) vs Agent-led (75%)
替代雷达图，每个维度独立 y 轴，1行5列布局

维度：
  1. Concreteness (Brysbaert et al., 2014)
  2. Propositional Idea Density (Brown et al., 2008)
  3. Analogical Ratio (Clark & Wilkes-Gibbs, 1986) — Likert 1-5
  4. Holistic Ratio (Clark & Wilkes-Gibbs, 1986) — Likert 1-5
  5. Event Framing Ratio (LLM annotation) — Likert 1-5
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import spacy
from scipy.stats import mannwhitneyu

# ══════════════════════════════════════════════════════════════
# 度量函数
# ══════════════════════════════════════════════════════════════

_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
def tokenize_en(text: str):
    return [m.group(0).lower() for m in _word_re.finditer(text)]


def concreteness_score_en(text: str, lex: dict):
    tokens = tokenize_en(text)
    if not tokens:
        return np.nan
    scored = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            phrase = f"{tokens[i]} {tokens[i+1]}"
            if phrase in lex:
                scored.append(lex[phrase])
                i += 2
                continue
        if tokens[i] in lex:
            scored.append(lex[tokens[i]])
        i += 1
    return np.mean(scored) if scored else np.nan


def pid_spacy(nlp_model, text: str):
    doc = nlp_model(text)
    words = [t for t in doc if t.is_alpha]
    n_words = len(words)
    if n_words == 0:
        return np.nan
    prop_pos = {"VERB", "ADJ", "ADV", "ADP", "CCONJ", "SCONJ"}
    n_props = sum(1 for t in words if t.pos_ in prop_pos
                  and not (t.pos_ == "VERB" and t.dep_ in {"aux", "auxpass"}))
    return (n_props / n_words) * 10.0


# ══════════════════════════════════════════════════════════════
# 加载数据
# ══════════════════════════════════════════════════════════════

ANNOT_CSV = "processed_data/intermediate_result/perspective_annotations_v3.csv"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# 标注数据已包含 expression + analogical/holistic/event，直接使用
df = pd.read_csv(ANNOT_CSV)
print("Preparing Fig. 2c boxplot...")

# ══════════════════════════════════════════════════════════════
# 计算每条表达的 Concreteness 和 PID
# ══════════════════════════════════════════════════════════════

xlsx_path = 'processed_data/13428_2013_403_MOESM1_ESM.xlsx'
df_lex = pd.read_excel(xlsx_path, sheet_name='Sheet1')
lex = {str(w).strip().lower(): float(s) for w, s in zip(df_lex['Word'], df_lex['Conc.M'])}

nlp_pid = spacy.load("en_core_web_sm", disable=["ner"])

# 对所有表达计算 concreteness 和 PID
df['concreteness'] = [concreteness_score_en(e, lex) for e in df['expression']]
df['pid'] = [pid_spacy(nlp_pid, e) for e in df['expression']]

# 取 12.5% 和 75% 的子集
df_125 = df[df['agent_ratio'] == 0.125].copy()
df_75 = df[df['agent_ratio'] == 0.75].copy()


# 诊断：打印 LLM 维度的分值分布
# ══════════════════════════════════════════════════════════════
# 绘图：1行5列 box plot
# ══════════════════════════════════════════════════════════════

COLOR_HUMAN = '#4292c6' # '#376b9e' # '#4a7cff' # '#5aaa6b'
COLOR_AGENT = '#b95a58' # '#376b9e' # '#e8923f' # '#e08c3a'

dims = [
    ('concreteness', 'Concreteness'),
    ('pid', 'Propositional\nIdea Density'),
    ('analogical', 'Analogical Ratio'),
    ('holistic', 'Holistic Ratio'),
    ('event', 'Event Framing Ratio'),
]

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(11, 5.8))
gs = GridSpec(2, 6, figure=fig, hspace=0.4, wspace=0.9, #hspace=0.65, wspace=0.8
              left=0.07, right=0.97, top=0.95, bottom=0.08)

# 上排：3个图各占2列
ax_positions = [
    fig.add_subplot(gs[0, 0:2]),  # Concreteness
    fig.add_subplot(gs[0, 2:4]),  # PID
    fig.add_subplot(gs[0, 4:6]),  # Analogical
    fig.add_subplot(gs[1, 1:3]),  # Holistic (居中)
    fig.add_subplot(gs[1, 3:5]),  # Event Framing (居中)
]

for idx, (col, title) in enumerate(dims):
    ax = ax_positions[idx]
    
    data_125 = df_125[col].dropna().values
    data_75 = df_75[col].dropna().values
    
    # Box plot
    bp = ax.boxplot(
        [data_125, data_75],
        positions=[1, 2],
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='#333333', linewidth=1.5),
        whiskerprops=dict(color='#666666', linewidth=1.0),
        capprops=dict(color='#666666', linewidth=1.0),
    )
    
    # Strip plot（jitter 散点叠加）
    # 填色
    bp['boxes'][0].set_facecolor(COLOR_HUMAN)
    #bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][0].set_edgecolor('#333333') # COLOR_HUMAN
    bp['boxes'][1].set_facecolor(COLOR_AGENT)
    #bp['boxes'][1].set_alpha(0.6)
    bp['boxes'][1].set_edgecolor('#333333') # COLOR_AGENT
    
    # X 轴
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Human-led\nConsensus', 'Agent-led\nConsensus'],
                       fontsize=15, fontfamily='Arial')
    
    # Y 轴标签
    ax.set_ylabel(title, fontsize=15, fontfamily='Arial', fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    
    # 网格
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 去掉上和右边框
    
    # 统计检验

plt.savefig(os.path.join(SAVE_DIR, "fig2c_boxplot.pdf"), bbox_inches='tight')
print(f"Generated figure: {SAVE_DIR}/fig2c_boxplot.pdf")
plt.close()
