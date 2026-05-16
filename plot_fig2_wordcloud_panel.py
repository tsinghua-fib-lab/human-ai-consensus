"""
plot_fig2_wordcloud_panel.py
============================
Fig 2 新 panel：
  上方：12.5% vs 75% 词云对比
  下方：各选一条最具代表性的 top-1 表达

依赖：pip install wordcloud matplotlib pandas
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from wordcloud import WordCloud
from collections import Counter
import re
import textwrap

# ── 配置 ──
CSV_PATH = "processed_data/intermediate_result/final_expressions_all.csv"
SAVE_DIR = "figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# 停用词：去掉太通用的描述词，保留有区分度的词
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'in', 'on', 'of', 'and', 'with', 'its',
    'it', 'to', 'by', 'that', 'this', 'was', 'for', 'from', 'at', 'as',
    'be', 'or', 'but', 'not', 'has', 'have', 'had', 'their', 'there',
    'like', 'which', 'image', 'picture', 'figure',  # 太通用
}

# ── 工具函数 ──
_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# British → American spelling mapping
BRIT_TO_AMER = {
    'grey': 'gray', 'colour': 'color', 'colours': 'color',
    'coloured': 'colored', 'favour': 'favor', 'favourite': 'favorite',
    'honour': 'honor', 'humour': 'humor', 'labour': 'labor',
    'neighbour': 'neighbor', 'behaviour': 'behavior', 'centre': 'center',
    'metre': 'meter', 'defence': 'defense', 'offence': 'offense',
    'licence': 'license', 'practise': 'practice',
    'recognise': 'recognize', 'realise': 'realize', 'organise': 'organize',
    'analyse': 'analyze', 'apologise': 'apologize',
}

# Lemmatization via nltk
import nltk
nltk.data.path.append("../../../nltk_data/corpora/")  # 无法联网下载，故手动从github下载packages后放在这里
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

_lemmatizer = WordNetLemmatizer()

def _get_wordnet_pos(tag):
    """Map POS tag to WordNet POS for lemmatizer."""
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def normalize_token(token):
    """British→American, then lemmatize (noun + verb, take shorter)."""
    # British → American
    token = BRIT_TO_AMER.get(token, token)
    # Lemmatize as noun and verb, pick the shorter form
    # (e.g. "sitting"→"sit" via verb, "shapes"→"shape" via noun)
    lem_noun = _lemmatizer.lemmatize(token, wordnet.NOUN)
    lem_verb = _lemmatizer.lemmatize(token, wordnet.VERB)
    # prefer the shorter lemma (more reduced form), break ties with noun
    if len(lem_verb) < len(lem_noun):
        return lem_verb
    return lem_noun

def tokenize(text):
    return [m.group(0).lower() for m in _word_re.finditer(text)]

def get_word_freq(expressions, stopwords=STOPWORDS):
    """统计词频，去除停用词，规范化拼写和词形"""
    counter = Counter()
    for expr in expressions:
        tokens = tokenize(expr)
        for t in tokens:
            t = normalize_token(t)
            if t not in stopwords and len(t) > 1:
                counter[t] += 1
    return counter


def find_top1_expression(expressions):
    """
    找到最具代表性的表达：与该条件所有其他表达的词重叠度最高的那一条。
    简单方法：每条表达的 token set 与所有其他表达 token set 的 Jaccard 相似度之和。
    """
    token_sets = [set(tokenize(e)) for e in expressions]
    best_idx = 0
    best_score = -1
    
    for i in range(len(expressions)):
        score = 0
        for j in range(len(expressions)):
            if i == j:
                continue
            intersection = len(token_sets[i] & token_sets[j])
            union = len(token_sets[i] | token_sets[j])
            if union > 0:
                score += intersection / union
        if score > best_score:
            best_score = score
            best_idx = i
    
    return expressions[best_idx]


# ── 加载数据 ──
df = pd.read_csv(CSV_PATH)
print("Preparing Fig. 2c wordcloud...")

# 只取最后5轮
for cond in df['condition'].unique():
    mask = df['condition'] == cond
    last_round = df.loc[mask, 'round'].max()
    cutoff_round = last_round - 4  # last 5 rounds
    df.loc[mask & (df['round'] < cutoff_round), 'is_last'] = False
    df.loc[mask & (df['round'] >= cutoff_round), 'is_last'] = True

df_last = df[df['is_last'] == True].copy()

# 分条件
cond_125 = df_last[df_last['agent_ratio'] == 0.125]['expression'].tolist()
cond_75 = df_last[df_last['agent_ratio'] == 0.75]['expression'].tolist()

# ── 词频 ──
freq_125 = get_word_freq(cond_125)
freq_75 = get_word_freq(cond_75)

# ── Top-1 表达 ──
top1_125 = find_top1_expression(cond_125)
top1_75 = find_top1_expression(cond_75)


# ══════════════════════════════════════════════════════════════
# 绘图
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(11, 5.8))
gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1.4], 
              hspace=0.02, wspace=0.15,
              left=0.04, right=0.96, top=0.93, bottom=0.03)

# ── 颜色方案 ──
COLOR_HUMAN = '#4292c6' # '#5aaa6b' # 绿色系
COLOR_AGENT = '#b95a58' # '#e08c3a' # 橙色系

# 自定义 color_func：按词频映射颜色深浅（频率越高越深）
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def make_freq_color_func(freq_dict, light_color, dark_color):
    """
    返回一个 color_func，根据词频线性映射颜色深浅。
    频率最高 → dark_color，频率最低 → light_color。
    """
    max_freq = max(freq_dict.values())
    min_freq = min(freq_dict.values())
    
    light_rgb = np.array(mcolors.to_rgb(light_color))
    dark_rgb = np.array(mcolors.to_rgb(dark_color))
    
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        freq = freq_dict.get(word.lower(), min_freq)
        if max_freq == min_freq:
            t = 1.0
        else:
            t = (freq - min_freq) / (max_freq - min_freq)
        rgb = light_rgb + t * (dark_rgb - light_rgb)
        r, g, b = [int(c * 255) for c in rgb]
        return f"rgb({r}, {g}, {b})"
    
    return color_func


# ── 圆形 mask：约束词云为圆形，高频词自然聚于中心 ──
def make_circle_mask(width=800, height=500):
    """Create an elliptical mask matching the canvas aspect ratio."""
    y, x = np.ogrid[:height, :width]
    cx, cy = width / 2, height / 2
    mask = np.full((height, width), 255, dtype=np.uint8)
    # ellipse matching canvas aspect ratio
    mask[((x - cx) / (width * 0.52)) ** 2 + ((y - cy) / (height * 0.52)) ** 2 <= 1] = 0
    return mask

circle_mask = make_circle_mask(800, 500)

# ── 词云 ──
wc_kwargs = dict(
    width=800, height=500,
    max_words=100, #80,
    mask=circle_mask,
    background_color='white',
    prefer_horizontal=0.7, #0.6,
    min_font_size=5, #20, #7,
    max_font_size=100,
    relative_scaling=0.4, #0.5,
    contour_width=0,
    random_state=42,
)




wc_125 = WordCloud(**wc_kwargs).generate_from_frequencies(freq_125)
wc_125.recolor(color_func=make_freq_color_func(freq_125, '#c6dbef', '#08519c'))  # light blue → dark blue
 
wc_75 = WordCloud(**wc_kwargs).generate_from_frequencies(freq_75)
wc_75.recolor(color_func=make_freq_color_func(freq_75, '#fcbba1', '#7f1b1b'))  # light red → dark red

# Panel 上方左：12.5% 词云
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(wc_125, interpolation='bilinear')
ax1.axis('off')
ax1.set_title("Human-led Consensus", fontsize=20, fontweight='bold',  # (12.5%)
              fontfamily='Arial', pad=8, color='#333333')

# Panel 上方右：75% 词云
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(wc_75, interpolation='bilinear')
ax2.axis('off')
ax2.set_title("Agent-led Consensus", fontsize=20, fontweight='bold',  # (75%)
              fontfamily='Arial', pad=8, color='#333333')

# ── 下方：Top-1 表达 ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')

# 背景框
rect = mpatches.FancyBboxPatch(
    (0.02, 0.12), 0.96, 0.82,
    boxstyle="round,pad=0.03",
    facecolor='#deebf7', edgecolor=COLOR_HUMAN, linewidth=1.5, alpha=0.6 #'#e8f5e9'
)
ax3.add_patch(rect)

ax3.text(0.5, 0.88, "Most Representative Expression", fontsize=18, 
         ha='center', va='top', fontfamily='Arial', color='#666666', style='italic')

wrapped_125 = textwrap.fill(top1_125, width=45)
ax3.text(0.5, 0.68, f'"{wrapped_125}"', fontsize=16, ha='center', va='top',
         fontfamily='Arial', color='#08519c', style='italic', linespacing=1.4, #'#1b5e20'
         wrap=True)

ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

rect2 = mpatches.FancyBboxPatch(
    (0.02, 0.12), 0.96, 0.82,
    boxstyle="round,pad=0.03",
    facecolor='#fee0d2', edgecolor=COLOR_AGENT, linewidth=1.5, alpha=0.6 #'#fff3e0'
)
ax4.add_patch(rect2)

ax4.text(0.5, 0.88, "Most Representative Expression", fontsize=18,
         ha='center', va='top', fontfamily='Arial', color='#666666', style='italic')

wrapped_75 = textwrap.fill(top1_75, width=45)
ax4.text(0.5, 0.68, f'"{wrapped_75}"', fontsize=16, ha='center', va='top',
         fontfamily='Arial', color='#7f1b1b', style='italic', linespacing=1.4, #'#bf360c'
         wrap=True)


# ── 保存 ──
plt.savefig(os.path.join(SAVE_DIR, "fig2c_wordcloud.pdf"), bbox_inches='tight')
print(f"Generated figure: {SAVE_DIR}/fig2c_wordcloud.pdf")
plt.close()
