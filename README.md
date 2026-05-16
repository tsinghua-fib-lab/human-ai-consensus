# Code for paper: AI agents reshape consensus formation in human groups

This repository contains the public figure-reproduction code and processed data for the Human-AI consensus manuscript.

## Layout

- Top-level `*.py`: scripts and helper modules for manuscript figures and tables.
- `processed_data/`: anonymized processed data used by the scripts.
- `figures/`: created when scripts are run; generated outputs are not committed.

## Requirements

The code was tested with the following Python packages:

| Package | Version |
| --- | --- |
| `matplotlib` | `3.10.5` |
| `modelscope` | `1.26.0` |
| `nltk` | `3.9.2` |
| `numpy` | `2.2.6` |
| `openpyxl` | `3.1.5` |
| `pandas` | `2.2.3` |
| `scikit-learn` | `1.7.1` |
| `scipy` | `1.15.3` |
| `seaborn` | `0.13.2` |
| `spacy` | `3.8.11` |
| `statsmodels` | `0.14.5` |
| `torch` | `2.6.0` |
| `transformers` | `4.51.3` |
| `wordcloud` | `1.9.6` |

Install the dependencies with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The word-cloud script also uses the NLTK WordNet corpus.

## Main Outputs

| Manuscript item | Script | Output |
| --- | --- | --- |
| Fig. 1b | `analyze_convergence_new.py` | `figures/fig1b.pdf` |
| Fig. 1c | `analyze_human_agent_separately.py` | `figures/fig1c.pdf` |
| Fig. 2a-b | `run_contribution_analysis_continuous.py` | `figures/fig2a.pdf`, `figures/fig2b.pdf` |
| Fig. 2c | `plot_fig2_boxplot_v5.py`, `plot_fig2_wordcloud_panel.py` | `figures/fig2c_boxplot.pdf`, `figures/fig2c_wordcloud.pdf` |
| Fig. 3a-c | `plot_adoption_persistence_dynamics_by_pair_type.py` | `figures/fig3a.pdf`, `figures/fig3bc.pdf` |
| Fig. 3d-g | `plot_fig3_combined_continuous.py` | `figures/fig3de.pdf`, `figures/fig3fg.pdf` |
| Fig. 4a | `plot_subjective_agreement_and_ai_leadership.py` | `figures/fig4a.pdf` |
| Fig. 4b-d and Supplementary Table 1 | `plot_ai_penalty_with_controls_with_centrality_distance.py` | `figures/fig4b.pdf`, `figures/fig4c.pdf`, `figures/fig4d.pdf`, `figures/table_s1.tex` |
