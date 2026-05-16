import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "analyze_convergence_new.py",
    "analyze_human_agent_separately.py",
    "run_contribution_analysis_continuous.py",
    "plot_fig2_boxplot_v5.py",
    "plot_fig2_wordcloud_panel.py",
    "plot_adoption_persistence_dynamics_by_pair_type.py",
    "plot_fig3_combined_continuous.py",
    "plot_subjective_agreement_and_ai_leadership.py",
    "plot_ai_penalty_with_controls_with_centrality_distance.py",
]


def main():
    Path("figures").mkdir(exist_ok=True)
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
