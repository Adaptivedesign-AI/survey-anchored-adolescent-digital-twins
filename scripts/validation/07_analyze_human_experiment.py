import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 0. Path Configuration
# ============================================================
class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        
        # Input Data Path (Assuming raw survey data is stored here)
        self.data_dir = self.root / "data" / "raw" / "human_experiment"
        self.input_file = self.data_dir / "DTHumanInteractionSu_DATA_2025-11-1.csv"
        
        # Output Figures Path
        self.figures_dir = self.root / "results" / "figures" / "validation" / "human_experiment"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Data Loading & Preprocessing
# ============================================================
def load_and_process_data(paths):
    if not paths.input_file.exists():
        print(f"Error: Input file not found at {paths.input_file}")
        return pd.DataFrame() # Return empty to handle gracefully

    df = pd.read_csv(paths.input_file)
    df.columns = df.columns.str.strip()

    # Define column mappings for each persona
    personas = {
        "lucas": {"a": [f"lucas_a{i}" for i in range(1, 5)], "b": [f"lucas_b{i}" for i in range(1, 4)],
                  "scenario": "scenario_with_Lucas", "rag": "rag_enabled_with_Lucas"},
        "hanna": {"a": [f"hana_a{i}" for i in range(1, 5)], "b": [f"hana_b{i}" for i in range(1, 4)],
                  "scenario": "scenario_with_Hanna", "rag": "rag_enabled_with_Hanna"},
        "amara": {"a": [f"amara_a{i}" for i in range(1, 5)], "b": [f"amara_b{i}" for i in range(1, 4)],
                  "scenario": "scenario_with_Amara", "rag": "rag_enabled_with_Amara"}
    }

    # Helper functions for recoding
    def recode_scenario(x):
        s = str(x).lower().strip()
        return "Toxic" if s in ["toxic", "1", "t", "tox"] else "Neutral"

    def recode_rag(x):
        s = str(x).lower().strip()
        return "On" if s in ["true", "1", "t", "yes", "on"] else "Off"

    def recode_major(val):
        # D1: 1=STEM, 2=Social Sciences, 3=Humanities, 4=Business, 5=Undecided
        mapping = {1: "STEM", 2: "Social Sci", 3: "Humanities", 4: "Business", 5: "Undecided"}
        return mapping.get(val, "Unknown")

    def recode_region(val):
        # D2: 1=Northeast, 2=Midwest, 3=South, 4=West, 5=Outside U.S.
        mapping = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West", 5: "Outside U.S."}
        return mapping.get(val, "Unknown")

    # Reshape to Long Format
    long_data = []

    for _, row in df.iterrows():
        major = recode_major(row.get('d1'))
        region = recode_region(row.get('d2'))
        pid = row.get('record_id', 'Unknown')

        for persona, info in personas.items():
            # Extract scores
            scores_a = [row[c] for c in info['a'] if c in row and pd.notna(row[c])]
            scores_b = [row[c] for c in info['b'] if c in row and pd.notna(row[c])]

            if not scores_a and not scores_b:
                continue

            pooled_score = np.mean(scores_a + scores_b)

            long_data.append({
                "Participant_ID": pid,
                "Persona": persona.capitalize(), # Lucas, Hanna, Amara
                "Scenario": recode_scenario(row.get(info['scenario'])),
                "RAG": recode_rag(row.get(info['rag'])),
                "Realism_Score": pooled_score,
                "Major": major,
                "Region": region
            })

    return pd.DataFrame(long_data)

# ============================================================
# 2. Visualization Logic
# ============================================================
def plot_results(df_long, paths):
    if df_long.empty:
        print("No data available to plot.")
        return

    sns.set(style="whitegrid", font_scale=1.2)

    # --- Figure 1: Interaction Effect (Boxplot + Strip) ---
    print("Generating Figure 1: Interaction Boxplot...")
    custom_palette = {"Neutral": "#A8D0E6", "Toxic": "#F76C6C"} # Light Blue vs Light Red

    g1 = sns.catplot(
        data=df_long,
        x="RAG", y="Realism_Score", hue="Scenario", col="Persona",
        kind="box", 
        palette=custom_palette,
        dodge=True,
        height=5, aspect=0.8,
        col_order=["Lucas", "Hanna", "Amara"],
        order=["Off", "On"],
        fliersize=0, # Hide outliers (handled by strip plot)
        linewidth=1.5
    )

    # Overlay Strip Plot
    g1.map_dataframe(
        sns.stripplot,
        x="RAG", y="Realism_Score", hue="Scenario",
        dodge=True,
        palette={"Neutral": "#374785", "Toxic": "#a40000"}, # Darker dots
        alpha=0.5, size=4, jitter=True,
        order=["Off", "On"]
    )

    g1.fig.suptitle("Impact of RAG & Environment on Realism (Boxplot)", y=1.05, fontsize=16, weight='bold')
    g1.set_axis_labels("Memory Enrichment (RAG)", "Realism Score")
    g1.set_titles("{col_name}")

    save_path_1 = paths.figures_dir / "human_experiment_interaction.png"
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path_1}")
    plt.close() # Close figure to free memory

    # --- Figure 2: Demographic Fairness ---
    print("Generating Figure 2: Demographic Analysis...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot A: Major
    sns.boxplot(data=df_long, x="Major", y="Realism_Score", ax=axes[0], palette="Set2")
    sns.stripplot(data=df_long, x="Major", y="Realism_Score", ax=axes[0], color="black", alpha=0.3, jitter=True)
    axes[0].set_title("Realism Scores by Major")
    axes[0].set_ylim(1, 7) 
    axes[0].set_xlabel("")

    # Subplot B: Region
    sns.boxplot(data=df_long, x="Region", y="Realism_Score", ax=axes[1], palette="Set3")
    sns.stripplot(data=df_long, x="Region", y="Realism_Score", ax=axes[1], color="black", alpha=0.3, jitter=True)
    axes[1].set_title("Realism Scores by Region")
    axes[1].set_ylim(1, 7)
    axes[1].set_xlabel("")

    plt.suptitle("Demographic Robustness Check (Fairness)", fontsize=16, weight='bold')
    plt.tight_layout()

    save_path_2 = paths.figures_dir / "human_experiment_demographics.png"
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path_2}")
    # plt.show() 

if __name__ == "__main__":
    paths = ProjectPaths()
    df_long = load_and_process_data(paths)
    plot_results(df_long, paths)