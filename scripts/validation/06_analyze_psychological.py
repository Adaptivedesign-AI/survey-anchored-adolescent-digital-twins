import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================
# 0. Path Configuration
# ============================================================
class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        # Output directory for figures
        self.figures_dir = self.root / "results" / "figures" / "validation" / "psychological"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Data Preparation
# ============================================================

# Correlation coefficients (r)
r_values = {
    "ASRI–BFI": {
        "Gemini 2.0":      {"Base-DT": 0.0005, "YRBS-DT": 0.6127, "YRBS-Memory-DT": 0.8311},
        "Gemini 2.5 Lite": {"Base-DT": -0.0285, "YRBS-DT": 0.8307, "YRBS-Memory-DT": 0.7401},
        "Gemini 2.5":      {"Base-DT": -0.4613, "YRBS-DT": 0.8142, "YRBS-Memory-DT": 0.7315},
    },
    "ASRI_long – Internalizing": {
        "Gemini 2.0":      {"Base-DT":  0.0291, "YRBS-DT": -0.2562, "YRBS-Memory-DT": -0.3420},
        "Gemini 2.5 Lite": {"Base-DT": -0.0356, "YRBS-DT": -0.4063, "YRBS-Memory-DT": -0.3758},
        "Gemini 2.5":      {"Base-DT":  0.0004, "YRBS-DT": -0.4063, "YRBS-Memory-DT": -0.5819},
    },
    "ASRI_long – Externalizing": {
        "Gemini 2.0":      {"Base-DT": -0.0095, "YRBS-DT": -0.1855, "YRBS-Memory-DT": -0.3849},
        "Gemini 2.5 Lite": {"Base-DT": -0.0808, "YRBS-DT": -0.3366, "YRBS-Memory-DT": -0.3595},
        "Gemini 2.5":      {"Base-DT": -0.0306, "YRBS-DT": -0.3366, "YRBS-Memory-DT": -0.5504},
    },
    "ASRI–Risk taking": {
        "Gemini 2.0":      {"Base-DT": -0.0361, "YRBS-DT": -0.1877, "YRBS-Memory-DT": -0.4798},
        "Gemini 2.5 Lite": {"Base-DT":  0.0129, "YRBS-DT": -0.3427, "YRBS-Memory-DT": -0.4854},
        "Gemini 2.5":      {"Base-DT":  0.0372, "YRBS-DT": -0.3153, "YRBS-Memory-DT": -0.4941},
    },
}

# Significance values (p)
p_values = {
    "ASRI–BFI": {
        "Gemini 2.0":      {"Base-DT": 0.9885, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5 Lite": {"Base-DT": 0.368,  "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5":      {"Base-DT": 0.180,  "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
    },
    "ASRI_long – Internalizing": {
        "Gemini 2.0":      {"Base-DT": 0.3585, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5 Lite": {"Base-DT": 0.2602, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5":      {"Base-DT": 0.9902, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
    },
    "ASRI_long – Externalizing": {
        "Gemini 2.0":      {"Base-DT": 0.7639, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5 Lite": {"Base-DT": 0.0106, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5":      {"Base-DT": 0.3337, "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
    },
    "ASRI–Risk taking": {
        "Gemini 2.0":      {"Base-DT": 0.254,  "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5 Lite": {"Base-DT": 0.684,  "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
        "Gemini 2.5":      {"Base-DT": 0.919,  "YRBS-DT": 0.0005, "YRBS-Memory-DT": 0.0005},
    },
}

# Human benchmarks (95% CI)
truth_ci = {
    "ASRI–BFI":                  (0.69, 0.79),
    "ASRI_long – Internalizing":  (-0.40, -0.12),
    "ASRI_long – Externalizing":  (-0.57, -0.33),
    "ASRI–Risk taking":           (-0.46, -0.36),
}

domains = list(truth_ci.keys())

# Formatted titles for plot
pretty_titles = {
    "ASRI–BFI":                  "Self-regulation and conscientiousness",
    "ASRI_long – Internalizing": "Long-term self-regulation and internalizing problems",
    "ASRI_long – Externalizing": "Long-term self-regulation and externalizing problems",
    "ASRI–Risk taking":          "Self-regulation and risk-taking",
}

def fisher_ci(r, n=1000):
    """Calculates Fisher z-transformation confidence intervals."""
    if abs(r) >= 0.999999:
        return (r, r)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    zl, zh = z - 1.96 * se, z + 1.96 * se
    rl = (np.exp(2 * zl) - 1) / (np.exp(2 * zl) + 1)
    rh = (np.exp(2 * zh) - 1) / (np.exp(2 * zh) + 1)
    return rl, rh

# ============================================================
# 2. Plotting Logic
# ============================================================

def generate_plot():
    models_list  = ["Gemini 2.0", "Gemini 2.5 Lite", "Gemini 2.5"]
    systems_list = ["Base-DT", "YRBS-DT", "YRBS-Memory-DT"]

    # Marker mapping for Systems
    markers = {
        "Base-DT": "^",         # Triangle
        "YRBS-DT": "o",         # Circle
        "YRBS-Memory-DT": "s",  # Square
    }

    # Color mapping for Domains (Foreground & Background)
    domain_colors = {
        "ASRI–BFI":                  {"fg": "indigo",      "bg": "purple", "alpha": 0.04, "ci_alpha": 0.2},
        "ASRI_long – Internalizing": {"fg": "firebrick",   "bg": "pink",   "alpha": 0.15, "ci_alpha": 0.5},
        "ASRI_long – Externalizing": {"fg": "darkgoldenrod","bg": "gold",   "alpha": 0.15, "ci_alpha": 0.2},
        "ASRI–Risk taking":          {"fg": "darkgreen",   "bg": "green",  "alpha": 0.04, "ci_alpha": 0.2},
    }

    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['figure.dpi'] = 300

    # Initialize figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add reference line (X=0)
    ax.axvline(0, color="#AAAAAA", linewidth=1.2, zorder=0)

    # Add vertical grid lines
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

    # Track Y positions
    current_y = 0
    yticks_pos = []
    yticks_labels = []
    prefixes = ["d.", "c.", "b.", "a."] # Reverse order

    # Iterate through Domains (Reverse order, 'a' at top)
    for i, domain in enumerate(domains[::-1]):
        prefix = prefixes[i]
        title_text = pretty_titles[domain]
        colors = domain_colors[domain]
        fg_color = colors["fg"]
        bg_color = colors["bg"]
        bg_alpha = colors["alpha"]
        current_ci_alpha = colors.get("ci_alpha", 0.2)

        # Calculate Y range for this domain
        y_start = current_y
        y_end = current_y + len(models_list)

        # 1. Draw background strip (translucent)
        ax.axhspan(y_start - 0.6, y_end - 0.4, color=bg_color, alpha=bg_alpha, zorder=0, edgecolor='none')

        # 2. Draw Ground Truth Region (Human 95% CI)
        gt_low, gt_high = truth_ci[domain]
        ax.fill_betweenx([y_start - 0.6, y_end - 0.4], gt_low, gt_high, color=bg_color, alpha=current_ci_alpha, zorder=1, edgecolor='none')

        # 3. Add subsection title (Top left)
        title_y = y_end - 0.2
        ax.text(-0.95, title_y, prefix, fontsize=12, fontweight='bold', color='black', ha='left', va='center')
        ax.text(-0.91, title_y, title_text, fontsize=12, fontweight='normal', fontstyle='italic', color='black', ha='left', va='center')

        # 4. Plot data points
        # Iterate Models (Reverse order)
        for j, model in enumerate(models_list[::-1]):
            y_pos = current_y + j
            yticks_pos.append(y_pos)
            yticks_labels.append(model)

            # Iterate Systems
            for sys in systems_list:
                r = r_values[domain][model][sys]
                p = p_values[domain][model][sys]

                # Fisher CI
                rl, rh = fisher_ci(r)
                xerr = np.array([[r - rl], [rh - r]])

                # Significance determines fill style
                sig = (p is not None) and (p < 0.05)
                mfc = fg_color if sig else "white" # Filled if significant, hollow if not

                ms = 8
                ax.errorbar(
                    r, y_pos,
                    xerr=xerr,
                    fmt=markers[sys], 
                    markersize=ms,
                    markeredgewidth=1.5,
                    markeredgecolor=fg_color,
                    markerfacecolor=mfc,      
                    ecolor=fg_color,          
                    elinewidth=1.5,
                    capsize=4,
                    zorder=3
                )

        current_y += len(models_list) + 1.5

    # Configure Y axis
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(yticks_labels, fontsize=11)
    ax.tick_params(axis='y', which='both', length=0)

    # Configure limits
    ax.set_ylim(yticks_pos[0] - 1.0, yticks_pos[-1] + 1.5)
    ax.set_xlim(-1.0, 1.1)
    ax.set_xlabel("Pearson correlation (r) with human benchmarks", fontsize=13, fontweight='bold', labelpad=10)
    ax.tick_params(axis='x', labelsize=11)

    # --- Build Legend ---
    legend_elements = []

    # System Type Legend
    legend_elements.append(Line2D([0], [0], marker='none', label=r'$\bf{System\ Type:}$', linestyle='none'))
    for sys in systems_list:
        legend_elements.append(Line2D([0], [0], marker=markers[sys], color='none',
                                      markeredgecolor='gray', markerfacecolor='gray',
                                      markersize=8, label=sys, linestyle='none'))
    legend_elements.append(Line2D([0], [0], marker='none', label='  ', linestyle='none'))

    # Significance Legend
    legend_elements.append(Line2D([0], [0], marker='none', label=r'$\bf{Significance:}$', linestyle='none'))
    legend_elements.append(Line2D([0], [0], marker='o', color='none',
                                  markeredgecolor='gray', markerfacecolor='gray',
                                  markersize=8, label='Filled ($p < .05$)', linestyle='none'))
    legend_elements.append(Line2D([0], [0], marker='o', color='none',
                                  markeredgecolor='gray', markerfacecolor='white',
                                  markersize=8, label='Hollow (n.s.)', linestyle='none'))
    legend_elements.append(Line2D([0], [0], marker='none', label='  ', linestyle='none'))

    # Human CI Legend
    legend_elements.append(mpatches.Patch(color='lightgray', alpha=0.5, label='Human 95% CI Region'))

    # Place Legend
    ax.legend(handles=legend_elements,
              loc="center left",       
              bbox_to_anchor=(1.02, 0.5), 
              frameon=False,
              ncol=1,                  
              fontsize=10,
              handletextpad=0.5,
              labelspacing=0.8         
             )

    plt.tight_layout()
    
    # Save figure
    paths = ProjectPaths()
    save_path = paths.figures_dir / "psychological_validation_results.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Figure saved to: {save_path}")
    
    # plt.show()

if __name__ == "__main__":
    generate_plot()