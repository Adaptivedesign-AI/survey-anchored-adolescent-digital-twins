import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

# =========================
# 0. Configuration
# =========================

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "meta_analysis"
        self.figures_dir = self.root / "results" / "figures" / "rct"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

MODELS = ['Memory-DT', 'Base-DT', 'LLM-only']

MODEL_DISPLAY = {
    'LLM-only':  'Base DT',
    'Base-DT':   'Survey DT',
    'Memory-DT': 'Survey+Memory DT',
}

COLOR_MAP = {
    'LLM-only':  "#2A9D8C",
    'Base-DT':   "#E9C46B",
    'Memory-DT': "#E66F51",
}
ORIGINAL_COLOR = "#264653"

MARKER_MAP = {
    'LLM-only':  '^',
    'Base-DT':   'o',
    'Memory-DT': 's',
}

EFFECT_TYPES = ['Overall', 'Main effect', 'Interaction effect']

# =========================
# 1. Data Aggregation
# =========================

def consolidate_studies():
    print("Aggregating results from all studies...")
    all_dfs = []
    
    # Scan study 01 to 10
    for i in range(1, 11):
        study_folder = paths.results_base / f"study_{i:02d}"
        summary_file = study_folder / f"study_{i:02d}_analysis_summary.csv"
        
        if summary_file.exists():
            try:
                df = pd.read_csv(summary_file)
                df['Study'] = i
                all_dfs.append(df)
                print(f"  Loaded Study {i}: {len(df)} effects")
            except Exception as e:
                print(f"  Error loading Study {i}: {e}")
        else:
            print(f"  Warning: Summary for Study {i} not found")

    if not all_dfs:
        return pd.DataFrame()

    master_df = pd.concat(all_dfs, ignore_index=True)
    
    # Normalize columns to match Plotting Logic expectations
    # 1. Rename 'd' columns to 'Effect Size (d)'
    # 2. Create 'Direction match' boolean columns
    # 3. Rename 'Success' to 'Replication success'
    
    for m in MODELS:
        # Rename d
        old_d = f"{m} d"
        new_d = f"{m} Effect Size (d)"
        if old_d in master_df.columns:
            master_df.rename(columns={old_d: new_d}, inplace=True)
            
        # Create Direction Match
        dir_col = f"{m} Direction"
        if dir_col in master_df.columns and "Original Direction" in master_df.columns:
            # Simple string match
            master_df[f"{m} Direction match"] = (master_df[dir_col] == master_df["Original Direction"]).astype(int)
            
        # Rename Success
        old_succ = f"{m} Success"
        new_succ = f"{m} Replication success"
        if old_succ in master_df.columns:
            master_df.rename(columns={old_succ: new_succ}, inplace=True)
            
    return master_df

# =========================
# 2. Plotting Logic
# =========================

def _extract_first_number(s: str):
    if pd.isna(s): return np.nan
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))
    return float(m.group()) if m else np.nan

def classify_original_sig(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().lower()
    
    # Handle Pr (Bayesian)
    if 'pr' in s or 'prob' in s:
        num = _extract_first_number(s)
        if np.isnan(num): return np.nan
        return num >= 0.95 # Threshold for posterior prob
        
    # Handle p-value
    if s in {'ns', 'n.s', 'n.s.', 'nonsignificant'}: return False
    if '<' in s:
        num = _extract_first_number(s)
        return True if np.isnan(num) else (num <= 0.05)
    
    num = _extract_first_number(s)
    return False if np.isnan(num) else (num < 0.05)

def normalize_effect_type(x):
    s = "" if pd.isna(x) else str(x).strip().lower()
    if s in {"overall", "overall effect", "main", "main effect"}: return "main effect"
    if "interaction" in s: return "interaction effect"
    return "main effect" # Default fallback

def subset_by_effect_type(data, effect_type):
    if data.empty: return data
    if effect_type == 'Overall': return data
    return data[data['_etype_norm'] == effect_type.strip().lower()]

def binom_ci_95(k, n, p):
    lo, hi = stats.binom.interval(0.95, n, p)
    return float(lo), float(hi)

def compute_rate_arrays(data, value_col_template):
    n_models = len(MODELS)
    n_types = len(EFFECT_TYPES)
    rate = np.full((n_models, n_types), np.nan)
    err_low = np.full_like(rate, np.nan)
    err_high = np.full_like(rate, np.nan)

    if data.empty: return rate, err_low, err_high

    for t_idx, et in enumerate(EFFECT_TYPES):
        sub = subset_by_effect_type(data, et)
        for m_idx, m in enumerate(MODELS):
            col = value_col_template(m)
            if col not in sub.columns: continue
            
            # Convert Yes/No/True/False to 1/0
            v = sub[col].astype(str).str.lower().map({'yes':1, 'no':0, 'true':1, 'false':0, '1':1, '0':0})
            v = v.dropna()
            
            n = len(v)
            if n == 0: continue
            
            k = float(v.sum())
            p = k / n
            r = p * 100.0
            lo_c, hi_c = binom_ci_95(k, n, p)
            
            rate[m_idx, t_idx] = r
            err_low[m_idx, t_idx] = max(0.0, r - lo_c/n*100.0)
            err_high[m_idx, t_idx] = max(0.0, hi_c/n*100.0 - r)
            
    return rate, err_low, err_high

def style_top_axis(ax, panel_letter, title, xlabel):
    span_h = 0.35
    bg_configs = {
        2: ('#FFF9C4', 'Overall effect'),
        1: ('#DCEDC8', 'Main effect'),
        0: ('#E1F5FE', 'Interaction effect')
    }
    for y_val, (bg_color, label) in bg_configs.items():
        ax.axhspan(y_val - span_h, y_val + span_h, color=bg_color, alpha=0.3, zorder=0, ec=None)
        ax.text(0.01, y_val + span_h + 0.02, label, transform=ax.get_yaxis_transform(),
                fontsize=12, fontweight='bold', color='#444444', ha='left', va='bottom')
    
    ax.set_yticks([])
    ax.set_ylim(-0.5, 2.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(50, ls='--', c='gray', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xlim(0, 118)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.text(-0.05, 1.08, panel_letter, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

def plot_top_points(ax, rate_arr, low_arr, high_arr):
    row_y = np.array([2, 1, 0], dtype=float)
    col_order = [0, 1, 2] # Overall, Main, Interaction
    offsets = np.linspace(-0.18, 0.18, len(MODELS))
    
    for m_idx, m in enumerate(MODELS):
        c = COLOR_MAP[m]
        mk = MARKER_MAP[m]
        y_pos = row_y + offsets[m_idx]
        rates = rate_arr[m_idx, col_order]
        xerr = [low_arr[m_idx, col_order], high_arr[m_idx, col_order]]
        
        ax.errorbar(rates, y_pos, xerr=xerr, fmt=mk, color=c, ecolor=c, capsize=4, markersize=8)
        
        for x, y in zip(rates, y_pos):
            if np.isnan(x): continue
            ax.text(x + 3.8, y, f'{x:.1f}%', va='center', fontsize=9)

def plot_square_scatter(ax, data, title, panel_label, d_cols, anchor=None):
    if data.empty or len(data) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        return

    # Standardize data for visual comparison
    z_cols = [d_cols['Original']] + [d_cols[m] for m in MODELS]
    valid_cols = [c for c in z_cols if c in data.columns]
    
    if len(valid_cols) < 2: return
    
    z_data = data[valid_cols].apply(pd.to_numeric, errors='coerce').dropna()
    # Normalize
    z_data = z_data.apply(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0)!=0 else 1))
    
    orig_z = z_data[d_cols['Original']]
    
    lim_min, lim_max = -3, 3
    x_line = np.linspace(lim_min, lim_max, 100)
    
    for m in MODELS:
        col_m = d_cols[m]
        if col_m not in z_data.columns: continue
        
        model_z = z_data[col_m]
        c = COLOR_MAP[m]
        mk = MARKER_MAP[m]
        
        ax.scatter(orig_z, model_z, alpha=0.4, color=c, s=40, marker=mk, edgecolors='none')
        
        if len(orig_z) > 1:
            r_val = orig_z.corr(model_z)
            ax.plot(x_line, r_val * x_line, color=c, lw=2, label=f'{MODEL_DISPLAY[m]} (r={r_val:.2f})')

    ax.plot(x_line, x_line, color='black', linestyle=':', alpha=0.4, label='Perfect match')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4)
    
    ax.set_xlabel('Standardized Original Effect (Z)', fontsize=12)
    ax.set_ylabel('Standardized Model Effect (Z)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes, fontsize=18, fontweight='bold', va='bottom')

# =========================
# 3. Main Execution
# =========================

def main():
    # 1. Consolidate
    df = consolidate_studies()
    if df.empty:
        print("No data found to analyze.")
        return
        
    # Pre-process Columns
    df['orig_sig_flag'] = df['Original p'].apply(classify_original_sig)
    df['_etype_norm'] = df['Effect Type'].apply(normalize_effect_type)
    
    # Save Master CSV
    master_csv = paths.output_dir / "all_studies_master.csv"
    df.to_csv(master_csv, index=False)
    print(f"\nMaster dataset saved to: {master_csv}")

    # 2. Prepare Subsets
    df_sig = df[df['orig_sig_flag'] == True].copy()
    df_null = df[df['orig_sig_flag'] == False].copy()

    # 3. Calculate Rates
    match_sig_rate, match_sig_low, match_sig_high = compute_rate_arrays(
        df_sig, lambda m: f'{m} Direction match'
    )
    repl_sig_rate, repl_sig_low, repl_sig_high = compute_rate_arrays(
        df_sig, lambda m: f'{m} Replication success'
    )
    repl_null_rate, repl_null_low, repl_null_high = compute_rate_arrays(
        df_null, lambda m: f'{m} Replication success'
    )

    # 4. Generate Plot
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.2], hspace=0.35, wspace=0.2, 
                          left=0.05, right=0.98, top=0.88, bottom=0.08)

    # Top Row
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    style_top_axis(ax_a, 'a', 'Direction match (Orig. Sig.)', 'Rate (%)')
    plot_top_points(ax_a, match_sig_rate, match_sig_low, match_sig_high)

    style_top_axis(ax_b, 'b', 'Replication (Orig. Sig.)', 'Rate (%)')
    plot_top_points(ax_b, repl_sig_rate, repl_sig_low, repl_sig_high)

    style_top_axis(ax_c, 'c', 'Replication (Orig. Null)', 'Rate (%)')
    plot_top_points(ax_c, repl_null_rate, repl_null_low, repl_null_high)

    # Legend
    handles = [Line2D([0],[0], marker='o', ls='none', c=ORIGINAL_COLOR, label='Original')]
    handles += [Line2D([0],[0], marker=MARKER_MAP[m], ls='none', c=COLOR_MAP[m], label=MODEL_DISPLAY[m]) for m in MODELS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False, fontsize=12)

    # Bottom Row (Scatter)
    d_cols = {'Original': 'Original Effect Size (d)'}
    for m in MODELS: d_cols[m] = f'{m} Effect Size (d)'
    
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    plot_square_scatter(ax_d, df, 'Overall Effect Sizes', 'd', d_cols)
    plot_square_scatter(ax_e, df_sig, 'Original Significant', 'e', d_cols)
    plot_square_scatter(ax_f, df_null, 'Original Null', 'f', d_cols)

    # Save
    save_path = paths.figures_dir / "all_rct_summary_plot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nFigure saved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    main()