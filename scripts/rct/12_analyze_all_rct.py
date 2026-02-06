#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import re
from pathlib import Path

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.input_excel = self.root / "data" / "processed" / "rct" / "DT_replication_reorganized_effects.xlsx"
        self.figures_dir = self.root / "results" / "figures" / "rct"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

FILE_STUDY = str(paths.input_excel)

models = ['Memory-DT', 'Base-DT', 'LLM-only']

model_display = {
    'LLM-only':  'Base DT',
    'Base-DT':   'Survey DT',
    'Memory-DT': 'Survey+Memory DT',
}

color_map = {
    'LLM-only':  "#2A9D8C",
    'Base-DT':   "#E9C46B",
    'Memory-DT': "#E66F51",
}
original_color = "#264653"

marker_map = {
    'LLM-only':  '^',
    'Base-DT':   'o',
    'Memory-DT': 's',
}

EFFECT_TYPES = ['Overall', 'Main effect', 'Interaction effect']

print(f"Reading data from: {FILE_STUDY}")
try:
    df = pd.read_excel(FILE_STUDY, sheet_name='All_effects')
    print(f"Successfully loaded {len(df)} rows.")
except Exception as e:
    print(f"Error reading file: {e}")
    df = pd.DataFrame()

def _extract_first_number(s: str):
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(s))
    return float(m.group()) if m else np.nan

def classify_original_sig(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.floating)):
        return float(val) < 0.05

    s = str(val).strip()
    if s == "":
        return np.nan
    sl = s.lower()

    if sl in {'ns', 'n.s', 'n.s.', 'nonsignificant', 'non-significant', 'not significant'}:
        return False

    if 'pr' in sl:
        pr = _extract_first_number(sl)
        if np.isnan(pr):
            return np.nan
        return pr >= 0.95

    if '<' in sl:
        num = _extract_first_number(sl)
        if np.isnan(num):
            return np.nan
        return num <= 0.05

    if '>' in sl:
        return False

    num = _extract_first_number(sl)
    if np.isnan(num):
        return np.nan
    return num < 0.05

def normalize_effect_type(x):
    s = "" if pd.isna(x) else str(x).strip().lower()
    s = re.sub(r'\s+', ' ', s)
    if s in {"overall", "overall effect"}:
        return "overall"
    if "main" in s:
        return "main effect"
    if "interaction" in s:
        return "interaction effect"
    return s

if not df.empty:
    for m in models:
        col_succ = f'{m} Replication success'
        col_dm   = f'{m} Direction match'

        if col_succ in df.columns:
            df[col_succ] = (
                df[col_succ].astype(str).str.strip().str.lower()
                .map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0})
            )

        if col_dm in df.columns:
            df[col_dm] = (
                df[col_dm].astype(str).str.strip().str.lower()
                .map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0})
            )

    d_cols = {'Original': 'Original Effect Size (d)'}
    for m in models:
        d_cols[m] = f'{m} Effect Size (d)'

    for col in d_cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['orig_sig_flag'] = df['Original p'].apply(classify_original_sig) if 'Original p' in df.columns else np.nan
    df['_etype_norm'] = df['Effect Type'].apply(normalize_effect_type) if 'Effect Type' in df.columns else ""

def subset_by_effect_type(data, effect_type):
    if data.empty:
        return data
    if effect_type == 'Overall':
        return data
    return data[data['_etype_norm'] == effect_type.strip().lower()]

def binom_ci_95(k, n, p):
    lo, hi = stats.binom.interval(0.95, n, p)
    return float(lo), float(hi)

def compute_rate_arrays(data, value_col_template):
    n_models = len(models)
    n_types = len(EFFECT_TYPES)

    rate = np.full((n_models, n_types), np.nan)
    err_low = np.full_like(rate, np.nan)
    err_high = np.full_like(rate, np.nan)

    if data.empty:
        return rate, err_low, err_high

    for t_idx, et in enumerate(EFFECT_TYPES):
        sub = subset_by_effect_type(data, et)

        for m_idx, m in enumerate(models):
            col = value_col_template(m)
            if col not in sub.columns:
                continue
            v = sub[col].dropna()
            v = v[v.isin([0, 1])]
            n = len(v)
            if n == 0:
                continue
            k = float(v.sum())
            p = k / n
            r = p * 100.0
            lo_c, hi_c = binom_ci_95(k, n, p)
            rate[m_idx, t_idx] = r
            err_low[m_idx, t_idx] = max(0.0, r - lo_c / n * 100.0)
            err_high[m_idx, t_idx] = max(0.0, hi_c / n * 100.0 - r)

    return rate, err_low, err_high

df_sig  = df[df['orig_sig_flag'] == True].copy()  if (not df.empty and 'orig_sig_flag' in df.columns) else pd.DataFrame()
df_null = df[df['orig_sig_flag'] == False].copy() if (not df.empty and 'orig_sig_flag' in df.columns) else pd.DataFrame()

match_sig_rate, match_sig_low, match_sig_high = compute_rate_arrays(
    df_sig, lambda m: f'{m} Direction match'
)

repl_sig_rate, repl_sig_low, repl_sig_high = compute_rate_arrays(
    df_sig, lambda m: f'{m} Replication success'
)

repl_null_rate, repl_null_low, repl_null_high = compute_rate_arrays(
    df_null, lambda m: f'{m} Replication success'
)

fig = plt.figure(figsize=(18, 13))

gs = fig.add_gridspec(
    2, 3,
    height_ratios=[1.25, 1.2],
    hspace=0.35,
    wspace=0.06,
    left=0.05,
    right=0.985,
    top=0.86,
    bottom=0.08
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])

ax_d = fig.add_subplot(gs[1, 0])
ax_e = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

row_y = np.array([2, 1, 0], dtype=float)
col_order = [0, 1, 2]
offsets = np.linspace(-0.18, 0.18, len(models))

span_h = 0.35
bg_configs = {
    2: ('#FFF9C4', 'Overall effect'),
    1: ('#DCEDC8', 'Main effect'),
    0: ('#E1F5FE', 'Interaction effect')
}

def style_top_axis(ax, panel_letter, title, xlabel):
    for y_val, (bg_color, label) in bg_configs.items():
        ax.axhspan(y_val - span_h, y_val + span_h,
                   color=bg_color, alpha=0.3, zorder=0, ec=None)
        ax.text(0.01, y_val + span_h + 0.02, label,
                transform=ax.get_yaxis_transform(),
                fontsize=15, fontweight='bold', color='#444444',
                ha='left', va='bottom')
    ax.set_yticks([])
    ax.set_ylim(-0.5, 2.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.axvline(50, ls='--', c='gray', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_xlim(0, 118)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.text(-0.05, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='bottom', ha='left')

def plot_top_points(ax, rate_arr, low_arr, high_arr, add_labels=True):
    for m_idx, m in enumerate(models):
        c = color_map[m]
        mk = marker_map[m]
        y_pos = row_y + offsets[m_idx]
        rates = rate_arr[m_idx, col_order]
        xerr  = [low_arr[m_idx, col_order], high_arr[m_idx, col_order]]

        ax.errorbar(
            rates, y_pos,
            xerr=xerr,
            fmt=mk,
            color=c,
            ecolor=c,
            capsize=4,
            markersize=9,
            elinewidth=1.6
        )
        if add_labels:
            for x, y in zip(rates, y_pos):
                if np.isnan(x):
                    continue
                ax.text(x + 3.8, y, f'{x:.1f}%', va='center', fontsize=15)

style_top_axis(ax_a, 'a', 'Direction match (Original Significant)', 'Direction match rate (%)')
plot_top_points(ax_a, match_sig_rate, match_sig_low, match_sig_high)

style_top_axis(ax_b, 'b', 'Replication (Original Significant)', 'Replication rate (%)')
plot_top_points(ax_b, repl_sig_rate, repl_sig_low, repl_sig_high)

style_top_axis(ax_c, 'c', 'Replication (Original Null)', 'Replication rate (%)')
plot_top_points(ax_c, repl_null_rate, repl_null_low, repl_null_high)

handles = [Line2D([0], [0], marker='o', ls='none', c=original_color,
                  label='Original', markersize=9)]
handles += [
    Line2D([0], [0], marker=marker_map[m], ls='none',
           c=color_map[m], label=model_display[m], markersize=9)
    for m in ['LLM-only', 'Base-DT', 'Memory-DT']
]

fig.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.975),
    ncol=4,
    frameon=True,
    edgecolor='gray',
    framealpha=1,
    borderpad=0.6,
    fontsize=15
)

if not df.empty:
    scatter_cols = [d_cols['Original']] + [d_cols[m] for m in models]
    valid_cols = [c for c in scatter_cols if c in df.columns]
    if len(valid_cols) == len(scatter_cols) and 'orig_sig_flag' in df.columns:
        sub_scatter = df[valid_cols + ['orig_sig_flag']].dropna()
        df_overall = sub_scatter.copy()
        df_sig_sc  = sub_scatter[sub_scatter['orig_sig_flag'] == True].copy()
        df_null_sc = sub_scatter[sub_scatter['orig_sig_flag'] == False].copy()
    else:
        df_overall = pd.DataFrame()
        df_sig_sc  = pd.DataFrame()
        df_null_sc = pd.DataFrame()
else:
    df_overall = pd.DataFrame()
    df_sig_sc  = pd.DataFrame()
    df_null_sc = pd.DataFrame()

def plot_square_scatter(ax, data, title, panel_label, anchor=None):
    if data.empty or len(data) < 3:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='bottom', ha='left')
        return

    z_cols = [d_cols['Original']] + [d_cols[m] for m in models]
    z_data = data[z_cols].apply(lambda x: (x - x.mean()) / x.std(ddof=0))
    orig_z = z_data[d_cols['Original']]

    all_z = np.concatenate([z_data[c].values for c in z_cols])
    z_min, z_max = np.nanmin(all_z), np.nanmax(all_z)
    z_range = z_max - z_min if z_max > z_min else 1.0
    z_buffer = z_range * 0.15
    lim_min = z_min - z_buffer
    lim_max = z_max + z_buffer
    x_line = np.linspace(lim_min, lim_max, 200)

    for m in models:
        col_m = d_cols[m]
        model_z = z_data[col_m]
        c = color_map[m]
        mk = marker_map[m]

        ax.scatter(orig_z, model_z,
                   alpha=0.35, color=c, s=50,
                   marker=mk, edgecolors='none')

        if len(orig_z) > 1 and orig_z.std() > 0 and model_z.std() > 0:
            r_val = orig_z.corr(model_z)
        else:
            r_val = 0.0

        ax.plot(x_line, r_val * x_line,
                color=c, lw=2.8,
                label=f'{model_display[m]} (r={r_val:.2f})')

    ax.plot(x_line, x_line,
            color='black', linestyle=':',
            alpha=0.4, linewidth=1.8,
            label='Perfect match')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1.2)

    ax.set_xlabel('Standardized original effect size (Z)', fontsize=15)
    ax.set_ylabel('Standardized model effect size (Z)', fontsize=15)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    ax.set_aspect('equal', adjustable='box')
    if anchor is not None:
        ax.set_anchor(anchor)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    ax.legend(loc='upper left', fontsize=12, frameon=False)
    ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes,
            fontsize=22, fontweight='bold', va='bottom', ha='left')

plot_square_scatter(ax_d, df_overall, 'Overall', 'd', anchor='E')
plot_square_scatter(ax_e, df_sig_sc,  'Original significant', 'e', anchor='C')
plot_square_scatter(ax_f, df_null_sc, 'Original null', 'f', anchor='W')

save_path = paths.figures_dir / "all_rct_summary_plot.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Figure saved to: {save_path}")
