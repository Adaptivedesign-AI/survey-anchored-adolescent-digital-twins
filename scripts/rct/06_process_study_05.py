import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path

# ============================================================
# 0. Paths and Configuration
# ============================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "study_05"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "5",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "5",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "5"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Filenames (JSON output from Step 01)
FILES = {
    "Pos": "5_positive_raw.json",
    "Neg": "5_negative_raw.json",
    "Ctrl": "5_control_raw.json"
}

# ============================================================
# 1. Ground Truth Benchmarks (Study 5)
# ============================================================
GROUND_TRUTH = [
    # --- 1. Primary Outcome: CCH (Scale) ---
    {"id": "CCH_Pos_Ctrl", "col": "CCH_Change", "A": "Pos", "B": "Ctrl", "flip": False, "desc": "Pos vs Ctrl (CCH)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.41},
    {"id": "CCH_Neg_Ctrl", "col": "CCH_Change", "A": "Neg", "B": "Ctrl", "flip": False, "desc": "Neg vs Ctrl (CCH)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.33},

    # --- 2. Hope (Slider) ---
    {"id": "Hope_Pos_Ctrl", "col": "Hope_Change", "A": "Pos", "B": "Ctrl", "flip": False, "desc": "Pos vs Ctrl (Hope Slider)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.54},
    {"id": "Hope_Neg_Ctrl", "col": "Hope_Change", "A": "Neg", "B": "Ctrl", "flip": False, "desc": "Neg vs Ctrl (Hope Slider)", "orig_dir": "Positive", "orig_p": "n.s.", "target_d": 0.05},

    # --- 3. Agency (Slider) ---
    {"id": "Agency_Pos_Ctrl", "col": "Agency_Change", "A": "Pos", "B": "Ctrl", "flip": False, "desc": "Pos vs Ctrl (Agency Slider)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.21},
    {"id": "Agency_Neg_Ctrl", "col": "Agency_Change", "A": "Neg", "B": "Ctrl", "flip": False, "desc": "Neg vs Ctrl (Agency Slider)", "orig_dir": "Positive", "orig_p": "n.s.", "target_d": 0.02},

    # --- 4. Anxiety (Scale) ---
    {"id": "Anx_Pos_Ctrl", "col": "CCA_Change", "A": "Pos", "B": "Ctrl", "flip": True, "desc": "Pos vs Ctrl (Anxiety)", "orig_dir": "Positive", "orig_p": "n.s.", "target_d": 0.25},
    {"id": "Anx_Neg_Ctrl", "col": "CCA_Change", "A": "Neg", "B": "Ctrl", "flip": True, "desc": "Neg vs Ctrl (Anxiety)", "orig_dir": "Positive", "orig_p": "n.s.", "target_d": 0.19},

    # --- 5. Behavioral Intentions (Scale) ---
    {"id": "BI_Pos_Ctrl", "col": "BI_Change", "A": "Pos", "B": "Ctrl", "flip": False, "desc": "Pos vs Ctrl (Behavior)", "orig_dir": "Positive", "orig_p": "n.s.", "target_d": 0.07},
    {"id": "BI_Neg_Ctrl", "col": "BI_Change", "A": "Neg", "B": "Ctrl", "flip": False, "desc": "Neg vs Ctrl (Behavior)", "orig_dir": "Positive", "orig_p": "n.s.", "target_d": 0.09},
]

# ============================================================
# 2. Data Loading & Feature Engineering
# ============================================================

def load_json_to_df(file_path: Path) -> pd.DataFrame:
    """Parses raw JSON simulation output into a DataFrame"""
    if not file_path.exists():
        return pd.DataFrame()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rows = []
        for entry in data:
            row = {'student_id': entry.get('student_id')}
            items = entry.get('json_items', {})
            if items:
                row.update(items)
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def safe_numeric(df, col):
    return pd.to_numeric(df[col], errors='coerce')

def calculate_cch_total(df, prefix):
    """Calculate CCH total score (with reverse coding for Q8-11)"""
    items = []
    # Positive items Q1-Q7
    for i in range(1, 8):
        col = f"{prefix}_Q{i}"
        if col in df.columns: items.append(safe_numeric(df, col))
    # Reverse items Q8-Q11 (Assuming 1-7 scale, so 8-x)
    for i in range(8, 12):
        col = f"{prefix}_Q{i}"
        if col in df.columns: items.append(8 - safe_numeric(df, col))
    
    if not items: return pd.Series(np.nan, index=df.index)
    return pd.concat(items, axis=1).sum(axis=1)

def safe_sum(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.Series(np.nan, index=df.index)
    return df[use].apply(pd.to_numeric, errors='coerce').sum(axis=1)

def load_and_process(folder: Path):
    if not folder.exists(): return None
    dfs = []
    
    for group, fname in FILES.items():
        fpath = folder / fname
        if fpath.exists():
            df = load_json_to_df(fpath)
            if not df.empty:
                df["Group"] = group
                dfs.append(df)
    
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)

    # --- Scoring ---
    df["CCH_PRE"] = calculate_cch_total(df, "CCH_PRE")
    df["CCH_POST"] = calculate_cch_total(df, "CCH_POST")

    pre_cca = [f"CCA_PRE_Q{i}" for i in range(1, 14)]
    post_cca = [f"CCA_POST_Q{i}" for i in range(1, 14)]
    df["CCA_PRE"] = safe_sum(df, pre_cca)
    df["CCA_POST"] = safe_sum(df, post_cca)

    pre_bi = [f"BI_PRE_Q{i}" for i in range(1, 9)]
    post_bi = [f"BI_POST_Q{i}" for i in range(1, 9)]
    df["BI_PRE"] = safe_sum(df, pre_bi)
    df["BI_POST"] = safe_sum(df, post_bi)

    # Sliders
    if "Hope_PRE_slider" in df.columns: df["Hope_PRE"] = safe_numeric(df, "Hope_PRE_slider")
    if "Hope_POST_slider" in df.columns: df["Hope_POST"] = safe_numeric(df, "Hope_POST_slider")
    if "Agency_PRE_slider" in df.columns: df["Agency_PRE"] = safe_numeric(df, "Agency_PRE_slider")
    if "Agency_POST_slider" in df.columns: df["Agency_POST"] = safe_numeric(df, "Agency_POST_slider")

    # --- Change Scores ---
    for metric in ["CCH", "CCA", "BI", "Hope", "Agency"]:
        pre = f"{metric}_PRE"
        post = f"{metric}_POST"
        if pre in df.columns and post in df.columns:
            df[f"{metric}_Change"] = df[post] - df[pre]

    return df

# ============================================================
# 3. Statistical Functions: ANOVA + Tukey HSD
# ============================================================

def calc_stat_test_strict(df, group_A, group_B, col, flip):
    """
    Computes Cohen's d and Tukey HSD p-value.
    Determines direction even if non-significant.
    """
    if col not in df.columns: return np.nan, np.nan, "N/A"

    subset = df[df["Group"].isin(["Pos", "Neg", "Ctrl"])].dropna(subset=[col]).copy()
    if len(subset) < 15: return np.nan, np.nan, "N < 15"

    # 1. Cohen's d (Raw Effect Size)
    a = subset[subset["Group"]==group_A][col]
    b = subset[subset["Group"]==group_B][col]

    if len(a)<2 or len(b)<2: return np.nan, np.nan, "N/A"

    sp = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    raw_d = (a.mean() - b.mean()) / sp if sp != 0 else 0.0
    d_val = -raw_d if flip else raw_d

    # 2. Tukey HSD (P-value)
    try:
        tukey = pairwise_tukeyhsd(endog=subset[col], groups=subset["Group"], alpha=0.05)
        res = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

        # Find row matching the two groups
        row = res[((res["group1"] == group_A) & (res["group2"] == group_B)) |
                  ((res["group1"] == group_B) & (res["group2"] == group_A))]

        if len(row) > 0:
            p_val = row["p-adj"].values[0]
        else:
            p_val = 1.0
    except:
        return np.nan, np.nan, "Error"

    # 3. Direction (Strict)
    if abs(d_val) < 1e-9:
        direction = "Null"
    elif d_val > 0:
        direction = "Positive"
    else:
        direction = "Negative"

    return d_val, p_val, direction

# ============================================================
# 4. Main Analysis Loop
# ============================================================

def main():
    data_cache = {}
    print("Loading data for Study 5...")
    
    for m in MODELS:
        folder = PATH_CONFIGS.get(m)
        if folder and folder.exists():
            data_cache[m] = load_and_process(folder)
        else:
            data_cache[m] = None

    final_rows = []

    for item in GROUND_TRUTH:
        row = {
            "Effect Type": "Main Effect",
            "Effect Desc": item["desc"],
            "Original Direction": item["orig_dir"],
            "Original p": item["orig_p"],
            "Original Effect Size (d)": item["target_d"]
        }

        for m_name in MODELS:
            df = data_cache.get(m_name)
            prefix = f"{m_name} "

            d_val, p_val, direction = np.nan, np.nan, "N/A"
            if df is not None:
                try:
                    d_val, p_val, direction = calc_stat_test_strict(df, item["A"], item["B"], item["col"], item["flip"])
                except: pass

            # --- Success Logic ---
            is_success = "No"
            if direction != "N/A":
                # Only check direction match
                if direction == item["orig_dir"]:
                    is_success = "Yes"

            row[prefix+"Direction"] = direction
            row[prefix+"p"] = round(p_val, 3) if not np.isnan(p_val) else "-"
            row[prefix+"d"] = round(d_val, 3) if not np.isnan(d_val) else "-"
            row[prefix+"Success"] = is_success

        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)

    # Reorder columns
    cols = ["Effect Type", "Effect Desc", "Original Direction", "Original p", "Original Effect Size (d)"]
    for m in MODELS:
        cols.extend([f"{m} Direction", f"{m} p", f"{m} d", f"{m} Success"])
    
    # Handle cases where cols might be missing if loop failed early
    available_cols = [c for c in cols if c in df_final.columns]
    df_final = df_final[available_cols]

    print("\n=== Study 5 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_05_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()