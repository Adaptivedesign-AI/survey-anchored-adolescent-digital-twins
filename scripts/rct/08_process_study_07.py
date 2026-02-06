import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ============================================================
# 0. Paths and Configuration
# ============================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "study_07"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "7",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "7",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "7"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Filenames (JSON)
FILES = {
    "Intervention": "7_intervention_raw.json",
    "Control": "7_control_raw.json"
}

# ============================================================
# 1. Ground Truth Benchmarks (Study 7)
# ============================================================
GROUND_TRUTH = [
    # 1. Appraisals (Post): Effective
    {"id": "Appr_Post", "type": "Main", "col": "Appraisal_POST", "A": "Intervention", "B": "Control", "flip": False, "desc": "Intervention -> Appraisals (Post)", "orig_dir": "Positive", "orig_p": "<.05", "target_d": 0.35},

    # 2. Help-Seeking (Post): Null
    {"id": "Help_Post", "type": "Main", "col": "GHSQ_POST", "A": "Intervention", "B": "Control", "flip": False, "desc": "Intervention -> Help-Seeking", "orig_dir": "Null", "orig_p": ".26", "target_d": 0.20},
]

# ============================================================
# 2. Data Loading & Feature Engineering
# ============================================================

# Map A-E to 1-5
LETTER_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

def safe_map(val):
    if pd.isna(val): return np.nan
    s = str(val).strip().upper()
    return LETTER_MAP.get(s, np.nan)

def safe_numeric(df, col):
    return pd.to_numeric(df[col], errors='coerce')

def safe_sum(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.Series(np.nan, index=df.index)
    return df[use].apply(pd.to_numeric, errors='coerce').sum(axis=1)

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

def load_and_process(folder: Path):
    if not folder.exists(): return None
    dfs = []

    for cond, fname in FILES.items():
        fpath = folder / fname
        if fpath.exists():
            df = load_json_to_df(fpath)
            if not df.empty:
                df["Condition"] = cond
                dfs.append(df)

    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)

    for t in ["PRE", "POST"]:
        # Appraisals (A-E -> 1-5)
        # Target Qs based on original script: Q1, Q5, Q6
        appr_cols_num = []
        target_qs = [1, 5, 6]
        for q in target_qs:
            col_str = f"Item_Appraisals_{t}_Q{q}"
            if col_str in df.columns:
                new_col = f"{col_str}_num"
                df[new_col] = df[col_str].apply(safe_map)
                appr_cols_num.append(new_col)

        if appr_cols_num:
            df[f"Appraisal_{t}"] = df[appr_cols_num].mean(axis=1)

        # GHSQ (1-7)
        ghsq_cols = [f"item_GHSQ_{t}_Q{i}" for i in range(1, 11)]
        # Try lowercase fallback if needed
        if not any(c in df.columns for c in ghsq_cols):
            ghsq_cols = [f"item_ghsq_{t.lower()}_q{i}" for i in range(1, 11)]

        df[f"GHSQ_{t}"] = safe_sum(df, ghsq_cols)

    return df

# ============================================================
# 3. Statistical Functions
# ============================================================

def calc_d_between(df, group_A, group_B, col):
    if col not in df.columns: return np.nan, np.nan
    a = df[df["Condition"]==group_A][col].dropna()
    b = df[df["Condition"]==group_B][col].dropna()
    
    if len(a)<5 or len(b)<5: return np.nan, np.nan

    if a.std()==0 and b.std()==0:
        if a.mean() == b.mean(): return 0.0, 1.0
        d = 10.0 if a.mean() > b.mean() else -10.0
        return d, 0.001

    sp = np.sqrt(((len(a)-1)*a.var() + (len(b)-1)*b.var()) / (len(a)+len(b)-2))
    
    if sp == 0: return 0.0, 1.0 # Avoid division by zero
    
    d_val = (a.mean() - b.mean()) / sp
    _, p_val = stats.ttest_ind(a, b, equal_var=False)
    return d_val, p_val

# ============================================================
# 4. Main Analysis Loop
# ============================================================

def main():
    data_cache = {}
    print("Loading data for Study 7...")
    for m in MODELS:
        folder = PATH_CONFIGS.get(m)
        if folder and folder.exists():
            data_cache[m] = load_and_process(folder)
        else:
            data_cache[m] = None

    final_rows = []

    for item in GROUND_TRUTH:
        row = {
            "Effect Type": item["type"],
            "Effect Desc": item["desc"],
            "Original Direction": item["orig_dir"],
            "Original p": item["orig_p"],
            "Original Effect Size (d)": item["target_d"]
        }

        for m_name in MODELS:
            df = data_cache.get(m_name)
            prefix = f"{m_name} "

            d_val, p_val = np.nan, np.nan

            if df is not None:
                d_val, p_val = calc_d_between(df, item["A"], item["B"], item["col"])

            # --- Success Logic ---
            is_success = "No"
            if not np.isnan(d_val):
                target = item["target_d"]

                # 1. Null Target Logic
                if abs(target) < 0.25 and (item["orig_p"] == ".26" or item["orig_p"] == "n.s."):
                    # Success if LLM is also Null (d<0.2) OR Non-significant (p>0.05)
                    if abs(d_val) < 0.2 or p_val > 0.05:
                        is_success = "Yes"
                # 2. Significant Target Logic
                else:
                    # Success if Direction matches AND Significant
                    if (d_val * target > 0) and (p_val < 0.05):
                        is_success = "Yes"

                dir_str = "Positive" if d_val > 0.05 else ("Negative" if d_val < -0.05 else "Null")
            else:
                dir_str = "N/A"

            row[prefix+"Direction"] = dir_str
            row[prefix+"p"] = round(p_val, 3) if not np.isnan(p_val) else "-"
            row[prefix+"d"] = round(d_val, 3) if not np.isnan(d_val) else "-"
            row[prefix+"Success"] = is_success

        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)
    print("\n=== Study 7 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_07_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()