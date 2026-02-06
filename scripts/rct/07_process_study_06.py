import os
import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from pathlib import Path

# ============================================================
# 0. Paths and Configuration
# ============================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.cohort_data = self.root / "data" / "processed" / "cohort" / "sampled_1000.csv"
        self.output_dir = self.results_base / "study_06"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "6",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "6",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "6"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Filenames
FILES = {
    "DEP": "6_DEP_raw.json",
    "ADJ": "6_ADJ_raw.json",
    "CONT": "6_CONT_raw.json",
}

# ============================================================
# 1. Ground Truth Benchmarks (Study 6)
# ============================================================
GROUND_TRUTH = [
    # --- A. Stigma (Race Moderation) ---
    {"id": "Stigma_NB_ADJ", "type": "Main Effect", "pop": "NonBlack", "A": "ADJ", "B": "CONT", "col": "DSS_POST", "pre": "DSS_PRE", "flip": True, "desc": "NonBlack: ADJ vs Ctrl (Stigma)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.25},
    {"id": "Stigma_B_ADJ",  "type": "Main Effect", "pop": "Black",    "A": "ADJ", "B": "CONT", "col": "DSS_POST", "pre": "DSS_PRE", "flip": True, "desc": "Black: ADJ vs Ctrl (Stigma)",     "orig_dir": "Positive", "orig_p": ".73",  "target_d": 0.09},

    # --- B. Help-Seeking (Main Effect) ---
    {"id": "Help_All_ADJ", "type": "Main Effect", "pop": "All", "A": "ADJ", "B": "CONT", "col": "GHSQ_POST", "pre": "GHSQ_PRE", "flip": False, "desc": "All: ADJ vs Ctrl (Help-Seek)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.07},
    {"id": "Help_All_DEP", "type": "Main Effect", "pop": "All", "A": "DEP", "B": "CONT", "col": "GHSQ_POST", "pre": "GHSQ_PRE", "flip": False, "desc": "All: DEP vs Ctrl (Help-Seek)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.08},

    # --- C. Warmth (Race Moderation) ---
    {"id": "Warmth_NB_ADJ", "type": "Main Effect", "pop": "NonBlack", "A": "ADJ", "B": "CONT", "col": "Thermo_POST", "pre": "Thermo_PRE", "flip": False, "desc": "NonBlack: ADJ vs Ctrl (Warmth)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.20},
    {"id": "Warmth_B_ADJ", "type": "Main Effect", "pop": "Black",    "A": "ADJ", "B": "CONT", "col": "Thermo_POST", "pre": "Thermo_PRE", "flip": False, "desc": "Black: ADJ vs Ctrl (Warmth)",     "orig_dir": "Positive", "orig_p": ".06",  "target_d": 0.21},

    # --- D. Interaction Test ---
    {"id": "INT_Race_Stigma", "type": "Interaction Effect", "term": "race_bin", "col": "DSS_POST", "pre": "DSS_PRE", "desc": "Race x Intervention (Stigma)", "orig_dir": "Significant", "orig_p": "<.001", "target_d": 0.25}
]

# ============================================================
# 2. Data Loading & Feature Engineering
# ============================================================

def load_demographics(file_path: Path):
    """Load demographic data to determine race"""
    if not file_path.exists():
        return pd.DataFrame(columns=["student_id", "race_bin"])
    
    df = pd.read_csv(file_path, dtype=str)
    
    # Clean ID
    id_col = next((c for c in df.columns if 'student' in c.lower()), 'student_id')
    df.rename(columns={id_col: 'student_id'}, inplace=True)
    
    def map_race(val):
        if pd.isna(val): return "NonBlack"
        # YRBS Q5 mapping (Simplified based on original script logic)
        # 'C' corresponds to Black or African American in standard YRBS
        tokens = str(val).replace(",", " ").split()
        if "C" in tokens: return "Black"
        return "NonBlack"

    if "Q5" in df.columns:
        df["race_bin"] = df["Q5"].apply(map_race)
    else:
        df["race_bin"] = "NonBlack"
        
    return df[["student_id", "race_bin"]]

def load_json_to_df(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = []
        for entry in data:
            row = {'student_id': str(entry.get('student_id'))} # ID as string for merging
            items = entry.get('json_items', {})
            if items:
                row.update(items)
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def safe_sum(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.Series(np.nan, index=df.index)
    return df[use].apply(pd.to_numeric, errors='coerce').sum(axis=1)

def safe_mean(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.Series(np.nan, index=df.index)
    return df[use].apply(pd.to_numeric, errors='coerce').mean(axis=1)

def load_and_process(folder: Path, demo_df: pd.DataFrame):
    if not folder.exists(): return None
    dfs = []
    
    for cond, fname in FILES.items():
        fpath = folder / fname
        df = load_json_to_df(fpath)
        if not df.empty:
            df["intervention"] = cond
            dfs.append(df)
            
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    
    # Merge Demographics
    if "student_id" in df.columns and not demo_df.empty:
        df = df.merge(demo_df, on="student_id", how="left")
        df["race_bin"] = df["race_bin"].fillna("NonBlack")
    else:
        df["race_bin"] = "NonBlack"

    # --- Scoring ---
    # Depression Stigma Scale (DSS)
    dss_pre = [f"item_DSS_PRE_Q{i}" for i in range(1, 10)]
    dss_post = [f"item_DSS_POST_Q{i}" for i in range(1, 10)]
    df["DSS_PRE"] = safe_sum(df, dss_pre)
    df["DSS_POST"] = safe_sum(df, dss_post)

    # General Help-Seeking Questionnaire (GHSQ)
    for t in ["PRE", "POST"]:
        # Q9 usually reverse scored if present? Original script logic:
        q9 = f"item_GHSQ_PE_{t}_Q9"
        if q9 in df.columns: 
            df[q9] = 8 - pd.to_numeric(df[q9], errors='coerce')
        
        gh = [f"item_GHSQ_PE_{t}_Q{i}" for i in range(1, 11)]
        df[f"GHSQ_{t}"] = safe_mean(df, gh)

    # Thermometer (Warmth)
    if "item_THERMO_PRE_Black" in df.columns: 
        df["Thermo_PRE"] = pd.to_numeric(df["item_THERMO_PRE_Black"], errors='coerce')
    if "item_THERMO_POST_Black" in df.columns: 
        df["Thermo_POST"] = pd.to_numeric(df["item_THERMO_POST_Black"], errors='coerce')

    return df

# ============================================================
# 3. Statistical Functions
# ============================================================

def calc_ancova_d(df, pop, group_treat, group_ctrl, col_post, col_pre, flip):
    """Calculates Cohen's d via ANCOVA for a specific sub-population."""
    if col_post not in df.columns or col_pre not in df.columns:
        return np.nan, np.nan, "N/A"

    # Filter Population
    if pop == "All": sub = df
    elif pop == "Black": sub = df[df["race_bin"] == "Black"]
    elif pop == "NonBlack": sub = df[df["race_bin"] == "NonBlack"]
    else: return np.nan, np.nan, "N/A"

    # Filter Conditions
    sub = sub[sub["intervention"].isin([group_treat, group_ctrl])].copy()
    if len(sub) < 5: return np.nan, np.nan, "N<5"
    if sub[col_post].std() == 0: return 0.0, 1.0, "Positive" # Degenerate case

    sub["is_treat"] = (sub["intervention"] == group_treat).astype(int)

    try:
        model = smf.ols(f"{col_post} ~ is_treat + {col_pre}", data=sub).fit()
        t_val = model.tvalues["is_treat"]
        p_val = model.pvalues["is_treat"]
        df_r = model.df_resid
        d_val = (2 * t_val) / np.sqrt(df_r)
    except:
        return np.nan, np.nan, "Error"

    if flip: d_val = -d_val
    dir_str = "Positive" if d_val > 0 else "Negative"
    return d_val, p_val, dir_str

def calc_interaction_p(df, col_post, col_pre, moderator):
    """Calculates Interaction P-value and Effect Size (d) from ANOVA."""
    sub = df.dropna(subset=[col_post, col_pre, moderator]).copy()
    sub = sub[sub["intervention"].isin(["DEP", "ADJ", "CONT"])]

    if len(sub) < 20: return np.nan, np.nan, "N<20"

    try:
        # Type 2 ANOVA for interaction
        model = smf.ols(f"{col_post} ~ C(intervention) * C({moderator}) + {col_pre}", data=sub).fit()
        aov_table = anova_lm(model, typ=2)

        # Find interaction row
        int_rows = [idx for idx in aov_table.index if ":" in idx]
        if not int_rows: return np.nan, np.nan, "Error"

        row_name = int_rows[0]
        p_int = aov_table.loc[row_name, "PR(>F)"]
        F_val = aov_table.loc[row_name, "F"]
        df_num = aov_table.loc[row_name, "df"]
        df_den = aov_table.loc["Residual", "df"]

        # Calculate Effect Size (d) from partial eta squared
        if np.isnan(F_val):
            d_int = 0.0
        else:
            eta_sq = (F_val * df_num) / (F_val * df_num + df_den)
            if eta_sq >= 1: d_int = np.nan
            else: d_int = 2 * np.sqrt(eta_sq / (1 - eta_sq))

        dir_str = "Significant" if p_int < 0.05 else "n.s."
        return d_int, p_int, dir_str

    except:
        return np.nan, np.nan, "Error"

# ============================================================
# 4. Main Analysis Loop
# ============================================================

def main():
    # Load Demographics
    demo_df = load_demographics(paths.cohort_data)
    
    # Load Results
    data_cache = {}
    print("Loading data for Study 6...")
    for m in MODELS:
        folder = PATH_CONFIGS.get(m)
        if folder and folder.exists():
            data_cache[m] = load_and_process(folder, demo_df)
        else:
            data_cache[m] = None

    final_rows = []

    for item in GROUND_TRUTH:
        row = {
            "Effect Type": item["type"],
            "Effect Desc": item["desc"],
            "Original Direction": item.get("orig_dir", ""),
            "Original p": item.get("orig_p", ""),
            "Original d": item.get("target_d", "")
        }

        for m_name in MODELS:
            df = data_cache.get(m_name)
            prefix = f"{m_name} "

            # Skip Race analysis for LLM-only (Base-DT) if desired, 
            # mirroring original logic. 'LLM-only' implies no persona/demographics.
            is_demographic_item = item.get("pop") in ["Black", "NonBlack"] or item["type"] == "Interaction Effect"
            
            if m_name == "LLM-only" and is_demographic_item:
                d_val, p_val, dir_str = np.nan, np.nan, "N/A"
                is_success = "N/A"
            else:
                d_val, p_val, dir_str = np.nan, np.nan, "N/A"

                if df is not None:
                    if item["type"] == "Interaction Effect":
                        d_val, p_val, dir_str = calc_interaction_p(df, item["col"], item["pre"], item["term"])
                    else:
                        d_val, p_val, dir_str = calc_ancova_d(
                            df, item["pop"], item["A"], item["B"], item["col"], item["pre"], item["flip"]
                        )

                # Success Logic
                is_success = "No"
                orig_is_sig = "<" in str(item["orig_p"]) or str(item["orig_p"]) in [".01", ".031"]
                model_is_sig = p_val < 0.05 if not np.isnan(p_val) else False

                if item["type"] == "Interaction Effect":
                    # For interaction, success matches significance status
                    if orig_is_sig == model_is_sig: is_success = "Yes"
                else:
                    if orig_is_sig:
                        # Orig Sig: Need Sig + Direction Match
                        if model_is_sig and (dir_str == item["orig_dir"]): is_success = "Yes"
                    else:
                        # Orig NS: Need NS (Reproducing null result)
                        if not model_is_sig: is_success = "Yes"

            row[prefix+"Dir"] = dir_str
            row[prefix+"p"] = round(p_val, 3) if not np.isnan(p_val) else "-"
            row[prefix+"d"] = round(d_val, 3) if not np.isnan(d_val) else "-"
            row[prefix+"Success"] = is_success

        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)
    print("\n=== Study 6 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_06_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()