import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ============================================================
# 0. Paths and Configuration
# ============================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "study_04"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "4",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "4",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "4"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Filenames (JSON output from Step 01)
FILES = {
    "Control": "4_Control_raw.json",
    "BA_SSI": "4_BA-SSI_raw.json",
    "GM_SSI": "4_GM-SSI_raw.json"
}

# ============================================================
# 1. Ground Truth Benchmarks (Study 4)
# ============================================================
GROUND_TRUTH = [
    # --- 1. Depression (CDI FU) ---
    {"id": "Dep_BA_Ctrl", "pre":"CDI_PRE", "post":"CDI_FU", "A":"BA_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":"<.001", "target_d":0.18, "desc":"BA vs Ctrl (Depression)"},
    {"id": "Dep_GM_Ctrl", "pre":"CDI_PRE", "post":"CDI_FU", "A":"GM_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":"<.001", "target_d":0.18, "desc":"GM vs Ctrl (Depression)"},
    {"id": "Dep_BA_GM",   "pre":"CDI_PRE", "post":"CDI_FU", "A":"BA_SSI", "B":"GM_SSI",  "flip":True, "orig_dir": "Positive", "orig_p":".85",  "target_d":0.01, "desc":"BA vs GM (Depression)"},

    # --- 2. Hopelessness Post (BHS POST) ---
    {"id": "HopeP_BA_Ctrl", "pre":"BHS_PRE", "post":"BHS_POST", "A":"BA_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":"<.001", "target_d":0.26, "desc":"BA vs Ctrl (Hope Post)"},
    {"id": "HopeP_GM_Ctrl", "pre":"BHS_PRE", "post":"BHS_POST", "A":"GM_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":"<.001", "target_d":0.28, "desc":"GM vs Ctrl (Hope Post)"},
    {"id": "HopeP_BA_GM",   "pre":"BHS_PRE", "post":"BHS_POST", "A":"BA_SSI", "B":"GM_SSI",  "flip":True, "orig_dir": "Negative", "orig_p":".50",  "target_d":-0.02, "desc":"BA vs GM (Hope Post)"},

    # --- 3. Hopelessness FU (BHS FU) ---
    {"id": "HopeF_BA_Ctrl", "pre":"BHS_PRE", "post":"BHS_FU", "A":"BA_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":"<.001", "target_d":0.17, "desc":"BA vs Ctrl (Hope FU)"},
    {"id": "HopeF_GM_Ctrl", "pre":"BHS_PRE", "post":"BHS_FU", "A":"GM_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":".002",  "target_d":0.15, "desc":"GM vs Ctrl (Hope FU)"},
    {"id": "HopeF_BA_GM",   "pre":"BHS_PRE", "post":"BHS_FU", "A":"BA_SSI", "B":"GM_SSI",  "flip":True, "orig_dir": "Positive", "orig_p":".59",  "target_d":0.02, "desc":"BA vs GM (Hope FU)"},

    # --- 4. Agency Post (AGENCY POST) ---
    {"id": "AgcP_BA_Ctrl", "pre":"AGENCY_PRE", "post":"AGENCY_POST", "A":"BA_SSI", "B":"Control", "flip":False, "orig_dir": "Positive", "orig_p":"<.001", "target_d":0.31, "desc":"BA vs Ctrl (Agency Post)"},
    {"id": "AgcP_GM_Ctrl", "pre":"AGENCY_PRE", "post":"AGENCY_POST", "A":"GM_SSI", "B":"Control", "flip":False, "orig_dir": "Positive", "orig_p":".001", "target_d":0.15, "desc":"GM vs Ctrl (Agency Post)"},
    {"id": "AgcP_BA_GM",   "pre":"AGENCY_PRE", "post":"AGENCY_POST", "A":"BA_SSI", "B":"GM_SSI",  "flip":False, "orig_dir": "Positive", "orig_p":".001", "target_d":0.16, "desc":"BA vs GM (Agency Post)"},

    # --- 5. Agency FU (AGENCY FU) ---
    {"id": "AgcF_BA_Ctrl", "pre":"AGENCY_PRE", "post":"AGENCY_FU", "A":"BA_SSI", "B":"Control", "flip":False, "orig_dir": "Positive", "orig_p":".12",  "target_d":0.08, "desc":"BA vs Ctrl (Agency FU)"},
    {"id": "AgcF_GM_Ctrl", "pre":"AGENCY_PRE", "post":"AGENCY_FU", "A":"GM_SSI", "B":"Control", "flip":False, "orig_dir": "Positive", "orig_p":".01",  "target_d":0.12, "desc":"GM vs Ctrl (Agency FU)"},
    {"id": "AgcF_BA_GM",   "pre":"AGENCY_PRE", "post":"AGENCY_FU", "A":"BA_SSI", "B":"GM_SSI",  "flip":False, "orig_dir": "Negative", "orig_p":".30",  "target_d":-0.04, "desc":"BA vs GM (Agency FU)"},

    # --- 6. Anxiety FU (GAD7) ---
    {"id": "Anx_BA_Ctrl", "pre":"GAD7_PRE", "post":"GAD7_FU", "A":"BA_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":".72",  "target_d":0.02, "desc":"BA vs Ctrl (Anxiety)"},
    {"id": "Anx_GM_Ctrl", "pre":"GAD7_PRE", "post":"GAD7_FU", "A":"GM_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":".038", "target_d":0.10, "desc":"GM vs Ctrl (Anxiety)"},
    {"id": "Anx_BA_GM",   "pre":"GAD7_PRE", "post":"GAD7_FU", "A":"BA_SSI", "B":"GM_SSI",  "flip":True, "orig_dir": "Negative", "orig_p":".044", "target_d":-0.08, "desc":"BA vs GM (Anxiety)"},

    # --- 7. Trauma FU (CTSRS) ---
    {"id": "Trau_BA_Ctrl", "pre":"CTSRS_PRE", "post":"CTSRS_FU", "A":"BA_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":".05",  "target_d":0.10, "desc":"BA vs Ctrl (Trauma)"},
    {"id": "Trau_GM_Ctrl", "pre":"CTSRS_PRE", "post":"CTSRS_FU", "A":"GM_SSI", "B":"Control", "flip":True, "orig_dir": "Positive", "orig_p":".037", "target_d":0.10, "desc":"GM vs Ctrl (Trauma)"},
    {"id": "Trau_BA_GM",   "pre":"CTSRS_PRE", "post":"CTSRS_FU", "A":"BA_SSI", "B":"GM_SSI",  "flip":True, "orig_dir": "Null",     "orig_p":".78",  "target_d":0.00, "desc":"BA vs GM (Trauma)"},
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

def cdi_to_num(x):
    """Converts CDI options (A/B/C) to scores (0/1/2)"""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return x
    # Try generic parsing
    s = str(x).strip().lower()
    mapping = {"a":0,"b":1,"c":2, "0":0, "1":1, "2":2}
    return mapping.get(s, np.nan)

def safe_mean(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.Series(np.nan, index=df.index)
    return df[use].apply(pd.to_numeric, errors="coerce").mean(axis=1)

def safe_sum(df, cols):
    use = [c for c in cols if c in df.columns]
    if not use: return pd.Series(np.nan, index=df.index)
    return df[use].apply(pd.to_numeric, errors="coerce").sum(axis=1)

def load_and_score(folder: Path):
    dfs = []
    if not folder.exists(): return None

    for cond_name, fname in FILES.items():
        fpath = folder / fname
        df = load_json_to_df(fpath)
        if not df.empty:
            df["condition"] = cond_name
            dfs.append(df)

    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)

    # Scoring CDI
    c_pre = [f"CDI_PRE_Q{i}" for i in range(1,13)]
    c_fu  = [f"CDI_FU_Q{i}"  for i in range(1,13)]
    for c in c_pre + c_fu:
        if c in df.columns: df[c] = df[c].apply(cdi_to_num)

    df["CDI_PRE"] = safe_sum(df, c_pre)
    df["CDI_FU"]  = safe_sum(df, c_fu)

    # Scoring BHS (Hopelessness)
    df["BHS_PRE"]  = safe_mean(df, [f"BHS_PRE_Q{i}" for i in range(1,5)])
    df["BHS_POST"] = safe_mean(df, [f"BHS_POST_Q{i}" for i in range(1,5)])
    df["BHS_FU"]   = safe_mean(df, [f"BHS_FU_Q{i}" for i in range(1,5)])

    # Scoring Agency
    df["AGENCY_PRE"]  = safe_mean(df, [f"AGENCY_PRE_Q{i}" for i in range(1,4)])
    df["AGENCY_POST"] = safe_mean(df, [f"AGENCY_POST_Q{i}" for i in range(1,4)])
    df["AGENCY_FU"]   = safe_mean(df, [f"AGENCY_FU_Q{i}" for i in range(1,4)])

    # Scoring GAD7 (Anxiety)
    df["GAD7_PRE"] = safe_mean(df, [f"GAD7_PRE_Q{i}" for i in range(1,8)])
    df["GAD7_FU"]  = safe_mean(df, [f"GAD7_FU_Q{i}" for i in range(1,8)])

    # Scoring CTSRS (Trauma)
    df["CTSRS_PRE"] = safe_mean(df, [f"CTSRS_PRE_Q{i}" for i in range(1,7)])
    df["CTSRS_FU"]  = safe_mean(df, [f"CTSRS_FU_Q{i}" for i in range(1,7)])

    return df

# ============================================================
# 3. Statistical Functions: ANCOVA
# ============================================================

def calc_ancova_d(df, group_treat, group_ctrl, col_pre, col_post, flip):
    """
    Computes Cohen's d using ANCOVA (OLS):
    Post ~ Condition + Pre
    """
    if col_pre not in df.columns or col_post not in df.columns:
        return "Col Missing", np.nan, np.nan

    # Filter data for two groups
    sub = df[df["condition"].isin([group_treat, group_ctrl])].copy()
    sub["is_treat"] = (sub["condition"] == group_treat).astype(int)
    sub = sub[[col_pre, col_post, "is_treat"]].dropna()

    if len(sub) < 10: return "N < 10", np.nan, np.nan
    if sub[col_post].std() == 0: return "Null", 1.0, 0.0

    X = sub[["is_treat", col_pre]]
    X = sm.add_constant(X)
    y = sub[col_post]

    try:
        model = sm.OLS(y, X).fit()
        t_val = model.tvalues["is_treat"]
        p_val = model.pvalues["is_treat"]
        df_resid = model.df_resid
    except:
        return "Error", np.nan, np.nan

    # Convert t to d
    d_val = (2 * t_val) / np.sqrt(df_resid)

    # Flip sign if improvement means LOWER score (e.g. Depression)
    if flip: d_val = -d_val

    # Direction Logic
    if pd.isna(d_val): direction = "Error"
    elif abs(d_val) < 1e-9: direction = "Null"
    elif d_val > 0: direction = "Positive"
    else: direction = "Negative"

    return direction, p_val, d_val

# ============================================================
# 4. Main Analysis Loop
# ============================================================

def main():
    data_cache = {}
    print("Loading data for Study 4...")
    for m in MODELS:
        folder = PATH_CONFIGS.get(m)
        if folder and folder.exists():
            data_cache[m] = load_and_score(folder)
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

            d_val, p_val = np.nan, np.nan
            direction = "-"
            is_success = "No"

            if df is not None:
                direction, p_val, d_val = calc_ancova_d(
                    df, item["A"], item["B"], item["pre"], item["post"], item["flip"]
                )

            # --- Success Logic ---
            if direction not in ["-", "Col Missing", "N < 10", "Error"]:
                is_model_sig = (p_val < 0.05)

                # Determine if original result was significant
                try:
                    if "<" in item["orig_p"]: orig_p_val = 0.001
                    elif "n.s." in item["orig_p"]: orig_p_val = 1.0
                    else: orig_p_val = float(item["orig_p"])
                except: orig_p_val = 1.0
                
                orig_is_sig = (orig_p_val < 0.05)

                if orig_is_sig:
                    # Original Significant: Need sig match + direction match
                    if is_model_sig and (direction == item["orig_dir"]):
                        is_success = "Yes"
                else:
                    # Original Non-Significant: Need non-sig match
                    if not is_model_sig:
                        is_success = "Yes"

            row[prefix+"Direction"] = direction
            row[prefix+"p"] = round(p_val, 3) if not np.isnan(p_val) else "-"
            row[prefix+"d"] = round(d_val, 3) if not np.isnan(d_val) else "-"
            row[prefix+"Success"] = is_success

        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)
    print("\n=== Study 4 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_04_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()