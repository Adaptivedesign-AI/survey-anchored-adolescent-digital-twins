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
        self.output_dir = self.results_base / "study_03"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "3",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "3",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "3"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Filenames (JSON output from Step 01)
ACCEPT_FILE = "3_acceptance_raw.json"
CONTROL_FILE = "3_control_raw.json"

# ============================================================
# 1. Ground Truth Benchmarks
# ============================================================
GROUND_TRUTH = [
    # --- A. Main Effects (T0 vs T1) ---
    {"id": "Main_NA", "type": "Within", "desc": "Ostracism -> NA Increase", "orig_dir": "Positive", "orig_p": ".003", "target_d": 0.35},
    {"id": "Main_PA", "type": "Within", "desc": "Ostracism -> PA Decrease", "orig_dir": "Negative", "orig_p": "<.001", "target_d": -0.45},

    # --- B. Interaction (Exp vs Ctrl) ---
    {"id": "Inter_NA", "type": "Between", "desc": "Acceptance vs Ctrl (NA Recovery)", "orig_dir": "Null", "orig_p": ".15", "target_d": 0.50},
    {"id": "Inter_PA", "type": "Between", "desc": "Acceptance vs Ctrl (PA Recovery)", "orig_dir": "Null", "orig_p": ".15", "target_d": 0.55},

    # --- C. Correlational (Traits) ---
    {"id": "Corr_Int_RS", "type": "Corr", "desc": "Internalizing <-> RS", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 1.03},
    {"id": "Corr_Int_Adapt", "type": "Corr", "desc": "Internalizing <-> Adaptive ER", "orig_dir": "Negative", "orig_p": "<.01", "target_d": -0.84},

    # --- D. Moderation (Traits predict Reactivity) ---
    {"id": "Mod_RS_NA", "type": "Regression", "desc": "RS -> NA Reactivity (T1-T0)", "orig_dir": "Null", "orig_p": ".26", "target_d": 0.28},
    {"id": "Mod_Int_NA", "type": "Regression", "desc": "Internalizing -> NA Reactivity", "orig_dir": "Null", "orig_p": ".47", "target_d": 0.20},
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

def compute_panas(df, t):
    """Compute PANAS Positive/Negative Affect scores"""
    pa_cols = [f"PANAS_{t}_Q{i}" for i in range(1, 6)]
    na_cols = [f"PANAS_{t}_Q{i}" for i in range(6, 11)]

    valid_pa = [c for c in pa_cols if c in df.columns]
    valid_na = [c for c in na_cols if c in df.columns]

    if valid_pa: df[f"PA_{t}"] = df[valid_pa].mean(axis=1)
    if valid_na: df[f"NA_{t}"] = df[valid_na].mean(axis=1)
    return df

def load_and_process(folder: Path):
    """Load Acceptance and Control files and merge"""
    f_acc = folder / ACCEPT_FILE
    f_ctl = folder / CONTROL_FILE

    if not f_acc.exists() or not f_ctl.exists():
        return None

    try:
        acc = load_json_to_df(f_acc)
        ctl = load_json_to_df(f_ctl)

        if acc.empty or ctl.empty: return None

        acc["Group"] = "Exp"
        ctl["Group"] = "Ctrl"

        df = pd.concat([acc, ctl], ignore_index=True)

        # Force numeric
        for col in df.columns:
            if any(x in col for x in ["PANAS", "FZEK", "FEEL", "BSCL"]):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- 1. PANAS Scores ---
        for t in ["T0", "T1", "T2"]:
            df = compute_panas(df, t)

        # Reactivity (Delta): T1 - T0
        if "NA_T1" in df.columns:
            df["Delta_NA"] = df["NA_T1"] - df["NA_T0"]
            df["Delta_PA"] = df["PA_T1"] - df["PA_T0"]

        # Recovery: T2 - T1
        if "NA_T2" in df.columns:
            df["Recovery_NA"] = df["NA_T2"] - df["NA_T1"]
            df["Recovery_PA"] = df["PA_T2"] - df["PA_T1"]

        # --- 2. Rejection Sensitivity (RS) ---
        rs_items = []
        for i in range(1, 10):
            anx = f"FZEK_T0_Q{i}_anxiety"
            exp = f"FZEK_T0_Q{i}_expectation"
            if anx in df.columns and exp in df.columns:
                rs_items.append(df[anx] * df[exp])
        if rs_items:
            df["RS"] = pd.concat(rs_items, axis=1).mean(axis=1)

        # --- 3. Internalizing (BSCL) ---
        bscl_cols = [f"BSCL_T0_Q{i}" for i in range(1, 18)]
        valid_bscl = [c for c in bscl_cols if c in df.columns]
        if valid_bscl:
            df["Internalizing"] = df[valid_bscl].mean(axis=1)

        # --- 4. Adaptive ER (FEEL) ---
        feel_cols = [f"FEEL_T0_Q{i}" for i in range(1, 7)]
        valid_feel = [c for c in feel_cols if c in df.columns]
        if valid_feel:
            df["Adaptive_ER"] = df[valid_feel].mean(axis=1)

        return df
    except Exception as e:
        print(f"Error loading {folder}: {e}")
        return None

# ============================================================
# 3. Statistical Functions
# ============================================================

def calc_d_within(df, col_pre, col_post):
    """Paired t-test -> Cohen's d"""
    if col_pre not in df.columns or col_post not in df.columns: return np.nan, np.nan
    diff = df[col_post] - df[col_pre]
    if len(diff.dropna()) < 10: return np.nan, np.nan

    if diff.std() == 0: return 0.0, 1.0

    d_val = diff.mean() / diff.std()
    _, p_val = stats.ttest_rel(df[col_post], df[col_pre])
    return d_val, p_val

def calc_d_between(df, y_col):
    """Independent t-test (Exp vs Ctrl) -> Cohen's d"""
    if y_col not in df.columns: return np.nan, np.nan
    exp = df[df["Group"]=="Exp"][y_col].dropna()
    ctrl = df[df["Group"]=="Ctrl"][y_col].dropna()

    if len(exp)<5 or len(ctrl)<5: return np.nan, np.nan

    if exp.std() == 0 and ctrl.std() == 0:
        return (0.0, 1.0) if exp.mean() == ctrl.mean() else (10.0, 0.001)

    # Pooled SD
    n1, n2 = len(exp), len(ctrl)
    sp = np.sqrt(((n1-1)*exp.var() + (n2-1)*ctrl.var()) / (n1+n2-2))

    d_val = (exp.mean() - ctrl.mean()) / sp
    _, p_val = stats.ttest_ind(exp, ctrl, equal_var=False)
    return d_val, p_val

def calc_beta_d(df, x_col, y_col):
    """Regression Beta -> Cohen's d"""
    if x_col not in df.columns or y_col not in df.columns: return np.nan, np.nan
    temp = df[[x_col, y_col]].dropna()

    if len(temp) < 10 or temp[x_col].std() == 0 or temp[y_col].std() == 0:
        return 0.0, 1.0

    X = (temp[[x_col]] - temp[x_col].mean()) / temp[x_col].std()
    y = (temp[y_col] - temp[y_col].mean()) / temp[y_col].std()

    try:
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
    except: return 0.0, 1.0

    n = len(y)
    if abs(beta) >= 0.99:
        d_val = 4.0
        p_val = 0.0
    else:
        d_val = (2 * beta) / np.sqrt(1 - beta**2)
        t_stat = beta * np.sqrt((n-2)/(1-beta**2))
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

    return d_val, p_val

# ============================================================
# 4. Main Loop
# ============================================================

def main():
    data_cache = {}
    print("Loading data for Study 3...")
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
            "Original d": item["target_d"]
        }

        for m_name in MODELS:
            df = data_cache.get(m_name)
            prefix = f"{m_name} "

            if df is None:
                row[prefix + "d"] = "N/A"
                row[prefix + "Success"] = "No Data"
                continue

            d_val, p_val = np.nan, np.nan

            try:
                # 1. Main Effect (T0->T1)
                if item["id"] == "Main_NA": d_val, p_val = calc_d_within(df, "NA_T0", "NA_T1")
                elif item["id"] == "Main_PA": d_val, p_val = calc_d_within(df, "PA_T0", "PA_T1")

                # 2. Interaction (T2 Recovery)
                elif item["id"] == "Inter_NA": d_val, p_val = calc_d_between(df, "Recovery_NA")
                elif item["id"] == "Inter_PA": d_val, p_val = calc_d_between(df, "Recovery_PA")

                # 3. Correlation / Moderation
                elif item["type"] in ["Corr", "Regression"]:
                    x, y = None, None
                    if "Int_RS" in item["id"]: x, y = "Internalizing", "RS"
                    elif "Int_Adapt" in item["id"]: x, y = "Internalizing", "Adaptive_ER"
                    elif "RS_NA" in item["id"]: x, y = "RS", "Delta_NA"
                    elif "Int_NA" in item["id"]: x, y = "Internalizing", "Delta_NA"

                    if x and y: d_val, p_val = calc_beta_d(df, x, y)

            except Exception:
                pass

            # Success Check
            target_d = item["target_d"]
            is_success = "No"

            if not np.isnan(d_val):
                if abs(d_val) < 0.1: direction = "Null"
                else: direction = "Positive" if d_val > 0 else "Negative"

                if abs(target_d) < 0.3 or "n.s." in str(item["orig_p"]):
                    if abs(d_val) < 0.3 or p_val > 0.05: is_success = "Yes"
                else:
                    if (d_val * target_d > 0) and (p_val < 0.05): is_success = "Yes"
            else:
                direction = "N/A"

            row[prefix + "Direction"] = direction
            row[prefix + "p"] = round(p_val, 3) if not np.isnan(p_val) else "-"
            row[prefix + "d"] = round(d_val, 3) if not np.isnan(d_val) else "-"
            row[prefix + "Success"] = is_success

        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)
    print("\n=== Study 3 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_03_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()