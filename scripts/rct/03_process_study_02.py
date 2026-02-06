import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ========================
# 0. Paths and Configuration
# ========================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "study_02"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
# Adjust these folder structures if your Script 01 output differs
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "2",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "2",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "2"
}

MODEL_ORDER = ["Memory-DT", "Base-DT", "LLM-only"]

# Files corresponding to the 4 conditions in Study 2
# Note: Script 01 outputs JSON files
FILES = {
    "basic_apology": "2_basic_apology_raw.json",
    "need_supportive": "2_need_supportive_apology_raw.json",
    "need_thwarting": "2_need_thwarting_apology_raw.json",
    "no_apology": "2_no_apology_raw.json"
}

# Map keys to Condition names used in analysis
COND_MAP = {
    "basic_apology": "Basic Apology",
    "need_supportive": "NS Apology",
    "need_thwarting": "NT Apology",
    "no_apology": "No Apology"
}

# ========================
# 1. Ground Truth Benchmarks
# ========================
GROUND_TRUTH = [
    # --- A. Global (Trait) ---
    {"id": "Trait_DISC_NS", "type": "Trait", "desc": "NS Apology -> Disclosure", "orig_dir": "Positive", "orig_p": "<.01", "target_d": 0.43},
    {"id": "Trait_LIE_NT",  "type": "Trait", "desc": "NT Apology -> Lying",      "orig_dir": "Positive", "orig_p": "<.01", "target_d": 0.45},
    {"id": "Trait_DISC_NT", "type": "Trait", "desc": "NT Apology -> Disclosure", "orig_dir": "Negative", "orig_p": "n.s.", "target_d": -0.22},
    {"id": "Trait_LIE_NS",  "type": "Trait", "desc": "NS Apology -> Lying",      "orig_dir": "Null",     "orig_p": "n.s.", "target_d": 0.02},
    {"id": "Trait_SECR_NS", "type": "Trait", "desc": "NS Apology -> Secrecy",    "orig_dir": "Null",     "orig_p": "n.s.", "target_d": -0.02},

    # --- B. Situational (Event) ---
    {"id": "Event_DISC_NS", "type": "Event", "desc": "NS Apology -> Disclosure", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 1.00},
    {"id": "Event_LIE_NT",  "type": "Event", "desc": "NT Apology -> Lying",      "orig_dir": "Positive", "orig_p": "<.001", "target_d": 0.87},
    {"id": "Event_SECR_NS", "type": "Event", "desc": "NS Apology -> Secrecy",    "orig_dir": "Negative", "orig_p": "<.05",  "target_d": -0.28},
    {"id": "Event_SECR_NT", "type": "Event", "desc": "NT Apology -> Secrecy",    "orig_dir": "Positive", "orig_p": "<.05",  "target_d": 0.28},
    {"id": "Event_LIE_NS",  "type": "Event", "desc": "NS Apology -> Lying",      "orig_dir": "Negative", "orig_p": "<.05",  "target_d": -0.26},

    # --- C. Hypothetical (Vignette) ---
    {"id": "Vig_Lying",     "type": "Vignette", "desc": "NS vs NT (Lying)",      "orig_dir": "Negative", "orig_p": "<.001", "target_d": -0.66},
    {"id": "Vig_Secrecy",   "type": "Vignette", "desc": "NS vs NT (Secrecy)",    "orig_dir": "Negative", "orig_p": "<.01",  "target_d": -0.59},
    {"id": "Vig_Disclosure","type": "Vignette", "desc": "NS vs NT (Disclosure)", "orig_dir": "Positive", "orig_p": ".08",   "target_d": 0.54},
    {"id": "Vig_Null",      "type": "Vignette", "desc": "NS vs Basic (Disclosure)","orig_dir": "Null",     "orig_p": ">.99",  "target_d": -0.04},
    {"id": "Vig_Check",     "type": "Vignette", "desc": "NS vs NT (Perception)", "orig_dir": "Positive", "orig_p": "<.001", "target_d": 1.78},
]

# ========================
# 2. Data Loading & Engineering
# ========================

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

def load_and_merge(folder: Path):
    """Loads and merges all 4 condition files from a model directory"""
    dfs = []
    if not folder.exists(): return None

    for key, fname in FILES.items():
        fpath = folder / fname
        if fpath.exists():
            df = load_json_to_df(fpath)
            if not df.empty:
                df["Condition"] = COND_MAP[key]
                dfs.append(df)

    if not dfs: return None
    full_df = pd.concat(dfs, ignore_index=True)

    # Convert numeric columns
    for col in full_df.columns:
        if "item_" in col:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    # --- Feature Engineering ---
    # Trait aggregates
    try:
        full_df["Trait_NS"] = full_df[[f"item_Family_Q{i}" for i in range(1, 6)]].mean(axis=1)
        full_df["Trait_NT"] = full_df[[f"item_Family_Q{i}" for i in range(6, 11)]].mean(axis=1)
        full_df["Trait_DISC"] = full_df[[f"item_Family_Q{i}" for i in range(11, 14)]].mean(axis=1)
        full_df["Trait_SECR"] = full_df[[f"item_Family_Q{i}" for i in range(14, 17)]].mean(axis=1)
        full_df["Trait_LIE"] = full_df[[f"item_Family_Q{i}" for i in range(17, 20)]].mean(axis=1)
    except Exception:
        pass

    # Event metrics
    if "item_Event_Q5" in full_df.columns:
        full_df["Event_DISC"] = full_df["item_Event_Q5"]
        full_df["Event_SECR"] = full_df["item_Event_Q6"]
        full_df["Event_LIE"] = full_df["item_Event_Q7"]

    # Vignette metrics
    if "item_Scenario1_Q5" in full_df.columns:
        full_df["Vig_DISC"] = full_df["item_Scenario1_Q3"]
        full_df["Vig_SECR"] = full_df["item_Scenario1_Q4"]
        full_df["Vig_LIE"] = full_df["item_Scenario1_Q5"]
        
        # Reverse score Q1 (Perception: "didn't care" -> "cared")
        full_df["Vig_CHECK"] = 8 - full_df["item_Scenario1_Q1"]

    return full_df

# ========================
# 3. Statistical Functions
# ========================

def calc_beta_d(df, x_col, y_col):
    """Regression Analysis: Beta -> Cohen's d"""
    if x_col not in df.columns or y_col not in df.columns:
        return "Col Missing", np.nan, np.nan

    df_clean = df[[x_col, y_col]].dropna()
    if len(df_clean) < 10: return "N < 10", np.nan, np.nan

    X = df_clean[[x_col]].values
    y = df_clean[y_col].values

    if np.std(X) == 0 or np.std(y) == 0:
        return "Null", 1.0, 0.0

    # Standardize
    X_z = (X - X.mean()) / X.std()
    y_z = (y - y.mean()) / y.std()

    try:
        model = LinearRegression().fit(X_z, y_z)
        beta = model.coef_[0]
    except:
        return "Error", np.nan, np.nan

    # P-value
    n = len(y)
    if abs(beta) >= 0.999:
        t_stat = 9999
    else:
        t_stat = beta * np.sqrt((n-2)/(1-beta**2))

    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

    # Convert to d
    if abs(beta) >= 0.99:
        d_val = 4.0
    else:
        d_val = (2 * beta) / np.sqrt(1 - beta**2)

    # Direction
    if abs(d_val) < 0.1: direction = "Null"
    else: direction = "Positive" if d_val > 0 else "Negative"

    return direction, p_val, d_val

def calc_group_d(df, cond_col, group_a, group_b, y_col):
    """Group Comparison: Cohen's d"""
    if y_col not in df.columns: return "Col Missing", np.nan, np.nan

    data_a = df[df[cond_col] == group_a][y_col].dropna()
    data_b = df[df[cond_col] == group_b][y_col].dropna()

    if len(data_a) < 5 or len(data_b) < 5: return "N < 5", np.nan, np.nan

    m1, m2 = data_a.mean(), data_b.mean()
    sd1, sd2 = data_a.std(), data_b.std()

    if sd1 == 0 and sd2 == 0:
        if m1 == m2: return "Null", 1.0, 0.0
        else:
            d_val = 10.0 if m1 > m2 else -10.0
            return ("Positive" if d_val > 0 else "Negative"), 0.001, d_val

    # Pooled SD
    n1, n2 = len(data_a), len(data_b)
    sd_pooled = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))

    if sd_pooled == 0: return "Null", 1.0, 0.0

    d_val = (m1 - m2) / sd_pooled

    try:
        if sd1 == 0 and sd2 == 0: p_val = 0.0
        else: _, p_val = stats.ttest_ind(data_a, data_b, equal_var=False)
    except:
        p_val = 1.0

    if abs(d_val) < 0.1: direction = "Null"
    else: direction = "Positive" if d_val > 0 else "Negative"

    return direction, p_val, d_val

# ========================
# 4. Main Analysis Loop
# ========================

def main():
    data_cache = {}
    print("Loading data for Study 2...")

    for m in MODEL_ORDER:
        folder = PATH_CONFIGS.get(m)
        if folder and folder.exists():
            data_cache[m] = load_and_merge(folder)
        else:
            print(f"Warning: Folder not found for {m}")
            data_cache[m] = None

    final_rows = []

    for item in GROUND_TRUTH:
        row = {
            "Effect Type": item["type"],
            "Effect Description": item["desc"],
            "Original Direction": item["orig_dir"],
            "Original p": item["orig_p"],
            "Original Effect Size (d)": item["target_d"]
        }

        for m_name in MODEL_ORDER:
            df = data_cache.get(m_name)
            prefix = f"{m_name} "

            if df is None:
                for k in ["Direction", "Dir Match?", "p", "Effect Size (d)", "Success (Yes/No)"]:
                    row[prefix + k] = "N/A"
                continue

            direction, p_val, d_val = "N/A", np.nan, np.nan

            try:
                # 1. Regression Logic
                if item["type"] in ["Trait", "Event"]:
                    x, y = None, None
                    if "DISC_NS" in item["id"]: x, y = "Trait_NS", "Trait_DISC"
                    elif "LIE_NT" in item["id"]: x, y = "Trait_NT", "Trait_LIE"
                    elif "DISC_NT" in item["id"]: x, y = "Trait_NT", "Trait_DISC"
                    elif "LIE_NS" in item["id"]: x, y = "Trait_NS", "Trait_LIE"
                    elif "SECR_NS" in item["id"]: x, y = "Trait_NS", "Trait_SECR"
                    elif "Event_DISC_NS" in item["id"]: x, y = "Trait_NS", "Event_DISC"
                    elif "Event_LIE_NT" in item["id"]: x, y = "Trait_NT", "Event_LIE"
                    elif "Event_SECR_NS" in item["id"]: x, y = "Trait_NS", "Event_SECR"
                    elif "Event_SECR_NT" in item["id"]: x, y = "Trait_NT", "Event_SECR"
                    elif "Event_LIE_NS" in item["id"]: x, y = "Trait_NS", "Event_LIE"

                    if x and y:
                        direction, p_val, d_val = calc_beta_d(df, x, y)

                # 2. Group Logic
                elif item["type"] == "Vignette":
                    y, a, b = None, None, None
                    if "Vig_Lying" in item["id"]: y, a, b = "Vig_LIE", "NS Apology", "NT Apology"
                    elif "Vig_Secrecy" in item["id"]: y, a, b = "Vig_SECR", "NS Apology", "NT Apology"
                    elif "Vig_Disclosure" in item["id"]: y, a, b = "Vig_DISC", "NS Apology", "NT Apology"
                    elif "Vig_Null" in item["id"]: y, a, b = "Vig_DISC", "NS Apology", "Basic Apology"
                    elif "Vig_Check" in item["id"]: y, a, b = "Vig_CHECK", "NS Apology", "NT Apology"

                    if y:
                        direction, p_val, d_val = calc_group_d(df, "Condition", a, b, y)

            except Exception as e:
                print(f"Error in {m_name} - {item['id']}: {e}")

            # Replication Success Logic
            target_d = item["target_d"]
            is_success = False
            is_match = False

            if direction not in ["N/A", "Col Missing", "N < 10", "Error"]:
                if abs(target_d) < 0.1: # Null Effect Target
                    if abs(d_val) < 0.2 or p_val > 0.05:
                        is_success = True
                        is_match = True
                else: # Significant Target
                    if (d_val * target_d > 0):
                        is_match = True
                        if p_val < 0.05: is_success = True

            # Populate Row
            row[prefix + "Direction"] = direction
            row[prefix + "Dir Match?"] = "Yes" if is_match else "No"
            row[prefix + "p"] = round(p_val, 3) if not np.isnan(p_val) else "-"
            row[prefix + "Effect Size (d)"] = round(d_val, 3) if not np.isnan(d_val) else "-"
            row[prefix + "Success (Yes/No)"] = "Yes" if is_success else "No"

        final_rows.append(row)

    df_wide = pd.DataFrame(final_rows)
    print("\n=== Study 2 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_wide)
    
    out_file = paths.output_dir / "study_02_analysis_summary.csv"
    df_wide.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()