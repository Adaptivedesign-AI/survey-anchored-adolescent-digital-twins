import os
import json
import re
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ============================================================
# 0. Paths and Configuration
# ============================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "study_10"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "10",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "10",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "10"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Filenames (JSON)
# 10_Turkish_immigrant_raw.json corresponds to the Immigrant condition
# 10_British_immigrant_raw.json corresponds to the British condition
FILES = {
    "Immigrant": "10_Turkish_immigrant_raw.json",
    "British": "10_British_immigrant_raw.json"
}

# ============================================================
# 1. Original Results (Benchmarks)
# ============================================================
ORIGINAL_RESULTS = {
    # 1) Adolescents: Immigrant condition, Individual > Group
    "Main_eval_Immigrant": {
        "desc": "Adolescents: Individual > Group (Immigrant condition)",
        "direction": "Individual > Group",
        "p": 0.001,   # Original p < .001
        "effect_d": 0.50
    },
    # 2) Adolescents: British condition, Individual > Group
    "Main_eval_British": {
        "desc": "Adolescents: Individual > Group (British condition)",
        "direction": "Individual > Group",
        "p": 0.001,   # Original p < .001
        "effect_d": 0.46
    },
    # 3) Condition main effect: Immigrant vs British (no difference expected)
    "Condition_main_effect": {
        "desc": "Immigrant vs British (no difference, approximated)",
        "direction": "≈0",
        "p": 0.50,    # Original F=0.46, p=.50
        "effect_d": 0.09
    },
    # 4) Condition x Evaluation: Gap larger in Immigrant condition
    "Condition_x_Eval": {
        "desc": "Gap larger in Immigrant condition (Condition x Eval)",
        "direction": "Immigrant gap > British gap",
        "p": 0.045,   # Original p=.045
        "effect_d": 0.24
    }
}

# ============================================================
# 2. Helper Functions
# ============================================================

def _norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Comprehension checks
def pass_friend_born_check(s):
    x = _norm(s)
    return any(k in x for k in ["britain","british","uk","united kingdom","england"])

def pass_newcomer_turkey(s):
    return "turkey" in _norm(s)

def pass_newcomer_british(s):
    x = _norm(s)
    return any(k in x for k in ["bristol","britain","uk","united kingdom","england"])

def pass_excluder_name(s):
    return _norm(s) == "sam"

def pass_excluder_in_group(s):
    return _norm(s) in {"yes","y","true","1"}

# Effect sizes
def cohen_dz_paired(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if diff.size < 2: return np.nan
    sd = diff.std(ddof=1)
    if sd == 0: return np.nan
    return diff.mean() / sd

def cohen_d_between(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2: return np.nan
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp <= 0: return np.nan
    return (x.mean() - y.mean()) / sp

# ============================================================
# 3. Data Loading & Cleaning
# ============================================================

def load_json_to_df(file_path: Path) -> pd.DataFrame:
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

def load_clean_df(folder_path: Path, fname: str, condition_label: str):
    path = folder_path / fname
    df = load_json_to_df(path)
    
    if df.empty: return pd.DataFrame()

    # Required columns
    needed = [
        "check_friends_born_where","check_newcomer_from_where",
        "check_excluder_name","check_excluder_in_your_group",
        "eval_individual_1_6","eval_group_1_6"
    ]
    
    # Check for missing columns (soft fail)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Warning: {fname} missing columns {missing}")
        return pd.DataFrame()

    # Numerical coercion
    for k in ["eval_individual_1_6","eval_group_1_6"]:
        df[k] = pd.to_numeric(df[k], errors="coerce")

    # Apply Comprehension Filters
    mask_friend = df["check_friends_born_where"].apply(pass_friend_born_check)

    if "Immigrant" in condition_label: 
        mask_newcomer = df["check_newcomer_from_where"].apply(pass_newcomer_turkey)
    else: 
        mask_newcomer = df["check_newcomer_from_where"].apply(pass_newcomer_british)

    mask_ex_name = df["check_excluder_name"].apply(pass_excluder_name)
    mask_ex_in   = df["check_excluder_in_your_group"].apply(pass_excluder_in_group)

    mask_scale = (
        df["eval_individual_1_6"].between(1,6, inclusive="both") &
        df["eval_group_1_6"].between(1,6, inclusive="both")
    )

    clean = df[mask_friend & mask_newcomer & mask_ex_name & mask_ex_in & mask_scale].copy()
    clean["condition"] = condition_label
    clean["gap"] = clean["eval_individual_1_6"] - clean["eval_group_1_6"]
    
    return clean

# ============================================================
# 4. Analysis Logic
# ============================================================

def analyze_system(name, folder):
    if not folder.exists(): return {}
    
    print(f"Analyzing {name}...")

    turk = load_clean_df(folder, FILES["Immigrant"], "Immigrant")
    brit = load_clean_df(folder, FILES["British"], "British")

    results = {}

    # 1) Main_eval_Immigrant: Adolescents, Immigrant condition
    if not turk.empty:
        ind_i = turk["eval_individual_1_6"]
        grp_i = turk["eval_group_1_6"]
        mask_i = ind_i.notna() & grp_i.notna()
        if mask_i.sum() > 2:
            t_i, p_i = stats.ttest_rel(ind_i[mask_i], grp_i[mask_i])
            dz_i = cohen_dz_paired(ind_i[mask_i] - grp_i[mask_i])
            dir_i = "Individual > Group" if ind_i.mean() > grp_i.mean() else "Individual < Group"
            results["Main_eval_Immigrant"] = (dir_i, p_i, dz_i)
        else:
            results["Main_eval_Immigrant"] = ("N<2", np.nan, np.nan)
    else:
        results["Main_eval_Immigrant"] = ("No Data", np.nan, np.nan)

    # 2) Main_eval_British: Adolescents, British condition
    if not brit.empty:
        ind_b = brit["eval_individual_1_6"]
        grp_b = brit["eval_group_1_6"]
        mask_b = ind_b.notna() & grp_b.notna()
        if mask_b.sum() > 2:
            t_b, p_b = stats.ttest_rel(ind_b[mask_b], grp_b[mask_b])
            dz_b = cohen_dz_paired(ind_b[mask_b] - grp_b[mask_b])
            dir_b = "Individual > Group" if ind_b.mean() > grp_b.mean() else "Individual < Group"
            results["Main_eval_British"] = (dir_b, p_b, dz_b)
        else:
            results["Main_eval_British"] = ("N<2", np.nan, np.nan)
    else:
        results["Main_eval_British"] = ("No Data", np.nan, np.nan)

    # 3) Condition_main_effect: Immigrant vs British on individual
    if not turk.empty and not brit.empty:
        t2, p2 = stats.ttest_ind(
            turk["eval_individual_1_6"], brit["eval_individual_1_6"],
            equal_var=False, nan_policy="omit"
        )
        d2 = cohen_d_between(turk["eval_individual_1_6"], brit["eval_individual_1_6"])
        dir2 = "Immigrant > British" if d2 > 0 else "Immigrant < British"
        
        if abs(d2) < 0.1: dir2 = "≈0"
        
        results["Condition_main_effect"] = (dir2, p2, d2)

        # 4) Condition_x_Eval: between-subjects d on gap
        t3, p3 = stats.ttest_ind(
            turk["gap"], brit["gap"],
            equal_var=False, nan_policy="omit"
        )
        d3 = cohen_d_between(turk["gap"], brit["gap"])
        dir3 = "Immigrant gap > British gap" if d3 > 0 else "Immigrant gap < British gap"
        results["Condition_x_Eval"] = (dir3, p3, d3)
    else:
        results["Condition_main_effect"] = ("No Data", np.nan, np.nan)
        results["Condition_x_Eval"] = ("No Data", np.nan, np.nan)

    return results

# ============================================================
# 5. Main Execution Loop
# ============================================================

def main():
    rows = []
    
    # Run Analysis
    sys_results = {}
    for m in MODELS:
        folder = PATH_CONFIGS.get(m)
        sys_results[m] = analyze_system(m, folder)

    # Compile Table
    for eff_key, orig in ORIGINAL_RESULTS.items():
        row = {
            "Effect ID": eff_key,
            "Description": orig["desc"],
            "Orig Dir": orig["direction"],
            "Orig p": orig["p"],
            "Orig d": orig["effect_d"]
        }

        for m in MODELS:
            if eff_key in sys_results[m]:
                res_dir, res_p, res_d = sys_results[m][eff_key]
            else:
                res_dir, res_p, res_d = "N/A", np.nan, np.nan

            # Success Logic
            is_success = "No"
            orig_p_thresh = 0.05
            
            # If Orig is Significant (p < .05) -> Need Sig & Direction match
            if orig["p"] < orig_p_thresh:
                if (not np.isnan(res_p)) and (res_p < 0.05):
                    # Check direction
                    if res_dir == orig["direction"]:
                        is_success = "Yes"
            else:
                # If Orig is Null -> Need Non-Sig
                if (not np.isnan(res_p)) and (res_p >= 0.05):
                    is_success = "Yes"

            row[f"{m} Dir"] = res_dir
            row[f"{m} p"] = round(res_p, 3) if not np.isnan(res_p) else "-"
            row[f"{m} d"] = round(res_d, 2) if not np.isnan(res_d) else "-"
            row[f"{m} Success"] = is_success

        rows.append(row)

    df_final = pd.DataFrame(rows)
    print("\n=== Study 10 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_10_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()