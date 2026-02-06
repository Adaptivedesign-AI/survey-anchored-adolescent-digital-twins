import os
import json
import math
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# =============================================================================
# 0. PATHS
# =============================================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.output_dir = self.results_base / "study_08"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "8",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "8",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "8"
}

MODELS = ["Memory-DT", "Base-DT", "LLM-only"]

# Files (JSON)
FILES = {
    "Control": "8_control_raw.json",
    "Intervention": "8_intervention_raw.json"
}

# =============================================================================
# 1. GLOBAL CONSTANTS & HELPERS
# =============================================================================

ALLOWED_COLS = [
    "student_id",
    "item_SMM_PRE_Q1","item_SMM_PRE_Q2","item_SMM_PRE_Q3","item_SMM_PRE_Q4",
    "item_SMM_PRE_Q5","item_SMM_PRE_Q6","item_SMM_PRE_Q7","item_SMM_PRE_Q8",
    "item_PSS_PRE_Q1","item_PSS_PRE_Q2","item_PSS_PRE_Q3","item_PSS_PRE_Q4",
    "item_PSS_PRE_Q5","item_PSS_PRE_Q6","item_PSS_PRE_Q7","item_PSS_PRE_Q8",
    "item_PSS_PRE_Q9","item_PSS_PRE_Q10",
    "item_GAD_PRE_Q1","item_GAD_PRE_Q2","item_GAD_PRE_Q3","item_GAD_PRE_Q4",
    "item_GAD_PRE_Q5","item_GAD_PRE_Q6","item_GAD_PRE_Q7","item_GAD_PRE_Q8",
    "item_GAD_PRE_Q9","item_GAD_PRE_Q10",
    "item_SMM_POST_Q1","item_SMM_POST_Q2","item_SMM_POST_Q3","item_SMM_POST_Q4",
    "item_SMM_POST_Q5","item_SMM_POST_Q6","item_SMM_POST_Q7","item_SMM_POST_Q8",
    "item_APPRAISAL_POST_Q1","item_APPRAISAL_POST_Q2","item_APPRAISAL_POST_Q3",
    "item_GAD_POST_Q1","item_GAD_POST_Q2","item_GAD_POST_Q3","item_GAD_POST_Q4",
    "item_GAD_POST_Q5","item_GAD_POST_Q6","item_GAD_POST_Q7","item_GAD_POST_Q8",
    "item_GAD_POST_Q9","item_GAD_POST_Q10"
]

def _to_float(x):
    try: return float(x)
    except Exception: return np.nan

def sign_label(x, tol=1e-8):
    if pd.isna(x) or abs(x) < tol: return "null"
    return "positive" if x > 0 else "negative"

def ensure_two_groups(df, cond_col="cond"):
    if cond_col not in df.columns: return False
    if df[cond_col].nunique() != 2: return False
    n1 = (df[cond_col] == 1).sum()
    n0 = (df[cond_col] == 0).sum()
    return (n1 > 0) and (n0 > 0)

def normal_cdf(x):
    """Standard normal CDF for posterior approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# =============================================================================
# 2. Original Effects
# =============================================================================

ORIGINAL_EFFECTS = {
    # ---------------- Main effects ----------------
    "SMM_POST_main": {
        "Effect type": "Main effect",
        "Effect Description": "T1 stress mindset (SMM_POST), intervention vs control",
        "Original Direction": "positive",     
        "Original p": np.nan,                 
        "Original Effect Size": 0.12,         
        "Original Significant": True          
    },
    "APPRAISAL_POST_main": {
        "Effect type": "Main effect",
        "Effect Description": "T1 negative stress appraisal (higher = more threatened)",
        "Original Direction": "negative",     
        "Original p": np.nan,                 
        "Original Effect Size": -0.05,        
        "Original Significant": True
    },
    "GAD_T1_main": {
        "Effect type": "Main effect",
        "Effect Description": "T1 GAD symptoms (GAD_POST), intervention vs control",
        "Original Direction": "negative",
        "Original p": 0.97,                   # Pr(ATE<0) = 0.97
        "Original Effect Size": -0.07,        
        "Original Significant": True          
    },

    # ---------------- Interactions ----------------
    "GAD_T1_CATE_SMM_-1SD": {
        "Effect type": "Interaction",
        "Effect Description": "T1 GAD: stress-is-debilitating mindset (-1 SD)",
        "Original Direction": "negative",
        "Original p": np.nan,                 
        "Original Effect Size": -0.07,
        "Original Significant": True
    },
    "GAD_T1_CATE_SMM_+1SD": {
        "Effect type": "Interaction",
        "Effect Description": "T1 GAD: stress-is-enhancing mindset (+1 SD)",
        "Original Direction": "negative",
        "Original p": np.nan,
        "Original Effect Size": -0.05,
        "Original Significant": True
    },
    "GAD_T1_CATE_PSS_high": {
        "Effect type": "Interaction",
        "Effect Description": "T1 GAD: high perceived stress at baseline",
        "Original Direction": "negative",
        "Original p": np.nan,
        "Original Effect Size": -0.11,
        "Original Significant": True
    },
    "GAD_T1_CATE_ANX_severe": {
        "Effect type": "Interaction",
        "Effect Description": "T1 GAD: baseline anxiety severe",
        "Original Direction": "negative",
        "Original p": np.nan,
        "Original Effect Size": -0.18,
        "Original Significant": True
    },
}

# =============================================================================
# 3. Data Loading
# =============================================================================

def load_json_to_df(file_path: Path) -> pd.DataFrame:
    if not file_path.exists(): return pd.DataFrame()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        rows = []
        for entry in data:
            row = {'student_id': entry.get('student_id')}
            items = entry.get('json_items', {})
            if items: row.update(items)
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def load_system_df(base_path):
    path_ctl = base_path / FILES["Control"]
    path_int = base_path / FILES["Intervention"]

    df_ctl = load_json_to_df(path_ctl)
    df_int = load_json_to_df(path_int)

    if df_ctl.empty or df_int.empty: return pd.DataFrame()

    keep_cols = [c for c in ALLOWED_COLS if c in df_ctl.columns or c in df_int.columns]
    
    # Ensure cols exist
    for c in keep_cols:
        if c not in df_ctl.columns: df_ctl[c] = np.nan
        if c not in df_int.columns: df_int[c] = np.nan

    df_ctl = df_ctl[keep_cols].copy()
    df_int = df_int[keep_cols].copy()

    df_ctl["cond"] = 0
    df_int["cond"] = 1

    return pd.concat([df_ctl, df_int], ignore_index=True)

def build_scales(df):
    SMM_PRE_COLS  = [f"item_SMM_PRE_Q{i}"  for i in range(1, 9)]
    SMM_POST_COLS = [f"item_SMM_POST_Q{i}" for i in range(1, 9)]
    PSS_PRE_COLS  = [f"item_PSS_PRE_Q{i}"  for i in range(1, 11)]
    GAD_PRE_COLS  = [f"item_GAD_PRE_Q{i}"  for i in range(1, 11)]
    GAD_POST_COLS = [f"item_GAD_POST_Q{i}" for i in range(1, 11)]
    APP_POST_COLS = [f"item_APPRAISAL_POST_Q{i}" for i in range(1, 4)]

    # Convert to numeric
    all_qs = SMM_PRE_COLS + SMM_POST_COLS + PSS_PRE_COLS + GAD_PRE_COLS + GAD_POST_COLS + APP_POST_COLS
    for c in all_qs:
        if c in df.columns: df[c] = df[c].apply(_to_float)

    # Reverse score SMM Q5-8: x -> 4 - x (Assuming 0-4 scale? Or 1-5? Assuming 0-4 based on prior context, if 1-5 then 6-x. Let's assume 0-4 as per standard or adjust)
    # Original script used 4 - x.
    for i in [5, 6, 7, 8]:
        for prefix in ["item_SMM_PRE_Q", "item_SMM_POST_Q"]:
            c = f"{prefix}{i}"
            if c in df.columns: df[c] = 4 - df[c]

    # Compute Means/Sums
    if set(SMM_PRE_COLS).issubset(df.columns): df["SMM_PRE"] = df[SMM_PRE_COLS].mean(axis=1)
    if set(SMM_POST_COLS).issubset(df.columns): df["SMM_POST"] = df[SMM_POST_COLS].mean(axis=1)
    
    if set(PSS_PRE_COLS).issubset(df.columns):
        df["PSS_PRE_mean"] = df[PSS_PRE_COLS].mean(axis=1)
        df["PSS_PRE_sum"]  = df[PSS_PRE_COLS].sum(axis=1)
        
    if set(APP_POST_COLS).issubset(df.columns): df["APPRAISAL_POST"] = df[APP_POST_COLS].mean(axis=1)
    
    if set(GAD_PRE_COLS).issubset(df.columns):
        df["GAD_PRE_mean"] = df[GAD_PRE_COLS].mean(axis=1)
        df["GAD_PRE_sum"]  = df[GAD_PRE_COLS].sum(axis=1)
        
    if set(GAD_POST_COLS).issubset(df.columns): df["GAD_POST"] = df[GAD_POST_COLS].mean(axis=1)

    # Groups
    def pss_group(score):
        if pd.isna(score): return np.nan
        if score <= 13: return "low"
        elif score <= 26: return "medium"
        else: return "high"
    df["PSS_GROUP"] = df["PSS_PRE_sum"].map(pss_group)

    def anx_group(score):
        if pd.isna(score): return np.nan
        if score < 20: return "slight" # Assuming sum metric logic matches scale
        elif score < 25: return "mild"
        elif score < 30: return "moderate"
        else: return "severe"
    df["ANX_GROUP"] = df["GAD_PRE_sum"].map(anx_group)

    return df

# =============================================================================
# 4. Statistical Models
# =============================================================================

def fit_main_effect(df, dv, covars):
    cols = [dv, "cond"] + covars
    d = df[cols].dropna().copy()
    if not ensure_two_groups(d): return {"effect_size": np.nan, "p": np.nan}

    # Standardize DV
    d["dv_std"] = (d[dv] - d[dv].mean()) / d[dv].std(ddof=1)
    
    formula = "dv_std ~ cond"
    if covars: formula += " + " + " + ".join(covars)

    try:
        model = smf.ols(formula, data=d).fit()
        beta = model.params.get("cond", np.nan)
        se = model.bse.get("cond", np.nan)
        
        if pd.isna(beta) or pd.isna(se) or se == 0:
            post_prob = np.nan
        else:
            post_prob = normal_cdf(-beta / se)
            
        return {"effect_size": beta, "p": post_prob, "N": len(d)}
    except:
        return {"effect_size": np.nan, "p": np.nan}

def fit_cate_smm(df):
    dv = "GAD_POST"
    covars = ["PSS_PRE_mean"]
    cols = [dv, "cond", "SMM_PRE"] + covars
    d = df[cols].dropna().copy()
    if not ensure_two_groups(d): return {}

    d["dv_std"] = (d[dv] - d[dv].mean()) / d[dv].std(ddof=1)
    
    formula = "dv_std ~ cond * SMM_PRE + " + " + ".join(covars)
    try:
        model = smf.ols(formula, data=d).fit()
        params = model.params
        covmat = model.cov_params()
        
        if "cond" not in params or "cond:SMM_PRE" not in params: return {}
        
        b_c = params["cond"]
        b_i = params["cond:SMM_PRE"]
        var_c = covmat.loc["cond", "cond"]
        var_i = covmat.loc["cond:SMM_PRE", "cond:SMM_PRE"]
        cov_ci = covmat.loc["cond", "cond:SMM_PRE"]
        
        smm_mean = d["SMM_PRE"].mean()
        smm_sd = d["SMM_PRE"].std()
        
        out = {}
        for label, smm_val, eff_id in [("-1SD", smm_mean - smm_sd, "GAD_T1_CATE_SMM_-1SD"), 
                                       ("+1SD", smm_mean + smm_sd, "GAD_T1_CATE_SMM_+1SD")]:
            eff = b_c + b_i * smm_val
            var_eff = var_c + (smm_val**2)*var_i + 2*smm_val*cov_ci
            se_eff = math.sqrt(var_eff) if var_eff > 0 else np.nan
            
            prob = normal_cdf(-eff/se_eff) if not pd.isna(se_eff) else np.nan
            out[eff_id] = {"effect_size": eff, "p": prob, "N": len(d)}
        return out
    except: return {}

def fit_cate_grouped(df, group_col, mapping):
    results = {}
    for grp_val, eff_id in mapping.items():
        d = df.loc[df[group_col] == grp_val, ["GAD_POST", "cond", "PSS_PRE_mean", "GAD_PRE_mean"]].dropna().copy()
        if not ensure_two_groups(d):
            results[eff_id] = {"effect_size": np.nan, "p": np.nan}
            continue
            
        d["dv_std"] = (d["GAD_POST"] - d["GAD_POST"].mean()) / d["GAD_POST"].std(ddof=1)
        try:
            model = smf.ols("dv_std ~ cond + GAD_PRE_mean + PSS_PRE_mean", data=d).fit()
            beta = model.params.get("cond", np.nan)
            se = model.bse.get("cond", np.nan)
            
            prob = normal_cdf(-beta/se) if not pd.isna(se) and se!=0 else np.nan
            results[eff_id] = {"effect_size": beta, "p": prob, "N": len(d)}
        except:
            results[eff_id] = {"effect_size": np.nan, "p": np.nan}
    return results

# =============================================================================
# 5. Main Execution
# =============================================================================

def run_system(base_path):
    if not base_path.exists(): return {}
    df = load_system_df(base_path)
    if df.empty: return {}
    df = build_scales(df)

    effects = {}
    effects["SMM_POST_main"] = fit_main_effect(df, "SMM_POST", ["SMM_PRE", "PSS_PRE_mean"])
    effects["APPRAISAL_POST_main"] = fit_main_effect(df, "APPRAISAL_POST", ["SMM_PRE", "PSS_PRE_mean"])
    effects["GAD_T1_main"] = fit_main_effect(df, "GAD_POST", ["GAD_PRE_mean", "PSS_PRE_mean"])
    
    effects.update(fit_cate_smm(df))
    effects.update(fit_cate_grouped(df, "PSS_GROUP", {"high": "GAD_T1_CATE_PSS_high"}))
    effects.update(fit_cate_grouped(df, "ANX_GROUP", {"severe": "GAD_T1_CATE_ANX_severe"}))
    
    return effects

def main():
    results = {}
    print("Processing Study 8...")
    
    for m in MODELS:
        path = PATH_CONFIGS.get(m)
        results[m] = run_system(path)

    rows = []
    for eff_id, o in ORIGINAL_EFFECTS.items():
        row = {
            "Effect ID": eff_id,
            "Description": o["Effect Description"],
            "Orig Dir": o["Original Direction"],
            "Orig p(Pr<0)": o["Original p"],
            "Orig d": o["Original Effect Size"]
        }

        for m in MODELS:
            res = results[m].get(eff_id, {})
            es = res.get("effect_size", np.nan)
            p = res.get("p", np.nan)
            
            direction = sign_label(es)
            
            # Success Logic (Bayesian Posterior)
            # Threshold 0.95 for posterior probability
            sig = False
            if not pd.isna(p):
                if o["Original Direction"] == "negative": sig = (p >= 0.95)
                elif o["Original Direction"] == "positive": sig = ((1-p) >= 0.95)
            
            # Match success
            is_success = False
            if o["Original Significant"]:
                if sig and direction == o["Original Direction"]: is_success = True
            else:
                if not sig: is_success = True

            row[f"{m} Dir"] = direction
            row[f"{m} Pr(<0)"] = round(p, 3) if not pd.isna(p) else "-"
            row[f"{m} d"] = round(es, 2) if not pd.isna(es) else "-"
            row[f"{m} Success"] = "YES" if is_success else "NO"

        rows.append(row)

    df_final = pd.DataFrame(rows)
    print("\n=== Study 8 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_08_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()