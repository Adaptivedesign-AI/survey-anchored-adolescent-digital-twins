import os
import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path

# ============================================================
# 0. Paths and Configuration
# ============================================================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.cohort_data = self.root / "data" / "processed" / "cohort" / "sampled_1000.csv"
        self.output_dir = self.results_base / "study_09"
        self.output_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define model paths
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "9",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "9",
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "9"
}

# Filenames
FILES = {
    "Control": "9_control_raw.json",
    "Intervention": "9_intervention_raw.json"
}

# ============================================================
# 1. Scale Definitions
# ============================================================
RFLA = list(range(1,33))
GHSQ = list(range(1,11))
CCSS = list(range(1,21))
ASS  = list(range(1,9))

# CCSS categories
CCSS_PRO  = [2,3,4,7,8,9,14,16,17]
CCSS_ANTI = [1,5,6,10,11,12,13,15,20]

# ASS mood
ASS_POS = [1,2,3,4]
ASS_NEG = [5,6,7,8]

# Outcome specs: (post, pre, higher_is_better)
OUTCOME_SPECS = {
    "RFLA_total":       ("RFLA_post", "RFLA_pre", True),
    "GHSQ_overall":     ("GHSQ_overall_post","GHSQ_overall_pre",True),
    "GHSQ_private":     ("GHSQ_private_post","GHSQ_private_pre",True),
    "GHSQ_professional":("GHSQ_prof_post","GHSQ_prof_pre",True),
    "CCSS_tolerance":   ("CCSS_tol_post","CCSS_tol_pre",False),
    "Mood_index":       ("Mood_post","Mood_pre",True),
}

# Original effects (for comparison)
ORIGINAL = {
    "RFLA_total":       ("Improved", "<.01", 0.03),
    "GHSQ_overall":     ("Improved", "<.001", 0.05),
    "GHSQ_private":     ("Improved", "<.01", 0.03),
    "GHSQ_professional":("Improved", ".01", 0.02),
    "CCSS_tolerance":   ("Improved", ".04", 0.01),
    "Mood_index":       ("Worse", ".02", 0.02),
}

# ============================================================
# 2. Data Loading & Feature Engineering
# ============================================================

def load_demographics(file_path: Path):
    if not file_path.exists():
        return pd.Series(dtype=int)
    
    demo = pd.read_csv(file_path, dtype=str)
    # Clean ID
    id_col = next((c for c in demo.columns if 'student' in c.lower()), 'student_id')
    demo.rename(columns={id_col: 'student_id'}, inplace=True)
    demo['student_id'] = pd.to_numeric(demo['student_id'], errors='coerce')
    
    # Q2: 1=Female, 2=Male. Map to 0/1 for regression
    # 1->0 (Female), 2->1 (Male)
    if 'Q2' in demo.columns:
        demo['gender'] = pd.to_numeric(demo['Q2'], errors='coerce').map({1: 0, 2: 1})
        return demo.set_index('student_id')['gender']
    return pd.Series(dtype=int)

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

def col(scale, ph, q):
    return f"item_{scale}_{ph}_Q{q}"

def reverse_series(x, lo, hi):
    return lo + hi - x

def load_raw(folder, demo_gender):
    f_ctl = folder / FILES["Control"]
    f_int = folder / FILES["Intervention"]
    
    df_c = load_json_to_df(f_ctl)
    df_i = load_json_to_df(f_int)
    
    if df_c.empty or df_i.empty: return None
    
    df_c["group"] = 0
    df_i["group"] = 1
    
    df = pd.concat([df_c, df_i], ignore_index=True)
    
    # Clean IDs for merge
    df['student_id'] = pd.to_numeric(df['student_id'], errors='coerce')
    df = df.dropna(subset=['student_id'])
    
    # Merge Gender
    df["gender"] = df["student_id"].map(demo_gender)
    
    return df

def score(df):
    # Convert numeric item columns
    for c in df.columns:
        if c.startswith("item_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------- RFLA ----------
    # Need to handle potential missing columns gracefully
    def mean_cols(prefix, indices):
        cols = [col(prefix[0], prefix[1], q) for q in indices]
        valid = [c for c in cols if c in df.columns]
        if not valid: return pd.Series(np.nan, index=df.index)
        return df[valid].mean(axis=1)

    df["RFLA_pre"]  = mean_cols(("RFLA","PRE"), RFLA)
    df["RFLA_post"] = mean_cols(("RFLA","POST"), RFLA)

    # ---------- GHSQ ----------
    df["GHSQ_overall_pre"]  = mean_cols(("GHSQ","PRE"), range(1,10))
    df["GHSQ_overall_post"] = mean_cols(("GHSQ","POST"), range(1,10))

    df["GHSQ_private_pre"]  = mean_cols(("GHSQ","PRE"), [1,2,3,7,8])
    df["GHSQ_private_post"] = mean_cols(("GHSQ","POST"), [1,2,3,7,8])

    df["GHSQ_prof_pre"]  = mean_cols(("GHSQ","PRE"), [4,5,6,9])
    df["GHSQ_prof_post"] = mean_cols(("GHSQ","POST"), [4,5,6,9])

    # ---------- CCSS ----------
    def mean_cols_list(prefix, q_list):
        cols = [col(prefix[0], prefix[1], q) for q in q_list]
        valid = [c for c in cols if c in df.columns]
        if not valid: return pd.Series(np.nan, index=df.index)
        return df[valid].mean(axis=1)

    pro_pre  = mean_cols_list(("CCSS","PRE"), CCSS_PRO)
    anti_pre = mean_cols_list(("CCSS","PRE"), CCSS_ANTI)
    anti_pre_r  = reverse_series(anti_pre,1,6)

    pro_post = mean_cols_list(("CCSS","POST"), CCSS_PRO)
    anti_post= mean_cols_list(("CCSS","POST"), CCSS_ANTI)
    anti_post_r = reverse_series(anti_post,1,6)

    df["CCSS_tol_pre"]  = pd.concat([pro_pre, anti_pre_r], axis=1).mean(axis=1)
    df["CCSS_tol_post"] = pd.concat([pro_post, anti_post_r], axis=1).mean(axis=1)

    # ---------- Mood ----------
    pos_pre  = mean_cols_list(("ASSmood","PRE"), ASS_POS)
    neg_pre  = mean_cols_list(("ASSmood","PRE"), ASS_NEG)
    neg_pre_r = reverse_series(neg_pre,1,4)

    pos_post = mean_cols_list(("ASSmood","POST"), ASS_POS)
    neg_post = mean_cols_list(("ASSmood","POST"), ASS_NEG)
    neg_post_r= reverse_series(neg_post,1,4)

    df["Mood_pre"]  = pd.concat([pos_pre, neg_pre_r], axis=1).mean(axis=1)
    df["Mood_post"] = pd.concat([pos_post, neg_post_r], axis=1).mean(axis=1)

    return df

# ============================================================
# 3. Analysis Functions
# ============================================================

def ancova(df, post, pre):
    # Ensure gender is present or drop it if totally missing
    cols = [post, pre, "group", "gender"]
    d = df[cols].copy()
    
    # If gender is all NaN (e.g. LLM-only without persona), drop gender from model
    has_gender = d["gender"].notna().sum() > 5
    
    if has_gender:
        d = d.dropna()
        f = f"{post} ~ group + {pre} + gender"
    else:
        d = d.drop(columns=["gender"]).dropna()
        f = f"{post} ~ group + {pre}"

    if d["group"].nunique()<2:
        return np.nan, "N/A", None
    
    try:
        m = smf.ols(f, data=d).fit()
        return m.pvalues["group"], m.params["group"], m
    except:
        return np.nan, "N/A", None

def compute_direction(coef, hib):
    if isinstance(coef, str) or pd.isna(coef):
        return "N/A"
    if hib:
        return "Improved" if coef>0 else "Worse"
    else:
        return "Improved" if coef<0 else "Worse"

def cohens_d(df, post, pre):
    cols = [post, pre, "group", "gender"]
    d = df[cols].copy()
    has_gender = d["gender"].notna().sum() > 5
    
    if has_gender:
        d = d.dropna()
        f = f"{post} ~ {pre} + gender"
    else:
        d = d.drop(columns=["gender"]).dropna()
        f = f"{post} ~ {pre}"

    if d["group"].nunique()<2:
        return np.nan
    
    try:
        base = smf.ols(f, data=d).fit()
        r = base.resid
        g0 = r[d["group"]==0]
        g1 = r[d["group"]==1]
        if len(g0)<2 or len(g1)<2: return np.nan
        pooled = np.sqrt(((g0.var(ddof=1)*(len(g0)-1)) + (g1.var(ddof=1)*(len(g1)-1))) / (len(g0)+len(g1)-2))
        if pooled<=0: return np.nan
        return (g1.mean() - g0.mean()) / pooled
    except:
        return np.nan

def classify(rep_dir, orig_dir, p, orig_p):
    if pd.isna(p): return "Fail"
    
    # Original Sig check
    try:
        if "<" in orig_p: orig_sig = True
        else: orig_sig = (float(orig_p) <= 0.05)
    except: orig_sig = False
    
    ok_dir = (rep_dir == orig_dir)
    ok_sig = (p < 0.05)
    
    if orig_sig:
        return "Success" if ok_dir and ok_sig else "Fail"
    else:
        return "Success" if not ok_sig else "Fail"

# ============================================================
# 4. Main Loop
# ============================================================

def main():
    demo_gender = load_demographics(paths.cohort_data)
    
    all_rows = []
    
    print("Processing Study 9...")
    
    for gname, folder in PATH_CONFIGS.items():
        if not folder.exists():
            continue
            
        print(f"  Analyzing {gname}...")
        df = load_raw(folder, demo_gender)
        if df is None: continue
        
        df = score(df)

        for outcome, (post, pre, hib) in OUTCOME_SPECS.items():
            d = df[[post, pre, "group"]].copy() # gender optional
            d_main = d.dropna()

            if d_main.empty or d_main["group"].nunique()<2 or len(d_main)<10:
                all_rows.append({
                    "Effect": outcome, "Group": gname,
                    "Direction": "N/A", "p": np.nan, "Effect Size": np.nan
                })
                continue

            p, coef, model = ancova(df, post, pre)
            direction = compute_direction(coef, hib)
            dval = cohens_d(df, post, pre)

            all_rows.append({
                "Effect": outcome,
                "Group": gname,
                "Direction": direction,
                "p": p,
                "Effect Size": dval
            })

    # Convert to wide format
    rep = pd.DataFrame(all_rows)
    final = []
    
    for outcome in OUTCOME_SPECS.keys():
        row = {
            "Effect Type": "Main Effect",
            "Effect Description": outcome,
            "Original Direction": ORIGINAL[outcome][0],
            "Original p": ORIGINAL[outcome][1],
            "Original Effect Size": ORIGINAL[outcome][2],
        }
        for g in ["Memory-DT","Base-DT","LLM-only"]:
            if rep.empty:
                sub = pd.DataFrame()
            else:
                sub = rep[(rep["Effect"]==outcome)&(rep["Group"]==g)]
            
            if len(sub)==0:
                row[f"{g} Direction"] = "N/A"
                row[f"{g} p"] = np.nan
                row[f"{g} Effect Size"] = np.nan
                row[f"{g} Success"] = "Fail"
            else:
                s=sub.iloc[0]
                row[f"{g} Direction"] = s["Direction"]
                row[f"{g} p"] = s["p"]
                row[f"{g} Effect Size"] = s["Effect Size"]
                row[f"{g} Success"] = classify(s["Direction"], ORIGINAL[outcome][0], s["p"], ORIGINAL[outcome][1])
        final.append(row)

    df_final = pd.DataFrame(final)
    print("\n=== Study 9 Full Replication Report ===")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    print(df_final)

    out_file = paths.output_dir / "study_09_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()