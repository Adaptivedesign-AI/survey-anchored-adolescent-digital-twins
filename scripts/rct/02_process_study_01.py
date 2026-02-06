import os
import sys
import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from pathlib import Path

# ========================
# 0. Paths and Configuration
# ========================

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.results_base = self.root / "results" / "rct_replication"
        self.study_dir = self.results_base / "study_01"
        self.study_dir.mkdir(parents=True, exist_ok=True)

paths = ProjectPaths()

# Define sub-directories for different model configurations
# You can adjust these folder names based on how you ran script 01
PATH_CONFIGS = {
    "Memory-DT": paths.results_base / "gemini-2.0-flash" / "survey_memory_dt" / "1",
    "Base-DT": paths.results_base / "gemini-2.0-flash" / "survey_dt" / "1", 
    "LLM-only": paths.results_base / "gemini-2.0-flash" / "base_dt" / "1"
}

MODEL_ORDER = ["Memory-DT", "Base-DT", "LLM-only"]

# Columns required for analysis (from Study 1 schema)
KEEP_COLS = [
    "student_id",
    "item_Belief_Q1", "item_Belief_Q2", "item_Belief_Q3",
    "item_Perception_Q12"
]

# ========================
# 1. Data Loading Helper (JSON -> DF)
# ========================

def load_json_to_df(file_path: Path) -> pd.DataFrame:
    """
    Reads the raw JSON output from 01_run_all_simulations.py and converts 
    the 'json_items' field into a flat DataFrame.
    """
    if not file_path.exists():
        return pd.DataFrame()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rows = []
        for entry in data:
            # Extract basic info
            row = {'student_id': entry.get('student_id')}
            
            # Extract parsed questionnaire items
            items = entry.get('json_items', {})
            if items:
                row.update(items)
            
            rows.append(row)
            
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# ========================
# 2. Core Statistical Functions
# ========================

def r_to_d(r):
    """Convert correlation r to Cohen's d."""
    if pd.isna(r):
        return 0.0
    r = np.clip(r, -0.99, 0.99)
    d = (2 * r) / np.sqrt(1 - r**2)
    return d

def fisher_z(r):
    """Fisher z-transform of correlation r."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def fisher_r_ci(r, n, alpha=0.05):
    """Compute 95% CI for r using Fisher's z."""
    if n is None or n <= 3 or pd.isna(r):
        return (np.nan, np.nan)
    z = fisher_z(r)
    z_crit = 1.96 
    se_z = 1.0 / np.sqrt(n - 3)
    z_low = z - z_crit * se_z
    z_high = z + z_crit * se_z
    r_low = (np.exp(2 * z_low) - 1) / (np.exp(2 * z_low) + 1)
    r_high = (np.exp(2 * z_high) - 1) / (np.exp(2 * z_high) + 1)
    return (r_low, r_high)

def format_ci(low, high):
    """Format confidence interval as string."""
    if pd.isna(low) or pd.isna(high):
        return "[-, -]"
    return f"[{low:.2f}, {high:.2f}]"

def get_direction(beta):
    """Determine effect direction based on beta."""
    if abs(beta) < 0.001:
        return "Null"
    return "Positive" if beta > 0 else "Negative"

def check_sig_match(orig_p_category, curr_p_val):
    """Check whether significance matches the original result."""
    is_curr_sig = curr_p_val < 0.05
    if orig_p_category == "sig":
        return is_curr_sig
    else:
        return not is_curr_sig

# ========================
# 3. Original Study Specifications
# ========================
# Benchmarks from PNAS Study 1
ORIGINAL_SPECS = {
    "A": {
        "label": "Growth",
        "category": "main",
        "N": 570,
        "beta": 0.11,
        "ci_beta": [0.06, 0.17],
        "p_str": "< .01",
        "p_cat": "sig",
        "dir": "Positive",
    },
    "B": {
        "label": "Fixed",
        "category": "main",
        "N": 524,
        "beta": -0.02,
        "ci_beta": [-0.07, 0.03],
        "p_str": "n.s.",
        "p_cat": "ns",
        "dir": "Negative",
    },
    "C": {
        "label": "Interaction",
        "category": "interaction",
        "N": 1094, 
        "beta": 0.13, 
        "ci_beta": [0.02, 0.24],
        "p_str": "< .01",
        "p_cat": "sig",
        "dir": "Positive",
    },
}

# Pre-compute original r, z, d
for k, v in ORIGINAL_SPECS.items():
    r = v["beta"]
    v["r"] = r
    v["z"] = fisher_z(r)
    r_low, r_high = v["ci_beta"]
    v["r_ci"] = (r_low, r_high)
    v["d"] = r_to_d(r)
    v["ci_d"] = [r_to_d(r_low), r_to_d(r_high)]

# ========================
# 4. Analysis Logic
# ========================

def force_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def clean_data(df):
    needed = ["item_Belief_Q1", "item_Belief_Q2", "item_Belief_Q3", "item_Perception_Q12"]
    if df.empty or not all(col in df.columns for col in needed):
        return pd.DataFrame()
    df = force_numeric(df, needed)
    return df.dropna(subset=needed).copy()

def get_stats_boot(df, n_boot=1000):
    """Compute statistics with bootstrapping."""
    if len(df) < 10:
        return None

    # Reverse-score and compute GM index (Growth Mindset)
    df = df.copy()
    # Assuming 1-6 scale? Or 1-7? Standard is usually 6 or 7.
    # The original script used 7 - value. Adjust if your scale is different.
    df["Q1_r"] = 7 - df["item_Belief_Q1"]
    df["Q3_r"] = 7 - df["item_Belief_Q3"]
    df["GM"] = df[["Q1_r", "item_Belief_Q2", "Q3_r"]].mean(axis=1)
    Y = df["item_Perception_Q12"]

    N = len(df)
    sd_gm = df["GM"].std()
    sd_choice = Y.std()

    if sd_gm == 0 or sd_choice == 0:
        return {
            "N": N, "beta": 0.0, "r": 0.0, "z": np.nan,
            "r_ci": (np.nan, np.nan), "d": 0.0, "d_ci": [0.0, 0.0],
            "p": 1.0, "boots_d": np.zeros(n_boot), "boots_r": np.zeros(n_boot),
        }

    # Standardize
    gm_z = (df["GM"] - df["GM"].mean()) / sd_gm
    y_z = (Y - Y.mean()) / sd_choice

    X = gm_z.values.reshape(-1, 1)
    y = y_z.values

    # OLS regression
    try:
        reg = LinearRegression().fit(X, y)
        beta = reg.coef_[0]
    except Exception:
        beta = 0.0

    # Correlation
    r = float(np.corrcoef(gm_z, y_z)[0, 1])
    z = fisher_z(r)
    r_low, r_high = fisher_r_ci(r, N)

    # Bootstrap
    rng = np.random.default_rng(42)
    boot_ds = []
    boot_rs = []

    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        x_b = X[idx]
        y_b = y[idx]

        if np.std(x_b) == 0 or np.std(y_b) == 0:
            b = 0.0
        else:
            try:
                b = LinearRegression().fit(x_b, y_b).coef_[0]
            except Exception:
                b = 0.0

        boot_rs.append(b)
        boot_ds.append(r_to_d(b))

    boot_rs = np.array(boot_rs)
    boot_ds = np.array(boot_ds)

    if beta > 0:
        p = 2 * (boot_rs < 0).mean()
    else:
        p = 2 * (boot_rs > 0).mean()
    if p == 0: p = 1.0 / n_boot

    d_ci = np.percentile(boot_ds, [2.5, 97.5])

    return {
        "N": N, "beta": beta, "r": r, "z": z,
        "r_ci": (r_low, r_high), "d": r_to_d(r), "d_ci": d_ci,
        "p": p, "boots_d": boot_ds, "boots_r": boot_rs,
    }

def get_interaction_p(df_c, df_i):
    """Compute interaction p-value."""
    try:
        df_c = df_c.copy(); df_c["Cond"] = 0
        df_i = df_i.copy(); df_i["Cond"] = 1
        df = pd.concat([df_c, df_i])

        df["GM_raw"] = ((7 - df["item_Belief_Q1"]) + df["item_Belief_Q2"] + (7 - df["item_Belief_Q3"])) / 3
        df["GM_z"] = (df["GM_raw"] - df["GM_raw"].mean()) / df["GM_raw"].std()
        df["Y_z"] = (df["item_Perception_Q12"] - df["item_Perception_Q12"].mean()) / df["item_Perception_Q12"].std()

        model = smf.ols("Y_z ~ GM_z * Cond", data=df).fit()
        return model.pvalues.get("GM_z:Cond", 1.0)
    except Exception:
        return 1.0

# ========================
# 5. Main Execution
# ========================

def main():
    results_map = {}

    print(f"Processing Study 1 Results...")
    
    for m_name in MODEL_ORDER:
        folder = PATH_CONFIGS.get(m_name)
        if not folder or not folder.exists():
            print(f"  Skipping {m_name}: Path not found ({folder})")
            results_map[m_name] = None
            continue

        print(f"  Analyzing {m_name}...")
        
        # Load JSONs and convert to DF
        # Assuming file naming convention from script 01: 1_control_raw.json / 1_intervention_raw.json
        file_c = folder / "1_control_raw.json"
        file_i = folder / "1_intervention_raw.json"
        
        df_c = clean_data(load_json_to_df(file_c))
        df_i = clean_data(load_json_to_df(file_i))

        if len(df_c) < 10 or len(df_i) < 10:
            print(f"    Insufficient data (N_c={len(df_c)}, N_i={len(df_i)})")
            results_map[m_name] = None
            continue

        # Stats
        res_c = get_stats_boot(df_c) # Fixed
        res_i = get_stats_boot(df_i) # Growth

        if res_c and res_i:
            int_beta = res_i["beta"] - res_c["beta"]
            int_r = res_i["r"] - res_c["r"]
            int_d = res_i["d"] - res_c["d"]
            int_p = get_interaction_p(df_c, df_i)

            boot_diff_d = res_i["boots_d"] - res_c["boots_d"]
            d_ci_int = np.percentile(boot_diff_d, [2.5, 97.5])
            boot_diff_r = res_i["boots_r"] - res_c["boots_r"]
            r_ci_int = np.percentile(boot_diff_r, [2.5, 97.5])

            res_int = {
                "N": res_c["N"] + res_i["N"],
                "beta": int_beta, "r": int_r, "z": fisher_z(int_r),
                "r_ci": (r_ci_int[0], r_ci_int[1]), "d": int_d, "d_ci": d_ci_int,
                "p": int_p
            }

            results_map[m_name] = {"A": res_i, "B": res_c, "C": res_int}
        else:
            results_map[m_name] = None

    # Build Table
    final_rows = []
    for key in ["A", "B", "C"]:
        spec = ORIGINAL_SPECS[key]
        row = {
            "Effect": spec["label"],
            "Category": spec["category"],
            "Orig_N": spec["N"],
            "Orig_Dir": spec["dir"],
            "Orig_p": spec["p_str"],
            "Orig_r": round(spec["r"], 2),
            "Orig_r_CI": format_ci(spec["r_ci"][0], spec["r_ci"][1]),
            "Orig_d": round(spec["d"], 2)
        }

        for m_name in MODEL_ORDER:
            res = results_map.get(m_name)
            prefix = f"{m_name}_"
            
            if res is None:
                for k in ["N", "r", "p", "Sig_Match", "Success"]:
                    row[prefix + k] = "-"
                continue

            curr_stats = res[key]
            curr_dir = get_direction(curr_stats["beta"])
            is_dir_match = (curr_dir == spec["dir"])
            is_sig_match = check_sig_match(spec["p_cat"], curr_stats["p"])
            
            # For null effects (Fixed condition), success is matching non-significance
            if spec["p_cat"] == "ns":
                is_success = is_sig_match
            else:
                is_success = (is_dir_match and is_sig_match)

            row[prefix + "N"] = curr_stats["N"]
            row[prefix + "r"] = f"{curr_stats['r']:.2f}"
            row[prefix + "p"] = f"{curr_stats['p']:.3f}"
            row[prefix + "Sig_Match"] = "YES" if is_sig_match else "NO"
            row[prefix + "Success"] = "✅" if is_success else "❌"

        final_rows.append(row)

    # Save
    df_final = pd.DataFrame(final_rows)
    print("\nSummary Table:")
    print(df_final.to_string())
    
    out_file = paths.study_dir / "study_01_analysis_summary.csv"
    df_final.to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == "__main__":
    main()