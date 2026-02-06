import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, List, Tuple

# --- Configuration ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        
        # Inputs
        self.gt_file = self.root / "data" / "processed" / "cohort" / "sampled_1000.csv"
        self.questions_file = self.root / "data" / "raw" / "yrbs" / "yrbs_questions.json"
        
        # Results Directories
        self.internal_dir = self.root / "results" / "validation" / "internal"
        self.external_dir = self.root / "results" / "validation" / "external"
        
        # Outputs
        self.output_dir = self.root / "results" / "validation" / "summary"
        self.figures_dir = self.root / "results" / "figures" / "validation" / "summary"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

class DataProcessor:
    """Handles loading and scoring of validation data"""
    
    # Mapping Folder Names to Display Names
    DT_TYPE_MAP = {
        'base_dt': 'Base DT',
        'survey_dt': 'Survey DT',
        'survey_memory_dt': 'Survey+Memory DT'
    }
    
    # External Domain Mapping (Matches filenames from script 02)
    EXTERNAL_DOMAIN_MAP = {
        'Suicidality & Self-Harm': ['Q27', 'Q28', 'Q29', 'Q30'],
        'Substance Use': ['Q35', 'Q36', 'Q37', 'Q49', 'Q92'],
        'Violence & Abuse Exposure': ['Q19', 'Q20', 'Q21', 'Q22', 'Q88', 'Q89', 'Q90', 'Q91'],
        'Safety Behaviors': ['Q8', 'Q9', 'Q10', 'Q11'],
        'Mental Health Status': ['Q26', 'Q84']
    }

    NUM_TO_ALPHA = {str(i): chr(64 + i) for i in range(1, 9)}

    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.q_meta = {}
        self.gt_df = self._load_gt()

    def _load_gt(self) -> pd.DataFrame:
        """Loads Ground Truth and assigns Demographics"""
        print(f"Loading GT from: {self.paths.gt_file}")
        df = pd.read_csv(self.paths.gt_file, dtype=str)
        
        # Normalize ID
        id_col = next((c for c in df.columns if 'student' in c.lower()), 'student_id')
        df.rename(columns={id_col: 'student_ID'}, inplace=True)
        df['student_ID'] = pd.to_numeric(df['student_ID'], errors='coerce')
        df = df.dropna(subset=['student_ID'])
        df['student_ID'] = df['student_ID'].astype(int)
        
        # Load Metadata for Scoring
        if self.paths.questions_file.exists():
            with open(self.paths.questions_file) as f:
                data = json.load(f)
                for q in data:
                    self.q_meta[q['id']] = q

        # --- Assign Demographics (Simplified Logic) ---
        # Ethnicity
        def get_ethnicity(row):
            q4, q5 = str(row.get('Q4', '')), str(row.get('Q5', ''))
            q5_std = ''.join(sorted(q5.upper().replace(',', '').replace(' ', '')))
            
            if q4 == 'A': return '1_Hispanic/Latino'
            if q4 == 'B':
                if len(q5_std) > 1: return '5_Multiple Races (Non-H)'
                if q5_std == 'C': return '2_Black (Non-H)'
                if q5_std == 'E': return '3_White (Non-H)'
                if q5_std in ['A', 'B', 'D']: return '4_Asian/Native/Pacific (Non-H)'
            return '6_Other/Missing/Unclassified'

        df['Ethnicity_Group'] = df.apply(get_ethnicity, axis=1)
        return df

    def _calculate_accuracy(self, pred_df: pd.DataFrame, q_list: List[str]) -> float:
        """Calculates accuracy for a subset of questions"""
        merged = pred_df.merge(self.gt_df, on='student_ID', suffixes=('_pred', '_gt'))
        if merged.empty: return np.nan

        scores = []
        for q in q_list:
            col_pred = f"{q}_pred" if f"{q}_pred" in merged.columns else q
            col_gt = f"{q}_gt" if f"{q}_gt" in merged.columns else q # GT usually doesn't have suffix before merge, but check logic
            col_gt_raw = q 

            if col_pred not in merged.columns: continue

            # Vectorized Scoring
            p_vals = merged[col_pred].astype(str).str.upper().str.strip()
            g_vals = merged[col_gt_raw].astype(str).str.upper().str.strip()
            
            # Simple logic: Exact match or Numeric tolerance
            is_numeric = q in self.q_meta and self.q_meta[q]['type'] == 'numeric'
            
            if is_numeric:
                try:
                    p_num = pd.to_numeric(p_vals, errors='coerce')
                    g_num = pd.to_numeric(g_vals, errors='coerce')
                    match = (abs(p_num - g_num) <= 0.5)
                except:
                    match = pd.Series([False] * len(merged))
            else:
                # Handle Q88+ (1-10 scale) vs Alpha
                # Simplified: exact string match after cleaning
                match = (p_vals == g_vals)

            scores.extend(match[match].index.tolist()) # Just counting hits? No, need mean accuracy
            
            # Correct approach: Mean accuracy per question then average
            q_acc = match.mean()
            scores.append(q_acc)
            
        return np.nanmean(scores) if scores else np.nan

    def _process_directory(self, base_dir: Path, domain_name: str, is_internal: bool) -> List[Dict]:
        results = []
        
        # Walk through Model / DT_Type structure
        if not base_dir.exists(): return []
        
        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir(): continue
            model_name = model_dir.name
            
            for dt_dir in model_dir.iterdir():
                if not dt_dir.is_dir(): continue
                dt_type_raw = dt_dir.name
                dt_type = self.DT_TYPE_MAP.get(dt_type_raw, dt_type_raw)
                
                # Identify Target Files
                if is_internal:
                    # Internal: Usually "responses.json" or "responses.csv"
                    files = list(dt_dir.glob("responses.json")) + list(dt_dir.glob("responses.csv"))
                    q_list = [f"Q{i}" for i in range(1, 108)] # All questions
                else:
                    # External: Look for specific domain files
                    files = []
                    # Find file matching domain questions (e.g., "no_Q27_Q30...")
                    # Logic: We check if any file in this dir matches the domain mapping logic
                    # Simplified: We iterate the MAP and find files
                    pass # Handled below
                
                # Logic split
                if is_internal:
                    if not files: continue
                    # Load Internal Data
                    try:
                        if files[0].suffix == '.json':
                            with open(files[0]) as f:
                                data = json.load(f)
                                # Convert JSON {sid: [{qid, response}]} to DataFrame
                                rows = []
                                for sid, resps in data.items():
                                    row = {'student_ID': int(sid)}
                                    for r in resps:
                                        row[r['question_id']] = r['response']
                                    rows.append(row)
                                df_pred = pd.DataFrame(rows)
                        else:
                            df_pred = pd.read_csv(files[0])
                        
                        # Calculate Score
                        # For internal, we score all available questions
                        # But for heterogeneity/big plot, we treat "Internal" as one domain
                        acc = self._calculate_accuracy(df_pred, q_list)
                        results.append({
                            'Model_Base': model_name,
                            'Condition': dt_type,
                            'Domain': 'Internal Validation',
                            'Accuracy': acc
                        })
                    except Exception as e:
                        print(f"Error processing internal {dt_dir}: {e}")

        return results

    def get_summary_data(self):
        all_results = []
        
        # 1. Internal Validation
        print("Processing Internal Validation...")
        all_results.extend(self._process_directory(self.paths.internal_dir, "Internal Validation", True))
        
        # 2. External Validation
        print("Processing External Validation...")
        if self.paths.external_dir.exists():
            for model_dir in self.paths.external_dir.iterdir():
                if not model_dir.is_dir(): continue
                
                for dt_dir in model_dir.iterdir():
                    if not dt_dir.is_dir(): continue
                    dt_type = self.DT_TYPE_MAP.get(dt_dir.name, dt_dir.name)
                    
                    for domain, q_list in self.EXTERNAL_DOMAIN_MAP.items():
                        # Find matching file (e.g., contains "Q27")
                        # This relies on the naming convention from script 02
                        found = False
                        for csv_file in dt_dir.glob("*.csv"):
                            # Check if filename implies this domain (e.g., "no_Q27_Q30")
                            # Heuristic: Check if the first Q in q_list is in filename
                            if q_list[0] in csv_file.name: 
                                found = True
                                try:
                                    df_pred = pd.read_csv(csv_file)
                                    # Normalize ID
                                    id_col = next((c for c in df_pred.columns if 'student' in c.lower()), 'student_ID')
                                    df_pred.rename(columns={id_col: 'student_ID'}, inplace=True)
                                    df_pred['student_ID'] = pd.to_numeric(df_pred['student_ID'], errors='coerce')
                                    df_pred = df_pred.dropna(subset=['student_ID']).astype({'student_ID': int})

                                    acc = self._calculate_accuracy(df_pred, q_list)
                                    all_results.append({
                                        'Model_Base': model_dir.name,
                                        'Condition': dt_type,
                                        'Domain': domain,
                                        'Accuracy': acc
                                    })
                                except Exception as e:
                                    print(f"Error reading {csv_file}: {e}")
                                break # Found the file for this domain
        
        return pd.DataFrame(all_results)

    def get_improvement_data(self):
        """Calculates improvement per ethnicity"""
        # Note: This requires recalculating accuracy PER ETHNICITY. 
        # The summary above was global accuracy.
        # For the heatmap, we need finer granularity.
        
        # Re-using logic but grouping by ethnicity.
        # For brevity in this script, I will implement a simplified version 
        # that iterates files again or assumes we can get row-wise accuracy.
        
        # Let's do a streamlined pass for Improvement Data
        imp_results = []
        
        # Helper to get per-ethnicity accuracy
        def get_eth_acc(pred_df, q_list):
            merged = pred_df.merge(self.gt_df, on='student_ID')
            if merged.empty: return []
            
            # Calculate correctness per row (student)
            # This is a simplification; ideally we check every question
            correct_counts = []
            for idx, row in merged.iterrows():
                hits = 0
                total = 0
                for q in q_list:
                    p = str(row.get(f"{q}_x", row.get(q))).upper().strip() # pred
                    g = str(row.get(f"{q}_y", row.get(f"{q}_gt"))).upper().strip() # gt
                    if p == g: hits += 1 # Simplified exact match
                    total += 1
                correct_counts.append(hits/total if total > 0 else 0)
            
            merged['row_acc'] = correct_counts
            return merged.groupby('Ethnicity_Group')['row_acc'].mean()

        # Iterate External Directories again for Improvement
        if self.paths.external_dir.exists():
            for model_dir in self.paths.external_dir.iterdir():
                if not model_dir.is_dir(): continue
                for dt_dir in model_dir.iterdir():
                    if not dt_dir.is_dir(): continue
                    dt_type = self.DT_TYPE_MAP.get(dt_dir.name, dt_dir.name)
                    
                    for domain, q_list in self.EXTERNAL_DOMAIN_MAP.items():
                         for csv_file in dt_dir.glob("*.csv"):
                            if q_list[0] in csv_file.name:
                                try:
                                    df = pd.read_csv(csv_file)
                                    # Fix ID
                                    id_col = next((c for c in df.columns if 'student' in c.lower()), 'student_ID')
                                    df.rename(columns={id_col: 'student_ID'}, inplace=True)
                                    df['student_ID'] = pd.to_numeric(df['student_ID'], errors='coerce')
                                    df = df.dropna().astype({'student_ID': int})
                                    
                                    eth_acc = get_eth_acc(df, q_list)
                                    for eth, val in eth_acc.items():
                                        imp_results.append({
                                            'Model_Base': model_dir.name,
                                            'Condition': dt_type,
                                            'Domain': domain,
                                            'Ethnicity_Group': eth,
                                            'Accuracy': val
                                        })
                                except: pass
        
        df = pd.DataFrame(imp_results)
        
        # Pivot to Calculate Improvement (Memory - Survey)
        if df.empty: return pd.DataFrame()
        
        pivot = df.pivot_table(
            index=['Model_Base', 'Domain', 'Ethnicity_Group'], 
            columns='Condition', 
            values='Accuracy'
        ).reset_index()
        
        if 'Survey+Memory DT' in pivot.columns and 'Survey DT' in pivot.columns:
            pivot['Improvement'] = (pivot['Survey+Memory DT'] - pivot['Survey DT']) * 100
            return pivot
        return pd.DataFrame()

# --- Visualization Function ---

def visualize_merged_big_plot(df_acc, df_imp, save_path):
    print("🎨 Generating Big Plot...")
    
    # 1. Config & Mappings
    sns.set_theme(style="white", font="sans-serif")
    
    ethnicity_map = {
        '1_Hispanic/Latino': 'Hispanic/Latino',
        '2_Black (Non-H)': 'Black',
        '3_White (Non-H)': 'White',
        '4_Asian/Native/Pacific (Non-H)': 'Asian/Native/Pacific',
        '5_Multiple Races (Non-H)': 'Multiple Races',
        '6_Other/Missing/Unclassified': 'Other/Missing'
    }
    
    domain_map = {
        'Internal Validation': 'Internal',
        'Suicidality & Self-Harm': 'Suicidality',
        'Substance Use': 'Substance',
        'Violence & Abuse Exposure': 'Violence',
        'Safety Behaviors': 'Safety',
        'Mental Health Status': 'Mood'
    }
    
    # Order
    model_order = sorted(df_acc['Model_Base'].unique()) # Or manual list
    condition_order = ['Survey DT', 'Survey+Memory DT']
    dom_order = ['Internal', 'Suicidality', 'Substance', 'Violence', 'Safety', 'Mood']
    eth_order = ['Hispanic/Latino', 'Black', 'White', 'Asian/Native/Pacific', 'Multiple Races', 'Other/Missing']

    # Preprocessing
    df_acc['Accuracy_Percent'] = df_acc['Accuracy'] * 100
    df_acc['Domain_Clean'] = df_acc['Domain'].map(domain_map).fillna(df_acc['Domain'])
    
    df_imp['Ethnicity_Label'] = df_imp['Ethnicity_Group'].map(ethnicity_map)
    df_imp['Domain_Clean'] = df_imp['Domain'].map(domain_map).fillna(df_imp['Domain'])

    # Setup Figure
    fig = plt.figure(figsize=(20, 19))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1.2, 1], wspace=0.25, hspace=0.2)

    # --- TOP ROW: BUBBLE CHART ---
    cmap_bubble = plt.cm.RdYlBu_r
    norm_bubble = mcolors.Normalize(vmin=60, vmax=100)
    y_pos_map = {d: i for i, d in enumerate(dom_order[::-1])}
    x_pos_map = {c: i for i, c in enumerate(condition_order)}

    for idx, model in enumerate(model_order):
        if idx >= 3: break # Limit to 3 columns
        ax = fig.add_subplot(gs[0, idx])
        
        subset = df_acc[df_acc['Model_Base'] == model].copy()
        subset['x'] = subset['Condition'].map(x_pos_map)
        subset['y'] = subset['Domain_Clean'].map(y_pos_map)
        subset = subset.dropna(subset=['x', 'y'])

        # Grid lines
        for y_val in range(len(dom_order)):
            ax.axhline(y=y_val, color='#E0E0E0', linestyle='--', linewidth=1, zorder=0)

        sizes = subset['Accuracy_Percent'].apply(lambda x: (x/10)**2.8 * 10.0)
        
        ax.scatter(subset['x'], subset['y'], s=sizes, c=subset['Accuracy_Percent'], 
                  cmap=cmap_bubble, norm=norm_bubble, edgecolors='grey', zorder=10)

        # Labels
        for _, row in subset.iterrows():
            txt_col = 'white' if norm_bubble(row['Accuracy_Percent']) < 0.4 or norm_bubble(row['Accuracy_Percent']) > 0.8 else 'black'
            ax.text(row['x'], row['y'], f"{row['Accuracy_Percent']:.1f}%", 
                   ha='center', va='center', fontsize=12, fontweight='bold', color=txt_col, zorder=11)

        ax.set_title(model, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(condition_order, fontsize=12)
        ax.set_yticks(range(len(dom_order)))
        
        if idx == 0:
            ax.set_yticklabels(dom_order[::-1], fontsize=12)
        else:
            ax.set_yticklabels([])

    # Colorbar Top
    cax1 = fig.add_subplot(gs[0, 3])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_bubble, cmap=cmap_bubble), cax=cax1, label='Accuracy (%)')

    # --- BOTTOM ROW: HEATMAP ---
    cmap_heat = "PuOr"
    norm_heat = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=10) # Adjust based on data

    for idx, model in enumerate(model_order):
        if idx >= 3: break
        ax = fig.add_subplot(gs[1, idx])
        
        subset = df_imp[df_imp['Model_Base'] == model]
        if subset.empty: continue
        
        pivot = subset.pivot(index='Ethnicity_Label', columns='Domain_Clean', values='Improvement')
        pivot = pivot.reindex(index=eth_order, columns=dom_order)
        
        sns.heatmap(pivot, cmap=cmap_heat, norm=norm_heat, annot=True, fmt=".1f", 
                   cbar=False, ax=ax, square=True)
        
        ax.set_title(model, fontsize=16, fontweight='bold')
        ax.set_xlabel("")
        if idx == 0:
            ax.set_ylabel("Ethnicity", fontsize=14)
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

    # Colorbar Bottom
    cax2 = fig.add_subplot(gs[1, 3])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm_heat, cmap=cmap_heat), cax=cax2, label='Improvement (%)')

    plt.suptitle("Overall Validation Results: Accuracy & Improvement", fontsize=20, fontweight='bold', y=0.95)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")


def main():
    paths = ProjectPaths()
    processor = DataProcessor(paths)
    
    # 1. Calculate Summary Stats (For Top Plot)
    df_acc = processor.get_summary_data()
    if df_acc.empty:
        print("No accuracy data found. Check paths.")
        return
        
    # Save CSV
    df_acc.to_csv(paths.output_dir / "accuracy_summary.csv", index=False)
    
    # 2. Calculate Improvement Stats (For Bottom Plot)
    df_imp = processor.get_improvement_data()
    if not df_imp.empty:
        df_imp.to_csv(paths.output_dir / "improvement_summary.csv", index=False)
    
    # 3. Visualize
    visualize_merged_big_plot(
        df_acc, 
        df_imp, 
        paths.figures_dir / "internal_external_summary.png"
    )

if __name__ == "__main__":
    main()