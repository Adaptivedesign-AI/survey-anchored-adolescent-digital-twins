import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional

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
        self.external_results_root = self.root / "results" / "validation" / "external"
        
        # Outputs
        self.output_dir = self.root / "results" / "validation" / "heterogeneity"
        self.figures_dir = self.root / "results" / "figures" / "validation" / "heterogeneity"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

class HeterogeneityAnalysis:
    
    # 映射逻辑：将代码里的 Key 映射到实际文件系统的 (Model, DT_Type)
    MODEL_MAPPING = {
        'gemini-2.0-flash-baseline': ('gemini-2.0-flash', 'survey_dt'),
        'gemini-2.0-flash-with-rag': ('gemini-2.0-flash', 'survey_memory_dt'),
        'gemini-2.5-flash-baseline': ('gemini-2.5-flash', 'survey_dt'),
        'gemini-2.5-flash-with-rag': ('gemini-2.5-flash', 'survey_memory_dt'),
        'gemini-2.5-flash-lite-baseline': ('gemini-2.5-flash-lite', 'survey_dt'),
        'gemini-2.5-flash-lite-with-rag': ('gemini-2.5-flash-lite', 'survey_memory_dt'),
    }
    
    # 显示用的标签信息
    MODEL_INFO = {
        'gemini-2.0-flash-baseline': {'Model_Base': 'gemini-2.0-flash', 'Condition': 'Survey DT'},
        'gemini-2.0-flash-with-rag': {'Model_Base': 'gemini-2.0-flash', 'Condition': 'Survey+Memory DT'},
        'gemini-2.5-flash-baseline': {'Model_Base': 'gemini-2.5-flash', 'Condition': 'Survey DT'},
        'gemini-2.5-flash-with-rag': {'Model_Base': 'gemini-2.5-flash', 'Condition': 'Survey+Memory DT'},
        'gemini-2.5-flash-lite-baseline': {'Model_Base': 'gemini-2.5-lite-flash', 'Condition': 'Survey DT'},
        'gemini-2.5-flash-lite-with-rag': {'Model_Base': 'gemini-2.5-lite-flash', 'Condition': 'Survey+Memory DT'},
    }

    EXTERNAL_DOMAIN_MAP = {
        'Suicidality & Self-Harm': {'file_pattern': 'no_Q27_Q30', 'questions': ['Q27', 'Q28', 'Q29', 'Q30']},
        'Substance Use': {'file_pattern': 'no_Q35_Q37_Q49_Q92', 'questions': ['Q35', 'Q36', 'Q37', 'Q49', 'Q92']},
        'Violence & Abuse Exposure': {'file_pattern': 'no_Q19_Q22_and_Q88_Q91', 'questions': ['Q19', 'Q20', 'Q21', 'Q22', 'Q88', 'Q89', 'Q90', 'Q91']},
        'Safety Behaviors': {'file_pattern': 'no_Q8_Q11', 'questions': ['Q8', 'Q9', 'Q10', 'Q11']},
        'Mental Health Status': {'file_pattern': 'no_Q26_Q84', 'questions': ['Q26', 'Q84']}
    }

    NUM_TO_ALPHA = {str(i): chr(64 + i) for i in range(1, 9)}
    SINGLE_CHOICE_ALPHA_COLS = [f"Q{i}" for i in list(range(1, 5)) + list(range(8, 88))]
    QUESTION_COLS = [f"Q{i}" for i in range(1, 108)]

    def __init__(self):
        self.paths = ProjectPaths()
        self.gt_df = None
        self.q_meta = {}

    # ==========================
    # 1. Data Preparation
    # ==========================
    def load_and_prepare_gt(self):
        """Loads and cleans Ground Truth data, assigning Demographics"""
        print(f"Loading GT from: {self.paths.gt_file}")
        df_gt = pd.read_csv(self.paths.gt_file, dtype=str)
        
        # ID Handling
        id_col = next((col for col in df_gt.columns if 'student_id' in col.lower()), None)
        if id_col: df_gt.rename(columns={id_col: 'student_ID'}, inplace=True)
        
        # Ensure ID format
        df_gt = df_gt[df_gt['student_ID'].astype(str).str.isdigit()].copy()
        df_gt['student_ID'] = df_gt['student_ID'].astype(int)
        df_gt = df_gt[(df_gt['student_ID'] >= 1) & (df_gt['student_ID'] <= 1000)].sort_values('student_ID').reset_index(drop=True)
        
        gt = df_gt[self.QUESTION_COLS].copy()
        
        # --- Gender Processing ---
        def assign_gender_group(val):
            if pd.isna(val): return 'Other/Missing'
            s = str(val).strip().split('.')[0]
            if s == '1': return 'Female'
            if s == '2': return 'Male'
            return 'Other/Missing'
        gt['Gender_Group'] = df_gt['Q2'].apply(assign_gender_group)

        # --- Option Formatting ---
        def convert_numeric_to_alpha_gt(val):
            if pd.isna(val) or not str(val).strip() or str(val).upper() in ['NAN', 'NONE']: return np.nan
            s_val = str(val).strip()
            if s_val.isalpha() and len(s_val) == 1: return s_val.upper()
            try:
                num = float(s_val)
                if num == int(num): return self.NUM_TO_ALPHA.get(str(int(num)), np.nan)
                return np.nan
            except ValueError: return np.nan

        for col in self.SINGLE_CHOICE_ALPHA_COLS:
            if col in gt.columns:
                gt.loc[:, col] = gt[col].apply(convert_numeric_to_alpha_gt).replace(np.nan, None).astype(object)

        # --- Load Metadata ---
        if self.paths.questions_file.exists():
            with open(self.paths.questions_file, 'r', encoding='utf-8') as f:
                questions_metadata_list = json.load(f)
            for q in questions_metadata_list:
                if q["id"] in self.QUESTION_COLS:
                    meta = {'type': q['type'], 'text': q['text']}
                    if q['type'] == 'choice':
                        # Rough parsing of choices from text
                        meta['choices'] = [line.strip().split('.')[0].strip().upper() 
                                         for line in q['text'].split('\n') if '.' in line]
                    self.q_meta[q["id"]] = meta

        # --- Ethnicity Processing ---
        def standardize_multichoice_internal(val):
            if pd.isna(val) or val in ['nan', 'NaN', '']: return ''
            return ''.join(sorted(str(val).upper().replace(',', '').replace(' ', '')))
        
        gt['Q5_standardized'] = gt['Q5'].apply(standardize_multichoice_internal)

        def assign_ethnicity_group(row):
            q4, q5 = row['Q4'], row['Q5_standardized']
            if q4 == 'A': return '1_Hispanic/Latino'
            if q4 is None or not q5 or q4 == 'nan': return '6_Other/Missing/Unclassified'
            if q4 == 'B':
                if len(q5) > 1: return '5_Multiple Races (Non-H)'
                if q5 == 'C': return '2_Black (Non-H)'
                if q5 == 'E': return '3_White (Non-H)'
                if q5 in ['A', 'B', 'D']: return '4_Asian/Native/Pacific (Non-H)'
                return '6_Other/Missing/Unclassified'
            return '6_Other/Missing/Unclassified'

        gt['Ethnicity_Group'] = gt.apply(assign_ethnicity_group, axis=1)
        gt['student_ID'] = df_gt['student_ID']
        
        self.gt_df = gt

    # ==========================
    # 2. Scoring Logic
    # ==========================
    def convert_student_id(self, student_id_str):
        if pd.isna(student_id_str): return None
        s = str(student_id_str).strip()
        if s.lower().startswith('student'):
            try: return int(s[7:])
            except: return None
        try: return int(s)
        except: return None

    def is_valid_response(self, val, qid):
        if pd.isna(val) or str(val).lower() in ['', 'nan', 'error', 'no_match', 'not matching']: return False
        val_str = str(val).strip().upper()
        if qid in ['Q6', 'Q7']:
            try: float(val); return True
            except: return False
        if qid == 'Q5': return all(c in 'ABCDE' for c in val_str) and len(val_str) > 0
        if qid in [f"Q{i}" for i in list(range(1, 5)) + list(range(8, 88))]:
            return len(val_str) == 1 and val_str.isalpha() and val_str.isupper()
        if qid in [f"Q{i}" for i in range(88, 108)]:
            try:
                num = int(round(float(val)))
                if qid in self.q_meta and 'choices' in self.q_meta[qid]:
                    valid_choices = [int(c) for c in self.q_meta[qid]['choices'] if c.isdigit()]
                    if valid_choices: return num in valid_choices
                return 1 <= num <= 10
            except: return False
        return True

    def build_score_map(self, qid):
        if qid not in self.q_meta: return {}
        meta = self.q_meta[qid]
        if meta['type'] == 'numeric': return 'numeric'
        choices = meta.get('choices', [])
        if len(choices) > 1:
            step = 1.0 / (len(choices) - 1)
            return {opt: i * step for i, opt in enumerate(choices)}
        return {choices[0]: 1.0} if choices else {}

    def calculate_scores(self, df_pred, df_gt, q_list, student_info):
        pred, gt = df_pred[q_list].copy(), df_gt[q_list].copy()
        valid_mask = pd.DataFrame(True, index=pred.index, columns=q_list)

        for q in q_list:
            valid_mask[q] = pred[q].apply(lambda x: self.is_valid_response(x, q)) & \
                            gt[q].apply(lambda x: self.is_valid_response(x, q))

        score_matrix = pd.DataFrame(np.nan, index=pred.index, columns=q_list)
        question_scores = {qid: self.build_score_map(qid) for qid in q_list}

        for q in q_list:
            valid_idx = valid_mask[q]
            if not valid_idx.any(): continue

            p_val, g_val = pred.loc[valid_idx, q], gt.loc[valid_idx, q]

            if question_scores[q] == 'numeric':
                try:
                    p_num = pd.to_numeric(p_val, errors='coerce')
                    g_num = pd.to_numeric(g_val, errors='coerce')
                    score_matrix.loc[valid_idx, q] = (abs(p_num - g_num) <= 0.5).astype(float)
                except: pass
            else:
                score_map = question_scores[q]
                if q in [f"Q{i}" for i in range(88, 108)]:
                    # Handle Q88+ numeric scaling
                    p_val = p_val.apply(lambda x: str(int(float(x))) if str(x).replace('.','').isdigit() else str(x))
                    g_val = g_val.apply(lambda x: str(int(float(x))) if str(x).replace('.','').isdigit() else str(x))
                else:
                    p_val, g_val = p_val.str.upper(), g_val.str.upper()

                p_score = p_val.map(score_map)
                g_score = g_val.map(score_map)

                valid_scores = p_score.notna() & g_score.notna()
                if valid_scores.any():
                    idx_sub = p_score[valid_scores].index
                    score_matrix.loc[idx_sub, q] = 1 - abs(p_score[valid_scores] - g_score[valid_scores])

        per_dt_accuracy = score_matrix.mean(axis=1).rename('Accuracy')
        return student_info.copy().loc[per_dt_accuracy.index].join(per_dt_accuracy)

    # ==========================
    # 3. Processing Pipeline
    # ==========================
    def process_all_results(self):
        self.load_and_prepare_gt()
        results = []
        
        print("\n=== Processing Analysis ===")
        
        for map_key, (model_folder, dt_type_folder) in self.MODEL_MAPPING.items():
            # 构建文件系统路径
            base_path = self.paths.external_results_root / model_folder / dt_type_folder
            
            if not base_path.exists():
                print(f"Skipping {map_key}: Path not found ({base_path})")
                continue

            info = self.MODEL_INFO[map_key]
            print(f"Processing {info['Model_Base']} - {info['Condition']}...")

            for domain, d_info in self.EXTERNAL_DOMAIN_MAP.items():
                # 模糊匹配文件名 (e.g., no_Q27_Q30_responses.csv)
                pattern = d_info['file_pattern']
                found_files = list(base_path.glob(f"*{pattern}*.csv"))
                
                if not found_files:
                    continue
                
                f_path = found_files[0]
                df_ext = pd.read_csv(f_path, dtype=str)
                
                # Normalize ID
                col = next((c for c in df_ext.columns if 'student_id' in c.lower()), None)
                if col: df_ext.rename(columns={col: 'student_ID'}, inplace=True)
                df_ext['student_ID'] = df_ext['student_ID'].apply(self.convert_student_id)
                df_ext = df_ext.dropna(subset=['student_ID'])
                df_ext['student_ID'] = df_ext['student_ID'].astype(int)

                q_list = d_info['questions']
                
                # Merge with GT
                gt_subset = self.gt_df[['student_ID', 'Ethnicity_Group', 'Gender_Group'] + q_list].copy()
                merged = df_ext.merge(gt_subset, on='student_ID', how='inner', suffixes=('_pred', '_gt'))

                if merged.empty: continue

                # Prepare Dataframes for Scoring
                pred_df = merged[[f"{q}_pred" for q in q_list]].rename(columns={f"{q}_pred": q for q in q_list})
                gt_score = merged[[f"{q}_gt" for q in q_list]].rename(columns={f"{q}_gt": q for q in q_list})
                info_df = merged[['student_ID', 'Ethnicity_Group', 'Gender_Group']]
                
                # Sync Indices
                pred_df.reset_index(drop=True, inplace=True)
                gt_score.index = pred_df.index
                info_df.index = pred_df.index

                # Calculate
                scores = self.calculate_scores(pred_df, gt_score, q_list, info_df)
                scores['Domain'] = domain
                scores['Model_Base'] = info['Model_Base']
                scores['Condition'] = info['Condition']
                results.append(scores)

        if results:
            final_df = pd.concat(results, ignore_index=True)
            output_csv = self.paths.output_dir / "subgroup_analysis.csv"
            final_df.to_csv(output_csv, index=False)
            print(f"Analysis saved to: {output_csv}")
            return final_df
        else:
            print("No matching results found.")
            return pd.DataFrame()

    # ==========================
    # 4. Visualization
    # ==========================
    def draw_grid_plot(self, df, group_col, group_order, title_suffix, 
                       fig_size=(20, 14), left_margin=0.08, domain_label_x=-0.18, top_margin=0.9):
        
        if df.empty: return

        # Configs
        FRAME_COLOR = '#666666'
        PALETTE = {'Survey DT': '#0056b3', 'Survey+Memory DT': '#d62728'}
        DOMAIN_ORDER = list(self.EXTERNAL_DOMAIN_MAP.keys())
        MODEL_ORDER = ['gemini-2.0-flash', 'gemini-2.5-lite-flash', 'gemini-2.5-flash']
        DOMAIN_SHORT_NAMES = {
            'Suicidality & Self-Harm': 'Suicidality &\nSelf-Harm',
            'Substance Use': 'Substance\nUse',
            'Violence & Abuse Exposure': 'Violence &\nAbuse',
            'Safety Behaviors': 'Safety\nBehaviors',
            'Mental Health Status': 'Mental\nHealth'
        }

        plot_df = df.copy()
        plot_df['Accuracy_Percent'] = plot_df['Accuracy'] * 100
        condition_order = ['Survey DT', 'Survey+Memory DT']
        means = plot_df.groupby(['Domain', 'Model_Base', group_col, 'Condition'])['Accuracy_Percent'].mean().reset_index()

        fig = plt.figure(figsize=fig_size)
        gs = fig.add_gridspec(len(DOMAIN_ORDER), len(MODEL_ORDER), 
                              hspace=0.1, wspace=0.1, 
                              left=left_margin, right=0.98, top=top_margin, bottom=0.05)

        for i, domain in enumerate(DOMAIN_ORDER):
            for j, model in enumerate(MODEL_ORDER):
                ax = fig.add_subplot(gs[i, j])
                subset = plot_df[(plot_df['Domain'] == domain) & (plot_df['Model_Base'] == model)]

                # Zebra Background
                for k in range(len(group_order)):
                    if k % 2 == 0:
                        ax.axhspan(k - 0.5, k + 0.5, color='#F0F0F0', alpha=0.6, zorder=0)

                if not subset.empty:
                    sns.pointplot(
                        data=subset, y=group_col, x='Accuracy_Percent', hue='Condition',
                        order=group_order, hue_order=condition_order, palette=PALETTE,
                        errorbar=('ci', 95), capsize=0.2, join=False, dodge=0.4,
                        markers=['o', 's'], scale=1.1,
                        err_kws={'linewidth': 2, 'alpha': 0.9}, ax=ax
                    )

                    # Value Labels
                    subset_means = means[(means['Domain'] == domain) & (means['Model_Base'] == model)]
                    for k, label in enumerate(group_order):
                        for cond in condition_order:
                            val_row = subset_means[(subset_means[group_col] == label) & (subset_means['Condition'] == cond)]
                            if not val_row.empty:
                                val = val_row['Accuracy_Percent'].values[0]
                                y_off = -0.15 if cond == 'Survey DT' else 0.15
                                ax.text(val + 0.5, k + y_off, f"{val:.1f}%", 
                                        va='center', ha='left', fontsize=8, color='black', fontweight='bold')

                # Styling
                if ax.get_legend(): ax.get_legend().remove()
                ax.set_ylabel("")
                ax.set_xlabel("")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(FRAME_COLOR)
                    spine.set_linewidth(1.2)
                ax.grid(axis='x', linestyle='--', alpha=0.5, color=FRAME_COLOR, linewidth=0.8)

                # Y Axis Labels (Left Column Only)
                if j == 0:
                    ax.set_yticklabels(group_order, fontsize=10, fontweight='bold')
                    short_name = DOMAIN_SHORT_NAMES.get(domain, domain)
                    ax.text(domain_label_x, 0.5, short_name, transform=ax.transAxes,
                            va='center', ha='center', fontsize=11, fontweight='bold',
                            rotation=90, color='#333333')
                else:
                    ax.set_yticklabels([])

                # X Axis Labels (Bottom Row Only)
                if i == len(DOMAIN_ORDER) - 1:
                    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
                    ax.set_xlabel("Accuracy (%)", fontsize=11, fontweight='bold')
                    ax.tick_params(axis='x', labelsize=9)
                else:
                    ax.set_xticklabels([])

                # Column Titles
                if i == 0:
                    ax.set_title(model, fontsize=14, fontweight='bold', pad=15)

                current_xlim = ax.get_xlim()
                ax.set_xlim(current_xlim[0], 101.5)

        # Legend
        legend_y = top_margin + 0.05
        handles = [mpatches.Patch(color=PALETTE[k], label=k) for k in condition_order]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, legend_y),
                   ncol=2, fontsize=14, frameon=False)
        
        # Save
        filename = f"heterogeneity_{title_suffix}.png"
        save_path = self.paths.figures_dir / filename
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")
        # plt.show() # Optional: Comment out if running in non-interactive environment

    def run(self):
        final_df = self.process_all_results()
        
        if final_df.empty:
            print("Skipping plotting due to lack of data.")
            return

        # 1. Gender Plot
        print("\nPlotting Gender Analysis...")
        gender_df = final_df[final_df['Gender_Group'].isin(['Female', 'Male'])].copy()
        self.draw_grid_plot(gender_df, 'Gender_Group', ['Female', 'Male'], "gender",
                            fig_size=(20, 14), left_margin=0.08, domain_label_x=-0.18, top_margin=0.9)

        # 2. Ethnicity Plot
        print("\nPlotting Ethnicity Analysis...")
        ethnicity_map = {
            '1_Hispanic/Latino': 'Hispanic/Latino',
            '2_Black (Non-H)': 'Black (Non-H)',
            '3_White (Non-H)': 'White (Non-H)',
            '4_Asian/Native/Pacific (Non-H)': 'Asian/Native',
            '5_Multiple Races (Non-H)': 'Multiple Races',
        }
        eth_df = final_df.copy()
        eth_df['Ethnicity_Label'] = eth_df['Ethnicity_Group'].map(ethnicity_map)
        eth_df = eth_df.dropna(subset=['Ethnicity_Label'])
        eth_order = sorted(list(ethnicity_map.values()), reverse=True)
        
        self.draw_grid_plot(eth_df, 'Ethnicity_Label', eth_order, "ethnicity",
                            fig_size=(20, 20), left_margin=0.15, domain_label_x=-0.3, top_margin=0.94)

if __name__ == "__main__":
    analysis = HeterogeneityAnalysis()
    analysis.run()