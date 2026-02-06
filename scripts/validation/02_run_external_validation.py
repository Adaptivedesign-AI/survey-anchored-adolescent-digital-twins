#!/usr/bin/env python3
"""
02_run_external_validation.py

Description:
    Performs external validation with strict data isolation.
    
    CRITICAL UPDATE:
    Loads specific 'Enriched Memory' files corresponding to each hold-out group 
    to prevent data leakage. Use memories generated from masked profiles, 
    not the full baseline.

Usage:
    python 02_run_external_validation.py
"""

import os
import sys
import json
import asyncio
import pandas as pd
import re
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional

import nest_asyncio
from google import genai
from google.genai import types

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ProjectPaths:
    """Path Management"""
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        
        self.config_file = self.root / "config.yaml"
        self.data_raw = self.root / "data" / "raw"
        self.data_processed = self.root / "data" / "processed"
        self.results_dir = self.root / "results" / "validation" / "external"
        self.prompts_dir = self.root / "configs" / "prompts"
        
        # Inputs
        self.yrbs_data = self.data_processed / "cohort" / "sampled_1000.csv"
        self.mapping_table = self.root / "configs" / "prompt_maps" / "yrbs_to_profile_map.csv"
        self.questions_json = self.data_raw / "yrbs" / "yrbs_questions.json"
        
        # Shared Prompt
        self.shared_prompt = self.prompts_dir / "personas" / "shared_base_prompt.txt"
        
        # [NEW] Directory specifically for the 5 sets of External Validation Memories
        # 假设你把那5个对应的memory json文件放在这个文件夹里
        self.external_memories_dir = self.data_processed / "cohort" / "external_memories"
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

class PromptGenerator:
    """Generates Masked Profiles (Data Cleaning & String Building)"""
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.data_df = pd.read_csv(self.paths.yrbs_data)
        self.mapping_df = pd.read_csv(self.paths.mapping_table)
        self.prompt_dict = self._build_prompt_dict()

    def _build_prompt_dict(self):
        self.mapping_df["code"] = self.mapping_df["code"].astype(str).str.replace(".0", "", regex=False)
        self.mapping_df["question"] = self.mapping_df["question"].astype(str)
        prompt_dict = {}
        for _, row in self.mapping_df.iterrows():
            q = row["question"]
            code = str(row["code"]).strip()
            frag = row["prompt_fragment"]
            if code.lower() in ["nan", "", "none"]: continue
            if q not in prompt_dict: prompt_dict[q] = {}
            prompt_dict[q][code] = frag
        return prompt_dict

    def _clean_code(self, val):
        if pd.isna(val): return None
        val_str = str(val).strip()
        return val_str.replace(".0", "") if val_str.endswith(".0") else val_str

    def generate_masked_profiles(self, drop_questions: List[str]) -> Dict[str, str]:
        """Generates baseline profiles EXCLUDING the dropped questions"""
        q_columns = [f"Q{i}" for i in range(1, 108) if f"Q{i}" not in drop_questions]
        profiles = {}
        for idx, row in self.data_df.iterrows():
            frags = []
            for q in q_columns:
                code = self._clean_code(row.get(q))
                if code is None: continue
                if q == "Q6":
                    try: frags.append(f"Your height without shoes is {float(code):.2f} meters.")
                    except: continue
                elif q == "Q7":
                    try: frags.append(f"Your weight without shoes is {float(code):.0f} kilograms.")
                    except: continue
                else:
                    frag = self.prompt_dict.get(q, {}).get(code)
                    if isinstance(frag, str) and frag.strip():
                        frags.append(frag)
            
            student_id = str(row.get('student_id', f"student{idx+1}"))
            profiles[student_id] = " ".join(frags)
        return profiles

class ExternalValidatorDT:
    """Async Digital Twin Agent"""
    def __init__(self, student_id: str, dt_type: str, model: str, 
                 masked_baseline: str, shared_prompt: str, enriched_data: dict, client: genai.Client):
        self.student_id = student_id
        self.dt_type = dt_type 
        self.model = model
        self.client = client
        self.system_prompt = self._build_system_prompt(masked_baseline, shared_prompt, enriched_data)

    def _build_system_prompt(self, baseline: str, shared: str, enriched: dict) -> str:
        parts = [shared]
        
        # 1. Baseline (Survey DT & Survey+Memory DT)
        if self.dt_type in ['survey_dt', 'survey_memory_dt']:
            parts.append(f"\n\n=== Your Personal Profile ===\n{baseline}")
        
        # 2. Memory (Survey+Memory DT ONLY)
        # 关键点：这里的 enriched 必须是针对该 Masked 组生成的，不包含泄露信息
        if self.dt_type == 'survey_memory_dt' and enriched:
            parts.append("\n\n=== Your Experiences and Mental Model ===\nBased on your life experiences:")
            for domain in enriched.get('enriched_domains', []):
                parts.append(f"\n**{domain['domain']}**: {domain['overall_domain_narrative']}")
            
            if enriched.get('daily_conversations'):
                parts.append("\n\n=== Daily Conversation Examples ===")
                for conv in enriched.get('daily_conversations', [])[:2]:
                    parts.append(f"\nSetting: {conv.get('setting')}")
                    for d in conv.get('dialogue', []):
                        parts.append(f"{d.get('speaker')}: {d.get('text')}")

        parts.append("\n\n=== INSTRUCTIONS ===\nYou are this person. Answer as yourself directly.")
        return "\n".join(parts)

    async def ask(self, question_text: str, q_num: int) -> dict:
        user_msg = f"""Question {q_num}: {question_text}
Return ONLY one line: "Answer: [Option Code]" (e.g., Answer: A or Answer: 2)."""
        
        try:
            response = await self.client.models.generate_content(
                model=self.model,
                contents=f"{self.system_prompt}\n\nUser: {user_msg}\nAssistant: ",
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=50)
            )
            raw = response.text.strip()
            match = re.search(r"Answer:\s*([A-H0-9]+)", raw, re.IGNORECASE)
            return {
                "student_id": self.student_id,
                f"Q{q_num}": match.group(1).upper() if match else "NO_MATCH",
                f"Q{q_num}_raw": raw
            }
        except Exception as e:
            return {"student_id": self.student_id, f"Q{q_num}": "ERROR", f"Q{q_num}_raw": str(e)}

class ValidationManager:
    """Manages the 5 External Validation Tasks"""
    
    # 定义5个分组任务
    HOLD_OUT_GROUPS = [
        {"name": "no_Q8_Q11", "qs": [8, 9, 10, 11], "drop": ["Q8", "Q9", "Q10", "Q11"]},
        {"name": "no_Q27_Q30", "qs": [27, 28, 29, 30], "drop": ["Q27", "Q28", "Q29", "Q30"]},
        {"name": "no_Q26_Q84", "qs": [26, 84], "drop": ["Q26", "Q84"]},
        {"name": "no_Q19_Q22_and_Q88_Q91", "qs": [19, 20, 21, 22, 88, 89, 90, 91], 
         "drop": ["Q19", "Q20", "Q21", "Q22", "Q88", "Q89", "Q90", "Q91"]},
        {"name": "no_Q35_Q37_Q49_Q92", "qs": [35, 36, 37, 49, 92], 
         "drop": ["Q35", "Q36", "Q37", "Q49", "Q92"]},
    ]

    def __init__(self):
        self.paths = ProjectPaths()
        self.prompt_gen = PromptGenerator(self.paths)
        
        with open(self.paths.config_file) as f:
            self.config = yaml.safe_load(f)
        self.client = genai.Client(api_key=self.config.get("gemini_api_key"))
        
        with open(self.paths.shared_prompt, 'r') as f:
            self.shared_prompt = f.read()
            
        with open(self.paths.questions_json, 'r') as f:
            self.questions_def = {q['id']: q['text'] for q in json.load(f)}

    def _load_task_specific_memories(self, task_name: str) -> Dict:
        """
        [CRITICAL] Dynamically loads the Enriched Memory file specific to the current task.
        Naming convention assumed: enriched_prompts_{task_name}.json
        """
        # 假设文件名是 enriched_prompts_no_Q8_Q11.json
        filename = f"enriched_prompts_{task_name}.json"
        file_path = self.paths.external_memories_dir / filename
        
        if not file_path.exists():
            # Fallback logic or Error
            logger.warning(f"⚠️ Specific memory file {filename} not found! Checking for generic or erroring out.")
            # 如果你为了测试想用通用的，可以取消下面注释，但科学上不严谨
            # return {} 
            raise FileNotFoundError(f"Missing required external memory file: {file_path}")

        logger.info(f"Loading task-specific memory: {filename}")
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Normalize to dict {student_id: data}
            return {str(p['student_id']): p for p in data.get('enrichments', [])}

    async def run_group(self, group_idx: int, dt_type: str, model: str, limit: int = None):
        group = self.HOLD_OUT_GROUPS[group_idx]
        task_name = group['name']
        
        logger.info(f"=== PROCESSING TASK: {task_name} | {dt_type} | {model} ===")

        # 1. Load Task-Specific Memories (防止数据泄露)
        # 只有 Survey+Memory DT 需要这个，为了效率，如果是其他类型可以跳过加载
        current_enriched_data = {}
        if dt_type == 'survey_memory_dt':
            try:
                current_enriched_data = self._load_task_specific_memories(task_name)
            except FileNotFoundError as e:
                logger.error(str(e))
                return # Skip this run if data is missing

        # 2. Generate Masked Profiles (从 Profile 中挖掉题)
        masked_profiles = self.prompt_gen.generate_masked_profiles(group['drop'])

        # 3. Setup Output
        output_dir = self.paths.results_dir / model / dt_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_name}_responses.csv"
        
        processed_ids = set()
        if output_file.exists():
            processed_ids = set(pd.read_csv(output_file)['student_id'].astype(str))

        # 4. Processing Loop
        student_ids = list(masked_profiles.keys())
        if limit: student_ids = student_ids[:limit]
        
        semaphore = asyncio.Semaphore(20)

        async def worker(sid):
            if sid in processed_ids: return None
            async with semaphore:
                dt = ExternalValidatorDT(
                    student_id=sid,
                    dt_type=dt_type,
                    model=model,
                    masked_baseline=masked_profiles.get(sid, ""),
                    shared_prompt=self.shared_prompt,
                    enriched_data=current_enriched_data.get(sid, {}), # Inject specific memory
                    client=self.client
                )
                
                res = {"student_id": sid}
                for q_num in group['qs']:
                    q_key = f"Q{q_num}"
                    ans = await dt.ask(self.questions_def.get(q_key, ""), q_num)
                    res.update(ans)
                return res

        # Batch Execution
        batch_size = 50
        for i in range(0, len(student_ids), batch_size):
            tasks = [worker(sid) for sid in student_ids[i:i+batch_size]]
            results = await asyncio.gather(*tasks)
            valid = [r for r in results if r]
            
            if valid:
                pd.DataFrame(valid).to_csv(output_file, mode='a', header=not output_file.exists(), index=False)
                logger.info(f"Saved batch {i//batch_size + 1}")
            await asyncio.sleep(1)

async def main():
    manager = ValidationManager()
    
    # 可以根据需要调整这里
    models = ['gemini-2.5-flash-lite'] 
    dt_types = ['survey_memory_dt', 'survey_dt', 'base_dt'] 
    
    for model in models:
        for dt_type in dt_types:
            for i in range(5): # 5 groups
                await manager.run_group(i, dt_type, model)

if __name__ == "__main__":
    asyncio.run(main())