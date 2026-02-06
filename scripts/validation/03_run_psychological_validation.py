#!/usr/bin/env python3
"""
03_run_psychological_validation.py

Description:
    Runs psychological validation by administering external questionnaires (CSV files)
    to the Digital Twins.
    
    Work flow:
    1. Load all CSV questionnaires from data/raw/questionnaires/
    2. For each Model (3) x DT Type (3):
       3. For each Questionnaire (8):
          4. Ask all 1000 DTs the questions in that questionnaire.
          5. Save results to results/validation/psychological/...

Usage:
    python 03_run_psychological_validation.py
"""

import sys
import os
import json
import asyncio
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import nest_asyncio
from google import genai
from google.genai import types

# Apply nest_asyncio
nest_asyncio.apply()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ProjectPaths:
    """Centralized Path Management"""
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.config_file = self.root / "config.yaml"
        
        # Inputs
        self.questionnaires_dir = self.root / "data" / "raw" / "questionnaires"
        self.cohort_dir = self.root / "data" / "processed" / "cohort"
        self.prompts_dir = self.root / "configs" / "prompts"
        
        # Data files
        self.baseline_profiles = self.cohort_dir / "baseline_profiles_1000.json"
        self.enriched_profiles = self.cohort_dir / "enriched_prompts_1000.json"
        self.shared_prompt = self.prompts_dir / "personas" / "shared_base_prompt.txt"
        
        # Outputs
        self.results_dir = self.root / "results" / "validation" / "psychological"
        self.results_dir.mkdir(parents=True, exist_ok=True)

class PsychologicalValidator:
    """Conducts psychological validation using external CSV questionnaires"""
    
    DT_TYPES = ['survey_memory_dt', 'survey_dt', 'base_dt']
    MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']
    
    def __init__(self, concurrency: int = 20):
        self.paths = ProjectPaths()
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        
        # Load Config & Client
        with open(self.paths.config_file) as f:
            self.config = yaml.safe_load(f)
        self.client = genai.Client(api_key=self.config.get("gemini_api_key"))
        
        # Load Data
        self._load_data()
        self._load_questionnaires()
        
    def _load_data(self):
        """Load prompts and profiles into memory"""
        logger.info("Loading DT profiles and prompts...")
        
        # Shared Prompt
        with open(self.paths.shared_prompt, 'r') as f:
            self.shared_prompt = f.read()
            
        # Baseline Profiles
        with open(self.paths.baseline_profiles, 'r') as f:
            data = json.load(f)
            self.baselines = {str(p['student_id']): p['baseline_prompt'] for p in data['profiles']}
            
        # Enriched Profiles (Memory)
        with open(self.paths.enriched_profiles, 'r') as f:
            data = json.load(f)
            self.memories = {str(p['student_id']): p for p in data.get('enrichments', [])}
            
    def _load_questionnaires(self):
        """Load all CSV questionnaires from the directory"""
        logger.info(f"Scanning for questionnaires in {self.paths.questionnaires_dir}...")
        self.questionnaires = {}
        
        if not self.paths.questionnaires_dir.exists():
            logger.error(f"Directory not found: {self.paths.questionnaires_dir}")
            return

        # Scan for CSVs
        for csv_file in self.paths.questionnaires_dir.glob("*.csv"):
            try:
                # Assuming CSV has columns like 'question_id', 'question_text', 'options'
                # If structure is simple (just a list of questions), we adapt.
                # Here we assume a simple format: column 'question' or 'text' contains the question.
                df = pd.read_csv(csv_file)
                
                # Normalize column names
                df.columns = [c.lower() for c in df.columns]
                
                # Identify text column
                text_col = next((c for c in df.columns if 'question' in c or 'text' in c), None)
                if not text_col:
                    logger.warning(f"Skipping {csv_file.name}: Could not identify question text column.")
                    continue
                
                questions = []
                for idx, row in df.iterrows():
                    q_text = row[text_col]
                    q_id = row.get('id', row.get('question_id', f"Q{idx+1}"))
                    questions.append({'id': str(q_id), 'text': str(q_text)})
                
                self.questionnaires[csv_file.stem] = questions
                logger.info(f"  ✓ Loaded {csv_file.stem}: {len(questions)} questions")
                
            except Exception as e:
                logger.error(f"Failed to load {csv_file.name}: {e}")

    def _build_system_prompt(self, student_id: str, dt_type: str) -> str:
        """Constructs the DT persona prompt"""
        parts = [self.shared_prompt]
        
        # 1. Baseline Profile
        if dt_type in ['survey_dt', 'survey_memory_dt']:
            baseline = self.baselines.get(student_id, "")
            parts.append(f"\n\n=== Your Personal Profile ===\n{baseline}")
            
        # 2. Enriched Memory
        if dt_type == 'survey_memory_dt':
            mem = self.memories.get(student_id, {})
            if mem:
                parts.append("\n\n=== Your Experiences and Mental Model ===\nBased on your life experiences:")
                for domain in mem.get('enriched_domains', []):
                    parts.append(f"\n**{domain['domain']}**: {domain['overall_domain_narrative']}")
                
                # Add conversations context
                if mem.get('daily_conversations'):
                    parts.append("\n\n=== Daily Conversation Examples ===")
                    for conv in mem.get('daily_conversations', [])[:2]:
                        parts.append(f"\nSetting: {conv.get('setting')}")
                        for d in conv.get('dialogue', []):
                            parts.append(f"{d.get('speaker')}: {d.get('text')}")

        parts.append("\n\n=== INSTRUCTIONS ===\nYou are taking a psychological survey. Answer honestly as yourself.")
        return "\n".join(parts)

    async def _process_student_questionnaire(self, student_id: str, dt_type: str, model: str, q_name: str, questions: List[Dict]):
        """Ask one student all questions in a single questionnaire"""
        
        system_prompt = self._build_system_prompt(student_id, dt_type)
        
        # We can optimize by asking questions in a conversational chain or one-by-one.
        # For psychological validity, one-by-one (stateless) is often cleaner to avoid context drift,
        # but slower. Here we do stateless requests for maximum isolation.
        
        student_responses = {"student_id": student_id}
        
        async with self.semaphore:
            for q in questions:
                prompt = f"""{system_prompt}

=== Survey Question ===
{q['text']}

Instructions:
Select the option that best describes you.
Return ONLY the answer (e.g., "Not at all", "3", "Agree").
Do not explain."""

                try:
                    response = await self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=50)
                    )
                    student_responses[q['id']] = response.text.strip()
                except Exception as e:
                    student_responses[q['id']] = "ERROR"
                    logger.warning(f"Error {student_id} {q_name} {q['id']}: {e}")
                    
        return student_responses

    async def run_specific_questionnaire(self, q_name: str, dt_type: str, model: str, student_ids: List[str]):
        """Run one specific questionnaire for a specific configuration"""
        
        questions = self.questionnaires.get(q_name)
        if not questions:
            return
            
        # Prepare output path
        output_dir = self.paths.results_dir / model / dt_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{q_name}_responses.csv"
        
        # Check existing
        processed_ids = set()
        if output_file.exists():
            processed_ids = set(pd.read_csv(output_file)['student_id'].astype(str))
            
        remaining_ids = [sid for sid in student_ids if sid not in processed_ids]
        if not remaining_ids:
            logger.info(f"Skipping {q_name} (All completed) for {model}/{dt_type}")
            return

        logger.info(f"Running {q_name} | {dt_type} | {model} | {len(remaining_ids)} students")

        # Batch processing
        batch_size = 50
        for i in range(0, len(remaining_ids), batch_size):
            batch = remaining_ids[i:i+batch_size]
            tasks = [self._process_student_questionnaire(sid, dt_type, model, q_name, questions) for sid in batch]
            
            results = await asyncio.gather(*tasks)
            
            # Save batch
            df = pd.DataFrame(results)
            # Reorder columns to put student_id first
            cols = ['student_id'] + [col for col in df.columns if col != 'student_id']
            df = df[cols]
            
            df.to_csv(output_file, mode='a', header=not output_file.exists(), index=False)
            logger.info(f"  Saved batch {i//batch_size + 1} to {output_file.name}")
            
            await asyncio.sleep(1) # Rate limit cooling

    async def run_all(self):
        """Main execution loop"""
        
        student_ids = [str(i) for i in range(1, 1001)]
        
        # Iterate all configurations
        total_steps = len(self.MODELS) * len(self.DT_TYPES) * len(self.questionnaires)
        current_step = 0
        
        for model in self.MODELS:
            for dt_type in self.DT_TYPES:
                for q_name in self.questionnaires.keys():
                    current_step += 1
                    logger.info(f"\n[{current_step}/{total_steps}] Validation Step")
                    
                    await self.run_specific_questionnaire(
                        q_name, dt_type, model, student_ids
                    )

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--concurrency', type=int, default=25)
    args = parser.parse_args()
    
    validator = PsychologicalValidator(concurrency=args.concurrency)
    await validator.run_all()

if __name__ == "__main__":
    asyncio.run(main())