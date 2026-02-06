import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from functools import lru_cache
from google import genai
import asyncio
import nest_asyncio
import time
import os
import signal
import re
from datetime import datetime

try:
    nest_asyncio.apply()
except RuntimeError:
    pass

class ProjectPaths:
    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.resolve()
        self.config_file = self.root / "config.yaml"
        self.data_raw_rct = self.root / "data" / "raw" / "rct_studies"
        self.prompts_dir = self.root / "configs" / "prompts" / "personas"
        self.results_dir = self.root / "results" / "rct_replication"
        
        self.results_dir.mkdir(parents=True, exist_ok=True)

def _is_error_response_text(s: Optional[str]) -> bool:
    if s is None: return True
    s = str(s).strip()
    if not s: return True
    return s.upper().startswith("ERROR")

def _study_prompt_columns(study_row: pd.Series) -> List[str]:
    cols = sorted(
        [c for c in study_row.index if c.startswith("prompt_")],
        key=lambda x: int(x.split("_")[1])
    )
    valid = []
    for c in cols:
        val = study_row[c]
        is_nan = pd.isna(val)
        is_empty = isinstance(val, str) and val.strip() == ""
        is_nan_str = isinstance(val, str) and val.strip().lower() == "nan"
        if is_nan or is_empty or is_nan_str:
            break
        valid.append(c)
    return valid

def _record_is_completed(record: Dict, expected_prompt_cols: List[str]) -> bool:
    if not isinstance(record, dict): return False
    raw_log = record.get("raw_log", [])
    if not isinstance(raw_log, list) or len(raw_log) != len(expected_prompt_cols):
        return False
    for i, col in enumerate(expected_prompt_cols):
        if i >= len(raw_log): return False
        entry = raw_log[i]
        if not isinstance(entry, dict): return False
        if entry.get("prompt_col") != col: return False
        if _is_error_response_text(entry.get("response_text")): return False
    return True

class DigitalTwin:
    def __init__(self, student_id: int, initialization_prompt: str, client: genai.Client):
        self.student_id = student_id
        self.initialization_prompt = initialization_prompt
        self.client = client
        self.conversation_history: List[Dict[str, str]] = []
        self.initialized = False

    async def initialize(self) -> bool:
        try:
            init_message = f"""You must roleplay as this specific character. This is your identity:

{self.initialization_prompt}

Rules:
- You ARE this person, answer as yourself
- Never mention being an AI
- Remember your background throughout our conversation

Confirm you understand by saying "Ready to answer as myself" """
            response = await self._call_api(init_message, is_initialization=True)
            if response and not _is_error_response_text(response):
                self.initialized = True
                return True
            return False
        except Exception:
            return False

    async def _call_api(self, new_message: str, is_initialization: bool = False) -> str:
        try:
            if is_initialization:
                full_context = f"""Character:
{self.initialization_prompt}

Question: {new_message}
Response: """
            else:
                full_context = "Conversation History:\n"
                for turn in self.conversation_history:
                    full_context += f"Question: {turn['human']}\n"
                    full_context += f"Response: {turn['assistant']}\n\n"
                full_context += f"Question: {new_message}\nResponse: "

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model="gemini-2.0-flash",
                    contents=full_context
                ),
                timeout=30.0
            )

            if hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
                self.conversation_history.append({
                    "human": new_message,
                    "assistant": response_text
                })
                return response_text
            return "ERROR: No response text"

        except asyncio.TimeoutError:
            return "ERROR: API timeout"
        except Exception as e:
            return f"ERROR: {str(e)}"

    async def ask_question(self, question: str) -> str:
        if not self.initialized: return "ERROR: NOT_INITIALIZED"
        return await self._call_api(question)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

class DigitalTwinExperiment:
    def __init__(self, concurrent_twins: int = 20, max_attempts: int = 3):
        self.paths = ProjectPaths()
        self.config = self._load_config()
        self.client = genai.Client(api_key=self.config.get("gemini_api_key"))
        
        self.concurrent_twins = concurrent_twins
        self.semaphore = asyncio.Semaphore(concurrent_twins)
        self.max_attempts = max_attempts
        self.shutdown_flag = False
        self.pending_results: List[Dict] = []

        self.study_age_ranges: Dict[int, str] = {
            1: "13–19", 2: "13–18", 3: "14–21", 4: "13–16", 5: "14–18",
            6: "14–18", 7: "12–13", 8: "12–18", 9: "14-19", 10: "15–18",
        }

        self.shared_prompt = self._load_shared_prompt()
        self.study_list = self._load_study_list()

    def _load_config(self) -> dict:
        with open(self.paths.config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_shared_prompt(self) -> str:
        prompt_path = self.paths.prompts_dir / "shared_base_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_study_list(self) -> pd.DataFrame:
        study_list_path = self.paths.data_raw_rct / "study_list.csv"
        if not study_list_path.exists():
            raise FileNotFoundError(f"Study list not found at {study_list_path}")
        df = pd.read_csv(study_list_path)
        df = df[df['study_id'].notna()].copy()
        df['study_id'] = df['study_id'].astype(int)
        return df

    def _get_age_prompt(self, study_id: int) -> str:
        if study_id not in self.study_age_ranges:
            return "You are an adolescent."
        return f"You are a {self.study_age_ranges[study_id]}-year-old adolescent."

    @lru_cache(maxsize=100)
    def _get_cached_initialization_prompt(self, study_id: int) -> str:
        age_prompt = self._get_age_prompt(study_id)
        return f"{self.shared_prompt}\n\n{age_prompt}"

    def _extract_json_from_response(self, response: str) -> Dict:
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match: return json.loads(json_match.group(1))
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match: return json.loads(json_match.group(0))
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot extract JSON")

    async def _run_single_attempt(self, study_id: int, student_id: int, study_row: pd.Series) -> Dict:
        async with self.semaphore:
            if self.shutdown_flag: return None
            try:
                init_prompt = self._get_cached_initialization_prompt(study_id)
                twin = DigitalTwin(student_id=student_id, initialization_prompt=init_prompt, client=self.client)

                if not await twin.initialize():
                    return {'student_id': student_id, 'error': 'Initialization failed', 'timestamp': datetime.now().isoformat()}

                results = {
                    'student_id': student_id,
                    'model_name': 'gemini-2.0-flash',
                    'timestamp': datetime.now().isoformat()
                }

                raw_log = []
                json_fields = {}
                json_items = {}
                prompt_cols = _study_prompt_columns(study_row)

                for col in prompt_cols:
                    if self.shutdown_flag:
                        results['error'] = 'Interrupted by user'
                        break

                    q = str(study_row[col])
                    instruction = q + "\n\nIMPORTANT: respond ONLY in valid JSON object. Do NOT add explanation text."
                    response = await twin.ask_question(instruction)

                    raw_log.append({
                        "prompt_col": col, "prompt_text": q, "response_text": response
                    })

                    if not _is_error_response_text(response):
                        try:
                            js = self._extract_json_from_response(response)
                            for k, v in js.items():
                                json_fields[k] = v
                                if isinstance(v, (int, float)): json_items[k] = v
                                elif isinstance(v, str) and v.isdigit(): json_items[k] = int(v)
                        except Exception:
                            pass

                results["raw_log"] = raw_log
                results["json_fields"] = json_fields
                results["json_items"] = json_items
                results["conversation_history"] = twin.get_conversation_history()
                return results

            except Exception as e:
                return {'student_id': student_id, 'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def run_single_dt_experiment(self, study_id: int, student_id: int, study_row: pd.Series) -> Dict:
        prompt_cols = _study_prompt_columns(study_row)
        attempt = 1
        last_result = None
        
        while attempt <= self.max_attempts and not self.shutdown_flag:
            result = await self._run_single_attempt(study_id, student_id, study_row)
            last_result = result
            if result is None: break
            if _record_is_completed(result, prompt_cols):
                return result
            attempt += 1

        if last_result is not None and 'error' not in last_result:
            last_result['error'] = f"Failed after {self.max_attempts} attempts"
        return last_result

    def _load_existing_results(self, study_id: int, group: str, study_row: pd.Series) -> Dict[int, bool]:
        study_dir = self.paths.results_dir / str(study_id)
        raw_filename = study_dir / f"{study_id}_{group}_raw.json"
        completed = {}
        if not raw_filename.exists(): return completed

        try:
            with open(raw_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list): data = []
            expected_cols = _study_prompt_columns(study_row)
            for rec in data:
                sid = rec.get("student_id")
                if sid is None: continue
                completed[int(sid)] = _record_is_completed(rec, expected_cols)
        except Exception:
            return {}
        return completed

    def _save_batch_results(self, study_id: int, group: str, results: List[Dict]):
        if not results: return
        study_dir = self.paths.results_dir / str(study_id)
        study_dir.mkdir(parents=True, exist_ok=True)
        raw_filename = study_dir / f"{study_id}_{group}_raw.json"
        existing = []
        if raw_filename.exists():
            try:
                with open(raw_filename, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list): existing = loaded
            except json.JSONDecodeError: existing = []
        
        new_ids = {r.get("student_id") for r in results}
        kept = [r for r in existing if r.get("student_id") not in new_ids]
        kept.extend(results)

        with open(raw_filename, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=2)

    async def run_study(self, study_id: int, group_name: str, student_ids: Optional[List[int]] = None):
        study_row_df = self.study_list[
            (self.study_list['study_id'] == study_id) &
            (self.study_list['group'] == group_name)
        ]
        if study_row_df.empty: return
        study_row = study_row_df.iloc[0]
        n_participants = int(study_row['N_participants'])
        
        if student_ids is None:
            student_ids = list(range(1, n_participants + 1))

        completed_students = self._load_existing_results(study_id, group_name, study_row)
        remaining_students = [sid for sid in student_ids if not completed_students.get(sid, False)]
        
        if not remaining_students: return

        batch_size = self.concurrent_twins
        all_results = []

        try:
            for i in range(0, len(remaining_students), batch_size):
                if self.shutdown_flag: break
                batch_students = remaining_students[i:i + batch_size]
                tasks = [
                    self.run_single_dt_experiment(study_id, student_id, study_row)
                    for student_id in batch_students
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                valid_results = []
                for result in batch_results:
                    if result and isinstance(result, dict):
                        valid_results.append(result)
                
                all_results.extend(valid_results)
                if valid_results:
                    self._save_batch_results(study_id, group_name, valid_results)
                
                if not self.shutdown_flag:
                    await asyncio.sleep(1)

        except KeyboardInterrupt:
            self.shutdown_flag = True
        finally:
            if self.pending_results:
                self._save_batch_results(study_id, group_name, self.pending_results)

    async def run_all_studies(self):
        for _, study_row in self.study_list.iterrows():
            if self.shutdown_flag: break
            await self.run_study(
                study_id=int(study_row['study_id']),
                group_name=str(study_row['group'])
            )

async def run_experiment(start_study_id: Optional[int] = None, concurrent_twins: int = 20, max_attempts: int = 3):
    experiment = DigitalTwinExperiment(concurrent_twins=concurrent_twins, max_attempts=max_attempts)
    
    def signal_handler(signum, frame):
        experiment.shutdown_flag = True

    try:
        signal.signal(signal.SIGINT, signal_handler)
    except Exception:
        pass

    try:
        if start_study_id is not None:
            study_rows = experiment.study_list[experiment.study_list['study_id'] == start_study_id]
            for _, study_row in study_rows.iterrows():
                if experiment.shutdown_flag: break
                await experiment.run_study(
                    study_id=int(study_row['study_id']),
                    group_name=str(study_row['group'])
                )
        else:
            await experiment.run_all_studies()
    except KeyboardInterrupt:
        experiment.shutdown_flag = True

def start_experiment_sync():
    try:
        asyncio.run(run_experiment(
            start_study_id=None,
            concurrent_twins=20,
            max_attempts=3
        ))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    start_experiment_sync()