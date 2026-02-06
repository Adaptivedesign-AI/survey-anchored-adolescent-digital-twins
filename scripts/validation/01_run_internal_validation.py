"""
Internal Validation: Re-ask YRBS Questions
Tests whether DTs can consistently reproduce their original survey answers
Runs 3 DT types × 3 models × 1000 students × 107 questions
"""

import sys
from pathlib import Path

# Add src to path to import DigitalTwinChat
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "scripts" / "cohort_generation"))

from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging

from google.genai import types
import nest_asyncio

# Import DigitalTwinChat from cohort_generation
from cohort_generation.dt_chat_interface import DigitalTwinChat

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InternalValidation:
    """Conducts internal validation by re-asking YRBS questions to DTs"""
    
    DT_TYPES = ['survey_memory_dt', 'survey_dt', 'base_dt']
    MODELS = ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']
    
    def __init__(self, concurrency: int = 20):
        self.project_root = Path(__file__).parent.parent.parent
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        
        self._setup_paths()
        self._load_questions()
        
        # Initialize DT chat interface (reuses all the prompt building logic)
        self.dt_chat = DigitalTwinChat()
        self.client = self.dt_chat.client
    
    def _setup_paths(self):
        """Setup paths for inputs and outputs"""
        self.data_dir = self.project_root / "data"
        self.results_dir = self.project_root / "results" / "validation" / "internal"
        
        # Input files
        self.questions_file = self.data_dir / "raw" / "yrbs" / "yrbs_questions.json"
    
    def _load_questions(self):
        """Load YRBS questions"""
        logger.info("Loading YRBS questions...")
        
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
            self.questions = questions_data if isinstance(questions_data, list) else questions_data.get('questions', [])
        
        logger.info(f"✓ Loaded {len(self.questions)} YRBS questions")
    
    async def _ask_single_question(
        self, 
        student_id: int,
        dt_type: str,
        model: str,
        question: Dict
    ) -> Dict:
        """
        Ask a single question to a DT
        Uses DigitalTwinChat's system prompt building
        """
        async with self.semaphore:
            try:
                # Build system prompt using DigitalTwinChat's method
                system_prompt = self.dt_chat._build_system_prompt(student_id, dt_type)
                
                # Add survey instructions
                survey_instruction = (
                    "\n\n=== Survey Instructions ===\n"
                    "Please answer the following survey question honestly as yourself.\n"
                    "For multiple choice questions, provide your answer in format: Answer: X\n"
                    "For numerical questions, provide the number: Answer: 1.75\n"
                    "Be clear and direct.\n\n"
                )
                
                full_prompt = f"{system_prompt}{survey_instruction}Question: {question['text']}\n\nYour answer:"
                
                # Call Gemini API
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=500
                        )
                    ),
                    timeout=30.0
                )
                
                answer = response.text.strip() if hasattr(response, 'text') else ''
                
                return {
                    'student_id': student_id,
                    'question_id': question['id'],
                    'question_text': question['text'],
                    'response': answer,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout: student {student_id}, Q{question['id']}")
                return {
                    'student_id': student_id,
                    'question_id': question['id'],
                    'question_text': question['text'],
                    'response': '',
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': 'timeout'
                }
            
            except Exception as e:
                logger.error(f"Error: student {student_id}, Q{question['id']}: {e}")
                return {
                    'student_id': student_id,
                    'question_id': question['id'],
                    'question_text': question['text'],
                    'response': '',
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': str(e)
                }
    
    async def _validate_student(
        self, 
        student_id: int, 
        dt_type: str, 
        model: str
    ) -> List[Dict]:
        """Validate one student by asking all 107 questions"""
        tasks = [
            self._ask_single_question(student_id, dt_type, model, question)
            for question in self.questions
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # Small delay between students
        await asyncio.sleep(0.1)
        
        return responses
    
    def _get_output_file(self, dt_type: str, model: str) -> Path:
        """Get output file path for a specific DT type and model"""
        output_dir = self.results_dir / model / dt_type
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "responses.json"
    
    def _load_existing_results(self, output_file: Path) -> Dict:
        """Load existing results if file exists"""
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing results: {e}")
        return {}
    
    def _save_results(self, output_file: Path, results: Dict):
        """Save results to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    async def run_validation(
        self, 
        dt_type: str, 
        model: str, 
        student_ids: Optional[List[int]] = None
    ):
        """
        Run validation for a specific DT type and model
        
        Args:
            dt_type: Type of DT ('survey_memory_dt', 'survey_dt', 'base_dt')
            model: Gemini model to use
            student_ids: List of student IDs to validate (None = all 1000)
        """
        if student_ids is None:
            student_ids = list(range(1, 1001))
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running internal validation:")
        logger.info(f"  DT Type: {dt_type}")
        logger.info(f"  Model: {model}")
        logger.info(f"  Students: {len(student_ids)}")
        logger.info(f"  Questions per student: {len(self.questions)}")
        logger.info(f"  Total questions: {len(student_ids) * len(self.questions)}")
        logger.info(f"  Concurrency: {self.concurrency}")
        logger.info(f"{'='*60}\n")
        
        output_file = self._get_output_file(dt_type, model)
        
        # Load existing results for resume capability
        all_results = self._load_existing_results(output_file)
        
        # Determine which students still need processing
        completed_students = set(all_results.keys())
        remaining_students = [
            sid for sid in student_ids 
            if str(sid) not in completed_students
        ]
        
        if not remaining_students:
            logger.info("✓ All students already completed. Skipping.")
            return
        
        logger.info(f"Processing {len(remaining_students)} remaining students...")
        logger.info(f"(Already completed: {len(completed_students)} students)\n")
        
        # Process in batches of 50 students
        batch_size = 50
        total_batches = (len(remaining_students) - 1) // batch_size + 1
        
        for i in range(0, len(remaining_students), batch_size):
            batch = remaining_students[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Batch {batch_num}/{total_batches}: Students {batch[0]}-{batch[-1]}")
            
            # Create tasks for this batch
            tasks = [
                self._validate_student(student_id, dt_type, model)
                for student_id in batch
            ]
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks)
            
            # Store results
            for student_id, responses in zip(batch, batch_results):
                all_results[str(student_id)] = responses
            
            # Save after each batch (checkpoint)
            self._save_results(output_file, all_results)
            
            # Log batch statistics
            successful = sum(
                1 for r in batch_results 
                if all(resp['success'] for resp in r)
            )
            total_questions = sum(len(r) for r in batch_results)
            successful_questions = sum(
                sum(1 for resp in r if resp['success']) 
                for r in batch_results
            )
            
            logger.info(f"  ✓ Students: {successful}/{len(batch)} fully completed")
            logger.info(f"  ✓ Questions: {successful_questions}/{total_questions} successful")
            logger.info(f"  ✓ Checkpoint saved\n")
            
            await asyncio.sleep(1)
        
        logger.info(f"{'='*60}")
        logger.info(f"✓ Validation complete for {dt_type} + {model}")
        logger.info(f"  Results saved to: {output_file}")
        logger.info(f"  Total students: {len(all_results)}")
        logger.info(f"{'='*60}\n")
    
    async def run_all_validations(self, student_ids: Optional[List[int]] = None):
        """Run validation for all combinations of DT types and models"""
        total_runs = len(self.DT_TYPES) * len(self.MODELS)
        current_run = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"INTERNAL VALIDATION - ALL CONFIGURATIONS")
        logger.info(f"Total configurations: {total_runs}")
        logger.info(f"Students per config: {1000 if student_ids is None else len(student_ids)}")
        logger.info(f"Questions per student: {len(self.questions)}")
        logger.info(f"{'='*60}\n")
        
        for model in self.MODELS:
            for dt_type in self.DT_TYPES:
                current_run += 1
                logger.info(f"\n>>> Configuration {current_run}/{total_runs}: {model} + {dt_type}")
                
                await self.run_validation(dt_type, model, student_ids)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ ALL VALIDATIONS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"\nResults saved in: {self.results_dir}")
        logger.info(f"Structure:")
        logger.info(f"  {self.results_dir}/")
        for model in self.MODELS:
            logger.info(f"    ├── {model}/")
            for dt_type in self.DT_TYPES:
                logger.info(f"    │   ├── {dt_type}/")
                logger.info(f"    │   │   └── responses.json")


async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run internal validation for Digital Twins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configurations (3 DT types × 3 models)
  python 01_run_internal_validation.py
  
  # Run specific DT type with all models
  python 01_run_internal_validation.py --dt_type survey_memory_dt
  
  # Run specific model with all DT types
  python 01_run_internal_validation.py --model gemini-2.5-flash
  
  # Run specific configuration
  python 01_run_internal_validation.py --dt_type survey_dt --model gemini-2.0-flash
  
  # Test with first 10 students only
  python 01_run_internal_validation.py --students 1 2 3 4 5 6 7 8 9 10
  
  # Adjust concurrency
  python 01_run_internal_validation.py --concurrency 30
        """
    )
    
    parser.add_argument(
        '--dt_type', 
        type=str, 
        choices=InternalValidation.DT_TYPES,
        help='Specific DT type to run (default: all)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        choices=InternalValidation.MODELS,
        help='Specific model to run (default: all)'
    )
    parser.add_argument(
        '--students', 
        type=int, 
        nargs='+',
        help='Specific student IDs to process (default: all 1000)'
    )
    parser.add_argument(
        '--concurrency', 
        type=int, 
        default=20,
        help='Number of concurrent requests (default: 20)'
    )
    
    args = parser.parse_args()
    
    validator = InternalValidation(concurrency=args.concurrency)
    
    if args.dt_type and args.model:
        # Run single configuration
        await validator.run_validation(args.dt_type, args.model, args.students)
    elif args.dt_type:
        # Run all models for one DT type
        for model in validator.MODELS:
            await validator.run_validation(args.dt_type, model, args.students)
    elif args.model:
        # Run all DT types for one model
        for dt_type in validator.DT_TYPES:
            await validator.run_validation(dt_type, args.model, args.students)
    else:
        # Run all combinations
        await validator.run_all_validations(args.students)


if __name__ == "__main__":
    asyncio.run(main())