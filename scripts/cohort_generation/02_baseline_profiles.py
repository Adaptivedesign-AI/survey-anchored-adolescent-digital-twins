"""
Generate Baseline DT Profiles from YRBS Survey Responses
Converts survey answers to natural language prompts using mapping table
"""

from pathlib import Path
import pandas as pd
import json


class BaselineProfileGenerator:
    """Generates baseline DT profiles from YRBS survey responses"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self._setup_paths()
        self._load_data()
    
    def _setup_paths(self):
        """Setup input and output paths"""
        # Input files
        self.survey_file = self.project_root / "data" / "processed" / "cohort" / "sampled_1000.csv"
        self.mapping_file = self.project_root / "prompts" / "mapping" / "yrbs_to_profile_map.json"
        
        # Output file
        self.output_file = self.project_root / "data" / "processed" / "cohort" / "baseline_profiles_1000.json"
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_data(self):
        """Load survey responses and mapping table"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load survey responses
        print(f"\nLoading survey responses: {self.survey_file}")
        self.survey_df = pd.read_csv(self.survey_file)
        print(f"✓ Loaded {len(self.survey_df)} student responses")
        
        # Load mapping table
        print(f"\nLoading mapping table: {self.mapping_file}")
        
        # Check if mapping file exists
        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Mapping file not found: {self.mapping_file}\n"
                f"Please place the YRBS-to-profile mapping JSON in: {self.mapping_file.parent}"
            )
        
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.mapping_data = json.load(f)
        
        # Build prompt dictionary: {question -> {code -> prompt_fragment}}
        self.prompt_dict = {}
        for item in self.mapping_data:
            question = item['question']
            code = str(item['code']).strip()
            fragment = item['prompt_fragment']
            
            if code.lower() in ['nan', '', 'none']:
                continue
            
            if question not in self.prompt_dict:
                self.prompt_dict[question] = {}
            
            self.prompt_dict[question][code] = fragment
        
        print(f"✓ Loaded mapping for {len(self.prompt_dict)} questions")
    
    def _clean_code(self, val) -> str:
        """Clean and normalize answer code"""
        if pd.isna(val):
            return None
        
        val_str = str(val).strip()
        
        # Remove trailing .0 for integer values
        if val_str.endswith('.0'):
            val_str = val_str[:-2]
        
        return val_str
    
    def _generate_single_prompt(self, row: pd.Series, drop_questions: list = None) -> str:
        """
        Generate prompt for a single student based on their survey answers
        
        Special handling for:
        - Q6: Height (continuous value)
        - Q7: Weight (continuous value)
        """
        if drop_questions is None:
            drop_questions = []
        
        # Get all question columns (Q1 to Q107)
        q_columns = [f"Q{i}" for i in range(1, 108) if f"Q{i}" not in drop_questions]
        
        fragments = []
        
        for q in q_columns:
            code = self._clean_code(row.get(q))
            
            if code is None:
                continue
            
            # Special handling for height (Q6)
            if q == "Q6":
                try:
                    height = float(code)
                    fragments.append(f"Your height without shoes is {height:.2f} meters.")
                except (ValueError, TypeError):
                    continue
            
            # Special handling for weight (Q7)
            elif q == "Q7":
                try:
                    weight = float(code)
                    fragments.append(f"Your weight without shoes is {weight:.0f} kilograms.")
                except (ValueError, TypeError):
                    continue
            
            # All other questions use mapping table
            else:
                fragment = self.prompt_dict.get(q, {}).get(code)
                if isinstance(fragment, str) and fragment.strip():
                    fragments.append(fragment)
        
        return " ".join(fragments)
    
    def generate_all_profiles(self, drop_questions: list = None) -> list:
        """
        Generate baseline profiles for all students
        Returns list of dicts with student_id and prompt
        """
        print("\n" + "="*60)
        print("GENERATING BASELINE PROFILES")
        print("="*60)
        
        if drop_questions:
            print(f"\nDropping questions: {', '.join(drop_questions)}")
        
        profiles = []
        
        for idx, row in self.survey_df.iterrows():
            student_id = int(row['student_id'])
            prompt = self._generate_single_prompt(row, drop_questions)
            
            profiles.append({
                'student_id': student_id,
                'baseline_prompt': prompt
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(self.survey_df)} students...")
        
        print(f"\n✓ Generated {len(profiles)} baseline profiles")
        
        return profiles
    
    def save_profiles(self, profiles: list):
        """Save profiles to JSON file"""
        print(f"\nSaving profiles to: {self.output_file}")
        
        output_data = {
            'metadata': {
                'num_students': len(profiles),
                'source_survey': 'YRBS',
                'description': 'Baseline DT profiles generated from YRBS survey responses'
            },
            'profiles': profiles
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved to: {self.output_file}")
    
    def run(self, drop_questions: list = None):
        """Execute the complete profile generation pipeline"""
        # Generate profiles
        profiles = self.generate_all_profiles(drop_questions)
        
        # Save to file
        self.save_profiles(profiles)
        
        # Print summary
        print("\n" + "="*60)
        print("PROFILE GENERATION COMPLETE")
        print("="*60)
        print(f"Generated baseline profiles for {len(profiles)} students")
        print(f"Each profile contains personalized information from YRBS survey")
        
        # Show example
        if profiles:
            print(f"\n--- Example Profile (Student {profiles[0]['student_id']}) ---")
            prompt_preview = profiles[0]['baseline_prompt'][:300]
            print(f"{prompt_preview}...")
        
        return profiles


def main():
    """Main execution function"""
    generator = BaselineProfileGenerator()
    generator.run()


if __name__ == "__main__":
    main()