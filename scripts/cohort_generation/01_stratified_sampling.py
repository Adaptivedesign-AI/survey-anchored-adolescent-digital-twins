"""
Stratified Sampling from YRBS Dataset
Samples 1000 students using stratified sampling based on demographic variables
"""

from pathlib import Path
import pandas as pd
import numpy as np


class YRBSSampler:
    """Handles stratified sampling from YRBS dataset"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup input and output paths"""
        self.input_file = self.project_root / "data" / "raw" / "yrbs" / "yrbs_original.csv"
        self.output_file = self.project_root / "data" / "processed" / "cohort" / "sampled_1000.csv"
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def preprocess_q5(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Q5 column - group non-ABCE values into 'Other'
        Q5 represents race/ethnicity categories
        """
        df = df.copy()
        valid_categories = ['A', 'B', 'C', 'E']
        df['Q5_processed'] = df['Q5'].apply(
            lambda x: x if x in valid_categories else 'Other'
        )
        return df
    
    def stratified_sample(
        self, 
        df: pd.DataFrame, 
        columns: list, 
        sample_size: int, 
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Perform stratified sampling based on multiple columns
        Ensures sample distribution matches population distribution
        """
        df_work = df.copy().reset_index(drop=True)
        
        # Create combined strata variable
        df_work['strata'] = df_work[columns].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        # Calculate proportions for each stratum
        strata_counts = df_work['strata'].value_counts()
        strata_proportions = strata_counts / len(df_work)
        
        # Calculate target sample size for each stratum
        target_counts = (strata_proportions * sample_size).round().astype(int)
        
        # Adjust to ensure exact sample_size
        diff = sample_size - target_counts.sum()
        if diff > 0:
            largest_strata = target_counts.nlargest(diff).index
            for stratum in largest_strata:
                target_counts[stratum] += 1
        elif diff < 0:
            largest_strata = target_counts.nlargest(-diff).index
            for stratum in largest_strata:
                target_counts[stratum] = max(0, target_counts[stratum] - 1)
        
        # Perform stratified sampling
        sampled_data = []
        np.random.seed(random_state)
        
        for stratum, target_count in target_counts.items():
            if target_count > 0:
                stratum_data = df_work[df_work['strata'] == stratum].copy()
                if len(stratum_data) >= target_count:
                    sampled = stratum_data.sample(n=target_count, random_state=random_state)
                else:
                    sampled = stratum_data
                sampled_data.append(sampled)
        
        if sampled_data:
            result = pd.concat(sampled_data, ignore_index=True)
            result = result.drop('strata', axis=1)
            return result
        else:
            return pd.DataFrame()
    
    def add_student_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sequential student IDs (1 to N)"""
        df_with_id = df.copy()
        df_with_id.insert(0, 'student_id', range(1, len(df) + 1))
        return df_with_id
    
    def print_distribution_comparison(
        self, 
        original_df: pd.DataFrame, 
        sample_df: pd.DataFrame, 
        columns: list
    ):
        """Print distribution comparison between original and sample"""
        print("\n" + "="*60)
        print("DISTRIBUTION COMPARISON")
        print("="*60)
        
        for col in columns:
            print(f"\n{col}:")
            print("-" * 40)
            
            print("Original distribution:")
            orig_dist = original_df[col].value_counts(normalize=True).sort_index()
            for cat, prop in orig_dist.items():
                print(f"  {cat}: {prop:.4f}")
            
            print("\nSample distribution:")
            sample_dist = sample_df[col].value_counts(normalize=True).sort_index()
            for cat, prop in sample_dist.items():
                print(f"  {cat}: {prop:.4f}")
            
            # Calculate L1 distance
            common_categories = set(orig_dist.index) & set(sample_dist.index)
            total_diff = sum(
                abs(orig_dist[cat] - sample_dist[cat]) 
                for cat in common_categories
            )
            print(f"\nL1 Distance: {total_diff:.4f}")
    
    def run(self, sample_size: int = 1000, random_state: int = 42):
        """Execute the complete sampling pipeline"""
        print("\n" + "="*60)
        print("YRBS STRATIFIED SAMPLING")
        print("="*60)
        
        # Load data
        print(f"\nLoading data from: {self.input_file}")
        df = pd.read_csv(self.input_file)
        print(f"Original dataset size: {len(df)} rows, {len(df.columns)} columns")
        
        # Process Q5
        df = self.preprocess_q5(df)
        
        # Define stratification columns
        # Q1: Age, Q2: Sex, Q5: Race/Ethnicity, Q24: Sexual Orientation, Q84: Gender Identity
        stratification_columns = ['Q1', 'Q2', 'Q5_processed', 'Q24', 'Q84']
        
        print(f"\nStratification variables: {', '.join(stratification_columns)}")
        
        # Check for required columns
        missing_cols = [col for col in stratification_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing values in stratification columns
        df_clean = df.dropna(subset=stratification_columns).copy()
        print(f"After removing missing values: {len(df_clean)} rows")
        
        # Perform stratified sampling
        print(f"\nPerforming stratified sampling (n={sample_size})...")
        sample_df = self.stratified_sample(
            df_clean, 
            stratification_columns, 
            sample_size, 
            random_state
        )
        
        # Add student IDs
        sample_df = self.add_student_ids(sample_df)
        
        print(f"Sample size: {len(sample_df)} students")
        print(f"Number of columns: {len(sample_df.columns)}")
        
        # Save results
        sample_df.to_csv(self.output_file, index=False)
        print(f"\n✓ Saved to: {self.output_file}")
        
        # Print distribution comparison
        self.print_distribution_comparison(df_clean, sample_df, stratification_columns)
        
        print("\n" + "="*60)
        print("SAMPLING COMPLETE")
        print("="*60)
        print(f"Generated {sample_size} Digital Twins with all {len(sample_df.columns)} YRBS questions")
        print(f"Each DT has a unique student_id (1-{sample_size})")
        
        return sample_df


def main():
    """Main execution function"""
    sampler = YRBSSampler()
    sampler.run(sample_size=1000, random_state=42)


if __name__ == "__main__":
    main()