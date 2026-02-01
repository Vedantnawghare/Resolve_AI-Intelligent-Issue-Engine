"""
STAGE 2 (Part 2): Dataset Preprocessing Integration
Apply NLP pipeline to entire dataset
"""

import pandas as pd
from pathlib import Path
from modules.nlp_preprocessing import NLPPreprocessor
from config import DATASET_PATH, PROCESSED_DATA_PATH



class DataProcessor:
    """Process entire dataset with NLP pipeline"""
    
    def __init__(self):
        self.preprocessor = NLPPreprocessor()
    
    def process_dataset(self, input_path: str = None, output_path: str = None):
        """
        Load dataset, apply NLP pipeline, save processed version
        
        Why this is necessary:
        - ML models need clean, standardized text
        - We process once, use many times (efficiency)
        - Preserves original data for auditing
        """
        # Use default paths if not provided
        input_path = input_path or DATASET_PATH
        output_path = output_path or PROCESSED_DATA_PATH
        
        print(f"Loading dataset from: {input_path}")
        df = pd.read_csv(input_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Processing {len(df)} issue texts...")
        
        # Apply preprocessing to issue_text column
        processed_results = self.preprocessor.batch_preprocess(df['issue_text'].tolist())
        
        # Add processed columns to original dataframe
        df['original_text'] = processed_results['original_text']
        df['cleaned_text'] = processed_results['cleaned_text']
        df['detected_language'] = processed_results['detected_language']
        df['token_count'] = processed_results['token_count']
        
        # Save processed dataset
        df.to_csv(output_path, index=False)
        
        print(f"Processed dataset saved to: {output_path}")
        
        # Show statistics
        print(f"\nProcessing Statistics:")
        print(f"   Language Distribution:")
        print(df['detected_language'].value_counts())
        print(f"\n   Average tokens per issue: {df['token_count'].mean():.2f}")
        print(f"   Min tokens: {df['token_count'].min()}")
        print(f"   Max tokens: {df['token_count'].max()}")
        
        # Show sample transformations
        print(f"\nSample Transformations:")
        sample = df[['issue_text', 'cleaned_text', 'detected_language']].head(10)
        for idx, row in sample.iterrows():
            print(f"\n   Original: {row['issue_text']}")
            print(f"   Cleaned:  {row['cleaned_text']}")
            print(f"   Language: {row['detected_language']}")
        
        return df

def main():
    """Process the dataset"""
    processor = DataProcessor()
    processor.process_dataset()

if __name__ == "__main__":
    main()
