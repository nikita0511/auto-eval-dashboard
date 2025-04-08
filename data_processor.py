import pandas as pd
import json
from pathlib import Path

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_data(self) -> pd.DataFrame:
        """Load and process evaluation data"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Add computed columns
        df['average_score'] = df[['accuracy', 'thoroughness', 'organization']].mean(axis=1)
        df['low_score_count'] = df[['accuracy', 'thoroughness', 'organization']].apply(
            lambda x: sum(x <= 2), axis=1
        )
        
        return df
    
    def get_summary_stats(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics"""
        return {
            'total_evaluations': len(df),
            'average_accuracy': df['accuracy'].mean(),
            'average_thoroughness': df['thoroughness'].mean(),
            'low_score_percentage': (df['low_score_count'] > 0).mean() * 100
        }