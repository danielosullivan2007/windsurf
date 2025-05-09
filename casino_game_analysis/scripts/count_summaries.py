#!/usr/bin/env python3
"""
Count the number of structured summaries in a CSV file with complex formatting.
"""
import pandas as pd
import csv

# Target file
FILE_PATH = "../data/bigwinboard_with_summaries_final.csv"

# Use robust CSV reading with custom parser
def count_structured_summaries(file_path):
    try:
        # Read with robust parsing options
        df = pd.read_csv(
            file_path, 
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            encoding='utf-8',
            low_memory=False
        )
        
        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Find the summary column
        summary_col = None
        for col in df.columns:
            if 'summary' in col.lower():
                summary_col = col
                break
        
        if not summary_col:
            print(f"No summary column found in {file_path}")
            return 0
        
        # Count non-empty summaries
        valid_summaries = df[df[summary_col].notna() & (df[summary_col].astype(str).str.len() > 5)]
        
        total_rows = len(df)
        valid_count = len(valid_summaries)
        
        print(f"Total rows in file: {total_rows}")
        print(f"Rows with valid structured summaries: {valid_count}")
        print(f"Percentage with summaries: {valid_count/total_rows*100:.2f}%")
        
        return valid_count
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0

if __name__ == "__main__":
    count_structured_summaries(FILE_PATH)
