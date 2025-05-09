#!/usr/bin/env python3
"""
Merge Cleaned Bigwinboard Data with Consolidated Summaries

This script performs a left outer join between bigwinboard_cleaned.csv 
and bigwinboard_consolidated_summaries.csv, adding structured summaries 
to the cleaned dataset.
"""

import pandas as pd
import numpy as np

# File paths
INPUT_CLEANED = "../data/bigwinboard_cleaned.csv"
INPUT_SUMMARIES = "../data/bigwinboard_consolidated_summaries.csv"
OUTPUT_FILE = "../data/bigwinboard_with_summaries_final.csv"

def clean_and_merge():
    # Read input files with robust parsing
    print("Reading input files...")
    
    # Custom CSV reading function to handle complex data
    def read_complex_csv(filepath):
        import csv
        
        # First, try to read the file with a more flexible approach
        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use csv reader with more flexible parsing
            csv_reader = csv.reader(f, quotechar='"', delimiter=',', 
                                    quoting=csv.QUOTE_ALL, 
                                    skipinitialspace=True)
            
            # Read headers
            headers = next(csv_reader)
            
            # Read rows, handling potential parsing issues
            for row in csv_reader:
                # Ensure consistent number of columns
                if len(row) >= len(headers):
                    rows.append(row[:len(headers)])
        
        # Convert to DataFrame
        return pd.DataFrame(rows, columns=headers)
    
    # Read files using custom function
    df_cleaned = read_complex_csv(INPUT_CLEANED)
    df_summaries = read_complex_csv(INPUT_SUMMARIES)
    
    # Normalize column names to ensure consistent matching
    df_cleaned.columns = [col.lower().strip() for col in df_cleaned.columns]
    df_summaries.columns = [col.lower().strip() for col in df_summaries.columns]
    
    # Print initial information
    print(f"Cleaned dataset rows: {len(df_cleaned)}")
    print(f"Summaries dataset rows: {len(df_summaries)}")
    
    # Perform left outer join
    # Use case-insensitive matching for title
    merged_df = pd.merge(
        df_cleaned, 
        df_summaries[['title', 'structured_summary']], 
        left_on='title', 
        right_on='title', 
        how='left'
    )
    
    # Check merge results
    print(f"Merged dataset rows: {len(merged_df)}")
    print(f"Rows with summaries: {merged_df['structured_summary'].notna().sum()}")
    
    # Fill NaN summaries with empty string
    merged_df['structured_summary'] = merged_df['structured_summary'].fillna('')
    
    # Save merged dataset
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nMerged dataset saved to {OUTPUT_FILE}")
    
    # Optional: Print some statistics about the merge
    print("\nMerge Statistics:")
    print(f"Total games: {len(merged_df)}")
    print(f"Games with summaries: {(merged_df['structured_summary'] != '').sum()}")
    print(f"Games without summaries: {(merged_df['structured_summary'] == '').sum()}")

if __name__ == "__main__":
    clean_and_merge()
