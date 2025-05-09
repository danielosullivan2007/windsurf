#!/usr/bin/env python3
"""
Consolidate Summaries Script

This script combines summaries from multiple CSV files into a single consolidated file
that will be used for generating embeddings.
"""

import pandas as pd
import os
import sys

# File paths
DATA_DIR = "../data"
OUTPUT_FILE = "../data/bigwinboard_consolidated_summaries.csv"

# Files that may contain summaries
summary_files = [
    "bigwinboard_name_summary.csv",
    "bigwinboard_name_summary_cleaned.csv",
    "bigwinboard_with_summaries.csv",
    "bigwinboard_with_summaries_cleaned.csv",
    "bigwinboard_with_summaries_complete.csv"
]

def main():
    print("Consolidating summaries from multiple files...")
    
    # Dictionary to track games by title and their summaries
    games_dict = {}
    
    # Track statistics
    stats = {file: {"total": 0, "with_summary": 0} for file in summary_files}
    
    # Process each file
    for filename in summary_files:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping.")
            continue
            
        print(f"Processing {filename}...")
        
        try:
            # Detect the correct column names by checking a few rows
            df = pd.read_csv(filepath, nrows=5)
            
            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Identify possible title and summary columns
            title_col = next((col for col in df.columns if col in ['title', 'game_name']), None)
            summary_col = next((col for col in df.columns if col in ['summary', 'structured_summary']), None)
            
            if not title_col or not summary_col:
                print(f"Warning: Could not identify title or summary columns in {filename}")
                print(f"Available columns: {list(df.columns)}")
                continue
                
            # Now read the full file
            df = pd.read_csv(filepath)
            df.columns = [col.lower() for col in df.columns]
            
            # Track statistics
            stats[filename]["total"] = len(df)
            stats[filename]["with_summary"] = df[df[summary_col].notna() & (df[summary_col] != "")].shape[0]
            
            # Process each row and add to dictionary if it has a summary
            for _, row in df.iterrows():
                title = row[title_col]
                summary = row[summary_col]
                
                # Skip if title is missing or summary is empty
                if pd.isna(title) or str(title).strip() == "":
                    continue
                    
                if not pd.isna(summary) and str(summary).strip() != "":
                    # Only add/update if the game doesn't exist or the current summary is empty
                    if title not in games_dict or not games_dict[title].get('structured_summary'):
                        # Create a standardized game entry
                        game_entry = {
                            'title': title,
                            'structured_summary': summary
                        }
                        
                        # Add other columns if available (like review, provider, etc.)
                        for col in df.columns:
                            if col not in [title_col, summary_col] and col not in game_entry:
                                game_entry[col] = row[col]
                                
                        games_dict[title] = game_entry
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Create final dataframe from dictionary values
    consolidated_df = pd.DataFrame(list(games_dict.values()))
    
    # Ensure we have the required columns for embedding generation
    if 'title' not in consolidated_df.columns or 'structured_summary' not in consolidated_df.columns:
        print("Error: Consolidated data is missing required columns (title, structured_summary)")
        sys.exit(1)
        
    # Save to CSV
    consolidated_df.to_csv(OUTPUT_FILE, index=False)
    
    # Print statistics
    print("\nSummary Statistics:")
    print(f"{'File':<40} {'Total Rows':<15} {'With Summaries':<15}")
    print("-" * 70)
    for file, counts in stats.items():
        print(f"{file:<40} {counts['total']:<15} {counts['with_summary']:<15}")
    
    print("\nConsolidation Results:")
    print(f"Total unique games with summaries: {len(consolidated_df)}")
    print(f"Consolidated file saved to: {OUTPUT_FILE}")
    
if __name__ == "__main__":
    main()
