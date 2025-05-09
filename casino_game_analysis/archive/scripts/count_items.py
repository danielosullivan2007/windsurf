#!/usr/bin/env python3
"""
Count the number of valid summaries and embeddings in the system
"""

import pandas as pd
import json
import os

# File paths
summaries_file = "../data/bigwinboard_with_summaries.csv"
embeddings_file = "../embeddings/game_summary_embeddings.csv"
tsne_file = "../embeddings/game_summary_embeddings_tsne.json"

print("Analyzing Casino Game Analysis Files...")
print("-" * 50)

# Count summaries
try:
    df = pd.read_csv(summaries_file)
    print(f"Total rows in summaries file: {len(df)}")
    
    # Count valid summaries (non-empty structured_summary column)
    summary_column = None
    for col in df.columns:
        if 'summary' in col.lower():
            summary_column = col
            break
    
    if summary_column:
        valid_summaries = df[df[summary_column].notna() & (df[summary_column].str.len() > 20)]
        print(f"Valid summaries column found: '{summary_column}'")
        print(f"Number of valid summaries: {len(valid_summaries)}")
    else:
        print("No summary column found in the file")
except Exception as e:
    print(f"Error analyzing summaries file: {e}")

# Count embeddings
try:
    embeddings_df = pd.read_csv(embeddings_file)
    print(f"\nNumber of embeddings: {len(embeddings_df)}")
except Exception as e:
    print(f"\nError analyzing embeddings file: {e}")

# Check TSNE file for game details
try:
    if os.path.exists(tsne_file):
        with open(tsne_file, 'r') as f:
            tsne_data = json.load(f)
        print(f"Games in TSNE visualization: {len(tsne_data)}")
    else:
        print("TSNE file not found")
except Exception as e:
    print(f"Error analyzing TSNE file: {e}")

print("\nAnalysis complete!")
