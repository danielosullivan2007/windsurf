#!/usr/bin/env python3
import csv
import os
import pandas as pd
import numpy as np

# File paths from pipeline
original_reviews_path = "../data/bigwinboard_with_summaries_final.csv"
summaries_path = "../data/bigwinboard_with_summaries_final.csv"  # Same file contains the summaries
embeddings_path = "../embeddings/game_summary_embeddings.csv"
unified_embeddings_path = "../embeddings/unified_game_embeddings.csv"
titles_mapping_path = "../embeddings/embedding_titles_mapping.csv"

# Function to safely load a CSV file to pandas DataFrame
def load_csv(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: File not found - {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

# Load the main data files
print("\n=== Casino Game Analysis Pipeline Counts ===")
print("Loading data files...")

# Load reviews/summaries file
reviews_df = load_csv(original_reviews_path)
if reviews_df is not None:
    total_reviews = len(reviews_df)
    print(f"\nOriginal game entries: {total_reviews:,}")
    
    # Check for title column
    title_col = None
    for col in reviews_df.columns:
        if 'title' in col.lower():
            title_col = col
            break
    
    # Check for summary columns
    summary_cols = []
    for col in reviews_df.columns:
        if 'summary' in col.lower():
            summary_cols.append(col)
    
    # Check for structured summaries specifically
    structured_summary_col = None
    for col in summary_cols:
        if 'structured' in col.lower():
            structured_summary_col = col
            break
    
    # Count non-empty summaries
    if structured_summary_col:
        valid_summaries = reviews_df[reviews_df[structured_summary_col].notna() & 
                                    (reviews_df[structured_summary_col].str.len() > 10)]
        print(f"Games with structured summaries: {len(valid_summaries):,} ({len(valid_summaries)/total_reviews*100:.1f}%)")
    elif summary_cols:
        for col in summary_cols:
            valid_summaries = reviews_df[reviews_df[col].notna() & 
                                        (reviews_df[col].str.len() > 10)]
            print(f"Games with {col}: {len(valid_summaries):,} ({len(valid_summaries)/total_reviews*100:.1f}%)")
    else:
        print("No summary columns found in the data file")
else:
    print("Could not analyze original reviews")

# Check embeddings file
embeddings_exist = os.path.exists(embeddings_path)
if embeddings_exist:
    try:
        # Load as numpy array (legacy format is just vectors with no headers)
        embeddings = np.loadtxt(embeddings_path, delimiter=',')
        embeddings_count = embeddings.shape[0] if len(embeddings.shape) > 1 else 1
        print(f"\nGame summary embeddings: {embeddings_count:,}")
        
        # Check if we have title mappings for these embeddings
        if os.path.exists(titles_mapping_path):
            titles_df = load_csv(titles_mapping_path)
            if titles_df is not None:
                print(f"Games with titles mapped to embeddings: {len(titles_df):,}")
                
    except Exception as e:
        # If load as numpy fails, try pandas
        try:
            embeddings_df = load_csv(embeddings_path)
            if embeddings_df is not None:
                print(f"Game summary embeddings: {len(embeddings_df):,}")
        except Exception as e2:
            print(f"Error analyzing embeddings: {e2}")
else:
    print("\nGame summary embeddings file not found")

# Check unified embeddings
if os.path.exists(unified_embeddings_path):
    unified_df = load_csv(unified_embeddings_path)
    if unified_df is not None:
        print(f"\nUnified embeddings: {len(unified_df):,}")
        
        # Check for title and embedding columns
        if 'title' in unified_df.columns and 'embedding' in unified_df.columns:
            print("  (Contains both title and embedding vectors in single file)")
        elif 'title' in unified_df.columns:
            print("  (Contains titles but no embedding vectors)")
        elif 'embedding' in unified_df.columns:
            print("  (Contains embedding vectors but no titles)")
else:
    print("\nUnified embeddings file not found")

print("\n==== Summary ====")
print(f"1. Original game entries: {total_reviews if 'total_reviews' in locals() else 'N/A'}")
print(f"2. Games with structured summaries: {len(valid_summaries) if 'valid_summaries' in locals() else 'N/A'}")
print(f"3. Games with embeddings: {embeddings_count if 'embeddings_count' in locals() else 'N/A'}")
print(f"4. Games with unified embeddings: {len(unified_df) if 'unified_df' in locals() and unified_df is not None else 'N/A'}")
print("===============================================")
