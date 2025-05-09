#!/usr/bin/env python3
"""
Merge Embeddings With Titles

This script consolidates existing embedding data and title mappings into a single
combined file for easier management. This is a one-time operation to streamline
the existing data structure.
"""

import pandas as pd
import numpy as np
import os
import logging
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('embedding_consolidation.log'),
        logging.StreamHandler()
    ]
)

# File paths
EMBEDDINGS_FILE = "../embeddings/game_summary_embeddings.csv"
TITLES_MAPPING_FILE = "../embeddings/embedding_titles_mapping.csv"
COMBINED_OUTPUT_FILE = "../embeddings/unified_game_embeddings.csv"

def merge_embeddings_with_titles():
    """
    Merge separate embeddings and titles files into a single unified file.
    """
    logging.info("Starting embeddings-titles consolidation process...")
    
    # Check if files exist
    if not os.path.exists(EMBEDDINGS_FILE):
        logging.error(f"Embeddings file not found: {EMBEDDINGS_FILE}")
        return False
        
    if not os.path.exists(TITLES_MAPPING_FILE):
        logging.error(f"Titles mapping file not found: {TITLES_MAPPING_FILE}")
        return False
    
    # Load the titles mapping
    try:
        titles_df = pd.read_csv(TITLES_MAPPING_FILE)
        logging.info(f"Loaded {len(titles_df)} titles from mapping file")
        
        # Make sure 'Title' column exists
        if 'Title' not in titles_df.columns:
            logging.error(f"No 'Title' column found in {TITLES_MAPPING_FILE}")
            return False
    except Exception as e:
        logging.error(f"Error loading titles mapping: {e}")
        return False
    
    # Load embeddings - these have no headers, just raw vectors
    try:
        embeddings_df = pd.read_csv(EMBEDDINGS_FILE, header=None)
        logging.info(f"Loaded {len(embeddings_df)} embeddings")
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return False
    
    # Check if dimensions match
    if len(embeddings_df) != len(titles_df):
        logging.warning(f"Number of embeddings ({len(embeddings_df)}) doesn't match titles ({len(titles_df)})")
        # Use minimum length to avoid index errors
        min_length = min(len(embeddings_df), len(titles_df))
        embeddings_df = embeddings_df.iloc[:min_length]
        titles_df = titles_df.iloc[:min_length]
        logging.info(f"Trimmed to {min_length} matching entries")
    
    # Create the combined dataset
    combined_df = pd.DataFrame()
    combined_df['title'] = titles_df['Title']
    
    # Convert embeddings to list format
    embeddings_list = embeddings_df.values.tolist()
    combined_df['embedding'] = embeddings_list
    
    # Save the combined file
    try:
        combined_df.to_csv(COMBINED_OUTPUT_FILE, index=False, quoting=csv.QUOTE_ALL)
        logging.info(f"Successfully saved {len(combined_df)} unified embeddings to {COMBINED_OUTPUT_FILE}")
        return True
    except Exception as e:
        logging.error(f"Error saving combined file: {e}")
        return False

def main():
    if merge_embeddings_with_titles():
        logging.info("Embeddings consolidation completed successfully")
        print(f"\nEmbeddings successfully merged into: {COMBINED_OUTPUT_FILE}")
        print("Next steps:")
        print("1. Update update_search_app.py to use the new unified file")
        print("2. Update generate_embeddings.py to output directly to the unified format")
    else:
        logging.error("Embeddings consolidation failed")
        print("\nFailed to merge embeddings. Check the log for details.")

if __name__ == "__main__":
    main()
