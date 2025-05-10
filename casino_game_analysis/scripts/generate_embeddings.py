#!/usr/bin/env python3
"""
Generate embeddings for casino game summaries with robust, resumable functionality.
Supports incremental embedding generation and checkpointing.
"""

import os
import json
import pandas as pd
import numpy as np
import openai
import csv
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('../data/logs/embedding_generation.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Default file paths
DEFAULT_INPUT_FILE = "../data/bigwinboard_with_summaries_final.csv"
DEFAULT_EMBEDDINGS_FILE = "../embeddings/unified_game_embeddings.csv"  # Now using unified format
DEFAULT_LEGACY_EMBEDDINGS_FILE = "../embeddings/game_summary_embeddings.csv"  # For backward compatibility
DEFAULT_CHECKPOINT_FILE = "../embeddings/embedding_checkpoint.json"

def load_checkpoint(checkpoint_file):
    """Load existing checkpoint data."""
    try:
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'processed_titles': [],
            'embeddings': []
        }

def save_checkpoint(checkpoint_file, processed_titles, embeddings):
    """Save checkpoint data."""
    checkpoint_data = {
        'processed_titles': processed_titles,
        'embeddings': embeddings
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def generate_embedding(text):
    """Generate embedding for a text using OpenAI's API"""
    try:
        # Ensure text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            return None
            
        # Add context about casino games to improve embedding quality
        enhanced_text = f"Casino game summary: {text}"
        
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=enhanced_text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def generate_game_embeddings(input_file=DEFAULT_INPUT_FILE, embeddings_file=DEFAULT_EMBEDDINGS_FILE, 
                     checkpoint_file=DEFAULT_CHECKPOINT_FILE, save_legacy=True, limit=None):
    """Generate embeddings for casino game summaries with checkpointing."""
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_file)
    processed_titles = checkpoint.get('processed_titles', [])
    existing_embeddings = checkpoint.get('embeddings', [])
    
    # Read the CSV with summaries
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        return
    
    # Normalize column names
    df.columns = [col.lower() for col in df.columns]
    
    # Identify title and summary columns
    title_col = 'title' if 'title' in df.columns else [col for col in df.columns if 'title' in col.lower()][0]
    summary_col = 'structured_summary' if 'structured_summary' in df.columns else [col for col in df.columns if 'summary' in col.lower()][0]
    
    # Filter out games already processed and with valid summaries
    unprocessed_df = df[
        (~df[title_col].isin(processed_titles)) & 
        (df[summary_col].notna()) & 
        (df[summary_col].str.len() > 5)
    ]
    
    # Apply limit if specified
    if limit is not None and limit > 0 and len(unprocessed_df) > limit:
        logging.info(f"Limiting processing to {limit} games (out of {len(unprocessed_df)} available)")
        unprocessed_df = unprocessed_df.head(limit)
    
    logging.info(f"Processing {len(unprocessed_df)} new games")
    
    # Prepare for embedding generation
    new_embeddings = []
    new_processed_titles = []
    
    # Generate embeddings with progress bar
    for _, row in tqdm(unprocessed_df.iterrows(), total=len(unprocessed_df), desc="Generating Embeddings"):
        title = row[title_col]
        summary = row[summary_col]
        
        embedding = generate_embedding(summary)
        
        if embedding is not None:
            new_embeddings.append({
                'title': title,
                'embedding': embedding
            })
            new_processed_titles.append(title)
    
    # Combine with existing embeddings
    all_embeddings = existing_embeddings + new_embeddings
    all_processed_titles = processed_titles + new_processed_titles
    
    # Ensure the data structure is correct for the unified format
    # This ensures we have proper title and embedding columns
    clean_embeddings = []
    for emb in all_embeddings:
        if isinstance(emb, dict) and 'title' in emb and 'embedding' in emb:
            clean_embeddings.append(emb)
        elif isinstance(emb, list) and len(emb) >= 2:
            # Handle possible legacy format
            clean_embeddings.append({
                'title': emb[0],
                'embedding': emb[1] if len(emb) > 1 else []
            })
            
    logging.info(f"Processed {len(clean_embeddings)} embeddings with correct format")
            
    # Save embeddings in unified format
    embeddings_df = pd.DataFrame(clean_embeddings)
    
    # Double-check we have the expected columns
    if 'title' not in embeddings_df.columns or 'embedding' not in embeddings_df.columns:
        logging.error("Embeddings dataframe missing required columns after cleanup")
        return
    
    # Save in unified format (title and embedding together)
    try:
        embeddings_df.to_csv(embeddings_file, index=False, quoting=csv.QUOTE_ALL)
        logging.info(f"Saved {len(all_embeddings)} embeddings to unified format: {embeddings_file}")
    except Exception as e:
        logging.error(f"Error saving unified embeddings: {e}")
        return
    
    # If requested, also save in legacy format for backward compatibility
    if save_legacy:
        try:
            # Extract just the embeddings as raw vectors without headers
            embeddings_array = np.array([emb['embedding'] for emb in all_embeddings])
            legacy_file = DEFAULT_LEGACY_EMBEDDINGS_FILE
            np.savetxt(legacy_file, embeddings_array, delimiter=',')
            
            # Save titles mapping
            titles_df = pd.DataFrame({'Title': [emb['title'] for emb in all_embeddings]})
            titles_mapping_file = "../embeddings/embedding_titles_mapping.csv"
            titles_df.to_csv(titles_mapping_file, index=False)
            
            logging.info(f"Saved {len(all_embeddings)} embeddings in legacy format for backward compatibility")
        except Exception as e:
            logging.warning(f"Could not save in legacy format: {e}")
    
    # Save checkpoint
    save_checkpoint(checkpoint_file, all_processed_titles, all_embeddings)
    logging.info(f"Saved checkpoint with {len(all_processed_titles)} processed titles")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate embeddings for casino game summaries")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Input CSV file with summaries (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', type=str, default=DEFAULT_EMBEDDINGS_FILE,
                        help=f'Output embeddings file (default: {DEFAULT_EMBEDDINGS_FILE})')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_FILE,
                        help=f'Checkpoint file (default: {DEFAULT_CHECKPOINT_FILE})')
    parser.add_argument('--legacy', action='store_true', default=True,
                        help='Also save in legacy format for backward compatibility')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of new embeddings to generate (default: no limit)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Generate embeddings
    generate_game_embeddings(
        input_file=args.input,
        embeddings_file=args.output,
        checkpoint_file=args.checkpoint,
        save_legacy=args.legacy,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
