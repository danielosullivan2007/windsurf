#!/usr/bin/env python3
"""
Update Search Index Script for Casino Game Analysis App

This script maintains the search index by:
1. Detecting new summaries that don't have embeddings yet
2. Generating embeddings only for those new summaries
3. Merging them with existing embeddings
4. Updating the search app's index to make new games searchable

Run this script periodically after generating new summaries.
"""

import os
import json
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import logging
import subprocess
import time
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('search_index_updates.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# File paths
SUMMARIES_FILE = "../data/bigwinboard_with_summaries.csv"
EMBEDDINGS_FILE = "../embeddings/game_summary_embeddings.csv"
INDEX_STATE_FILE = "../embeddings/search_index_state.json"
EMBEDDINGS_TSNE_FILE = "../embeddings/game_summary_embeddings_tsne.json"

def load_api_key():
    """Load OpenAI API key from .env file"""
    # Try different locations for .env file
    for env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(env_path):
            logging.info(f"API key found in {env_path}")
            load_dotenv(env_path)
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}"
                logging.info(f"API key loaded: {masked_key}")
                return api_key
    
    logging.error("No API key found in .env files")
    return None

def generate_embedding(text):
    """Generate embedding for a text using OpenAI's API"""
    if not isinstance(text, str) or not text.strip():
        return None
        
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        time.sleep(1)  # Rate limiting backup
        return None

def load_summaries():
    """Load the summaries from CSV file"""
    try:
        df = pd.read_csv(SUMMARIES_FILE)
        logging.info(f"Loaded {len(df)} summaries from {SUMMARIES_FILE}")
        
        # Normalize column names
        if 'Title' in df.columns:
            df.rename(columns={'Title': 'title'}, inplace=True)
        if 'structured_summary' in df.columns:
            df.rename(columns={'structured_summary': 'summary'}, inplace=True)
        elif 'summary' not in df.columns:
            # Try to find the summary column
            for col in df.columns:
                if 'summary' in col.lower():
                    df.rename(columns={col: 'summary'}, inplace=True)
                    break
                    
        return df
    except Exception as e:
        logging.error(f"Error loading summaries: {e}")
        return pd.DataFrame()

def load_existing_embeddings():
    """Load existing embeddings"""
    if not os.path.exists(EMBEDDINGS_FILE):
        logging.warning(f"Embeddings file {EMBEDDINGS_FILE} not found")
        return pd.DataFrame(), set()
        
    try:
        embeddings_df = pd.read_csv(EMBEDDINGS_FILE)
        logging.info(f"Loaded {len(embeddings_df)} embeddings from {EMBEDDINGS_FILE}")
        
        # Load the state file to get titles with embeddings
        if os.path.exists(INDEX_STATE_FILE):
            with open(INDEX_STATE_FILE, 'r') as f:
                state = json.load(f)
                titles_with_embeddings = set(state.get('processed_titles', []))
                logging.info(f"Found {len(titles_with_embeddings)} titles with embeddings in state file")
        else:
            # If no state file, try to get titles from TSNE file
            if os.path.exists(EMBEDDINGS_TSNE_FILE):
                with open(EMBEDDINGS_TSNE_FILE, 'r') as f:
                    tsne_data = json.load(f)
                    titles_with_embeddings = set(item['title'] for item in tsne_data)
                    logging.info(f"Found {len(titles_with_embeddings)} titles with embeddings in TSNE file")
            else:
                titles_with_embeddings = set()
                logging.warning("No state file or TSNE file found to determine which titles have embeddings")
                
        return embeddings_df, titles_with_embeddings
    except Exception as e:
        logging.error(f"Error loading existing embeddings: {e}")
        return pd.DataFrame(), set()

def save_index_state(processed_titles):
    """Save the state of the index"""
    state = {
        'processed_titles': list(processed_titles),
        'last_update': datetime.now().isoformat(),
        'embeddings_count': len(processed_titles)
    }
    
    with open(INDEX_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    logging.info(f"Saved index state with {len(processed_titles)} processed titles")

def update_search_index(force_regenerate=False):
    """Update the search index with new summaries"""
    # Load API key
    api_key = load_api_key()
    if not api_key:
        logging.error("Cannot proceed without API key")
        return False
        
    # Load summaries
    summaries_df = load_summaries()
    if summaries_df.empty:
        logging.error("No summaries found, cannot update index")
        return False
        
    # Load existing embeddings
    existing_embeddings_df, titles_with_embeddings = load_existing_embeddings()
    
    # Find titles that need embeddings
    all_titles = set(summaries_df['title'].tolist())
    if force_regenerate:
        titles_needing_embeddings = all_titles
        logging.info(f"Force regenerating embeddings for all {len(all_titles)} titles")
    else:
        titles_needing_embeddings = all_titles - titles_with_embeddings
        logging.info(f"Found {len(titles_needing_embeddings)} titles needing embeddings")
    
    if not titles_needing_embeddings:
        logging.info("No new titles need embeddings, search index is up-to-date")
        return True
        
    # Generate embeddings for new titles
    new_embeddings = []
    new_processed_titles = []
    
    logging.info(f"Generating embeddings for {len(titles_needing_embeddings)} titles")
    
    # Filter summaries to only those needing embeddings
    filtered_df = summaries_df[summaries_df['title'].isin(titles_needing_embeddings)]
    
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        title = row['title']
        summary = row['summary']
        
        if not isinstance(summary, str) or len(summary.strip()) < 10:
            logging.warning(f"Skipping '{title}' - summary too short or not a string")
            continue
            
        logging.info(f"[{i+1}/{len(filtered_df)}] Generating embedding for '{title}'")
        
        embedding = generate_embedding(summary)
        if embedding:
            new_embeddings.append(embedding)
            new_processed_titles.append(title)
            logging.info(f"Successfully generated embedding for '{title}'")
        else:
            logging.error(f"Failed to generate embedding for '{title}'")
    
    if not new_embeddings:
        logging.warning("No new embeddings were generated")
        return False
        
    # Convert new embeddings to DataFrame
    new_embeddings_df = pd.DataFrame(new_embeddings)
    
    # Merge with existing embeddings if they exist
    if not existing_embeddings_df.empty:
        all_embeddings_df = pd.concat([existing_embeddings_df, new_embeddings_df], ignore_index=True)
        logging.info(f"Merged {len(new_embeddings_df)} new embeddings with {len(existing_embeddings_df)} existing embeddings")
    else:
        all_embeddings_df = new_embeddings_df
        logging.info(f"Created new embeddings file with {len(new_embeddings_df)} embeddings")
    
    # Save merged embeddings
    all_embeddings_df.to_csv(EMBEDDINGS_FILE, index=False)
    logging.info(f"Saved {len(all_embeddings_df)} embeddings to {EMBEDDINGS_FILE}")
    
    # Update index state
    all_processed_titles = set(titles_with_embeddings) | set(new_processed_titles)
    save_index_state(all_processed_titles)
    
    # Generate full embedding visualization
    logging.info("Running full embedding generation script to update visualizations...")
    try:
        subprocess.run(["python", "generate_summary_embeddings.py", 
                        "--input", SUMMARIES_FILE, 
                        "--output", EMBEDDINGS_FILE], check=True)
        logging.info("Successfully updated embeddings visualization")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error updating embeddings visualization: {e}")
        return False

def update_search_api():
    """Update the search API to use the new embeddings"""
    # This function would restart your search API service or trigger a reload
    # The implementation depends on how your search API is deployed
    logging.info("Notifying search API to reload embeddings...")
    
    # Example: touching a file that triggers API reload
    reload_trigger_file = "../api/reload_embeddings.trigger"
    with open(reload_trigger_file, 'w') as f:
        f.write(datetime.now().isoformat())
    
    logging.info("Search API reload triggered")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update search index for casino game analysis app")
    parser.add_argument("--force-regenerate", action="store_true", help="Force regenerate all embeddings")
    args = parser.parse_args()
    
    logging.info("Starting search index update")
    
    if update_search_index(args.force_regenerate):
        update_search_api()
        logging.info("Search index update completed successfully")
    else:
        logging.error("Search index update failed")
