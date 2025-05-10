#!/usr/bin/env python3
"""
Optimized Update Search App Script

This script updates the search application with only what it needs:
1. Game embeddings for semantic search
2. Essential metadata from the game data CSV (Developer, volatility)

The script:
1. Loads embeddings from the embeddings file
2. Extracts required metadata from the game data CSV
3. Combines the data into a single compact JSON file for the frontend
4. Updates the API data directory
"""

import os
import json
import logging
import shutil
import pandas as pd
import csv
from datetime import datetime
import argparse
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('../data/logs/search_app_updates.log'),
        logging.StreamHandler()
    ]
)

# File paths
UNIFIED_EMBEDDINGS_FILE = "../embeddings/unified_game_embeddings.csv"
LEGACY_EMBEDDINGS_FILE = "../embeddings/game_summary_embeddings.csv" # Kept for backward compatibility
GAME_DATA_FILE = "../data/bigwinboard_with_summaries_final.csv"
SHORT_SUMMARIES_FILE = "../data/bigwinboard_short_summaries.csv"
SEARCH_APP_DATA_DIR = "../api/data"
COMBINED_DATA_FILE = "../api/data/game_data_with_embeddings.json"
LAST_UPDATE_FILE = "../api/last_update.json"

def update_search_app(force=False, unified_embeddings_file=UNIFIED_EMBEDDINGS_FILE, game_data_file=GAME_DATA_FILE):
    """Update the search app with embeddings and essential game metadata"""
    logging.info("Checking for new embeddings and game data...")
    
    # Check if required files exist
    if not os.path.exists(unified_embeddings_file):
        logging.error(f"Unified embeddings file {unified_embeddings_file} not found!")
        return False
        
    if not os.path.exists(game_data_file):
        logging.error(f"Game data file {game_data_file} not found!")
        return False
    
    # Check if we need to update
    need_update = force
    
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, 'r') as f:
                last_update = json.load(f)
                
            embeddings_mtime = os.path.getmtime(unified_embeddings_file)
            game_data_mtime = os.path.getmtime(game_data_file)
            last_update_time = last_update.get('timestamp', 0)
            
            if max(embeddings_mtime, game_data_mtime) > last_update_time:
                logging.info("New data detected!")
                need_update = True
            else:
                logging.info("No new data found.")
        except Exception as e:
            logging.error(f"Error checking last update time: {e}")
            need_update = True
    else:
        logging.info("No previous update record found. Will update.")
        need_update = True
    
    if not need_update and not force:
        logging.info("Search app is already up-to-date.")
        return True
    
    # Create app data directory if it doesn't exist
    os.makedirs(SEARCH_APP_DATA_DIR, exist_ok=True)
    
    # Load unified embeddings file (title and embedding vectors together)
    try:
        unified_embeddings_df = pd.read_csv(unified_embeddings_file, quoting=csv.QUOTE_ALL)
        
        # Check for required columns
        if 'title' not in unified_embeddings_df.columns or 'embedding' not in unified_embeddings_df.columns:
            logging.error(f"Unified embeddings file missing required columns. Found: {list(unified_embeddings_df.columns)}")
            return False
            
        logging.info(f"Loaded {len(unified_embeddings_df)} unified embeddings")
        
        # Convert embedding strings to lists if they're stored as strings
        if unified_embeddings_df['embedding'].dtype == 'object':
            try:
                unified_embeddings_df['embedding'] = unified_embeddings_df['embedding'].apply(
                    lambda x: eval(x) if isinstance(x, str) else x
                )
                logging.info("Converted embedding strings to lists")
            except Exception as e:
                logging.error(f"Error converting embeddings to lists: {e}")
                return False
    except Exception as e:
        logging.error(f"Error loading unified embeddings: {e}")
        return False
    
    # Load game metadata from CSV, maintaining essential fields
    try:
        # Use our custom CSV reading approach
        game_metadata = pd.read_csv(game_data_file, quoting=csv.QUOTE_ALL)
        
        # Use only required columns to reduce memory usage
        required_cols = ['title', 'structured_summary', 'developer', 'volatility']
        available_cols = [col for col in required_cols if col in game_metadata.columns]
        
        logging.info(f"Using columns: {available_cols} from game data file")
        
        if 'title' not in available_cols:
            logging.error("Title column missing from game data file")
            return False
            
        game_metadata = game_metadata[available_cols]
        logging.info(f"Loaded metadata for {len(game_metadata)} games")
        
        # Try to load short summaries if they exist
        short_summaries = {}
        if os.path.exists(SHORT_SUMMARIES_FILE):
            try:
                # Read the CSV file without headers and assign column names manually
                # The file has 3 columns: game_name, full_summary, short_summary
                short_summaries_df = pd.read_csv(
                    SHORT_SUMMARIES_FILE, 
                    quoting=csv.QUOTE_ALL, 
                    header=None, 
                    names=['game_name', 'full_summary', 'short_summary']
                )
                
                # Clean up quotation marks from the game names and short summaries
                short_summaries_df['game_name'] = short_summaries_df['game_name'].str.strip('"')
                short_summaries_df['short_summary'] = short_summaries_df['short_summary'].str.strip('"')
                
                # Create dictionary mapping from game name to short summary
                short_summaries = dict(zip(
                    short_summaries_df['game_name'],
                    short_summaries_df['short_summary']
                ))
                
                # Log some debug info
                logging.info(f"Loaded {len(short_summaries)} short summaries")
                if len(short_summaries) > 0:
                    sample_keys = list(short_summaries.keys())[:3]
                    logging.info(f"Sample short summary keys: {sample_keys}")
                    if 'Hugo 2' in short_summaries:
                        logging.info(f"Hugo 2 short summary is available")
            except Exception as e:
                logging.warning(f"Error loading short summaries: {e}")
                short_summaries = {}
    except Exception as e:
        logging.error(f"Error loading game metadata: {e}")
        return False
    
    # Normalize column names in game data
    game_metadata.columns = [col.lower().strip() for col in game_metadata.columns]
    
    # Print column names for debugging
    logging.info(f"Game data columns: {list(game_metadata.columns)}")
    logging.info(f"Unified embeddings columns: {list(unified_embeddings_df.columns)}")
    
    # Make sure title column exists in game data
    if 'title' not in game_metadata.columns:
        # Try to find a suitable title column
        title_candidates = [col for col in game_metadata.columns if 'title' in col or 'name' in col]
        if title_candidates:
            logging.info(f"Found potential title columns: {title_candidates}")
            # Use the first matching column as title
            game_metadata = game_metadata.rename(columns={title_candidates[0]: 'title'})
            logging.info(f"Renamed '{title_candidates[0]}' to 'title'")
        else:
            logging.error("No title column found in game data")
            return False
    
    # Extract only needed metadata
    required_columns = ['title', 'developer', 'volatility', 'structured_summary']
    filtered_columns = [col for col in required_columns if col in game_metadata.columns]
    
    if len(filtered_columns) < 2:
        logging.warning(f"Minimal game metadata available. Only {filtered_columns} found.")
    
    game_metadata_slim = game_metadata[filtered_columns].copy()
    
    # Print before merge
    logging.info(f"Game metadata shape: {game_metadata_slim.shape}, columns: {list(game_metadata_slim.columns)}")
    logging.info(f"Unified embeddings shape: {unified_embeddings_df.shape}, columns: {list(unified_embeddings_df.columns)}")
    
    # Merge with embeddings
    combined_data = pd.merge(
        game_metadata_slim, 
        unified_embeddings_df, 
        left_on='title', 
        right_on='title', 
        how='inner'
    )
    
    logging.info(f"Combined {len(combined_data)} games with embeddings and metadata")
    
    # Convert embeddings from string to list
    combined_data['embedding'] = combined_data['embedding'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )
    
    # Convert to dict format for JSON
    game_data_list = []
    for _, row in combined_data.iterrows():
        game_entry = {
            'title': row['title'],
            'embedding': row['embedding']
        }
        
        # Add optional metadata if available
        if 'developer' in row:
            game_entry['developer'] = row['developer']
        if 'volatility' in row:
            game_entry['volatility'] = row['volatility']
        if 'structured_summary' in row:
            game_entry['summary'] = row['structured_summary']
            
        # Add short summary if available for this title
        if row['title'] in short_summaries:
            game_entry['short_summary'] = short_summaries[row['title']]
        
        game_data_list.append(game_entry)
    
    # Save combined data
    try:
        with open(COMBINED_DATA_FILE, 'w') as f:
            json.dump(game_data_list, f)
        logging.info(f"Saved combined data to {COMBINED_DATA_FILE}")
        
        # Also copy the unified embeddings file
        unified_dest = os.path.join(SEARCH_APP_DATA_DIR, os.path.basename(unified_embeddings_file))
        shutil.copy2(unified_embeddings_file, unified_dest)
        logging.info(f"Copied unified embeddings to {unified_dest}")
        
        # Copy legacy format file if it exists (for backward compatibility)
        if os.path.exists(LEGACY_EMBEDDINGS_FILE):
            legacy_dest = os.path.join(SEARCH_APP_DATA_DIR, os.path.basename(LEGACY_EMBEDDINGS_FILE))
            shutil.copy2(LEGACY_EMBEDDINGS_FILE, legacy_dest)
            logging.info(f"Copied legacy embeddings format for compatibility to {legacy_dest}")
        
        # Save update timestamp
        update_time = {
            'timestamp': max(os.path.getmtime(unified_embeddings_file), os.path.getmtime(game_data_file)),
            'datetime': datetime.now().isoformat(),
            'games_count': len(game_data_list),
            'unified_format': True  # Flag indicating we're using the new format
        }
        
        with open(LAST_UPDATE_FILE, 'w') as f:
            json.dump(update_time, f, indent=2)
        logging.info(f"Update recorded at {update_time['datetime']}")
        
        # Signal API to reload (create trigger file)
        with open(os.path.join(SEARCH_APP_DATA_DIR, "reload_data.trigger"), 'w') as f:
            f.write(datetime.now().isoformat())
        logging.info("Reload trigger created for API")
        
        return True
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Update search app with game data and embeddings")
    parser.add_argument('--force', action='store_true', help='Force update even if files have not changed')
    parser.add_argument('--embeddings', type=str, default=UNIFIED_EMBEDDINGS_FILE,
                        help=f'Path to unified embeddings file (default: {UNIFIED_EMBEDDINGS_FILE})')
    parser.add_argument('--gamedata', type=str, default=GAME_DATA_FILE,
                        help=f'Path to game data CSV (default: {GAME_DATA_FILE})')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use argument values for file paths
    unified_embeddings_file = args.embeddings  # Now using the unified file
    game_data_file = args.gamedata
    
    # Update the search app
    if update_search_app(force=args.force, unified_embeddings_file=unified_embeddings_file, game_data_file=game_data_file):
        logging.info("Search app update successful!")
    else:
        logging.error("Search app update failed!")
        exit(1)

if __name__ == "__main__":
    main()
