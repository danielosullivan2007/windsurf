#!/usr/bin/env python
import pandas as pd
import json
import os
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# File paths
BASE_DIR = "/Users/danielosullivan/Desktop/windsurf_testing/windsurf/casino_game_analysis"

# New unified format files
UNIFIED_EMBEDDINGS_FILE = f"{BASE_DIR}/embeddings/unified_game_embeddings.csv"
GAME_DATA_FILE = f"{BASE_DIR}/data/bigwinboard_with_summaries_final.csv"
API_DATA_DIR = f"{BASE_DIR}/api/data"

# Original files used by the API
COMBINED_EMBEDDINGS_FILE = f"{BASE_DIR}/embeddings/combined_game_summary_embeddings.csv"
EMBEDDINGS_TITLES_MAPPING_FILE = f"{BASE_DIR}/embeddings/embedding_titles_mapping.csv"
SUMMARIES_COMPLETE_FILE = f"{BASE_DIR}/data/bigwinboard_with_summaries_complete.csv"
CLEANED_DATA_FILE = f"{BASE_DIR}/data/bigwinboard_cleaned.csv"

def fix_titles_in_all_files():
    old_titles = ["Lights, Camera, Cash!", "\"Lights", "Lights"]
    new_title = "Lights, Camera, Action!"
    
    # Keep track of changes made
    changes_made = {}
    
    # 0. Check if any of the older API files exist and fix them first
    old_api_files = [
        (COMBINED_EMBEDDINGS_FILE, False),  # CSV without header
        (EMBEDDINGS_TITLES_MAPPING_FILE, True),  # CSV with header
        (SUMMARIES_COMPLETE_FILE, True),  # CSV with header
        (CLEANED_DATA_FILE, True)  # CSV with header
    ]
    
    for file_path, has_header in old_api_files:
        try:
            if os.path.exists(file_path):
                logging.info(f"Processing older API file: {file_path}")
                
                # Handle CSV files with different structures
                if file_path == EMBEDDINGS_TITLES_MAPPING_FILE:
                    # This file maps titles to indices
                    df = pd.read_csv(file_path)
                    title_col = 'Title'  # This is the column name in the mapping file
                    
                    for old_title in old_titles:
                        mask = df[title_col].str.contains(old_title, regex=False, na=False)
                        if mask.any():
                            old_values = df.loc[mask, title_col].tolist()
                            count = len(old_values)
                            df.loc[mask, title_col] = new_title
                            logging.info(f"Updated {count} occurrences in {file_path}")
                            changes_made[file_path] = f"Updated {count} titles from {old_values} to {new_title}"
                    
                    df.to_csv(file_path, index=False)
                    
                elif file_path == SUMMARIES_COMPLETE_FILE or file_path == CLEANED_DATA_FILE:
                    # This is a CSV with game data including titles
                    df = pd.read_csv(file_path)
                    title_col = 'Title'  # This is typically the column name
                    
                    for old_title in old_titles:
                        mask = df[title_col].str.contains(old_title, regex=False, na=False)
                        if mask.any():
                            old_values = df.loc[mask, title_col].tolist()
                            count = len(old_values)
                            df.loc[mask, title_col] = new_title
                            logging.info(f"Updated {count} occurrences in {file_path}")
                            changes_made[file_path] = f"Updated {count} titles from {old_values} to {new_title}"
                    
                    # Also fix in any summary or description columns
                    for col in df.columns:
                        if col.lower() in ['structured_summary', 'summary', 'description', 'review']:
                            for old_title in old_titles:
                                df[col] = df[col].astype(str).str.replace(old_title, new_title, regex=False)
                    
                    df.to_csv(file_path, index=False)
                    
                elif file_path == COMBINED_EMBEDDINGS_FILE:
                    # This is a headerless CSV with embeddings
                    # We need to check if the title exists in the mapping file and make sure they align
                    if os.path.exists(EMBEDDINGS_TITLES_MAPPING_FILE):
                        titles_df = pd.read_csv(EMBEDDINGS_TITLES_MAPPING_FILE)
                        # If we updated the mapping file, we don't need to modify the embeddings
                        if EMBEDDINGS_TITLES_MAPPING_FILE in changes_made:
                            logging.info(f"Skipping {file_path} as the mapping file was already updated")
                        else:
                            logging.info(f"No changes needed for {file_path}")
                    else:
                        logging.warning(f"Cannot process {file_path} without the titles mapping file")
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    
    # 1. Fix in unified embeddings CSV
    try:
        logging.info(f"Processing {UNIFIED_EMBEDDINGS_FILE}")
        if os.path.exists(UNIFIED_EMBEDDINGS_FILE):
            df = pd.read_csv(UNIFIED_EMBEDDINGS_FILE)
            for old_title in old_titles:
                mask = df['title'].str.contains(old_title, regex=False)
                if mask.any():
                    old_values = df.loc[mask, 'title'].tolist()
                    count = len(old_values)
                    df.loc[mask, 'title'] = new_title
                    logging.info(f"Updated {count} occurrences in unified embeddings file")
                    changes_made[UNIFIED_EMBEDDINGS_FILE] = f"Updated {count} titles from {old_values} to {new_title}"
            df.to_csv(UNIFIED_EMBEDDINGS_FILE, index=False)
    except Exception as e:
        logging.error(f"Error updating unified embeddings file: {e}")
    
    # 2. Fix in all JSON files in the API directory
    json_files = glob.glob(f"{API_DATA_DIR}/*.json")
    for json_file in json_files:
        try:
            logging.info(f"Processing {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Track changes for this file
            file_changes = 0
            
            # Check if data is a list of objects
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'title' in item:
                        for old_title in old_titles:
                            if old_title in item['title']:
                                item['title'] = new_title
                                file_changes += 1
                    
                    # Also check for the title in structured_summary or any description field
                    for field in ['structured_summary', 'summary', 'description']:
                        if isinstance(item, dict) and field in item and isinstance(item[field], str):
                            for old_title in old_titles:
                                if old_title in item[field]:
                                    item[field] = item[field].replace(old_title, new_title)
                                    file_changes += 1
            
            # Special case for any non-list JSON structures
            elif isinstance(data, dict):
                # Handle any dictionary format if needed
                for key, value in data.items():
                    if isinstance(value, str):
                        for old_title in old_titles:
                            if old_title in value:
                                data[key] = value.replace(old_title, new_title)
                                file_changes += 1
            
            if file_changes > 0:
                logging.info(f"Made {file_changes} changes in {json_file}")
                changes_made[json_file] = f"Made {file_changes} changes"
                # Write updated data back to file
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}")
    
    # 3. Also check and force update the API reload file to ensure changes take effect
    reload_file = f"{API_DATA_DIR}/reload_trigger.txt"
    try:
        import datetime
        with open(reload_file, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        logging.info(f"Updated reload trigger file at {reload_file}")
    except Exception as e:
        logging.error(f"Error updating reload trigger: {e}")
    
    # Summary of changes
    if changes_made:
        logging.info("== Summary of Changes ==")
        for file, changes in changes_made.items():
            logging.info(f"{file}: {changes}")
        logging.info("All updates complete. Please restart the search app to see changes.")
        return True
    else:
        logging.info("No changes were required.")
        return False

if __name__ == "__main__":
    fix_titles_in_all_files()
