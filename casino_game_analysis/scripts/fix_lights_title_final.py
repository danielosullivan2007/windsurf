#!/usr/bin/env python
import json
import os
import glob
import logging
import pandas as pd
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Target files to check
BASE_DIR = "/Users/danielosullivan/Desktop/windsurf_testing/windsurf/casino_game_analysis"
API_DATA_DIR = f"{BASE_DIR}/api/data"
SEMANTIC_APP_DIR = f"{BASE_DIR}/semantic-search-app"

def check_and_fix_all_instances():
    """Find and fix all instances of the incorrect title in all relevant files"""
    # The incorrect titles to search for
    incorrect_titles = ['"Lights', 'Lights,', 'Lights Camera', 'Lights\\s', '"Lights"', '^Lights$']
    correct_title = "Lights, Camera, Action!"
    files_fixed = 0
    
    # 1. First pass: JSON files in API data directory - these affect the search results
    json_files = glob.glob(f"{API_DATA_DIR}/*.json")
    for json_file in json_files:
        logging.info(f"Checking JSON file: {json_file}")
        try:
            with open(json_file, 'r') as f:
                content = f.read()
                
            # Check if any version of the incorrect title exists
            needs_fixing = False
            for pattern in incorrect_titles:
                if re.search(pattern, content):
                    needs_fixing = True
                    logging.info(f"Found pattern '{pattern}' in {json_file}")
            
            if needs_fixing:
                # Load JSON content
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Fix the content based on the structure
                if isinstance(data, list):
                    # List of objects (most common format)
                    for item in data:
                        if isinstance(item, dict):
                            # Fix in title field
                            if 'title' in item and isinstance(item['title'], str):
                                # Check if it's the game we want to fix
                                if any(re.search(pattern, item['title']) for pattern in incorrect_titles):
                                    item['title'] = correct_title
                                    logging.info(f"Fixed title in {json_file}")
                            
                            # Fix in game_name field (sometimes used instead of title)
                            if 'game_name' in item and isinstance(item['game_name'], str):
                                if any(re.search(pattern, item['game_name']) for pattern in incorrect_titles):
                                    item['game_name'] = correct_title
                                    logging.info(f"Fixed game_name in {json_file}")
                            
                            # Fix in structured_summary or any other field that might contain the title
                            for field in ['structured_summary', 'summary', 'description']:
                                if field in item and isinstance(item[field], str):
                                    for pattern in incorrect_titles:
                                        if re.search(pattern, item[field]):
                                            item[field] = re.sub(pattern, correct_title, item[field])
                                            logging.info(f"Fixed {field} in {json_file}")
                
                # Save the corrected data
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                files_fixed += 1
        
        except Exception as e:
            logging.error(f"Error processing {json_file}: {e}")
    
    # 2. Regenerate the reload trigger for the API
    reload_file = f"{API_DATA_DIR}/reload_trigger.txt"
    try:
        import datetime
        with open(reload_file, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        logging.info(f"Updated reload trigger file at {reload_file}")
    except Exception as e:
        logging.error(f"Error updating reload trigger: {e}")
    
    # Report results
    if files_fixed > 0:
        logging.info(f"Total files fixed: {files_fixed}")
    else:
        logging.info("No files needed fixing.")
    
    return files_fixed > 0

if __name__ == "__main__":
    check_and_fix_all_instances()
