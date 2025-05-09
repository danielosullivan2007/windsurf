#!/usr/bin/env python
import pandas as pd
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# File paths
UNIFIED_EMBEDDINGS_FILE = "../embeddings/unified_game_embeddings.csv"
GAME_DATA_FILE = "../data/bigwinboard_with_summaries_final.csv"
APP_DATA_FILE = "../api/data/game_data_with_embeddings.json"

def fix_game_title():
    logging.info("Loading unified embeddings file")
    try:
        # Load the unified embeddings file
        embeddings_df = pd.read_csv(UNIFIED_EMBEDDINGS_FILE)
        
        # Check if the title needs to be fixed
        title_to_fix = "Lights, Camera, Cash!"
        target_title = "Lights, Camera, Action!"
        
        if title_to_fix in embeddings_df['title'].values:
            logging.info(f"Found title to fix: '{title_to_fix}'")
            
            # Fix the title
            embeddings_df.loc[embeddings_df['title'] == title_to_fix, 'title'] = target_title
            
            # Save back to file
            embeddings_df.to_csv(UNIFIED_EMBEDDINGS_FILE, index=False)
            logging.info(f"Updated unified embeddings file with corrected title: '{target_title}'")
            
            # Also fix in the search app data
            try:
                with open(APP_DATA_FILE, 'r') as f:
                    app_data = json.load(f)
                
                # Fix title in app data
                found = False
                for game in app_data:
                    if game.get('title') == title_to_fix:
                        game['title'] = target_title
                        found = True
                
                if found:
                    with open(APP_DATA_FILE, 'w') as f:
                        json.dump(app_data, f, indent=2)
                    logging.info(f"Updated search app data with corrected title: '{target_title}'")
                else:
                    logging.info("Title not found in search app data")
                
            except Exception as e:
                logging.error(f"Error updating search app data: {e}")
            
            return True
        else:
            logging.info(f"Title '{title_to_fix}' not found in embeddings file")
            
            # Look for similar titles
            similar_titles = [t for t in embeddings_df['title'].values if "Lights" in t]
            if similar_titles:
                logging.info(f"Found similar titles: {similar_titles}")
                
            return False
    
    except Exception as e:
        logging.error(f"Error fixing game title: {e}")
        return False

if __name__ == "__main__":
    fix_game_title()
