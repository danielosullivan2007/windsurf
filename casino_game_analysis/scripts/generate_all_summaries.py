#!/usr/bin/env python3
"""
Generate structured summaries for all casino games that don't have them yet.
This script builds on generate_reliable_summaries.py but is optimized for batch processing.
"""

import os
import time
import sys
import json
import pandas as pd
import openai
from dotenv import load_dotenv
from tqdm import tqdm
import threading
import concurrent.futures
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('all_summaries_generation.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    logging.critical("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)

# File paths
INPUT_FILE = "../data/bigwinboard_with_summaries.csv"
OUTPUT_FILE = "../data/bigwinboard_with_summaries_complete.csv"
PROGRESS_FILE = "all_summaries_progress.json"

# Configuration
BATCH_SIZE = 50         # Number of games to process in each batch
MAX_WORKERS = 5         # Number of concurrent workers
API_RATE_LIMIT = 0.5    # Time between API calls in seconds
MAX_GAMES = None        # Maximum number of games to process (None for all)
SUMMARY_LENGTH = 300    # Maximum tokens for each summary

# Thread-safe locks
api_lock = threading.Lock()
csv_lock = threading.Lock()
api_last_call = [0]  # Using list for mutable reference in threads

def load_progress():
    """Load the last processed index from progress file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('last_index', 0)
    return 0

def save_progress(index):
    """Save the current progress to a JSON file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({'last_index': index, 'timestamp': time.time()}, f)

def create_summary(title, review_text):
    """Generate a structured summary for a review using OpenAI's API"""
    # Skip empty reviews
    if not review_text or pd.isna(review_text) or len(str(review_text).strip()) < 50:
        return f"{title} - A casino slot game with no detailed description available."
        
    # Truncate extremely long reviews to 8000 chars to speed up API calls
    review_text = str(review_text)
    if len(review_text) > 8000:
        review_text = review_text[:8000] + "..."
    
    prompt = f"""
    Please provide a concise 4-line summary of the following casino game review for '{title}'.
    
    Line 1: Overview focusing on the game theme
    Line 2: Focus on the game features
    Line 3: Summary of the reviewer's verdict
    Line 4: Description of the game aesthetics and audio (if mentioned)
    
    Make each line brief but informative.
    
    Review: {review_text}
    """
    try:
        # Implement rate limiting
        with api_lock:
            current_time = time.time()
            time_since_last_call = current_time - api_last_call[0]
            if time_since_last_call < API_RATE_LIMIT:
                time.sleep(API_RATE_LIMIT - time_since_last_call)
            api_last_call[0] = time.time()
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at creating concise, structured summaries of casino game reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=SUMMARY_LENGTH,
            temperature=0.7
        )
        
        # Extract the summary text
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        logging.error(f"Error generating summary for {title}: {e}")
        return None

def process_game(row_data):
    """Process a single game to generate a summary"""
    index, row = row_data
    
    try:
        title = str(row['Title']).strip() if 'Title' in row else ""
        
        # Skip if title is empty
        if not title:
            return None
            
        # Get review text from either 'review' or 'Review' column
        review_text = None
        if 'review' in row and pd.notna(row['review']):
            review_text = row['review']
        elif 'Review' in row and pd.notna(row['Review']):
            review_text = row['Review']
            
        # Skip if we already have a structured summary
        if 'structured_summary' in row and pd.notna(row['structured_summary']):
            return None
            
        # Generate summary
        summary = create_summary(title, review_text)
        
        if not summary:
            return None
            
        # Escape quotes for CSV
        safe_title = title.replace('"', '""')
        safe_summary = summary.replace('"', '""')
        safe_review = str(review_text).replace('"', '""') if review_text else ""
        
        return (index, safe_title, safe_summary, safe_review)
    except Exception as e:
        logging.error(f"Error processing row {index}: {e}")
        return None

def process_all_summaries(df, start_index=0):
    """Process summaries for all games that don't have them yet"""
    # Filter to only include games without structured summaries
    missing_summaries_df = df[df['structured_summary'].isna()].copy()
    
    if len(missing_summaries_df) == 0:
        logging.info("All games already have structured summaries!")
        return 0
        
    logging.info(f"Found {len(missing_summaries_df)} games without structured summaries")
    
    # Calculate how many rows to process
    max_index = min(start_index + MAX_GAMES, len(missing_summaries_df)) if MAX_GAMES else len(missing_summaries_df)
    rows_to_process = missing_summaries_df.iloc[start_index:max_index]
    
    logging.info(f"Processing {len(rows_to_process)} games starting from index {start_index}")
    
    # Create a progress bar
    start_time = time.time()
    pbar = tqdm(total=len(rows_to_process), desc="Generating Summaries", unit="game")
    
    # Create a copy of the original dataframe to update
    output_df = df.copy()
    
    # Process rows in parallel
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_game, (i, row)): i 
                         for i, row in rows_to_process.iterrows()}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_row):
            try:
                result = future.result()
                if not result:
                    pbar.update(1)
                    continue
                    
                index, safe_title, safe_summary, safe_review = result
                completed += 1
                
                # Update the dataframe with the new summary
                output_df.at[index, 'structured_summary'] = safe_summary
                
                # Save progress every BATCH_SIZE games
                if completed % BATCH_SIZE == 0:
                    with csv_lock:
                        output_df.to_csv(OUTPUT_FILE, index=False)
                        save_progress(index)
                        logging.info(f"Saved progress: {completed}/{len(rows_to_process)} games processed")
            except Exception as e:
                logging.error(f"Error processing result: {e}")
            
            # Update progress bar
            pbar.update(1)
    
    # Save final results
    with csv_lock:
        output_df.to_csv(OUTPUT_FILE, index=False)
        save_progress(max_index)
    
    pbar.close()
    end_time = time.time()
    logging.info(f"Total time: {end_time - start_time:.2f} seconds")
    logging.info(f"Processed {completed} summaries")
    
    return completed

def main():
    # These variables can be modified by command line arguments
    global MAX_GAMES, MAX_WORKERS, BATCH_SIZE
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate structured summaries for all casino games")
    parser.add_argument('-n', '--next', type=int, default=None, 
                        help='Number of games to process from the last processed index')
    parser.add_argument('-w', '--workers', type=int, default=MAX_WORKERS,
                        help=f'Number of concurrent workers (default: {MAX_WORKERS})')
    parser.add_argument('-b', '--batch', type=int, default=BATCH_SIZE,
                        help=f'Batch size for saving progress (default: {BATCH_SIZE})')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update configuration based on arguments
    if args.next is not None:
        MAX_GAMES = args.next
    
    if args.workers is not None:
        MAX_WORKERS = args.workers
    
    if args.batch is not None:
        BATCH_SIZE = args.batch
    
    # Load the starting index from progress file
    start_index = load_progress()
    
    # Create output file as a copy of input file if it doesn't exist
    if not os.path.exists(OUTPUT_FILE):
        if os.path.exists(INPUT_FILE):
            import shutil
            shutil.copy(INPUT_FILE, OUTPUT_FILE)
            logging.info(f"Created output file as a copy of input file")
        else:
            logging.critical(f"Input file {INPUT_FILE} does not exist!")
            sys.exit(1)
    
    # Read the CSV file
    try:
        df = pd.read_csv(OUTPUT_FILE)
        logging.info(f"Loaded {len(df)} games from {OUTPUT_FILE}")
    except Exception as e:
        logging.critical(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Make sure we have a structured_summary column
    if 'structured_summary' not in df.columns:
        df['structured_summary'] = None
        logging.info("Added structured_summary column to dataframe")
    
    # Process summaries, starting from the last processed index
    processed_count = process_all_summaries(df, start_index)
    
    logging.info(f"Completed! Processed {processed_count} summaries, starting from index {start_index}")

if __name__ == "__main__":
    main()
