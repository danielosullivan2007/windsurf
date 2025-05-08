import os
import time
import sys
import json
import argparse
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
from tqdm import tqdm
import csv
import concurrent.futures
import threading
from functools import lru_cache

# Input and output file paths
input_file = "../data/bigwinboard_cleaned.csv"
output_file = "../data/bigwinboard_with_summaries.csv"
embeddings_file = "../embeddings/game_summary_embeddings.csv"
progress_file = "summary_progress.json"

import logging

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('summary_generation.log'),
        logging.StreamHandler()
    ]
)

# Direct read from .env file for API key
def read_api_key_from_file():
    # Try current directory first
    env_paths = [
        '.env',  # Current directory
        '../.env',  # Project root
        os.path.expanduser('~/.env')  # Home directory
    ]
    
    for env_path in env_paths:
        try:
            logging.info(f"Attempting to read .env file from: {os.path.abspath(env_path)}")
            if not os.path.exists(env_path):
                logging.warning(f"File does not exist: {env_path}")
                continue
            
            with open(env_path, 'r') as file:
                logging.info(f"Successfully opened file: {env_path}")
                for line in file:
                    if line.strip().startswith('OPENAI_API_KEY='):
                        api_key = line.strip().split('=', 1)[1]
                        # Remove any quotes if present
                        api_key = api_key.strip('\'"')
                        
                        # Basic validation
                        if not api_key or len(api_key) < 10:
                            logging.error("Invalid API key found")
                            return None
                        
                        return api_key
        except Exception as e:
            logging.error(f"Error reading .env file {env_path}: {e}")
    
    return None

# Get API key from .env file
api_key = read_api_key_from_file()

if not api_key:
    logging.critical("No valid OpenAI API key found. Please check your .env file.")
    sys.exit(1)

# Mask API key for logging
masked_key = f"{api_key[:5]}...{api_key[-4:]}"
logging.info(f"API key loaded: {masked_key}")

# Create OpenAI client with error handling
try:
    client = openai.OpenAI(api_key=api_key)
    # Validate client by checking available models
    available_models = client.models.list()
    logging.info(f"Available models: {[model.id for model in available_models.data[:5]]}")
except Exception as e:
    logging.critical(f"Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Configuration
BATCH_SIZE = 20       # Number of reviews to process in a batch before saving
MAX_GAMES = 2000      # Maximum number of games to process (None for all)
SUMMARY_LENGTH = 300  # Maximum tokens for each summary

import time

# Thread-safe lock for API rate limiting
api_lock = threading.Lock()
api_last_call = [0]  # Using list for mutable reference in threads
API_RATE_LIMIT = 0.5  # 0.5 seconds between API calls (20 requests per 10 seconds)

# Cache for summaries to avoid duplicate API calls
@lru_cache(maxsize=1000)
def create_summary(title, review_text):
    """Generate a structured summary for a review using OpenAI's API with caching"""
    # Skip empty reviews
    if not review_text or len(review_text.strip()) < 50:
        return None
        
    # Truncate extremely long reviews to 8000 chars to speed up API calls
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
        # Implement smarter rate limiting
        with api_lock:
            current_time = time.time()
            time_since_last_call = current_time - api_last_call[0]
            if time_since_last_call < API_RATE_LIMIT:
                time.sleep(API_RATE_LIMIT - time_since_last_call)
            api_last_call[0] = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at creating concise, structured summaries of casino game reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=SUMMARY_LENGTH,
            temperature=0.7
        )
        
        # Extract the summary text
        summary = response.choices[0].message.content.strip()
        
        return summary
    except Exception as e:
        logging.error(f"Error generating summary for {title}: {e}")
        return None

def load_progress():
    """Load the last processed index from progress file"""
    try:
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            return progress.get('last_processed_index', 0)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0

def save_progress(index):
    """Save the current progress to a JSON file"""
    progress = {
        'last_processed_index': index,
        'timestamp': time.time()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

# Thread-safe CSV writer
csv_lock = threading.Lock()

# Process a single row and return the result
def process_single_row(row_data):
    """Process a single row to generate a summary"""
    index, row = row_data
    
    # Handle different column name formats
    title = row['title'] if 'title' in row else row.get('Title', '')
    
    # Try different possible column names for review text
    if 'review_text' in row:
        review_text = row['review_text']
    elif 'review' in row:
        review_text = row['review']
    else:
        # Try to find any column that might contain review text
        for col in row.index:
            if 'review' in col.lower():
                review_text = row[col]
                break
        else:
            # No review column found
            return None
    
    # Skip if no review text
    if not isinstance(review_text, str) or len(review_text.strip()) < 100:
        return None
    
    # Generate summary
    try:
        summary = create_summary(title, review_text)
        if summary:
            # Safely escape summary for CSV
            safe_title = title.replace('"', '""')
            safe_summary = summary.replace('"', '""')
            safe_review = review_text.replace('"', '""') if isinstance(review_text, str) else ""
            
            return (index, safe_title, safe_summary, safe_review)
    except Exception as e:
        logging.error(f"Error processing row {index}: {e}")
        # Log the error to a separate file
        with open('summary_errors.log', 'a') as error_log:
            error_log.write(f"Error at index {index}: {e}\n")
        return None

def process_summaries(df, start_index=0):
    """Process summaries with resume functionality and parallel processing"""
    processed_count = 0
    
    # Check if output file exists and load existing data
    existing_summaries = {}
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            print(f"Found existing output file with {len(existing_df)} rows")
            
            # Check if we have the structured_summary column
            if 'structured_summary' in existing_df.columns and 'Title' in existing_df.columns:
                existing_summaries = {row['Title']: True for _, row in existing_df.iterrows() if pd.notna(row['structured_summary'])}
                print(f"Loaded {len(existing_summaries)} existing summaries to avoid duplicates")
        except Exception as e:
            print(f"Error reading existing summaries: {e}")
    
    # Create output file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write('Title,review,structured_summary\n')
    
    # Calculate how many rows to process
    max_index = min(start_index + MAX_GAMES, len(df)) if MAX_GAMES else len(df)
    rows_to_process = df.iloc[start_index:max_index]
    
    # Create a progress bar
    start_time = time.time()
    pbar = tqdm(total=len(rows_to_process), desc="Generating Summaries", unit="game")
    
    # Determine optimal number of workers based on CPU cores
    # Use fewer workers for API-bound tasks to avoid rate limits
    max_workers = min(4, os.cpu_count() or 4)  # Limit to 4 workers to avoid API rate limits
    
    # Process rows in parallel
    results = []
    completed = 0
    skipped = 0
    
    print(f"\nProcessing {len(rows_to_process)} rows with {max_workers} workers")
    print("Progress will be shown for each completed summary")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {executor.submit(process_single_row, (i, row)): i 
                         for i, row in rows_to_process.iterrows()}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_row):
            try:
                result = future.result()
                if not result:
                    continue
                    
                index, safe_title, safe_summary, safe_review = result
                completed += 1
                print(f"\r[{completed}/{len(rows_to_process)}] Generated summary for: {safe_title[:40]}{'...' if len(safe_title) > 40 else ''}    ", end="")
                
                # Skip if we already have a summary for this title
                if safe_title in existing_summaries:
                    skipped += 1
                    print(f"\nSkipping {safe_title} - already has a summary")
                    continue
            except Exception as e:
                print(f"\nError processing row: {e}")
                continue
                    
                # Thread-safe write to CSV
                with csv_lock:
                    with open(output_file, 'a', newline='', encoding='utf-8') as f:
                        f.write(f'"{safe_title}","{safe_review}","{safe_summary}"\n')
                    
                    # Save progress
                    save_progress(index)
                    processed_count += 1
                    
                    # Add to existing summaries to avoid duplicates
                    existing_summaries[safe_title] = safe_summary
            
            # Update progress bar
            pbar.update(1)
    
    pbar.close()
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    return processed_count

def custom_csv_reader():
    """Custom CSV reader for problematic files with improved performance"""
    try:
        # First try using pandas directly for better performance
        try:
            # Try to read the file without specifying columns first to detect available columns
            temp_df = pd.read_csv(input_file, nrows=1, encoding='utf-8')
            available_columns = temp_df.columns.tolist()
            
            # Find the title and review columns
            title_col = None
            review_col = None
            
            # Look for title column
            for col in available_columns:
                if col.lower() == 'title':
                    title_col = col
                    break
            
            # Look for review column
            for col in available_columns:
                if col.lower() in ['review', 'review_text']:
                    review_col = col
                    break
            
            if title_col and review_col:
                df = pd.read_csv(input_file, usecols=[title_col, review_col], encoding='utf-8')
                df.rename(columns={title_col: 'title', review_col: 'review_text'}, inplace=True)
                print(f"Loaded {len(df)} rows using pandas")
                
                # Limit to MAX_GAMES if specified
                if MAX_GAMES and len(df) > MAX_GAMES:
                    df = df.iloc[:MAX_GAMES]
                    
                return df
            else:
                raise Exception(f"Could not find title and review columns. Available columns: {available_columns}")
        except Exception as e:
            print(f"Pandas direct read failed: {e}\nFalling back to custom reader")
        
        # Use custom CSV reader to handle complex formatting
        data = []
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            csv_reader = csv.reader(f, quotechar='"', delimiter=',', 
                                    quoting=csv.QUOTE_ALL, skipinitialspace=True)
            
            # Read header
            headers = next(csv_reader)
            print(f"CSV Headers: {headers}")
            
            # Find column indices for title and review
            try:
                title_index = headers.index('Title')
                review_index = headers.index('Review')
            except ValueError:
                # Fallback to common column names or indices
                title_index = 0  # First column
                review_index = -1  # Last column
            
            # Process rows in batches for better performance
            batch_size = 1000
            batch = []
            
            for row_num, row in enumerate(csv_reader, 1):
                try:
                    # Handle rows with fewer columns than expected
                    if len(row) <= max(title_index, review_index):
                        continue
                    
                    title = row[title_index].strip()
                    review_text = row[review_index].strip()
                    
                    batch.append({
                        'title': title,
                        'review_text': review_text
                    })
                    
                    # Process in batches for better memory efficiency
                    if len(batch) >= batch_size:
                        data.extend(batch)
                        batch = []
                    
                    # Stop if we've reached MAX_GAMES
                    if MAX_GAMES and len(data) + len(batch) >= MAX_GAMES:
                        data.extend(batch[:MAX_GAMES - len(data)])
                        break
                
                except Exception as e:
                    logging.error(f"Error processing row {row_num}: {e}")
            
            # Add any remaining batch items
            if batch:
                data.extend(batch)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} rows using custom reader")
        return df
    
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate summaries for casino games")
    parser.add_argument('-n', '--next', type=int, default=None, 
                        help='Number of games to process from the last processed index')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the starting index from progress file
    start_index = load_progress()
    
    # Read the CSV file
    df = custom_csv_reader()
    
    # If --next is specified, update MAX_GAMES
    if args.next is not None:
        MAX_GAMES = args.next
    
    # Process summaries, starting from the last processed index
    processed_count = process_summaries(df, start_index)
    
    print(f"Processed {processed_count} summaries, starting from index {start_index}")
    start_time = time.time()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")
