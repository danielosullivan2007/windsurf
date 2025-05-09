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
output_file = "../data/bigwinboard_consolidated_summaries.csv"
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
    # First try the new OpenAI client format
    try:
        client = openai.OpenAI(api_key=api_key)
        # Validate client by checking available models
        available_models = client.models.list()
        logging.info(f"Available models: {[model.id for model in available_models.data[:5]]}")
        client_version = "new"
    except AttributeError:
        # Fall back to older API format
        openai.api_key = api_key
        client = openai
        logging.info("Using legacy OpenAI client")
        client_version = "legacy"
except Exception as e:
    logging.critical(f"Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Configuration
BATCH_SIZE = 20       # Number of reviews to process in a batch before saving
MAX_GAMES = None      # Maximum number of games to process (None for all)
SUMMARY_LENGTH = 300  # Maximum tokens for each summary
OVERRIDE_EXISTING = False  # Whether to override existing summaries

# Cost estimation configuration
MODEL_COSTS = {
    "gpt-4o": {"input": 5.0, "output": 20.0},  # $5.00 per 1M input tokens, $20.00 per 1M output tokens
    "gpt-4o-mini": {"input": 0.6, "output": 2.4},  # $0.60 per 1M input tokens, $2.40 per 1M output tokens
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},  # $0.40 per 1M input tokens, $1.60 per 1M output tokens
    "gpt-4": {"input": 30.0, "output": 60.0},   # $30 per 1M input tokens, $60 per 1M output tokens
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}  # $0.50 per 1M input tokens, $1.50 per 1M output tokens
}
# Note: We don't account for cached input tokens in this estimation as it's hard to predict cache hits
DEFAULT_MODEL = "gpt-4.1-mini"  # Default model to use - most cost-effective option with good quality
AVG_TOKENS_PER_CHAR = 0.25  # Rough estimate of tokens per character for English text
PROMPT_OVERHEAD_TOKENS = 100  # Estimated tokens for system message and formatting

import time

# Thread-safe lock for API rate limiting
api_lock = threading.Lock()
api_last_call = [0]  # Using list for mutable reference in threads
API_RATE_LIMIT = 0.5  # 0.5 seconds between API calls (20 requests per 10 seconds)

# Model to use will be set in the main function
model_to_use = DEFAULT_MODEL

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
        
        # Handle both new and old client formats
        if client_version == "new":
            # New client format
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, structured summaries of casino game reviews."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=SUMMARY_LENGTH,
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
        else:
            # Old client format - some models might not be available in older client
            fallback_model = model_to_use
            if model_to_use == "gpt-4o" and client_version == "legacy":
                fallback_model = "gpt-4"  # Use gpt-4 if gpt-4o isn't available in older client
                
            response = client.ChatCompletion.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, structured summaries of casino game reviews."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=SUMMARY_LENGTH,
                temperature=0.7
            )
            summary = response.choices[0].message.content.strip()
        
        # Summary is already extracted in the conditional blocks above
        
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
    
    # Skip if no title
    if not title or not isinstance(title, str) or len(title.strip()) < 2:
        return None
    
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
    # Reduced minimum length from 20 to 10 characters to process more games
    if not isinstance(review_text, str) or len(review_text.strip()) < 10:
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

def read_existing_summaries():
    """Read existing summaries from the output file using robust CSV parsing"""
    existing_summaries = {}
    if os.path.exists(output_file):
        try:
            # Use the same robust CSV parsing approach as the input file
            with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
                csv_reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
                
                # Read header
                headers = next(csv_reader)
                print(f"Output file headers: {headers}")
                
                # Find column indices for title and summary
                title_index = None
                summary_index = None
                
                # Look for title column
                for i, col in enumerate(headers):
                    if col.lower() == 'title':
                        title_index = i
                        break
                
                # Look for summary column
                for i, col in enumerate(headers):
                    if col.lower() in ['structured_summary', 'summary']:
                        summary_index = i
                        break
                
                # If we can't find the columns, use fallbacks
                if title_index is None:
                    title_index = 0  # First column
                if summary_index is None:
                    summary_index = 2  # Assume it's the third column
                
                print(f"Using column '{headers[title_index]}' for titles and '{headers[summary_index]}' for summaries")
                
                # Read all rows
                valid_summaries = 0
                for row in csv_reader:
                    # Skip rows that are too short
                    if len(row) <= max(title_index, summary_index):
                        continue
                    
                    title = row[title_index].strip() if title_index < len(row) else ""
                    summary = row[summary_index].strip() if summary_index < len(row) else ""
                    
                    # Only count non-empty summaries
                    if title and summary:
                        existing_summaries[title] = True
                        valid_summaries += 1
                
                print(f"Found {valid_summaries} existing valid summaries")
        except Exception as e:
            print(f"Error reading existing summaries: {e}")
    
    return existing_summaries

def process_summaries(df, start_index=0):
    """Process summaries with resume functionality and parallel processing"""
    processed_count = 0
    
    # Check if output file exists and load existing data
    existing_summaries = read_existing_summaries()
    if OVERRIDE_EXISTING:
        print(f"Found {len(existing_summaries)} existing summaries - will override if needed")
    else:
        print(f"Loaded {len(existing_summaries)} existing summaries to avoid duplicates")
    
    # Create output file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            f.write('Title,review,structured_summary\n')
    
    # Filter out rows with missing titles or reviews before calculating range
    valid_rows = []
    for i, row in df.iterrows():
        title = row['title'] if 'title' in row else row.get('Title', '')
        if not title or not isinstance(title, str) or len(title.strip()) < 2:
            continue
            
        review_col = next((col for col in row.index if 'review' in col.lower()), None)
        if not review_col or not isinstance(row[review_col], str) or len(str(row[review_col]).strip()) < 10:
            continue
            
        valid_rows.append(i)
    
    # Filter dataframe to only include valid rows
    df = df.loc[valid_rows]
    print(f"Found {len(df)} games with valid titles and reviews")
    
    # Apply limit if specified
    if MAX_GAMES:
        max_index = min(start_index + MAX_GAMES, len(df))
        rows_to_process = df.iloc[start_index:max_index]
        print(f"Processing {len(rows_to_process)} games (limited by MAX_GAMES={MAX_GAMES})")
    else:
        rows_to_process = df.iloc[start_index:]
        print(f"Processing {len(rows_to_process)} games starting from index {start_index}")
    
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
                
                # Skip if we already have a summary for this title and not overriding
                if not OVERRIDE_EXISTING and safe_title in existing_summaries:
                    skipped += 1
                    print(f"\nSkipping {safe_title} - already has a summary")
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
            except Exception as e:
                print(f"\nError processing row: {e}")
                continue
            
            # Update progress bar
            pbar.update(1)
    
    pbar.close()
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")
    return processed_count

def custom_csv_reader():
    """Custom CSV reader for problematic files with improved performance"""
    try:
        # Use a more robust CSV parsing approach
        data = []
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
            # Use CSV reader with proper quoting
            csv_reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
            
            # Read header
            headers = next(csv_reader)
            print(f"CSV Headers: {headers}")
            
            # Find column indices for title and review
            title_index = None
            review_index = None
            
            # Look for title column
            for i, col in enumerate(headers):
                if col.lower() == 'title':
                    title_index = i
                    break
            
            # Look for review column
            for i, col in enumerate(headers):
                if col.lower() in ['review', 'review_text']:
                    review_index = i
                    break
            
            # If we can't find the columns, use fallbacks
            if title_index is None:
                title_index = 0  # First column
            if review_index is None:
                # Try to find any column with 'review' in the name
                for i, col in enumerate(headers):
                    if 'review' in col.lower():
                        review_index = i
                        break
                if review_index is None:
                    review_index = -1  # Last column
            
            logging.info(f"Using column '{headers[title_index]}' for titles and '{headers[review_index]}' for reviews")
            
            # Process all rows
            row_count = 0
            for row in csv_reader:
                row_count += 1
                
                try:
                    # Handle rows that are too short
                    if len(row) <= max(title_index, review_index):
                        # Pad with empty values
                        row = row + [''] * (max(title_index, review_index) + 1 - len(row))
                    
                    # Extract title and review
                    title = row[title_index].strip() if title_index < len(row) else ""
                    review_text = row[review_index].strip() if review_index < len(row) else ""
                    
                    # Add to data
                    data.append({
                        'title': title,
                        'review_text': review_text
                    })
                    
                    # Stop if we've reached MAX_GAMES
                    if MAX_GAMES and len(data) >= MAX_GAMES:
                        break
                        
                except Exception as e:
                    logging.error(f"Error processing row {row_count}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} rows using robust CSV reader")
        return df
    
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

def estimate_api_cost(df, start_index=0, model_name=DEFAULT_MODEL):
    """Estimate the cost of API calls for generating summaries"""
    # Standard tokens per character assumption
    AVG_TOKENS_PER_CHAR = 0.25  # About 4 characters per token for English text
    PROMPT_OVERHEAD_TOKENS = 100  # Fixed overhead tokens per prompt
    
    # Get cost rates from model configuration
    if model_name not in MODEL_COSTS:
        print(f"Warning: Unknown model {model_name}, using default {DEFAULT_MODEL} for cost estimation")
        model_name = DEFAULT_MODEL
    
    input_cost_per_token = MODEL_COSTS[model_name]["input"] / 1000000  # Convert to cost per token
    output_cost_per_token = MODEL_COSTS[model_name]["output"] / 1000000
    
    # Initialize counters
    total_input_chars = 0
    rows_to_actually_process = 0
    skipped_no_title = 0
    skipped_no_review = 0
    skipped_existing = 0
    
    # Check if output file exists and load existing data
    existing_summaries = read_existing_summaries()
    print(f"Found {len(existing_summaries)} existing summaries for cost estimation")
    
    # Filter out rows with missing titles or reviews first
    valid_rows = []
    for i, row in df.iterrows():
        # Get title
        title = row['title'] if 'title' in row else row.get('Title', '')
        if not title or not isinstance(title, str) or len(title.strip()) < 2:
            skipped_no_title += 1
            continue
            
        # Get review text
        review_col = next((col for col in row.index if 'review' in col.lower()), None)
        if not review_col or not isinstance(row[review_col], str) or len(str(row[review_col]).strip()) < 10:
            skipped_no_review += 1
            continue
            
        valid_rows.append(i)
    
    # Get filtered dataframe
    filtered_df = df.loc[valid_rows]
    print(f"Found {len(filtered_df)} games with valid titles and reviews")
    print(f"Skipped: {skipped_no_title} with no title, {skipped_no_review} with no review")
    
    # Calculate how many rows we'll actually process
    if MAX_GAMES:
        max_index = min(start_index + MAX_GAMES, len(filtered_df))
        rows_to_check = filtered_df.iloc[start_index:max_index]
    else:
        rows_to_check = filtered_df.iloc[start_index:]
    
    print(f"Estimating cost for {len(rows_to_check)} rows starting at index {start_index}")
    
    # For each row, estimate input tokens
    for i, row in rows_to_check.iterrows():
        # Get title and review text 
        title = row['title'] if 'title' in row else row.get('Title', '')
        
        # Try to find the review column
        review_col = next((col for col in row.index if 'review' in col.lower()), None)
        review_text = row[review_col] if review_col else ''
        
        # Skip if we already have a summary for this title and not overriding
        if not OVERRIDE_EXISTING and title in existing_summaries:
            skipped_existing += 1
            continue
            
        # Count this row
        rows_to_actually_process += 1
        
        # Truncate extremely long reviews to 8000 chars to match the actual processing
        if len(str(review_text)) > 8000:
            review_text = str(review_text)[:8000] + "..."
            
        # Add to total input characters
        total_input_chars += len(str(review_text)) + len(str(title)) + 200  # 200 chars for prompt template
    
    # Estimate tokens
    estimated_input_tokens = total_input_chars * AVG_TOKENS_PER_CHAR + (rows_to_actually_process * PROMPT_OVERHEAD_TOKENS)
    estimated_output_tokens = rows_to_actually_process * SUMMARY_LENGTH  # Max tokens per summary
    
    # Calculate costs
    input_cost = estimated_input_tokens * input_cost_per_token
    output_cost = estimated_output_tokens * output_cost_per_token
    total_cost = input_cost + output_cost
    
    # Print detailed statistics
    print(f"\nDETAILED STATISTICS:")
    print(f"Total games in dataset: {len(df)}")
    print(f"Valid games (with title & review): {len(filtered_df)}")
    print(f"Skipped due to no title: {skipped_no_title}")
    print(f"Skipped due to no/short review: {skipped_no_review}")
    print(f"Skipped due to existing summary: {skipped_existing}")
    print(f"Games that will be processed: {rows_to_actually_process}")
    
    if rows_to_actually_process == 0:
        print("\nWARNING: No games will be processed. Consider using --override to regenerate existing summaries.")
    
    return {
        "rows_to_process": rows_to_actually_process,
        "estimated_input_tokens": int(estimated_input_tokens),
        "estimated_output_tokens": int(estimated_output_tokens),
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "model": model_name,
        "filtered_count": len(filtered_df),
        "skipped_no_title": skipped_no_title,
        "skipped_no_review": skipped_no_review,
        "skipped_existing": skipped_existing
    }

def confirm_processing(cost_estimate):
    """Ask for user confirmation to proceed based on cost estimate"""
    print("\n===== COST ESTIMATE =====")
    print(f"Model: {cost_estimate['model']}")
    print(f"Games to process: {cost_estimate['rows_to_process']}")
    print(f"Estimated input tokens: {cost_estimate['estimated_input_tokens']:,}")
    print(f"Estimated output tokens: {cost_estimate['estimated_output_tokens']:,}")
    print(f"Estimated input cost: ${cost_estimate['input_cost']:.2f}")
    print(f"Estimated output cost: ${cost_estimate['output_cost']:.2f}")
    print(f"Estimated total cost: ${cost_estimate['total_cost']:.2f}")
    print("========================\n")
    
    while True:
        response = input("Do you want to proceed with generating summaries? (Y/N): ").strip().upper()
        if response == 'Y':
            return True
        elif response == 'N':
            return False
        else:
            print("Please enter Y or N.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate summaries for casino games")
    parser.add_argument('-n', '--next', type=int, default=None, 
                        help='Number of games to process from the last processed index')
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL,
                        help=f'Model to use for generating summaries (default: {DEFAULT_MODEL})')
    parser.add_argument('--estimate-only', action='store_true',
                        help='Only estimate the cost without generating summaries')
    parser.add_argument('--override', action='store_true',
                        help='Override existing summaries instead of skipping them')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load the starting index from progress file
    start_index = load_progress()
    
    # Read the CSV file
    df = custom_csv_reader()
    
    # If --next is specified, update MAX_GAMES
    if args.next is not None:
        MAX_GAMES = args.next
    
    # Set the model to use
    globals()['model_to_use'] = args.model
    
    # Set override flag
    globals()['OVERRIDE_EXISTING'] = args.override
    if args.override:
        print("Will override existing summaries instead of skipping them")
    
    # Estimate cost
    cost_estimate = estimate_api_cost(df, start_index, args.model)
    
    # If estimate only, just print the estimate and exit
    if args.estimate_only:
        print(f"Estimated cost for processing {cost_estimate['rows_to_process']} games with {args.model}: ${cost_estimate['total_cost']:.2f}")
        sys.exit(0)
    
    # Ask for confirmation
    if not confirm_processing(cost_estimate):
        print("Operation cancelled by user.")
        sys.exit(0)
    
    # Process summaries, starting from the last processed index
    processed_count = process_summaries(df, start_index)
    
    print(f"Processed {processed_count} summaries, starting from index {start_index}")
    start_time = time.time()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time} seconds")
