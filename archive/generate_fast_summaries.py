import pandas as pd
import os
import time
import sys
import json
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Input and output file paths
input_file = "bigwinboard_cleaned.csv"
output_file = "bigwinboard_with_summaries.csv"
progress_file = "summary_progress.json"

# Direct read from .env file for API key
def read_api_key_from_file():
    try:
        with open('.env', 'r') as file:
            for line in file:
                if line.strip().startswith('OPENAI_API_KEY='):
                    api_key = line.strip().split('=', 1)[1]
                    # Remove any quotes if present
                    api_key = api_key.strip('\'"')
                    return api_key
    except Exception as e:
        print(f"Error reading .env file: {e}")
    return None

# Get API key from .env file
api_key = read_api_key_from_file()

if api_key and len(api_key) > 10:
    print(f"API key from .env file: {api_key[:5]}...{api_key[-4:]}")
    print(f"API key length: {len(api_key)}")
    
    # Set it in the environment for current process
    os.environ["OPENAI_API_KEY"] = api_key
else:
    print("No valid API key found in .env file")
    sys.exit(1)

# Configuration
BATCH_SIZE = 5       # Number of reviews to batch in a single API call
MAX_WORKERS = 10     # Maximum number of concurrent API calls
MAX_GAMES = None     # Maximum number of games to process (None for all)

def create_structured_summary_batch(reviews_data):
    """
    Generate structured summaries for a batch of reviews using OpenAI's API
    
    Args:
        reviews_data: List of tuples (index, title, review_text)
    
    Returns:
        List of tuples (index, summary)
    """
    # Create a client with the API key
    client = openai.OpenAI(api_key=api_key)
    
    # Create batch messages
    messages = [
        {"role": "system", "content": "You are a professional game analyst creating concise summaries of casino game reviews."}
    ]
    
    # Add each review as a separate user message
    for idx, title, review in reviews_data:
        prompt = f"""
        Please provide a concise 4-line summary of the following casino game review for '{title}'.
        
        Line 1: Overview focusing on the game theme
        Line 2: Focus on the game features
        Line 3: Summary of the reviewer's verdict
        Line 4: Description of the game aesthetics and audio (if mentioned)
        
        Make each line brief but informative.
        
        Review: {review}
        """
        messages.append({"role": "user", "content": prompt})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300 * len(reviews_data),  # Adjust tokens based on batch size
            temperature=0.5
        )
        
        # Extract summaries from response
        summaries = []
        for i, choice in enumerate(response.choices):
            if i < len(reviews_data):
                idx = reviews_data[i][0]
                summary = choice.message.content.strip()
                summaries.append((idx, summary))
        
        return summaries
    except Exception as e:
        print(f"Error generating batch summaries: {e}")
        return [(idx, "") for idx, _, _ in reviews_data]

def process_game_batch(batch_data):
    """Process a batch of games to get summaries"""
    try:
        return create_structured_summary_batch(batch_data)
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [(idx, "") for idx, _, _ in batch_data]

def read_csv_with_fallbacks():
    """Read the CSV file with multiple fallback methods"""
    try:
        # Try standard reading
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"CSV read successfully with standard method: {len(df)} rows")
        return df
    except Exception as e:
        print(f"Standard CSV reading failed: {e}")
        
        try:
            # Try with Python engine
            df = pd.read_csv(input_file, engine='python', encoding='utf-8')
            print(f"CSV read successfully with python engine: {len(df)} rows")
            return df
        except Exception as e2:
            print(f"Python engine CSV reading failed: {e2}")
            
            try:
                # Try with explicit quoting
                df = pd.read_csv(input_file, quoting=pd.io.common.csv.QUOTE_MINIMAL, 
                                escapechar='\\', encoding='utf-8')
                print(f"CSV read successfully with explicit quoting: {len(df)} rows")
                return df
            except Exception as e3:
                print(f"All CSV reading methods failed. Final error: {e3}")
                
                # Custom CSV reader as last resort
                print("Attempting custom CSV parsing...")
                return custom_csv_reader()

def custom_csv_reader():
    """Custom CSV reader for problematic files"""
    try:
        # Read the header
        with open(input_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        
        # Read the data
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                # Simple parsing, assumes no commas in quoted fields
                row = []
                in_quotes = False
                current_field = ""
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        row.append(current_field)
                        current_field = ""
                    else:
                        current_field += char
                
                # Add the last field
                if current_field:
                    row.append(current_field)
                
                # Ensure we have the right number of columns
                if len(row) >= len(header):
                    data.append(row[:len(header)])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=header)
        print(f"Custom CSV parsing successful: {len(df)} rows")
        return df
    
    except Exception as e:
        print(f"Custom CSV parsing failed: {e}")
        print("Could not read the input file. Please check the file format.")
        sys.exit(1)

def get_progress():
    """Get the progress from the JSON file"""
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading progress file: {e}")
    return {"completed_indices": [], "total": 0}

def save_progress(completed_indices, total):
    """Save the progress to the JSON file"""
    try:
        with open(progress_file, 'w') as f:
            json.dump({"completed_indices": completed_indices, "total": total}, f)
    except Exception as e:
        print(f"Error saving progress file: {e}")

def process_csv():
    """Process the CSV file and generate summaries"""
    start_time = time.time()
    
    # Read the CSV file
    df = read_csv_with_fallbacks()
    
    # Determine column names
    title_column = 'Title' if 'Title' in df.columns else df.columns[0]
    
    # Find the review column
    review_column = None
    for col in ['review', 'Review', 'review_text', 'content']:
        if col in df.columns:
            review_column = col
            break
    
    # If no specific review column found, use the column with the longest text
    if not review_column:
        max_length = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].str.len().mean()
                if avg_len > max_length:
                    max_length = avg_len
                    review_column = col
    
    print(f"Using column '{title_column}' for titles and '{review_column}' for reviews")
    
    # Add structured_summary column if it doesn't exist
    if 'structured_summary' not in df.columns:
        df['structured_summary'] = ""
    
    # Get the previous progress
    progress = get_progress()
    completed_indices = set(progress["completed_indices"])
    print(f"Found {len(completed_indices)} previously completed summaries")
    
    # Limit the number of games if specified
    if MAX_GAMES:
        df = df.head(MAX_GAMES)
    
    # Prepare batches
    batches = []
    indices_to_process = []
    
    for i, row in df.iterrows():
        if i not in completed_indices:
            indices_to_process.append(i)
            
            # Get title and review
            title = str(row[title_column])
            review = str(row[review_column])
            
            # Skip rows with empty reviews
            if len(review) < 50:
                print(f"Skipping row {i} due to short review: {len(review)} chars")
                continue
            
            # Truncate very long reviews
            if len(review) > 4000:
                review = review[:4000] + "..."
            
            # Add to batch
            batches.append((i, title, review))
            
            # Process in smaller batches for API efficiency
            if len(batches) >= BATCH_SIZE:
                yield batches
                batches = []
    
    # Process any remaining reviews
    if batches:
        yield batches

def main():
    """Main function to orchestrate the process"""
    # Read the CSV file
    df = read_csv_with_fallbacks()
    
    # Get the previous progress
    progress = get_progress()
    completed_indices = set(progress["completed_indices"])
    
    # Create a generator for batches
    batch_generator = process_csv()
    total_batches = (len(df) - len(completed_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Processing {len(df)} games in batches of {BATCH_SIZE}")
    print(f"Already processed: {len(completed_indices)} games")
    print(f"Remaining: {len(df) - len(completed_indices)} games")
    print(f"Estimated batches: {total_batches}")
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        batch_count = 0
        
        # Submit initial batches up to MAX_WORKERS
        try:
            for _ in range(min(MAX_WORKERS, total_batches)):
                batch = next(batch_generator, None)
                if batch:
                    futures.append(executor.submit(process_game_batch, batch))
                    batch_count += 1
                else:
                    break
        except StopIteration:
            pass
        
        # Process results and submit new batches
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            while futures:
                # Wait for the next future to complete
                done, futures = as_completed(futures, timeout=1), [f for f in futures if not f.done()]
                
                # Process completed futures
                for future in done:
                    try:
                        results = future.result()
                        
                        # Update the dataframe with the summaries
                        for idx, summary in results:
                            if summary:  # Only update if summary was generated
                                df.at[idx, 'structured_summary'] = summary
                                completed_indices.add(idx)
                        
                        # Save progress
                        save_progress(list(completed_indices), len(df))
                        print(f"Progress saved: {len(completed_indices)}/{len(df)} summaries generated")
                        
                        # Submit a new batch if available
                        try:
                            batch = next(batch_generator, None)
                            if batch:
                                futures.append(executor.submit(process_game_batch, batch))
                                batch_count += 1
                        except StopIteration:
                            pass
                        
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing batch: {e}")
    
    # Save the final dataframe
    df.to_csv(output_file, index=False)
    print(f"All summaries generated and saved to {output_file}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
