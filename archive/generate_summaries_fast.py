import pandas as pd
import csv
import os
import time
import openai
import json
import sys
import concurrent.futures
from tqdm import tqdm

# Get API key from environment or prompt user
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        sys.exit(1)

# Set the API key - we'll use it directly in the API calls
# Don't set it globally to avoid issues with threading
print(f"Using API key: {api_key[:5]}...{api_key[-5:] if api_key else ''}")

# Configuration
MAX_WORKERS = 5  # Number of parallel workers
BATCH_SIZE = 5    # Number of reviews to batch in a single API call
MAX_GAMES = None  # Maximum number of games to process (None for all)

def create_structured_summary_batch(reviews_data, api_key):
    """
    Generate structured summaries for a batch of reviews using OpenAI's API
    
    Args:
        reviews_data: List of tuples (index, title, review_text)
        api_key: OpenAI API key to use
    
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
        
        # Extract summaries from the response
        results = []
        for i, choice in enumerate(response.choices):
            if i < len(reviews_data):  # Safety check
                idx = reviews_data[i][0]
                summary = choice.message.content.strip()
                results.append((idx, summary))
        
        return results
    except Exception as e:
        print(f"Error generating batch summaries: {e}")
        # Return empty summaries for the batch
        return [(idx, "") for idx, _, _ in reviews_data]

def process_game_batch(batch_data):
    """Process a batch of games to get summaries"""
    try:
        # Pass the API key explicitly to the function
        return create_structured_summary_batch(batch_data, api_key)
    except Exception as e:
        print(f"Error processing batch: {e}")
        return [(idx, "") for idx, _, _ in batch_data]

def process_csv():
    """
    Process the original CSV and generate structured summaries for each game review
    """
    # Path to the original cleaned CSV file
    input_file = "bigwinboard_cleaned.csv"
    output_file = "bigwinboard_with_summaries.csv"
    
    # Read the original CSV file with custom parsing
    try:
        # Try to manually read and parse the file
        print(f"Reading {input_file} line by line...")
        with open(input_file, 'r', encoding='utf-8') as f:
            # Read header line
            header_line = f.readline().strip()
            headers = header_line.split(',')
            
            data = []
            raw_lines = f.readlines()
            
            # Process the file in chunks to handle quotation marks correctly
            i = 0
            while i < len(raw_lines):
                line = raw_lines[i].strip()
                
                # Check if we have an open quote that might span multiple lines
                if line.count('"') % 2 != 0:
                    j = i + 1
                    while j < len(raw_lines) and line.count('"') % 2 != 0:
                        line += raw_lines[j].strip()
                        j += 1
                    i = j
                else:
                    i += 1
                    
                # Split by comma, respecting quoted fields
                fields = []
                field = ''
                in_quotes = False
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        fields.append(field.strip())
                        field = ''
                    else:
                        field += char
                        
                # Add the last field
                fields.append(field.strip())
                
                # Add row if it has the right number of fields
                if len(fields) == len(headers):
                    data.append(fields)
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            print(f"Successfully loaded {len(df)} rows from {input_file}")
            
    except Exception as e:
        print(f"Custom parsing failed. Trying standard methods...")
        try:
            # Try with different parsing settings
            df = pd.read_csv(input_file, quoting=csv.QUOTE_NONE, sep=',', encoding='utf-8', on_bad_lines='skip')
            print(f"Successfully loaded {len(df)} rows using fallback method")
        except Exception as e2:
            print(f"All parsing methods failed. Error: {e2}")
            return
    
    # Print available columns to debug
    print("Available columns in the dataframe:")
    print(df.columns.tolist())
    
    # Determine the correct column names
    title_column = 'Title' if 'Title' in df.columns else df.columns[0]  # Assume first column is title if 'Title' not found
    
    # Find the review column - it's likely to be the longest text field
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    review_column = None
    
    if 'review' in df.columns:
        review_column = 'review'
    elif 'Review' in df.columns:
        review_column = 'Review'
    else:
        # Find the column with the longest average text length
        max_length = 0
        for col in text_columns:
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > max_length:
                max_length = avg_length
                review_column = col
    
    print(f"Using '{title_column}' as the title column and '{review_column}' as the review column")
    
    # Create a new column for structured summaries
    df['structured_summary'] = ""
    
    # Limit the number of games if specified
    if MAX_GAMES:
        df = df.head(MAX_GAMES)
        print(f"Limited to processing {MAX_GAMES} games for faster testing")
    
    # Prepare batches for processing
    batches = []
    current_batch = []
    
    for i, row in df.iterrows():
        try:
            game_title = row[title_column]
            review_text = row[review_column]
            
            # Skip if review is empty or too short
            if pd.isna(review_text) or len(str(review_text).strip()) < 50:
                continue
                
            # Add to current batch
            current_batch.append((i, game_title, str(review_text)))
            
            # If batch is full, add to batches list
            if len(current_batch) >= BATCH_SIZE:
                batches.append(current_batch)
                current_batch = []
        except Exception as e:
            print(f"Error preparing row {i}: {e}")
            continue
    
    # Add any remaining items as the last batch
    if current_batch:
        batches.append(current_batch)
    
    print(f"Prepared {len(batches)} batches of approx. {BATCH_SIZE} games each")
    
    # Process batches in parallel
    results = []
    
    # Create a ThreadPoolExecutor to process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches to the executor
        future_to_batch = {executor.submit(process_game_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_batch), 
                                   total=len(batches), 
                                   desc="Processing batches")):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                results.extend(batch_results)
                
                # Update dataframe with results
                for idx, summary in batch_results:
                    df.at[idx, 'structured_summary'] = summary
                
                # Save progress every 5 batches
                if i % 5 == 0:
                    df.to_csv(output_file, index=False)
                    print(f"Progress saved: {len(results)}/{len(df)} summaries generated")
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    # Save the final result
    df.to_csv(output_file, index=False)
    print(f"All summaries generated and saved to {output_file}")
    return df

if __name__ == "__main__":
    start_time = time.time()
    process_csv()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
