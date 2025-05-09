import pandas as pd
import csv
import os
import time
import openai
import json
import sys
from tqdm import tqdm

# Get API key from environment or prompt user
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        sys.exit(1)

# Set the API key
openai.api_key = api_key
print(f"Using API key: {api_key[:5]}...{api_key[-5:] if api_key else ''}")

def create_structured_summary(review_text, game_title):
    """
    Generate a structured 4-line summary of a game review using OpenAI's API
    """
    prompt = f"""
    Please provide a concise 4-line summary of the following casino game review for '{game_title}'.
    
    Line 1: Overview focusing on the game theme
    Line 2: Focus on the game features
    Line 3: Summary of the reviewer's verdict
    Line 4: Description of the game aesthetics and audio (if mentioned)
    
    Make each line brief but informative.
    
    Review: {review_text}
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional game analyst creating concise summaries of casino game reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary for {game_title}: {e}")
        return ""

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
    
    if 'Review' in df.columns:
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
    
    # Process each review and generate a summary
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
        try:
            game_title = row[title_column]
            review_text = row[review_column]
            
            # Skip if review is empty or too short
            if pd.isna(review_text) or len(str(review_text).strip()) < 50:
                continue
        
            # Generate structured summary
            summary = create_structured_summary(str(review_text), game_title)
        
            # Save the summary to the dataframe
            df.at[i, 'structured_summary'] = summary
            
            # Save progress after each batch of 10 to avoid losing work
            if (i + 1) % 10 == 0:
                df.to_csv(output_file, index=False)
                print(f"Progress saved: {i+1}/{len(df)} summaries generated")
            
            # Respect API rate limits
            time.sleep(1)
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    
    # Save the final result
    df.to_csv(output_file, index=False)
    print(f"All summaries generated and saved to {output_file}")

if __name__ == "__main__":
    process_csv()
