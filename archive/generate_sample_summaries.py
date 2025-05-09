import pandas as pd
import os
import time
import openai
import sys
import json
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Get API key - try different methods to get it
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY environment variable not set.")
    api_key = input("Please enter your OpenAI API key: ").strip()
    if not api_key:
        sys.exit(1)

# Print API key info for debugging
print(f"API Key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 9 else 'too short'}")

# Create OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

# Configuration
SAMPLE_SIZE = 10  # Number of games to process

def create_summary(title, review_text):
    """
    Generate a structured 4-line summary using OpenAI's API
    """
    prompt = f"""
    Please provide a concise 4-line summary of the following casino game review for '{title}':
    
    Line 1: Overview focusing on the game theme
    Line 2: Focus on the game features
    Line 3: Summary of the reviewer's verdict
    Line 4: Description of the game aesthetics and audio (if mentioned)
    
    Make each line brief but informative.
    
    Review: {review_text}
    """
    
    try:
        # Test API key with a simple call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional game analyst creating concise summaries of casino game reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.5
        )
        
        # Return the generated summary
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def process_sample():
    """
    Process a sample of the CSV and generate structured summaries
    """
    # Path to the original cleaned CSV file
    input_file = "bigwinboard_cleaned.csv"
    output_file = "sample_summaries.csv"
    
    # Try to read the CSV file
    try:
        df = pd.read_csv(input_file, quoting=pd.io.common.csv.QUOTE_MINIMAL, escapechar='\\', encoding='utf-8')
        print(f"Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        try:
            # Try an alternative method
            df = pd.read_csv(input_file, engine='python', encoding='utf-8')
            print(f"Loaded with python engine: {len(df)} rows")
        except Exception as e2:
            print(f"All CSV reading methods failed: {e2}")
            return
    
    # Print columns
    print(f"Columns in the CSV: {df.columns.tolist()}")
    
    # Determine the correct column names
    title_column = 'Title' if 'Title' in df.columns else df.columns[0]
    
    # Try to find the review column
    review_column = None
    for col_name in ['review', 'Review', 'review_text', 'content']:
        if col_name in df.columns:
            review_column = col_name
            break
    
    # If no column was found, use the column with the longest text
    if not review_column:
        max_length = 0
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > max_length:
                    max_length = avg_len
                    review_column = col
    
    print(f"Using column '{title_column}' for titles and '{review_column}' for reviews")
    
    # Take a sample
    sample_df = df.head(SAMPLE_SIZE)
    sample_df['structured_summary'] = ""
    
    # Process each row
    for i, row in sample_df.iterrows():
        print(f"\nProcessing game {i+1}/{SAMPLE_SIZE}: {row[title_column]}")
        
        try:
            # Get title and review text
            title = row[title_column]
            review = str(row[review_column])
            
            if len(review) < 50:
                print(f"Review text too short: {len(review)} chars")
                continue
                
            # Truncate very long reviews
            if len(review) > 4000:
                review = review[:4000] + "..."
                
            # Generate summary
            print(f"Generating summary with OpenAI API...")
            summary = create_summary(title, review)
            
            # Update dataframe
            sample_df.at[i, 'structured_summary'] = summary
            
            print(f"Summary generated ({len(summary)} chars)")
            print(f"Summary: {summary[:100]}...")
            
            # Small delay
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing row: {e}")
    
    # Save the results
    sample_df.to_csv(output_file, index=False)
    print(f"\nSample summaries saved to {output_file}")
    
    # Also save as JSON for easier inspection
    sample_data = []
    for _, row in sample_df.iterrows():
        if row['structured_summary']:
            sample_data.append({
                'title': row[title_column],
                'summary': row['structured_summary']
            })
    
    with open('sample_summaries.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample data also saved as JSON")
    
    return sample_df

if __name__ == "__main__":
    start_time = time.time()
    process_sample()
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
