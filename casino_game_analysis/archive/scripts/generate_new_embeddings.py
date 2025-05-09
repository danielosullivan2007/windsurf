#!/usr/bin/env python3
"""
Generate embeddings for casino game summaries that don't have embeddings yet.
This script builds on generate_summary_embeddings.py but focuses only on new summaries.
"""

import os
import json
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Default file paths
INPUT_FILE = "../data/bigwinboard_with_summaries_complete.csv"
EMBEDDINGS_FILE = "../embeddings/game_summary_embeddings.csv"
NEW_EMBEDDINGS_FILE = "../embeddings/new_game_summary_embeddings.csv"
COMBINED_EMBEDDINGS_FILE = "../embeddings/combined_game_summary_embeddings.csv"

def generate_embedding(text):
    """Generate embedding for a text using OpenAI's API"""
    try:
        # Ensure text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            return None
            
        # Add context about casino games to improve embedding quality
        enhanced_text = f"Casino game summary: {text}"
        
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=enhanced_text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_new_embeddings(input_file=INPUT_FILE, 
                          embeddings_file=EMBEDDINGS_FILE,
                          new_embeddings_file=NEW_EMBEDDINGS_FILE,
                          combined_file=COMBINED_EMBEDDINGS_FILE):
    """Process games with summaries but no embeddings yet"""
    
    # Read the CSV with summaries
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Filter to only include games with structured summaries
    df = df[pd.notna(df['structured_summary'])].copy()
    print(f"Found {len(df)} games with structured summaries")
    
    # Load existing embeddings if available
    existing_embeddings = []
    existing_titles = []
    
    if os.path.exists(embeddings_file):
        try:
            embeddings_df = pd.read_csv(embeddings_file, header=None)
            print(f"Loaded {len(embeddings_df)} existing embeddings")
            
            # Get the corresponding titles from the first N rows of the input file
            # where N is the number of existing embeddings
            existing_titles = df['Title'].iloc[:len(embeddings_df)].tolist()
            existing_embeddings = embeddings_df.values.tolist()
        except Exception as e:
            print(f"Error loading existing embeddings: {e}")
    
    # Find games that have summaries but no embeddings
    games_with_embeddings = set(existing_titles)
    new_games = df[~df['Title'].isin(games_with_embeddings)].copy()
    
    print(f"Found {len(new_games)} games with summaries but no embeddings")
    
    if len(new_games) == 0:
        print("No new embeddings to generate")
        return
    
    # Generate embeddings for new games
    new_embeddings = []
    processed_titles = []
    
    print("Generating embeddings for new games...")
    for i, row in tqdm(new_games.iterrows(), total=len(new_games)):
        title = row['Title']
        summary = row['structured_summary']
        
        # Skip if summary is missing
        if pd.isna(summary):
            print(f"Skipping {title} - missing summary")
            continue
        
        # Generate embedding
        embedding = generate_embedding(summary)
        
        if embedding:
            new_embeddings.append(embedding)
            processed_titles.append(title)
        else:
            print(f"Failed to generate embedding for {title}")
    
    # Save new embeddings to CSV
    if new_embeddings:
        new_embeddings_df = pd.DataFrame(new_embeddings)
        new_embeddings_df.to_csv(new_embeddings_file, index=False, header=False)
        print(f"Saved {len(new_embeddings)} new embeddings to {new_embeddings_file}")
        
        # Combine existing and new embeddings
        combined_embeddings = existing_embeddings + new_embeddings
        combined_df = pd.DataFrame(combined_embeddings)
        combined_df.to_csv(combined_file, index=False, header=False)
        print(f"Saved {len(combined_embeddings)} combined embeddings to {combined_file}")
        
        # Create a mapping file to track which titles correspond to which embeddings
        titles_mapping = existing_titles + processed_titles
        titles_df = pd.DataFrame({'Title': titles_mapping})
        titles_df.to_csv("../embeddings/embedding_titles_mapping.csv", index=False)
        print(f"Saved titles mapping with {len(titles_mapping)} entries")
        
        return len(new_embeddings)
    else:
        print("No new embeddings were generated")
        return 0

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate embeddings for new casino game summaries")
    parser.add_argument('--input', type=str, default=INPUT_FILE,
                        help=f'Input CSV file with summaries (default: {INPUT_FILE})')
    parser.add_argument('--existing', type=str, default=EMBEDDINGS_FILE,
                        help=f'Existing embeddings file (default: {EMBEDDINGS_FILE})')
    parser.add_argument('--output', type=str, default=NEW_EMBEDDINGS_FILE,
                        help=f'Output file for new embeddings (default: {NEW_EMBEDDINGS_FILE})')
    parser.add_argument('--combined', type=str, default=COMBINED_EMBEDDINGS_FILE,
                        help=f'Output file for combined embeddings (default: {COMBINED_EMBEDDINGS_FILE})')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process new embeddings
    process_new_embeddings(
        input_file=args.input,
        embeddings_file=args.existing,
        new_embeddings_file=args.output,
        combined_file=args.combined
    )

if __name__ == "__main__":
    main()
