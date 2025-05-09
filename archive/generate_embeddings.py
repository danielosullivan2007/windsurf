import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API client
client = OpenAI()

def generate_embeddings(texts, model="text-embedding-3-small"):
    """Generate embeddings for given texts using OpenAI API
    
    Args:
        texts (list): List of text strings to embed
        model (str): OpenAI embedding model to use
    
    Returns:
        list: List of embeddings
    """
    # Validate inputs
    if not texts:
        return []
    
    # Remove empty or NaN texts
    texts = [str(text).strip() for text in texts if pd.notna(text) and str(text).strip()]
    
    if not texts:
        return []
    
    # Generate embeddings using OpenAI API
    try:
        response = client.embeddings.create(
            input=texts, 
            model=model
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def main():
    # Input file path
    input_file = '/Users/danielosullivan/Desktop/windsurf_testing/windsurf/bigwinboard_cleaned.csv'
    
    # Potential output files
    possible_output_files = [
        '/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/game_embeddings.csv',
        '/Users/danielosullivan/Desktop/windsurf_testing/windsurf/game_embeddings.csv'
    ]
    
    # Find first writable output file
    output_file = next((f for f in possible_output_files if os.access(os.path.dirname(f), os.W_OK)), None)
    if not output_file:
        raise PermissionError(f"No writable output file found. Tried: {possible_output_files}")
    
    # Read input CSV
    df = pd.read_csv(input_file)
    
    # Prepare for full dataset embedding
    total_records = len(df)
    batch_size = 100  # Process in batches to manage memory and API calls
    all_embeddings = []
    processed_records = []
    
    # Track progress and cost
    total_cost = 0
    embedding_cost_per_1k = 0.02  # OpenAI pricing for text-embedding-3-small per 1000 tokens
    
    # Confirm before proceeding
    print(f"Total records to process: {total_records}")
    print(f"Estimated total cost: ${(total_records / 1000) * embedding_cost_per_1k:.2f}")
    
    user_confirmation = input("Do you want to proceed with generating embeddings? (yes/no): ").lower()
    if user_confirmation != 'yes':
        print("Embedding generation cancelled.")
        return
    
    # Generate embeddings in batches
    for i in range(0, total_records, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        # Skip rows with empty reviews
        batch_reviews = [str(review).strip() for review in batch_df['review'] if pd.notna(review) and str(review).strip()]
        
        if not batch_reviews:
            continue
        
        # Generate embeddings for this batch's reviews
        batch_embeddings = generate_embeddings(batch_reviews)
        
        # Accumulate results
        all_embeddings.extend(batch_embeddings)
        processed_records.append(batch_df)
        
        # Calculate and track cost
        batch_cost = len(batch_reviews) * embedding_cost_per_1k / 1000
        total_cost += batch_cost
        
        # Print progress
        print(f"Processed {i + len(batch_reviews)}/{total_records} records. Cumulative cost: ${total_cost:.4f}")
    
    # Combine all processed records and embeddings
    result_df = pd.concat(processed_records, ignore_index=True)
    
    # Create embedding columns
    embedding_columns = [f'embedding_{i}' for i in range(len(all_embeddings[0]))]
    embedding_df = pd.DataFrame(all_embeddings, columns=embedding_columns)
    
    # Combine results
    final_df = pd.concat([result_df, embedding_df], axis=1)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    
    print(f"Output file: {output_file}")
    print(f"Total records processed: {total_records}")
    print(f"Total embedding generation cost: ${total_cost:.4f}")

if __name__ == '__main__':
    main()
