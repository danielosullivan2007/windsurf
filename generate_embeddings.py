import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_embeddings(texts, model="text-embedding-3-small"):
    """
    Generate embeddings for given texts using OpenAI's API
    
    Args:
        texts (list): List of text strings to embed
        model (str): OpenAI embedding model to use
    
    Returns:
        list: List of embeddings
    """
    # Validate inputs
    if not texts:
        return []
    
    # Truncate texts to avoid API limits
    texts = [text[:8192] for text in texts]
    
    try:
        # Batch embeddings to manage rate limits
        embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            response = client.embeddings.create(
                input=batch, 
                model=model
            )
            embeddings.extend([data.embedding for data in response.data])
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def main():
    # Load the CSV file
    input_file = '/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/Data/BigGameBoard.csv'
    output_file = '/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/game_embeddings.csv'
    
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Combine text columns for embedding
    text_columns = [col for col in df.columns if 'Text' in col]
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    # Prepare for full dataset embedding
    total_records = len(df)
    batch_size = 100  # Process in batches to manage memory and API calls
    all_embeddings = []
    processed_records = []
    
    # Track progress and cost
    total_cost = 0
    batch_cost_per_100 = 0.0001  # OpenAI pricing
    
    # Generate embeddings in batches
    for i in range(0, total_records, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        
        # Generate embeddings for this batch
        batch_embeddings = generate_embeddings(batch_df['combined_text'].tolist())
        
        # Accumulate results
        all_embeddings.extend(batch_embeddings)
        processed_records.append(batch_df)
        
        # Calculate and track cost
        batch_cost = len(batch_df) * batch_cost_per_100
        total_cost += batch_cost
        
        # Print progress
        print(f"Processed {i + len(batch_df)}/{total_records} records. Cumulative cost: ${total_cost:.4f}")
    
    # Combine all processed records and embeddings
    result_df = pd.concat(processed_records, ignore_index=True)
    embedding_df = pd.DataFrame(all_embeddings)
    final_df = pd.concat([result_df, embedding_df], axis=1)
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    
    print(f"\nTotal records processed: {total_records}")
    print(f"Total embedding generation cost: ${total_cost:.4f}")

if __name__ == '__main__':
    main()
