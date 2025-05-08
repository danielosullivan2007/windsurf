import os
import json
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import dotenv
import argparse

# Default input and output file paths
DEFAULT_INPUT_FILE = "../data/bigwinboard_with_summaries.csv"
DEFAULT_OUTPUT_FILE = "../embeddings/game_summary_embeddings.csv"

# Load API key from .env file
def load_api_key():
    # Try different locations for .env file
    for env_path in ['.env', '../.env', '../../.env']:
        if os.path.exists(env_path):
            print(f"API key found in {env_path}")
            dotenv.load_dotenv(env_path)
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}"
                print(f"API key from .env file: {masked_key}")
                return api_key
    
    # If no API key found, prompt user
    print("No API key found in .env files")
    return None

# Initialize OpenAI client
api_key = load_api_key()
if api_key:
    openai.api_key = api_key
else:
    print("No API key found. Please create a .env file with OPENAI_API_KEY")
    exit(1)

def generate_embedding(text):
    """Generate embedding for a text using OpenAI's API"""
    try:
        # Ensure text is a string and not empty
        if not isinstance(text, str) or not text.strip():
            return None
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def find_optimal_clusters(embeddings, valid_rows, max_clusters=10):
    """Find the optimal number of clusters using silhouette score"""
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    # Create a DataFrame from the valid rows
    valid_df = pd.DataFrame(valid_rows)
    
    # Perform K-means clustering
    print("Finding optimal number of clusters...")
    
    # Find optimal number of clusters using silhouette score
    X = np.array(embeddings)
    best_score = -1
    best_k = 2  # Default to 2 clusters
    
    # Try different numbers of clusters
    for k in tqdm(range(2, 11), desc="Finding optimal clusters"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        score = silhouette_score(X, cluster_labels)
        print(f"K={k}, Silhouette Score={score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal number of clusters: {best_k}")
    
    # Final clustering with optimal k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    return best_k, cluster_labels

def process_embeddings(input_file=DEFAULT_INPUT_FILE, output_file=DEFAULT_OUTPUT_FILE):
    """Process the CSV with summaries and generate embeddings and clusters"""
    checkpoint_file = "../embeddings/embedding_checkpoint.json"
    tsne_output = "../embeddings/game_summary_embeddings_tsne.json"
    # No limit on the number of games to process
    MAX_GAMES = None
    
    # Read the CSV with summaries
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Normalize column names (handle case sensitivity and variations)
    df.columns = [col.lower() for col in df.columns]
    
    # Check if we have 'title' column, if not, try to find it
    if 'title' not in df.columns:
        # Try to identify the title column
        possible_title_cols = [col for col in df.columns if 'title' in col.lower()]
        if possible_title_cols:
            df.rename(columns={possible_title_cols[0]: 'title'}, inplace=True)
        else:
            # If no title column found, use the first column as title
            df.rename(columns={df.columns[0]: 'title'}, inplace=True)
            print(f"Warning: No title column found, using {df.columns[0]} as title")
    
    # Ensure structured_summary column exists
    if 'structured_summary' not in df.columns:
        # Check for summary or similar columns
        summary_cols = [col for col in df.columns if 'summary' in col.lower()]
        if summary_cols:
            df.rename(columns={summary_cols[0]: 'structured_summary'}, inplace=True)
        else:
            print("Error: No structured_summary column found")
            return
    
    # Filter rows with valid summaries
    df = df[df['structured_summary'].notna() & (df['structured_summary'] != "")]
    print(f"Processing {len(df)} valid games with summaries")
    
    # Check for existing checkpoint
    embeddings = []
    valid_rows = []
    processed_titles = []
    start_index = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                embeddings = checkpoint.get('embeddings', [])
                processed_titles = checkpoint.get('processed_titles', [])
                start_index = checkpoint.get('last_processed_index', 0)
                print(f"Resuming from checkpoint. Already processed {len(processed_titles)} games.")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")
            embeddings = []
            processed_titles = []
    
    # Generate embeddings for summaries
    print("Generating embeddings...")
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        # Skip if already processed
        if row['title'] in processed_titles:
            continue
        
        summary = row['structured_summary']
        embedding = generate_embedding(summary)
        
        if embedding:
            embeddings.append(embedding)
            valid_rows.append(row.to_dict())
            processed_titles.append(row['title'])
            
            # Create checkpoint every 10 iterations
            if len(embeddings) % 10 == 0:
                checkpoint = {
                    'embeddings': embeddings,
                    'processed_titles': processed_titles,
                    'last_processed_index': i
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f)
                print(f"Checkpoint saved at index {i}")
    
    # Save embeddings to CSV
    embedding_df = pd.DataFrame(embeddings)
    embedding_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")
    
    # Skip further processing if no embeddings
    if len(embeddings) == 0:
        print("No embeddings to process")
        return
    
    # Perform t-SNE dimensionality reduction
    print("Performing t-SNE dimensionality reduction...")
    
    # Convert embeddings to numpy array
    X = np.array(embeddings)
    print(f"Embeddings shape: {X.shape}")
    
    # Set perplexity based on number of samples
    perplexity = min(30, len(X) - 1)
    print(f"Using perplexity: {perplexity}")
    
    # Perform t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(X)
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    optimal_k, cluster_labels = find_optimal_clusters(X, valid_rows)
    
    # Create t-SNE visualization data
    tsne_data = []
    
    # Create a DataFrame from valid_rows for easier access
    valid_df = pd.DataFrame(valid_rows)
    
    for i, row in enumerate(valid_rows):
        tsne_data.append({
            "title": row['title'],
            "provider": row.get('provider', ""),
            "cluster": int(cluster_labels[i]),
            "tsneX": float(tsne_results[i, 0]),
            "tsneY": float(tsne_results[i, 1]),
            "tsneZ": float(tsne_results[i, 2]),
            "summary": row['structured_summary']
        })
    
    # Save t-SNE results to JSON for visualization
    with open(tsne_output, 'w') as f:
        json.dump(tsne_data, f, indent=2)
    
    print(f"t-SNE results and clusters saved to {tsne_output}")
    
    # Save cluster information for analysis
    # Ensure all arrays have the same length
    num_rows = len(valid_rows)
    
    # Create DataFrame with consistent lengths
    cluster_data = {
        'Title': [row['title'] for row in valid_rows[:num_rows]],
        'Cluster': cluster_labels[:num_rows],
        'Summary': [row['structured_summary'] for row in valid_rows[:num_rows]]
    }
    
    # Add Provider column if it exists
    if any('provider' in row for row in valid_rows):
        cluster_data['Provider'] = [row.get('provider', "") for row in valid_rows[:num_rows]]
    else:
        cluster_data['Provider'] = [""] * num_rows
        
    cluster_df = pd.DataFrame(cluster_data)
    
    cluster_df.to_csv("game_clusters.csv", index=False)
    print("Cluster assignments saved to game_clusters.csv")
    
    return tsne_data, cluster_labels, valid_rows

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate embeddings for casino game summaries")
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE,
                        help=f'Input CSV file with summaries (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE,
                        help=f'Output CSV file for embeddings (default: {DEFAULT_OUTPUT_FILE})')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process embeddings with specified input/output files
    process_embeddings(input_file=args.input, output_file=args.output)
