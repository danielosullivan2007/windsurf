import os
import json
import pandas as pd
import numpy as np
import openai
import time
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
DEFAULT_INPUT_FILE = "../data/bigwinboard_with_summaries_final.csv"
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
    progress_log = "embedding_progress.log"
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
            # Use the first title-like column
            df.rename(columns={possible_title_cols[0]: 'title'}, inplace=True)
        else:
            print("Could not find a title column in the CSV")
            return
    
    # Check if we have 'structured_summary' column or find a suitable alternative
    if 'structured_summary' not in df.columns:
        # Try to identify a summary column
        possible_summary_cols = [col for col in df.columns if 'summary' in col.lower()]
        if possible_summary_cols:
            # Use the first summary-like column
            df.rename(columns={possible_summary_cols[0]: 'structured_summary'}, inplace=True)
        else:
            print("Could not find a summary column in the CSV")
            return
    
    # Filter out games with invalid or missing summaries
    df = df[df['structured_summary'].notna() & (df['structured_summary'].str.len() > 5)]
    df = df.reset_index(drop=True)  # Reset index after filtering
    print(f"Found {len(df)} valid games with summaries")
    
    # Prepare variables for storing embeddings and valid rows
    valid_rows = []
    embeddings = []
    processed_titles = []
    start_index = 0
    
    # Check if there's a checkpoint file to resume from
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                embeddings = checkpoint.get('embeddings', [])
                processed_titles = checkpoint.get('processed_titles', [])
                start_index = checkpoint.get('last_processed_index', 0) + 1
                print(f"Resuming from checkpoint at index {start_index}")
                print(f"Already processed {len(embeddings)} embeddings")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            embeddings = []
            processed_titles = []
            start_index = 0
    else:
        embeddings = []
        processed_titles = []
        start_index = 0
        
    # Open progress log for writing
    with open(progress_log, 'w') as log_file:
        log_file.write(f"Starting embedding generation at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Input file: {input_file}\n")
        log_file.write(f"Output file: {output_file}\n")
    
    start_time = time.time()
    print("Generating embeddings...")
    
    # Calculate the number of rows to process (only those starting from the start_index)
    rows_to_process = df.iloc[start_index:].shape[0]
    processed_count = 0
    start_time = time.time()
    last_update_time = start_time
    total_time_per_item = 0
    
    # Get a set of titles that need processing (titles with summaries but no embeddings yet)
    titles_to_process = set(df['title']) - set(processed_titles)
    print(f"Found {len(titles_to_process)} games that need embeddings")
    
    # If no new titles to process, we're done
    if len(titles_to_process) == 0:
        print("No new games need embeddings. All summaries already have embeddings.")
        return embeddings, None, valid_rows
    
    # Filter dataframe to only include games that need processing
    df_to_process = df[df['title'].isin(titles_to_process)]
    print(f"Processing {len(df_to_process)} games...")
    
    # Process with progress bar and detailed reporting
    with tqdm(total=len(df_to_process), desc="Generating embeddings", unit="game") as pbar:
        for i, row in df_to_process.iterrows():
            iter_start = time.time()
            
            title = row['title']
            summary = row['structured_summary']
            
            # Generate embedding
            embedding = generate_embedding(summary)
            
            if embedding:
                embeddings.append(embedding)
                valid_rows.append(row.to_dict())
                processed_titles.append(title)
                processed_count += 1
                
                # Calculate time statistics
                iter_time = time.time() - iter_start
                total_time_per_item += iter_time
                avg_time = total_time_per_item / processed_count
                
                # Display detailed progress every 10 items or at least every 30 seconds
                current_time = time.time()
                if processed_count % 10 == 0 or (current_time - last_update_time) > 30:
                    # Calculate time estimates
                    elapsed = current_time - start_time
                    total_to_process = len(df_to_process)
                    items_remaining = total_to_process - pbar.n
                    estimated_remaining = avg_time * items_remaining
                    
                    # Print detailed progress
                    print(f"\nProgress: {processed_count}/{total_to_process} games processed ({processed_count/total_to_process*100:.1f}%)")
                    print(f"Time elapsed: {elapsed/60:.1f} minutes")
                    print(f"Estimated time remaining: {estimated_remaining/60:.1f} minutes")
                    print(f"Last processed: {title[:40]}" + ('...' if len(title) > 40 else ''))
                    
                    last_update_time = current_time
                
                # Create checkpoint every 10 iterations
                if processed_count % 10 == 0:
                    checkpoint = {
                        'embeddings': embeddings,
                        'processed_titles': processed_titles,
                        'last_processed_index': i
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint, f)
            
            # Update progress bar
            pbar.update(1)
    
    # Log completion of embedding generation
    with open(progress_log, 'a') as log_file:
        total_time = time.time() - start_time
        log_file.write(f"\nCompleted embedding generation at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total embeddings generated: {len(embeddings)}\n")
        log_file.write(f"Total time taken: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        if processed_count > 0:
            log_file.write(f"Average time per embedding: {total_time/processed_count:.2f} seconds\n")
    
    # Save embeddings to CSV
    embedding_df = pd.DataFrame(embeddings)
    embedding_df.to_csv(output_file, index=False)
    print(f"\nEmbeddings saved to {output_file}")
    print(f"Generated {len(embeddings)} embeddings in {total_time/60:.2f} minutes")
    print(f"Detailed progress log saved to {progress_log}")
    
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
    
    # Find optimal number of clusters (show progress)
    print("\nFinding optimal number of clusters...")
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
