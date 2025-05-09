import pandas as pd
import numpy as np
import json
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def fix_visualization_data():
    """
    Fix the visualization data by processing all embeddings and generating
    consistent t-SNE visualization and cluster data.
    """
    print("Fixing visualization data...")
    
    # Input and output files
    embedding_file = "../embeddings/game_summary_embeddings.csv"
    summary_file = "../data/all_summaries_merged.csv"
    tsne_output = "../embeddings/game_summary_embeddings_tsne.json"
    cluster_output = "game_clusters.csv"
    
    # Load embeddings
    try:
        embedding_df = pd.read_csv(embedding_file)
        print(f"Loaded {len(embedding_df)} embeddings from {embedding_file}")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
    
    # Load summaries
    try:
        summary_df = pd.read_csv(summary_file)
        print(f"Loaded {len(summary_df)} summaries from {summary_file}")
    except Exception as e:
        print(f"Error loading summaries: {e}")
        return
    
    # Normalize column names
    summary_df.columns = [col.lower() for col in summary_df.columns]
    
    # Create a dictionary of summaries by title
    summary_dict = {}
    for _, row in summary_df.iterrows():
        if 'title' in row and 'structured_summary' in row:
            summary_dict[row['title']] = {
                'summary': row['structured_summary'],
                'provider': row.get('provider', "")
            }
    
    print(f"Created summary dictionary with {len(summary_dict)} entries")
    
    # Convert embeddings to numpy array
    X = embedding_df.values
    print(f"Embeddings shape: {X.shape}")
    
    # Perform t-SNE dimensionality reduction
    print("Performing t-SNE dimensionality reduction...")
    perplexity = min(30, len(X) - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(X)
    print(f"t-SNE results shape: {tsne_results.shape}")
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
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
    
    # Create t-SNE visualization data
    tsne_data = []
    titles = list(summary_dict.keys())
    
    for i in range(len(titles)):
        if i < len(titles) and i < len(tsne_results) and i < len(cluster_labels):
            title = titles[i]
            if title in summary_dict:
                tsne_data.append({
                    "title": title,
                    "provider": summary_dict[title].get('provider', ""),
                    "cluster": int(cluster_labels[i]),
                    "tsneX": float(tsne_results[i, 0]),
                    "tsneY": float(tsne_results[i, 1]),
                    "tsneZ": float(tsne_results[i, 2]),
                    "summary": summary_dict[title].get('summary', "")
                })
    
    print(f"Created {len(tsne_data)} items for t-SNE visualization")
    
    # Save t-SNE results to JSON for visualization
    with open(tsne_output, 'w') as f:
        json.dump(tsne_data, f, indent=2)
    
    print(f"t-SNE results saved to {tsne_output}")
    
    # Create cluster data for CSV
    cluster_data = {
        'Title': [item['title'] for item in tsne_data],
        'Cluster': [item['cluster'] for item in tsne_data],
        'Summary': [item['summary'] for item in tsne_data]
    }
    
    # Add Provider column if available
    if any('provider' in item and item['provider'] for item in tsne_data):
        cluster_data['Provider'] = [item.get('provider', "") for item in tsne_data]
    
    # Create and save cluster DataFrame
    cluster_df = pd.DataFrame(cluster_data)
    cluster_df.to_csv(cluster_output, index=False)
    
    print(f"Cluster assignments saved to {cluster_output}")
    print(f"Fixed visualization data for {len(tsne_data)} games")

if __name__ == "__main__":
    fix_visualization_data()
