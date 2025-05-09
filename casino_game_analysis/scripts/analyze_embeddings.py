#!/usr/bin/env python3
"""
Analyze game embeddings using t-SNE visualization and clustering.
Separate script to perform advanced embedding analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('embeddings_analysis.log'),
        logging.StreamHandler()
    ]
)

# Default file paths
DEFAULT_EMBEDDINGS_FILE = "../embeddings/game_summary_embeddings.csv"
DEFAULT_SUMMARIES_FILE = "../data/bigwinboard_with_summaries_final.csv"
DEFAULT_TSNE_OUTPUT = "../embeddings/game_summary_embeddings_tsne.json"
DEFAULT_CLUSTERS_OUTPUT = "../embeddings/game_clusters.csv"

def find_optimal_clusters(embeddings, max_clusters=10):
    """
    Find the optimal number of clusters using silhouette score.
    
    Args:
        embeddings (np.array): Numpy array of embeddings
        max_clusters (int): Maximum number of clusters to test
    
    Returns:
        int: Optimal number of clusters
    """
    # Standardize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Test different cluster numbers
    silhouette_scores = []
    for n_clusters in range(2, min(max_clusters + 1, len(embeddings))):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_embeddings)
            score = silhouette_score(scaled_embeddings, cluster_labels)
            silhouette_scores.append(score)
        except Exception as e:
            logging.warning(f"Clustering failed for {n_clusters} clusters: {e}")
            break
    
    # Find optimal number of clusters
    if silhouette_scores:
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        logging.info(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    return 3  # Default fallback

def analyze_embeddings(
    embeddings_file=DEFAULT_EMBEDDINGS_FILE,
    summaries_file=DEFAULT_SUMMARIES_FILE,
    tsne_output=DEFAULT_TSNE_OUTPUT,
    clusters_output=DEFAULT_CLUSTERS_OUTPUT
):
    """
    Analyze game embeddings using t-SNE and clustering.
    
    Args:
        embeddings_file (str): Path to CSV with game embeddings
        summaries_file (str): Path to CSV with game summaries
        tsne_output (str): Path to save t-SNE visualization data
        clusters_output (str): Path to save cluster assignments
    """
    # Load embeddings
    try:
        embeddings_df = pd.read_csv(embeddings_file)
        logging.info(f"Loaded {len(embeddings_df)} embeddings")
    except Exception as e:
        logging.error(f"Error reading embeddings: {e}")
        return
    
    # Load summaries for additional context
    try:
        summaries_df = pd.read_csv(summaries_file)
        logging.info(f"Loaded {len(summaries_df)} summaries")
    except Exception as e:
        logging.error(f"Error reading summaries: {e}")
        return
    
    # Merge embeddings with summaries
    merged_df = pd.merge(embeddings_df, summaries_df[['title', 'structured_summary']], 
                         on='title', how='left')
    
    # Prepare embeddings for analysis
    embeddings = np.array(merged_df['embedding'].apply(eval).tolist())
    titles = merged_df['title'].tolist()
    summaries = merged_df['structured_summary'].tolist()
    
    # Perform t-SNE dimensionality reduction
    logging.info("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_results = tsne.fit_transform(embeddings)
    
    # Perform clustering
    logging.info("Performing clustering...")
    optimal_clusters = find_optimal_clusters(embeddings)
    
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(StandardScaler().fit_transform(embeddings))
    
    # Prepare t-SNE visualization data
    tsne_data = []
    for i, (x, y) in enumerate(tsne_results):
        tsne_data.append({
            'title': titles[i],
            'x': float(x),
            'y': float(y),
            'cluster': int(cluster_labels[i]),
            'summary': summaries[i]
        })
    
    # Save t-SNE data
    with open(tsne_output, 'w') as f:
        json.dump(tsne_data, f, indent=2)
    logging.info(f"Saved t-SNE visualization data to {tsne_output}")
    
    # Create cluster CSV
    cluster_df = pd.DataFrame({
        'Title': titles,
        'Cluster': cluster_labels,
        'Summary': summaries
    })
    cluster_df.to_csv(clusters_output, index=False)
    logging.info(f"Saved cluster assignments to {clusters_output}")
    
    # Optional: Visualize clusters (can be commented out)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1], 
        c=cluster_labels, 
        cmap='viridis'
    )
    plt.title('t-SNE Visualization of Game Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('../embeddings/game_embeddings_tsne.png')
    plt.close()
    logging.info("Saved t-SNE visualization plot")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze game embeddings")
    parser.add_argument('--embeddings', type=str, default=DEFAULT_EMBEDDINGS_FILE,
                        help=f'Input embeddings CSV (default: {DEFAULT_EMBEDDINGS_FILE})')
    parser.add_argument('--summaries', type=str, default=DEFAULT_SUMMARIES_FILE,
                        help=f'Input summaries CSV (default: {DEFAULT_SUMMARIES_FILE})')
    parser.add_argument('--tsne', type=str, default=DEFAULT_TSNE_OUTPUT,
                        help=f'Output t-SNE JSON (default: {DEFAULT_TSNE_OUTPUT})')
    parser.add_argument('--clusters', type=str, default=DEFAULT_CLUSTERS_OUTPUT,
                        help=f'Output clusters CSV (default: {DEFAULT_CLUSTERS_OUTPUT})')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Analyze embeddings
    analyze_embeddings(
        embeddings_file=args.embeddings,
        summaries_file=args.summaries,
        tsne_output=args.tsne,
        clusters_output=args.clusters
    )

if __name__ == "__main__":
    main()
