import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load embeddings
input_file = '/Users/danielosullivan/Desktop/windsurf_testing/windsurf/game_embeddings.csv'
df = pd.read_csv(input_file)

# Extract embedding columns
embedding_columns = df.columns[-1536:]  # Adjust based on actual embedding dimensions
embeddings = df[embedding_columns].values

# Standardize embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Basic embedding statistics
print("Embedding Shape:", embeddings.shape)
print("\nEmbedding Statistics:")
print(pd.DataFrame(embeddings).describe())

def find_optimal_clusters(embeddings, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
        print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
    return silhouette_scores

# Find optimal clusters
silhouette_scores = find_optimal_clusters(embeddings_scaled)

# Select optimal number of clusters (highest silhouette score)
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal number of clusters: {optimal_clusters}")

# Perform clustering with optimal clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_scaled)

# Add cluster labels to dataframe
df['cluster'] = cluster_labels

# t-SNE Dimensionality Reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title(f't-SNE Visualization of {optimal_clusters} Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.savefig('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/cluster_visualization.png')
plt.close()

# Cluster Analysis
def analyze_clusters(df, cluster_column='cluster', text_column='review'):
    cluster_analysis = {}
    for cluster in df[cluster_column].unique():
        cluster_df = df[df[cluster_column] == cluster]
        
        # Aggregate game stats
        cluster_games = cluster_df[['game', 'provider']]
        top_games = cluster_games.groupby('game').size().nlargest(5)
        top_providers = cluster_games['provider'].value_counts().nlargest(3)
        
        # Text analysis
        reviews = cluster_df[text_column]
        
        cluster_analysis[cluster] = {
            'size': len(cluster_df),
            'top_games': top_games.to_dict(),
            'top_providers': top_providers.to_dict(),
            'sample_review': reviews.sample(1).values[0] if len(reviews) > 0 else 'No reviews'
        }
    
    return cluster_analysis

# Perform cluster analysis
cluster_details = analyze_clusters(df)

# Print cluster analysis
print("\nCluster Analysis:")
for cluster, details in cluster_details.items():
    print(f"\nCluster {cluster}:")
    print(f"Size: {details['size']} games")
    print("Top Games:", details['top_games'])
    print("Top Providers:", details['top_providers'])
    print("Sample Review:", details['sample_review'][:300] + '...')

# Save cluster details to file
with open('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/cluster_analysis.txt', 'w') as f:
    for cluster, details in cluster_details.items():
        f.write(f"Cluster {cluster}:\n")
        f.write(f"Size: {details['size']} games\n")
        f.write(f"Top Games: {details['top_games']}\n")
        f.write(f"Top Providers: {details['top_providers']}\n")
        f.write(f"Sample Review: {details['sample_review'][:300] + '...'}\n")

print("\nAnalysis complete. Check cluster_visualization.png and cluster_analysis.txt for details.")
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('silhouette_scores.png')
    plt.close()
    
    return silhouette_scores

# Find optimal clusters
silhouette_scores = find_optimal_clusters(embeddings)
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"\nOptimal number of clusters: {optimal_clusters}")

# Perform clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(embeddings)

# Reduce dimensionality for visualization
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# Cosine Similarity Function
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Find similar games within clusters
def find_similar_games(df, embeddings, cluster, top_n=5):
    """Find top N most similar games within a cluster"""
    cluster_mask = df['cluster'] == cluster
    cluster_embeddings = embeddings[cluster_mask]
    cluster_games = df[cluster_mask]['Text'].values
    
    similarities = []
    for i, game_embedding in enumerate(cluster_embeddings):
        game_sims = [
            (cosine_similarity(game_embedding, other_embedding), cluster_games[j])
            for j, other_embedding in enumerate(cluster_embeddings)
            if j != i
        ]
        game_sims.sort(reverse=True)
        similarities.append({
            'game': cluster_games[i],
            'similar_games': game_sims[:top_n]
        })
    
    return similarities

# Analyze cluster characteristics and similarities
print("\nCluster Summary:")
cluster_summary = df.groupby('cluster')['Text'].agg(['count', 'first', 'last'])
print(cluster_summary)

print("\nSimilar Games in Each Cluster:")
for cluster in range(optimal_clusters):
    print(f"\nCluster {cluster} Similar Games:")
    similar_games = find_similar_games(df, embeddings, cluster)
    for game_info in similar_games[:3]:  # Show first 3 games in detail
        print(f"\nGame: {game_info['game']}")
        print("Similar Games:")
        for sim, similar_game in game_info['similar_games']:
            print(f"  - {similar_game} (Similarity: {sim:.4f})")

# Save cluster information
cluster_summary.to_csv('cluster_summary.csv')
