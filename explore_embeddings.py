import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load embeddings
df = pd.read_csv('game_embeddings.csv')

# Extract embedding columns
embedding_columns = df.columns[-1536:]  # Adjust based on actual embedding dimensions
embeddings = df[embedding_columns].values

# Basic embedding statistics
print("Embedding Shape:", embeddings.shape)
print("\nEmbedding Statistics:")
print(pd.DataFrame(embeddings).describe())

# Clustering Analysis
def find_optimal_clusters(embeddings, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Cluster Counts')
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
