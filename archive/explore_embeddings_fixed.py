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
df = pd.read_csv(input_file, low_memory=False)

# Extract embedding columns
embedding_columns = [col for col in df.columns if 'embedding_' in col]
print(f"Number of embedding columns: {len(embedding_columns)}")
embeddings = df[embedding_columns].values
df_cleaned = df.dropna(subset=embedding_columns)
embeddings = df_cleaned[embedding_columns].values

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
    
    # Visualize silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Cluster Counts')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/silhouette_scores.png')
    plt.close()
    
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
df_cleaned['cluster'] = cluster_labels

# t-SNE Dimensionality Reduction
tsne = TSNE(n_components=3, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df_cleaned['cluster'], cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title(f't-SNE Visualization of {optimal_clusters} Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.tight_layout()
plt.savefig('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/cluster_visualization.png')
plt.close()

def analyze_clusters(df, cluster_column='cluster', text_column='Title'):
    """Analyze characteristics of each cluster"""
    cluster_analysis = {}
    for cluster in df[cluster_column].unique():
        cluster_df = df[df[cluster_column] == cluster]
        
        # Aggregate game stats
        cluster_games = cluster_df[['Title', 'Developer']]
        top_games = cluster_games.groupby('Title').size().nlargest(5)
        top_providers = cluster_games['Developer'].value_counts().nlargest(3)
        
        # Text analysis
        titles = cluster_df[text_column]
        
        # Get representative reviews (if available)
        try:
            # Filter out NaN and non-string reviews
            valid_reviews = cluster_df['review'][cluster_df['review'].notna() & cluster_df['review'].apply(lambda x: isinstance(x, str))]
            representative_reviews = valid_reviews.sample(min(3, len(valid_reviews))) if len(valid_reviews) > 0 else ['No valid reviews']
        except KeyError:
            representative_reviews = ['No reviews available']
        
        cluster_analysis[cluster] = {
            'size': len(cluster_df),
            'top_games': top_games.to_dict(),
            'top_providers': top_providers.to_dict(),
            'sample_title': titles.sample(1).values[0] if len(titles) > 0 else 'No titles',
            'representative_reviews': representative_reviews.tolist()
        }
    
    return cluster_analysis

# Perform cluster analysis
cluster_details = analyze_clusters(df_cleaned)

# Print cluster analysis
print("\nCluster Analysis:")
for cluster, details in cluster_details.items():
    print(f"\nCluster {cluster}:")
    print(f"Size: {details['size']} games")
    print("Top Games:", details['top_games'])
    print("Top Providers:", details['top_providers'])
    print("Sample Title:", details['sample_title'])
    print("Representative Reviews:")
    for review in details['representative_reviews']:
        if isinstance(review, str):
            print(f"  - {review[:200]}...")
        else:
            print(f"  - Invalid review: {review}")

# Save cluster details to file
with open('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/cluster_analysis.txt', 'w') as f:
    for cluster, details in cluster_details.items():
        f.write(f"\nCluster {cluster}:\n")
        f.write(f"Size: {details['size']} games\n")
        f.write(f"Top Games: {details['top_games']}\n")
        f.write(f"Top Providers: {details['top_providers']}\n")
        f.write(f"Sample Title: {details['sample_title']}\n")
        f.write("Representative Reviews:\n")
        for review in details['representative_reviews']:
            if isinstance(review, str):
                f.write(f"  - {review[:200]}...\n")
            else:
                f.write(f"  - Invalid review: {review}\n")

print("\nAnalysis complete. Check cluster_visualization.png, silhouette_scores.png, and cluster_analysis.txt for details.")
