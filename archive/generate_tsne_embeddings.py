import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Potential input file paths
possible_paths = [
    '/Users/danielosullivan/Desktop/windsurf_testing/windsurf/game_embeddings.csv',
    '/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/game_embeddings.csv'
]

# Find the first existing file
input_file = next((path for path in possible_paths if os.path.exists(path)), None)

if not input_file:
    raise FileNotFoundError(f"No game embeddings file found. Tried paths: {possible_paths}")

# Load embeddings
df = pd.read_csv(input_file)

# Extract embedding columns
embedding_columns = [col for col in df.columns if col.startswith('embedding_')]
embeddings = df[embedding_columns].values

# Normalize embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

# Adjust perplexity based on number of samples
perplexity = min(30, len(normalized_embeddings) - 1)

# Perform t-SNE
tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
tsne_embeddings = tsne.fit_transform(normalized_embeddings)

# Perform clustering
kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
clusters = kmeans.fit_predict(normalized_embeddings)

# Create new DataFrame with t-SNE coordinates and clusters
tsne_df = pd.DataFrame({
    'name': df['name'],
    'tsne_x': tsne_embeddings[:, 0],
    'tsne_y': tsne_embeddings[:, 1],
    'tsne_z': tsne_embeddings[:, 2],
    'cluster': clusters
})

# Ensure output directory exists
output_file = '/Users/danielosullivan/Desktop/windsurf_testing/windsurf/embedding-viewer/public/tsne_embeddings.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save t-SNE embeddings
tsne_df.to_csv(output_file, index=False)

print("t-SNE embeddings generated successfully!")
print(f"Input file: {input_file}")
print(f"Shape of t-SNE embeddings: {tsne_embeddings.shape}")
print(f"Number of clusters: {len(np.unique(clusters))}")
