import pandas as pd
import numpy as np
import json

# Load embeddings
input_file = '/Users/danielosullivan/Desktop/windsurf_testing/windsurf/game_embeddings.csv'
df = pd.read_csv(input_file, low_memory=False)

# Extract embedding columns
embedding_columns = [col for col in df.columns if 'embedding_' in col]
df_cleaned = df.dropna(subset=embedding_columns)

# Perform t-SNE (same as in explore_embeddings_fixed.py)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Standardize embeddings
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(df_cleaned[embedding_columns].values)

# Perform clustering
kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_scaled)

# t-SNE Dimensionality Reduction
tsne = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne.fit_transform(embeddings_scaled)

# Prepare data for export
react_embeddings = []
for i, (title, tsne_coords, cluster) in enumerate(zip(df_cleaned['Title'], embeddings_3d, cluster_labels)):
    react_embeddings.append({
        'title': title,
        'cluster': int(cluster),
        'tsneX': float(tsne_coords[0]),
        'tsneY': float(tsne_coords[1]),
        'tsneZ': float(tsne_coords[2])
    })

# Export to JSON
with open('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/game_embeddings_tsne.json', 'w') as f:
    json.dump(react_embeddings, f, indent=2)

print(f"Exported {len(react_embeddings)} game embeddings to game_embeddings_tsne.json")
