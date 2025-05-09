import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

# Load the data
df = pd.read_csv('game_embeddings.csv', low_memory=False)

# Extract embedding columns
embedding_columns = df.columns[-1536:]  # Adjust based on actual embedding dimensions
embeddings = df[embedding_columns].values

# Normalize embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

# Perform clustering
kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
clusters = kmeans.fit_predict(normalized_embeddings)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=3, random_state=42, perplexity=30)
embeddings_3d = tsne.fit_transform(normalized_embeddings)

# Prepare data for export
export_data = []
cluster_colors = [
    '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', 
    '#9966FF', '#FF9F40', '#E7E9ED', '#FF6384', '#4BC0C0'
]

for i, (point, cluster) in enumerate(zip(embeddings_3d, clusters)):
    export_data.append({
        'id': i,
        'x': float(point[0]),
        'y': float(point[1]),
        'z': float(point[2]),
        'cluster': int(cluster),
        'color': cluster_colors[cluster],
        'name': df.iloc[i]['Text'] if 'Text' in df.columns else f'Game {i}',
        'description': df.iloc[i]['Description'] if 'Description' in df.columns else ''
    })

# Export to JSON
with open('embedding-viewer/src/data/game_embeddings.json', 'w') as f:
    json.dump(export_data, f, indent=2)

# Export cluster information
cluster_info = {}
for cluster in range(9):
    cluster_games = [game for game in export_data if game['cluster'] == cluster]
    cluster_info[cluster] = {
        'count': len(cluster_games),
        'color': cluster_colors[cluster],
        'sample_games': [game['name'] for game in cluster_games[:5]]
    }

with open('embedding-viewer/src/data/cluster_info.json', 'w') as f:
    json.dump(cluster_info, f, indent=2)

print("Embeddings prepared for React visualization.")
