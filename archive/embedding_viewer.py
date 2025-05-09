import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('game_embeddings.csv', low_memory=False)

# Extract embedding columns
embedding_columns = df.columns[-1536:]  # Adjust based on actual embedding dimensions
embeddings = df[embedding_columns].values

# Perform clustering
kmeans = KMeans(n_clusters=9, random_state=42, n_init=10)
clusters = kmeans.fit_predict(embeddings)

# Reduce dimensionality for visualization
tsne = TSNE(n_components=3, random_state=42)
embeddings_3d = tsne.fit_transform(embeddings)

# Create a DataFrame with the reduced embeddings and additional game info
embedding_df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
embedding_df['game_name'] = df['Text']
embedding_df['cluster'] = clusters

# Create interactive 3D scatter plot
fig = px.scatter_3d(
    embedding_df, 
    x='x', y='y', z='z', 
    color='cluster',  # Color by cluster
    hover_data=['game_name'],
    title='Casino Game Embeddings Visualization',
    labels={'x': 'Dimension 1', 'y': 'Dimension 2', 'z': 'Dimension 3'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)

# Customize layout for better readability
fig.update_traces(
    marker=dict(
        size=5,
        opacity=0.8,
        line=dict(width=0.5, color='DarkSlateGrey')
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title='Semantic Dimension 1',
        yaxis_title='Semantic Dimension 2', 
        zaxis_title='Semantic Dimension 3'
    ),
    title_font_size=20,
    width=1200,
    height=800,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Rockwell"
    )
)

# Save the interactive HTML
fig.write_html('casino_game_embeddings_viewer.html')
print("Interactive embedding viewer saved to casino_game_embeddings_viewer.html")

# Generate cluster summary
cluster_summary = embedding_df.groupby('cluster').agg({
    'game_name': ['count', 'first', 'last']
})
cluster_summary.columns = ['Game Count', 'Sample Game 1', 'Sample Game 2']
cluster_summary.to_csv('cluster_summary.csv')
print("\nCluster Summary:")
print(cluster_summary)
