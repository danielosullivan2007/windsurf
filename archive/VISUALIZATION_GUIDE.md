# Casino Game Embeddings Visualization Guide

## Interaction Modes

### 1. Navigation
- **Left-Click and Drag**: Rotate the 3D space
- **Scroll Wheel**: Zoom in and out
- **Right-Click and Drag**: Pan the view

### 2. Game Point Interactions
- **Hover**: 
  - Enlarges the selected game point
  - Triggers detailed information display
- **Click**: 
  - Reveals full game details in the sidebar
  - Shows cluster-specific information

### 3. Visualization Features
- **Starry Background**: Represents the vast game universe
- **Color-Coded Points**: Each color represents a different game cluster
- **Dynamic Scaling**: Points grow when hovered, creating visual emphasis

## Cluster Exploration

### Sidebar Details
When you hover or click a game point, the sidebar will show:
- Game Name
- Cluster Number
- Cluster Color
- Number of Games in Cluster
- Sample Games from the Same Cluster

## Recommended Exploration Techniques
1. Start by rotating the view to see the overall distribution
2. Zoom in on interesting clusters
3. Hover over points to get quick insights
4. Click points for detailed information
5. Look for color patterns and spatial relationships

## Technical Details
- **Visualization Engine**: React Three Fiber
- **Dimensionality Reduction**: t-SNE
- **Clustering**: K-means with 9 clusters
- **Total Games Visualized**: 3,549

## Pro Tips
- The closer points are, the more semantically similar the games
- Different colors indicate distinct game themes
- Use orbit controls to explore from multiple angles
