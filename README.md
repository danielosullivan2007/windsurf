# Casino Game Embeddings Project

## Project Overview
This project generates high-dimensional embeddings for casino game data using OpenAI's embedding API, enabling advanced text analysis and visualization.

## Key Components
- `generate_embeddings.py`: Script to generate OpenAI embeddings
- `embedding_analysis.ipynb`: Jupyter notebook for embedding visualization
- `game_embeddings.csv`: Output file with game data and embeddings

## Embedding Generation Process
- Model: OpenAI's text-embedding-3-small
- Total Records: 3,549
- Total Cost: $0.35
- Embedding Dimensions: 1,536

## Visualization
The `embedding_analysis.ipynb` notebook creates:
- 3D t-SNE visualization
- Interactive HTML output
- Clustering insights

## Setup Instructions
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate embeddings
python generate_embeddings.py

# Open Jupyter notebook
jupyter lab embedding_analysis.ipynb
```

## Embedding Use Cases
- Semantic search
- Game recommendation
- Clustering similar games
- Dimensional reduction for ML models

## Cost Management
- Batch processing to control API costs
- Transparent cost tracking
- Flexible batch size configuration

## Embedding Exploration
Detailed analysis of game embeddings includes:
- `EMBEDDING_EXPLORATION.md`: Comprehensive insights
- `casino_game_embeddings_viewer.html`: Interactive 3D visualization
- `cluster_summary.csv`: Cluster characteristics

### Visualization Features
- Interactive 3D React-based embedding viewer
- Immersive dark mode design
- Real-time cluster exploration
- Hover-based game and cluster details

### Embedding Viewer Highlights
- Powered by React Three Fiber
- 9 distinct game theme clusters
- Dynamic point scaling and interaction
- Starry background visualization

### Cluster Insights
- Largest cluster: High-Energy Games (741 games)
- Themes range from adventure to strategic
- Detailed analysis in `CLUSTER_INSIGHTS.md`

## Visualization Setup
```bash
# Navigate to embedding viewer
cd embedding-viewer

# Install dependencies
npm install

# Start development server
npm start
```

## Visualization Guide
Detailed interaction instructions are available in `VISUALIZATION_GUIDE.md`:
- Navigation techniques
- Game point interactions
- Cluster exploration tips
- Technical visualization details

## Security and Maintenance

### Security Analysis
A comprehensive security vulnerability assessment is available in `SECURITY_ANALYSIS.md`:
- Detailed vulnerability breakdown
- Mitigation strategies
- Best practices for dependency management

### Project History
Detailed project development journey is documented in `HISTORY.md`:
- Conception and initial development
- Technical challenges and solutions
- Project metrics and milestones

## Next Steps
- Enhance interactive visualization
- Develop AI-driven game recommendations
- Implement advanced clustering techniques
- Continuous security monitoring

### Recommended Security Actions
1. Regularly update dependencies
2. Run `npm audit` before deployment
3. Review `SECURITY_ANALYSIS.md` for detailed guidance
