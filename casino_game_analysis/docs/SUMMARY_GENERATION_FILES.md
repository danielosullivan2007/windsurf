# Casino Game Summary Generation Files

## Core Generation Scripts
1. `generate_reliable_summaries.py`
   - **Purpose**: Main script for generating structured, high-quality summaries of casino game reviews
   - **Key Features**:
     - Uses OpenAI's GPT-4o model
     - Creates 4-line summaries focusing on theme, features, verdict, and aesthetics
     - Handles batch processing and error recovery

2. `generate_summary_embeddings.py`
   - **Purpose**: Converts generated summaries into embeddings for cluster analysis
   - **Key Features**:
     - Generates vector representations of game summaries
     - Performs dimensionality reduction (t-SNE)
     - Prepares data for visualization and clustering

## Input and Output Files
3. `bigwinboard_cleaned.csv`
   - **Purpose**: Original dataset of casino game reviews
   - **Content**: Raw review data before summary generation

4. `bigwinboard_with_summaries.csv`
   - **Purpose**: CSV file containing original reviews with generated summaries
   - **Content**: Augmented dataset with AI-generated structured summaries

## Embedding and Analysis Files
5. `game_summary_embeddings.csv`
   - **Purpose**: Stores vector embeddings generated from game summaries
   - **Content**: Numerical representations of summary semantics

6. `game_summary_embeddings_tsne.json`
   - **Purpose**: Stores t-SNE reduced embeddings for visualization
   - **Content**: Low-dimensional coordinates for 3D/2D plotting

## Visualization and Exploration
7. `embedding-viewer/`
   - **Purpose**: React application for interactive embedding visualization
   - **Features**: 
     - 3D visualization of game clusters
     - Interactive exploration of game similarities

## Supporting Files
8. `.env`
   - **Purpose**: Stores OpenAI API key securely
   - **Content**: Environment variable for API authentication

## Analysis Outputs
9. `CLUSTER_ANALYSIS_REPORT.md`
   - **Purpose**: Detailed report of clustering analysis
   - **Content**: Insights, statistics, and interpretations of game clusters

## Recommended Workflow
1. Run `generate_reliable_summaries.py` to create summaries
2. Use `generate_summary_embeddings.py` to create embeddings
3. Visualize results in the `embedding-viewer` React app
