import pandas as pd
import json
import os
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def extract_key_themes(summaries, n_themes=5):
    """Extract key themes from a list of summaries using word frequency analysis"""
    # Create a CountVectorizer to extract words
    vectorizer = CountVectorizer(
        stop_words='english', 
        min_df=2,  # Minimum document frequency
        max_df=0.9,  # Maximum document frequency
        ngram_range=(1, 2)  # Include single words and bigrams
    )
    
    # Fit and transform the summaries
    X = vectorizer.fit_transform(summaries)
    
    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Sum the occurrences of each word across all documents
    word_counts = X.sum(axis=0).A1
    
    # Sort words by frequency
    sorted_indices = word_counts.argsort()[::-1]
    
    # Get the top N themes (words)
    top_themes = [feature_names[i] for i in sorted_indices[:n_themes]]
    
    return top_themes

def analyze_cluster(cluster_games, cluster_id):
    """Analyze a cluster of games and return insights"""
    # Count providers
    providers = Counter([game.get('provider', 'Unknown') for game in cluster_games])
    top_providers = providers.most_common(5)
    
    # Extract summaries for theme analysis
    summaries = [game.get('summary', '') for game in cluster_games if game.get('summary')]
    
    # Extract key themes
    key_themes = extract_key_themes(summaries)
    
    # Sample games (first 5)
    sample_games = cluster_games[:5]
    
    return {
        'cluster_id': cluster_id,
        'game_count': len(cluster_games),
        'top_providers': top_providers,
        'key_themes': key_themes,
        'sample_games': sample_games
    }

def generate_report():
    """Generate a comprehensive cluster analysis report"""
    # Load the cluster data
    try:
        df = pd.read_csv('game_clusters.csv')
        print(f"Loaded {len(df)} games from game_clusters.csv")
    except Exception as e:
        print(f"Error loading game_clusters.csv: {e}")
        return
    
    # Load the t-SNE visualization data
    try:
        with open('../embeddings/game_summary_embeddings_tsne.json', 'r') as f:
            tsne_data = json.load(f)
        print(f"Loaded {len(tsne_data)} games from t-SNE data")
    except Exception as e:
        print(f"Error loading t-SNE data: {e}")
        return
    
    # Get the number of clusters
    clusters = sorted(list(set(item["cluster"] for item in tsne_data)))
    print(f"Found {len(clusters)} clusters: {clusters}")
    
    # Analyze each cluster
    cluster_analyses = []
    for cluster_id in clusters:
        cluster_games = [item for item in tsne_data if item['cluster'] == cluster_id]
        analysis = analyze_cluster(cluster_games, cluster_id)
        cluster_analyses.append(analysis)
    
    # Sort clusters by size (descending)
    cluster_analyses.sort(key=lambda x: x['game_count'], reverse=True)
    
    # Generate the report
    report = generate_markdown_report(cluster_analyses, tsne_data)
    
    # Save the report
    report_path = '../docs/CLUSTER_ANALYSIS_REPORT.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to {report_path}")
    return report_path

def generate_markdown_report(cluster_analyses, tsne_data):
    """Generate a markdown report from the cluster analyses"""
    total_games = len(tsne_data)
    
    # Start with a header
    report = f"""# Casino Game Cluster Analysis Report

## Overview

This report analyzes the clustering results of {total_games} casino games based on their semantic embeddings. 
The embeddings were generated from structured summaries of each game, and then clustered using K-means 
with t-SNE dimensionality reduction for visualization.

## Summary

- **Total Games Analyzed**: {total_games}
- **Number of Clusters**: {len(cluster_analyses)}
- **Date of Analysis**: May 8, 2025
- **Embedding Model**: OpenAI's text-embedding-ada-002

## Cluster Distribution

| Cluster ID | Number of Games | Percentage | Key Themes |
|------------|----------------|------------|------------|
"""
    
    # Add cluster distribution table
    for analysis in cluster_analyses:
        cluster_id = analysis['cluster_id']
        game_count = analysis['game_count']
        percentage = round((game_count / total_games) * 100, 1)
        themes = ', '.join(analysis['key_themes'][:3])  # Show top 3 themes
        report += f"| {cluster_id} | {game_count} | {percentage}% | {themes} |\n"
    
    # Add detailed cluster analysis
    report += "\n## Detailed Cluster Analysis\n\n"
    
    for analysis in cluster_analyses:
        cluster_id = analysis['cluster_id']
        game_count = analysis['game_count']
        
        report += f"### Cluster {cluster_id} ({game_count} games)\n\n"
        
        # Add key themes
        report += "#### Key Themes\n\n"
        for theme in analysis['key_themes']:
            report += f"- {theme}\n"
        
        # Add top providers
        report += "\n#### Top Providers\n\n"
        for provider, count in analysis['top_providers']:
            if provider and provider != "Unknown" and provider != "":
                percentage = round((count / game_count) * 100, 1)
                report += f"- {provider}: {count} games ({percentage}%)\n"
        
        # Add sample games
        report += "\n#### Sample Games\n\n"
        for game in analysis['sample_games']:
            title = game.get('title', 'Untitled')
            provider = game.get('provider', 'Unknown')
            summary = game.get('summary', 'No summary available')
            
            # Format the summary for better readability
            summary = summary.replace('\n', '\n  ')
            
            report += f"**{title}**"
            if provider and provider != "Unknown" and provider != "":
                report += f" by {provider}"
            report += "\n\n"
            report += f"  {summary}\n\n"
        
        report += "---\n\n"
    
    # Add conclusion
    report += """## Conclusion

This cluster analysis reveals distinct patterns in casino games based on their thematic elements, 
gameplay features, and aesthetic characteristics. The clusters identified represent natural groupings 
of games that share similar attributes as described in their summaries.

The visualization of these clusters in 3D space using t-SNE allows for an intuitive exploration of the 
relationships between different games and game types. Games that are positioned closer together in the 
visualization share more similarities in their features and themes.

This analysis can be valuable for:

1. **Game Recommendation Systems**: Suggesting similar games to players based on cluster membership
2. **Market Analysis**: Identifying popular game themes and features
3. **Portfolio Gap Analysis**: Finding underrepresented game types in a casino's offerings
4. **Trend Identification**: Recognizing emerging patterns in game design and player preferences
"""
    
    return report

if __name__ == "__main__":
    report_path = generate_report()
    print(f"Report generated at: {report_path}")
