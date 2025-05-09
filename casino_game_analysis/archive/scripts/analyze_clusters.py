import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

def generate_cluster_analysis():
    """Generate a detailed analysis of the clusters and save as markdown"""
    # Load the cluster data
    try:
        cluster_df = pd.read_csv("game_clusters.csv")
        print(f"Loaded {len(cluster_df)} games with cluster assignments")
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return
        
    # Load the t-SNE data
    try:
        with open("game_summary_embeddings_tsne.json", "r") as f:
            tsne_data = json.load(f)
    except Exception as e:
        print(f"Error loading t-SNE data: {e}")
        return
    
    # Count games per cluster
    cluster_counts = Counter(cluster_df['Cluster'])
    num_clusters = len(cluster_counts)
    
    # Create a folder for word clouds if it doesn't exist
    os.makedirs("cluster_wordclouds", exist_ok=True)
    
    # Create markdown report
    report = f"""# Casino Game Cluster Analysis

## Overview

This report presents the results of clustering {len(cluster_df)} casino games based on structured summaries of their reviews. 
The summaries were created using OpenAI's GPT-3.5 and focused on four key aspects of each game:

1. Game theme overview
2. Game features
3. Reviewer verdict
4. Aesthetics and audio

These summaries were then converted to vector embeddings using OpenAI's text-embedding-ada-002 model, 
and clustered using K-means clustering. The optimal number of clusters was determined to be {num_clusters} 
using silhouette score analysis.

## Cluster Distribution

The {len(cluster_df)} games were distributed across {num_clusters} clusters as follows:

| Cluster | Number of Games | Percentage |
|---------|----------------|------------|
"""
    
    # Add cluster distribution table
    for cluster_id in sorted(cluster_counts.keys()):
        count = cluster_counts[cluster_id]
        percentage = count / len(cluster_df) * 100
        report += f"| {cluster_id} | {count} | {percentage:.2f}% |\n"
    
    report += "\n## Cluster Analysis\n\n"
    
    # Analyze each cluster
    for cluster_id in sorted(cluster_counts.keys()):
        # Get games in this cluster
        cluster_games = cluster_df[cluster_df['Cluster'] == cluster_id]
        
        # Sample games from the cluster (up to 5)
        sample_games = cluster_games.sample(min(5, len(cluster_games)))
        
        # Generate word cloud for the cluster
        all_summaries = " ".join(cluster_games['Summary'].tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(all_summaries)
        
        # Save the word cloud
        wordcloud_filename = f"cluster_wordclouds/cluster_{cluster_id}_wordcloud.png"
        wordcloud.to_file(wordcloud_filename)
        
        # Extract top providers in this cluster
        provider_counts = Counter(cluster_games['Provider'])
        top_providers = provider_counts.most_common(3)
        top_providers_str = ", ".join([f"{provider} ({count})" for provider, count in top_providers])
        
        # Extract common words/themes
        vectorizer = CountVectorizer(stop_words='english', max_features=20)
        try:
            X = vectorizer.fit_transform(cluster_games['Summary'])
            words = vectorizer.get_feature_names_out()
            word_counts = np.asarray(X.sum(axis=0)).ravel()
            top_words = [words[i] for i in word_counts.argsort()[-10:][::-1]]
            top_words_str = ", ".join(top_words)
        except:
            top_words_str = "N/A"
        
        # Add cluster section to report
        report += f"### Cluster {cluster_id} ({cluster_counts[cluster_id]} games)\n\n"
        report += f"**Top Providers:** {top_providers_str}\n\n"
        report += f"**Common Themes/Features:** {top_words_str}\n\n"
        report += "**Sample Games:**\n\n"
        
        for _, game in sample_games.iterrows():
            report += f"- **{game['Title']}** ({game['Provider']})\n"
            report += f"  *Summary:* {game['Summary']}\n\n"
        
        report += f"![Cluster {cluster_id} Word Cloud]({wordcloud_filename})\n\n"
        report += "---\n\n"
    
    # Add conclusion
    report += """## Conclusion

The clustering analysis reveals distinct groups of casino games based on their themes, features, and overall reception. 
These clusters can provide valuable insights for:

1. **Game Development:** Understanding which game types and features are most prevalent or underrepresented
2. **Marketing:** Tailoring promotions to players based on their preferred game clusters
3. **Player Recommendations:** Building recommendation systems that suggest games from similar clusters
4. **Portfolio Analysis:** Identifying gaps or oversaturation in the game portfolio

This analysis demonstrates the power of embedding-based clustering for understanding the casino game landscape in a more nuanced way than traditional categorizations.
"""
    
    # Save the report
    with open("CLUSTER_ANALYSIS_REPORT.md", "w") as f:
        f.write(report)
    
    print("Cluster analysis report saved to CLUSTER_ANALYSIS_REPORT.md")
    return report

if __name__ == "__main__":
    generate_cluster_analysis()
