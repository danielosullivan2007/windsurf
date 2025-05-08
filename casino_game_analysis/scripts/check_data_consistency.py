import pandas as pd
import json
import os

def check_file_consistency():
    """Check consistency between summary, embedding, and visualization files"""
    print("Checking data consistency across files...\n")
    
    # Check summary file
    print("Summary file check:")
    summary_file = "../data/all_summaries_merged.csv"
    print(f"- {summary_file} exists: {os.path.exists(summary_file)}")
    
    summary_count = 0
    if os.path.exists(summary_file):
        try:
            df = pd.read_csv(summary_file)
            summary_count = len(df)
            print(f"- Number of rows: {summary_count}")
            valid_summaries = df["structured_summary"].notna().sum() if "structured_summary" in df.columns else "Column not found"
            print(f"- Number of valid summaries: {valid_summaries}")
        except Exception as e:
            print(f"- Error reading summary file: {e}")
    
    # Check original summary file
    print("\nOriginal summary file check:")
    orig_summary_file = "../data/bigwinboard_with_summaries.csv"
    print(f"- {orig_summary_file} exists: {os.path.exists(orig_summary_file)}")
    
    orig_summary_count = 0
    if os.path.exists(orig_summary_file):
        try:
            df = pd.read_csv(orig_summary_file)
            orig_summary_count = len(df)
            print(f"- Number of rows: {orig_summary_count}")
            valid_summaries = df["structured_summary"].notna().sum() if "structured_summary" in df.columns else "Column not found"
            print(f"- Number of valid summaries: {valid_summaries}")
        except Exception as e:
            print(f"- Error reading original summary file: {e}")
    
    # Check embedding file
    print("\nEmbedding file check:")
    embedding_file = "../embeddings/game_summary_embeddings.csv"
    print(f"- {embedding_file} exists: {os.path.exists(embedding_file)}")
    
    embedding_count = 0
    if os.path.exists(embedding_file):
        try:
            df = pd.read_csv(embedding_file)
            embedding_count = len(df)
            print(f"- Number of rows: {embedding_count}")
        except Exception as e:
            print(f"- Error reading embedding file: {e}")
    
    # Check TSNE file
    print("\nTSNE file check:")
    tsne_file = "../embeddings/game_summary_embeddings_tsne.json"
    print(f"- {tsne_file} exists: {os.path.exists(tsne_file)}")
    
    tsne_count = 0
    if os.path.exists(tsne_file):
        try:
            with open(tsne_file, "r") as f:
                data = json.load(f)
                tsne_count = len(data)
                print(f"- Number of items: {tsne_count}")
                
                # Check for unique titles
                titles = [item.get("title", "") for item in data]
                unique_titles = len(set(titles))
                print(f"- Number of unique titles: {unique_titles}")
                
                # Check clusters
                clusters = [item.get("cluster", -1) for item in data]
                unique_clusters = sorted(list(set(clusters)))
                cluster_counts = {cluster: clusters.count(cluster) for cluster in unique_clusters}
                print(f"- Clusters: {unique_clusters}")
                print(f"- Cluster distribution: {cluster_counts}")
        except Exception as e:
            print(f"- Error reading TSNE file: {e}")
    
    # Check cluster file
    print("\nCluster file check:")
    cluster_file = "game_clusters.csv"
    print(f"- {cluster_file} exists: {os.path.exists(cluster_file)}")
    
    cluster_count = 0
    if os.path.exists(cluster_file):
        try:
            df = pd.read_csv(cluster_file)
            cluster_count = len(df)
            print(f"- Number of rows: {cluster_count}")
            
            # Check clusters
            if "Cluster" in df.columns:
                unique_clusters = sorted(df["Cluster"].unique())
                cluster_counts = df["Cluster"].value_counts().to_dict()
                print(f"- Clusters: {unique_clusters}")
                print(f"- Cluster distribution: {cluster_counts}")
        except Exception as e:
            print(f"- Error reading cluster file: {e}")
    
    # Check checkpoint file
    print("\nCheckpoint file check:")
    checkpoint_file = "../embeddings/embedding_checkpoint.json"
    print(f"- {checkpoint_file} exists: {os.path.exists(checkpoint_file)}")
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)
                embeddings = data.get("embeddings", [])
                processed_titles = data.get("processed_titles", [])
                last_index = data.get("last_processed_index", 0)
                print(f"- Number of embeddings in checkpoint: {len(embeddings)}")
                print(f"- Number of processed titles: {len(processed_titles)}")
                print(f"- Last processed index: {last_index}")
        except Exception as e:
            print(f"- Error reading checkpoint file: {e}")
    
    # Summary of findings
    print("\nSummary of findings:")
    print(f"- Number of summaries in merged file: {summary_count}")
    print(f"- Number of summaries in original file: {orig_summary_count}")
    print(f"- Number of embeddings: {embedding_count}")
    print(f"- Number of items in TSNE visualization: {tsne_count}")
    print(f"- Number of items in cluster file: {cluster_count}")
    
    # Check for mismatches
    if summary_count != embedding_count or embedding_count != tsne_count or tsne_count != cluster_count:
        print("\nMISMATCH DETECTED! The numbers don't align:")
        
        if summary_count != embedding_count:
            print(f"- Mismatch between summaries ({summary_count}) and embeddings ({embedding_count})")
        
        if embedding_count != tsne_count:
            print(f"- Mismatch between embeddings ({embedding_count}) and TSNE visualization ({tsne_count})")
        
        if tsne_count != cluster_count:
            print(f"- Mismatch between TSNE visualization ({tsne_count}) and cluster file ({cluster_count})")
            
        print("\nPossible causes:")
        print("1. The embedding generation process didn't process all summaries")
        print("2. The TSNE visualization only includes a subset of the embeddings")
        print("3. The cluster file was generated from a different dataset")
        print("4. Files were generated at different times with different data")
        
        print("\nRecommended actions:")
        print("1. Re-run the embedding generation script with the merged summary file")
        print("2. Ensure all scripts use the same input and output files")
        print("3. Check for any filtering or limiting in the scripts")
    else:
        print("\nNo mismatches detected. All counts are consistent.")

if __name__ == "__main__":
    check_file_consistency()
