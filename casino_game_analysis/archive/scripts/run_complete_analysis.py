import os
import subprocess
import time
import sys

def check_file_exists(filename):
    """Check if a file exists and return its size"""
    if os.path.exists(filename):
        size = os.path.getsize(filename) / (1024 * 1024)  # Size in MB
        print(f"File {filename} exists ({size:.2f} MB)")
        return True
    else:
        print(f"File {filename} does not exist")
        return False

def main():
    """Run the complete analysis pipeline"""
    print("\n🚀 Starting complete analysis pipeline...")
    
    # Step 1: Generate structured summaries
    if not check_file_exists("bigwinboard_with_summaries.csv"):
        print("\n--- Step 1: Generating summaries ---")
        print("Running the summary generation script...")
        summary_process = subprocess.run(["python", "generate_reliable_summaries.py"], check=True)
        if summary_process.returncode != 0:
            print("❌ Error: Summary generation failed")
            return
    else:
        print("\n✅ Skipping summary generation - file already exists")
    
    # Step 2: Generate embeddings and clusters
    print("\n--- Step 2: Generating embeddings and clusters ---")
    if not check_file_exists("game_summary_embeddings_tsne.json"):
        print("Running the embedding generation script...")
        embedding_process = subprocess.run(["python", "generate_summary_embeddings.py"], check=True)
        if embedding_process.returncode != 0:
            print("❌ Error: Embedding generation failed")
            return
    else:
        print("✅ Skipping embedding generation - file already exists")
    
    # Step 3: Analyze clusters
    print("\n--- Step 3: Analyzing clusters ---")
    if not check_file_exists("CLUSTER_ANALYSIS_REPORT.md"):
        print("Running the cluster analysis script...")
        analysis_process = subprocess.run(["python", "analyze_clusters.py"], check=True)
        if analysis_process.returncode != 0:
            print("❌ Error: Cluster analysis failed")
            return
    else:
        print("✅ Skipping cluster analysis - report already exists")
    
    # Step 5: Start the React app
    print("\n--- Step 4: Starting visualization ---")
    if os.path.exists("visualization"):
        print("Found visualization directory")
        current_dir = os.getcwd()
        os.chdir("visualization")
        
        # Check if node_modules exists, if not run npm install
        if not os.path.exists("node_modules"):
            print("Installing dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Copy the latest TSNE data to the visualization folder
        print("Copying t-SNE data to visualization folder...")
        subprocess.run(["cp", "../game_summary_embeddings_tsne.json", "public/data/"], check=True)
        
        # Start the React app
        print("Starting the React application...")
        subprocess.Popen(["npm", "start"])
        
        # Change back to original directory
        os.chdir(current_dir)
    else:
        print("Error: Visualization directory not found")
        print("Please make sure the 'visualization' folder exists with the React application")
        return
    
    print("\n🎉 Complete analysis pipeline finished successfully!")
    print("\nYou can now view:")
    print("1. Cluster analysis report: CLUSTER_ANALYSIS_REPORT.md")
    print("2. Interactive visualization: http://localhost:3000")

if __name__ == "__main__":
    main()
