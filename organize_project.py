import os
import shutil

# Base directory
base_dir = '/Users/danielosullivan/Desktop/windsurf_testing/windsurf'

# Create directory structure
project_structure = {
    'data': [
        'bigwinboard_cleaned.csv',
        'bigwinboard_with_summaries.csv'
    ],
    'scripts': [
        'generate_reliable_summaries.py',
        'generate_summary_embeddings.py', 
        'analyze_clusters.py',
        'run_complete_analysis.py'
    ],
    'embeddings': [
        'game_summary_embeddings.csv',
        'game_summary_embeddings_tsne.json'
    ],
    'visualization': [
        'embedding-viewer'
    ],
    'docs': [
        'SUMMARY_GENERATION_FILES.md',
        'CLUSTER_ANALYSIS_REPORT.md',
        'PROJECT_STATUS.md'
    ]
}

# Create project directory
project_dir = os.path.join(base_dir, 'casino_game_analysis')
os.makedirs(project_dir, exist_ok=True)

# Create subdirectories and move files
for folder, files in project_structure.items():
    folder_path = os.path.join(project_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    
    for file in files:
        src_path = os.path.join(base_dir, file)
        dst_path = os.path.join(folder_path, file)
        
        # Handle directories differently
        if os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        elif os.path.exists(src_path):
            shutil.move(src_path, dst_path)

# Move .env file
env_src = os.path.join(base_dir, '.env')
env_dst = os.path.join(project_dir, '.env')
if os.path.exists(env_src):
    shutil.move(env_src, env_dst)

print("Project reorganization complete.")
