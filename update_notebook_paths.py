import json
import os

# Define the path to the root directory
root_dir = '/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project'

# Files to update
notebook_files = [
    os.path.join(root_dir, 'casino_game_analysis', 'embedding_analysis.ipynb'),
    os.path.join(root_dir, 'casino_game_analysis', 'game_embedding_analysis_updated.ipynb')
]

# Old path patterns to replace
old_path_patterns = [
    '/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/casino_game_analysis/data/'
]

for notebook_file in notebook_files:
    if not os.path.exists(notebook_file):
        print(f"File {notebook_file} does not exist. Skipping...")
        continue
    
    print(f"Processing {notebook_file}...")
    
    # Load the notebook
    with open(notebook_file, 'r') as f:
        notebook = json.load(f)
    
    # Process each cell
    modified = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            
            # Look for data file references and update them
            for i, line in enumerate(source):
                modified_line = line
                
                # Check if this line contains a data file reference
                for pattern in old_path_patterns:
                    if pattern in line:
                        # If line loads data with the pattern, replace with Data_Dir variable
                        if "pd.read_csv" in line and pattern in line:
                            # Extract the filename from the path
                            filename_start = line.find(pattern) + len(pattern)
                            filename_end = line.find("'", filename_start) if "'" in line[filename_start:] else line.find('"', filename_start)
                            if filename_end > filename_start:
                                filename = line[filename_start:filename_end]
                                
                                # If this is the first data file reference in the cell, add Data_Dir definition
                                if i == 0 or "Data_Dir" not in "".join(new_source):
                                    data_dir_def = 'Data_Dir = "/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/Data/"\n'
                                    modified_line = data_dir_def + line.replace(pattern + filename, 'Data_Dir + "' + filename)
                                else:
                                    modified_line = line.replace(pattern + filename, 'Data_Dir + "' + filename)
                                
                                modified = True
                
                new_source.append(modified_line)
            
            if new_source != source:
                cell['source'] = new_source
    
    if modified:
        # Save the modified notebook
        with open(notebook_file, 'w') as f:
            json.dump(notebook, f, indent=1)
        print(f"Updated {notebook_file}")
    else:
        print(f"No changes made to {notebook_file}")

# Update Python script
py_file = os.path.join(root_dir, 'casino_game_analysis', 'game_embedding_analysis.py')
if os.path.exists(py_file):
    with open(py_file, 'r') as f:
        content = f.read()
    
    # Add Data_Dir definition and update references
    modified_content = content
    for pattern in old_path_patterns:
        if pattern in content:
            # Find the position right after the import statements
            import_end = content.find("\n# Load the data")
            if import_end > 0:
                # Add Data_Dir definition
                data_dir_def = '\n# Define the data directory\nData_Dir = "/Users/osulldan/Library/CloudStorage/OneDrive-TheStarsGroup/Desktop/Windsurf_agent_test/CascadeProjects/windsurf-project/Data/"\n'
                
                # Split the content at import_end
                before_import = content[:import_end]
                after_import = content[import_end:]
                
                # Replace all instances of the pattern in the after_import part
                for pattern in old_path_patterns:
                    # Find occurrences of the pattern
                    after_import_modified = after_import
                    while pattern in after_import_modified:
                        pattern_start = after_import_modified.find(pattern)
                        if pattern_start >= 0:
                            # Find the end of the filename
                            filename_start = pattern_start + len(pattern)
                            filename_end = after_import_modified.find("'", filename_start) if "'" in after_import_modified[filename_start:] else after_import_modified.find('"', filename_start)
                            if filename_end > filename_start:
                                filename = after_import_modified[filename_start:filename_end]
                                old_path = pattern + filename
                                new_path = 'Data_Dir + "' + filename
                                after_import_modified = after_import_modified.replace(old_path, new_path, 1)
                    
                    after_import = after_import_modified
                
                # Combine the parts
                modified_content = before_import + data_dir_def + after_import
    
    if modified_content != content:
        with open(py_file, 'w') as f:
            f.write(modified_content)
        print(f"Updated {py_file}")
    else:
        print(f"No changes made to {py_file}")

# Check the casino-game-viewer directory for references to data files
casino_game_viewer_dir = os.path.join(root_dir, 'casino-game-viewer')
for root, dirs, files in os.walk(casino_game_viewer_dir):
    for file in files:
        if file.endswith(('.js', '.jsx', '.ts', '.tsx', '.html', '.css')):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                try:
                    content = f.read()
                    
                    # Check if the file references data from casino_game_analysis
                    for pattern in old_path_patterns:
                        if pattern in content:
                            print(f"Found data reference in {file_path}")
                            # Replace references (implementation depends on how the data is used in the viewer)
                            # This would need to be customized based on the actual code
                except UnicodeDecodeError:
                    print(f"Could not read {file_path} due to encoding issues")

print("Path update complete!")
