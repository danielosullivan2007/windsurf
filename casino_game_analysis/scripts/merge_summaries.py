import pandas as pd
import os
import glob
from tqdm import tqdm

def merge_summaries():
    """
    Merge all summary files into a single file with unique entries.
    This ensures we have all summaries in one place for embedding generation.
    """
    # Path to the main summary file
    main_file = "../data/bigwinboard_with_summaries.csv"
    
    # Check if the main file exists
    if not os.path.exists(main_file):
        print(f"Error: Main file {main_file} not found")
        return
    
    # Load the main file
    try:
        main_df = pd.read_csv(main_file)
        print(f"Loaded main file with {len(main_df)} rows")
    except Exception as e:
        print(f"Error loading main file: {e}")
        return
    
    # Normalize column names
    main_df.columns = [col.lower() for col in main_df.columns]
    
    # Ensure we have the necessary columns
    required_columns = ['title', 'structured_summary']
    missing_columns = [col for col in required_columns if col not in main_df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns in main file: {missing_columns}")
        return
    
    # Create a dictionary of existing summaries
    existing_summaries = {}
    for _, row in main_df.iterrows():
        if pd.notna(row['structured_summary']):
            existing_summaries[row['title']] = row['structured_summary']
    
    print(f"Found {len(existing_summaries)} existing summaries in main file")
    
    # Count total summaries
    total_summaries = len(existing_summaries)
    
    # Save the merged data
    output_file = "../data/all_summaries_merged.csv"
    
    # Create the output file with headers
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Title,review,structured_summary\n")
    
    # Write all summaries to the output file
    with open(output_file, 'a', encoding='utf-8') as f:
        for title, summary in tqdm(existing_summaries.items(), desc="Writing summaries"):
            # Escape quotes in CSV
            safe_title = title.replace('"', '""')
            safe_summary = summary.replace('"', '""')
            
            # Get the review if available
            review = ""
            row = main_df[main_df['title'] == title]
            if not row.empty and 'review' in row.columns and pd.notna(row['review'].iloc[0]):
                review = row['review'].iloc[0].replace('"', '""')
            
            # Write to CSV
            f.write(f'"{safe_title}","{review}","{safe_summary}"\n')
    
    print(f"Merged {total_summaries} summaries into {output_file}")
    return output_file

if __name__ == "__main__":
    output_file = merge_summaries()
    
    if output_file:
        print(f"\nTo generate embeddings for all summaries, run:")
        print(f"python generate_summary_embeddings.py --input {output_file}")
