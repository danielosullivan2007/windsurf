import os
import sys
import csv

# Input and output file paths (same as in generate_reliable_summaries.py)
input_file = "../data/bigwinboard_cleaned.csv"
output_file = "../data/bigwinboard_with_summaries.csv"

# Check files existence
if not os.path.exists(input_file):
    print(f"Input file not found: {input_file}")
    sys.exit(1)

print(f"Analyzing filtering process for casino game summaries...")

# Function to safely load a CSV file with complex formatting
def load_csv_safely(file_path):
    data = []
    headers = None
    row_count = 0
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # Use CSV reader with proper quoting
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        
        try:
            # Get headers
            headers = next(reader)
            print(f"CSV Headers: {headers}")
            
            # Process each row
            for row in reader:
                row_count += 1
                
                # Skip rows that are too short
                if len(row) < len(headers):
                    # Pad with empty values
                    row = row + [''] * (len(headers) - len(row))
                elif len(row) > len(headers):
                    # Truncate extra fields
                    row = row[:len(headers)]
                    
                # Add to data
                data.append(dict(zip(headers, row)))
                
        except Exception as e:
            print(f"Error reading CSV file at row {row_count}: {e}")
            
    print(f"Successfully loaded {len(data)} rows from {file_path}")
    return data, headers

# Load input data
try:
    input_data, input_headers = load_csv_safely(input_file)
except Exception as e:
    print(f"Error loading input file: {e}")
    sys.exit(1)

# Load output data if it exists
existing_summaries = {}
output_data = None
if os.path.exists(output_file):
    try:
        output_data, output_headers = load_csv_safely(output_file)
        print(f"Output file columns: {output_headers}")
        
        # Try to identify the title and summary columns
        title_col = None
        summary_col = None
        
        for col in output_headers:
            if col.lower() == 'title':
                title_col = col
            if col.lower() in ['structured_summary', 'summary', 'summary_text']:
                summary_col = col
        
        if title_col and summary_col:
            # Count existing summaries
            valid_summaries = [row for row in output_data if row[summary_col] and row[summary_col].strip()]
            print(f"Found {len(valid_summaries)} rows with valid summaries in column '{summary_col}'")
            
            # Build dictionary of existing summaries
            existing_summaries = {row[title_col]: True for row in valid_summaries}
            print(f"Identified {len(existing_summaries)} unique game titles with existing summaries")
        else:
            print(f"Warning: Could not identify title or summary columns in output file")
    except Exception as e:
        print(f"Error analyzing output file: {e}")

# Analyze filtering process
filtered_count = 0
title_filtered = 0
review_missing = 0
review_short = 0
duplicates = 0
valid_to_process = 0

# Display some statistics
review_lengths = []

for row in input_data:
    filtered_count += 1
    
    # Check title
    title = row.get('Title', row.get('title', ''))
    if not title:
        title_filtered += 1
        continue
    
    # Check if already has summary
    if title in existing_summaries:
        duplicates += 1
        continue
    
    # Check review text
    review_text = None
    if 'review' in row and row['review']:
        review_text = row['review']
    elif 'review_text' in row and row['review_text']:
        review_text = row['review_text']
    else:
        # Try to find any column with 'review' in the name
        for col in row:
            if 'review' in col.lower() and row[col]:
                review_text = row[col]
                break
    
    if not review_text:
        review_missing += 1
        continue
    
    # Check review length
    review_length = len(str(review_text).strip())
    review_lengths.append(review_length)
    
    if review_length < 20:  # Using 20 characters as the minimum (from the script)
        review_short += 1
        continue
    
    # If we get here, this game would be processed
    valid_to_process += 1

print(f"\nFiltering analysis:")
print(f"Total games in input file: {len(input_data)}")
print(f"Games with missing/invalid titles: {title_filtered}")
print(f"Games already having summaries: {duplicates}")
print(f"Games with missing reviews: {review_missing}")
print(f"Games with reviews shorter than 20 chars: {review_short}")
print(f"Games valid for processing: {valid_to_process}")

# Analyze review length distribution
if review_lengths:
    print(f"\nReview length statistics:")
    print(f"Minimum review length: {min(review_lengths)}")
    print(f"Maximum review length: {max(review_lengths)}")
    print(f"Average review length: {sum(review_lengths) / len(review_lengths):.1f}")
    
    # Create bins
    bins = [0, 20, 50, 100, 500, 1000, 5000, float('inf')]
    bin_labels = ['0-19', '20-49', '50-99', '100-499', '500-999', '1000-4999', '5000+']
    bin_counts = [0] * len(bin_labels)
    
    for length in review_lengths:
        for i in range(len(bins)-1):
            if bins[i] <= length < bins[i+1]:
                bin_counts[i] += 1
                break
    
    print("\nDistribution of review lengths:")
    for i in range(len(bin_labels)):
        count = bin_counts[i]
        percentage = count / len(review_lengths) * 100
        print(f"   {bin_labels[i]} characters: {count} reviews ({percentage:.1f}%)")

print("\nBased on this analysis, you should expect to process approximately", valid_to_process, "games.")
print("\nIf you want to process more games, you may need to modify the script to:")
print("1. Include games with shorter reviews (currently filtered if < 20 chars)")
print("2. Fix column name detection for existing summaries")
print("3. Handle CSV parsing issues more robustly")

print("\nThis explains why the script is only showing", valid_to_process, "games in the progress bar.")

