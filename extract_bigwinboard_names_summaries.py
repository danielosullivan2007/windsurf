import csv
import os

def extract_game_name_and_summary(input_file, output_file):
    """
    Extract just the game name (Title) and summary (structured_summary) 
    from the bigwinboard_with_summaries.csv file
    """
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        # Use csv.reader to properly handle CSV with quoted fields containing commas
        reader = csv.reader(infile)
        headers = next(reader)  # Read header row
        
        # Verify we have the expected columns
        if 'Title' in headers and 'structured_summary' in headers:
            title_index = headers.index('Title')
            summary_index = headers.index('structured_summary')
        else:
            # If column names are different, make educated guesses
            print(f"Warning: Expected column names not found. Using columns {headers[0]} and {headers[2]}")
            title_index = 0  # Assuming Title is the first column
            summary_index = 2  # Assuming structured_summary is the third column
        
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['game_name', 'summary'])  # Write header with new column names
            
            # Extract and write only game name and summary for each row
            row_count = 0
            for row in reader:
                if len(row) > max(title_index, summary_index):
                    writer.writerow([row[title_index], row[summary_index]])
                    row_count += 1
            
            print(f"Successfully processed {row_count} games")

if __name__ == "__main__":
    # File paths
    input_file = "/Users/danielosullivan/Desktop/windsurf_testing/windsurf/casino_game_analysis/data/bigwinboard_with_summaries.csv"
    output_file = "/Users/danielosullivan/Desktop/windsurf_testing/windsurf/casino_game_analysis/data/bigwinboard_name_summary.csv"
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run the extraction
    extract_game_name_and_summary(input_file, output_file)
    print("Extraction complete!")
