#!/usr/bin/env python3
"""
Analyze the bigwinboard_name_summary.csv file to understand the available data
and compare it with the existing summaries.
"""

import os
import pandas as pd
import csv
from collections import Counter

# File paths
name_summary_file = "../data/bigwinboard_name_summary.csv"
summaries_file = "../data/bigwinboard_with_summaries.csv"

def load_csv_safely(file_path):
    """Load CSV file with proper quoting to handle embedded commas"""
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

def analyze_summaries_overlap():
    """Analyze the overlap between name_summary and existing summaries"""
    
    # Load the name summary data
    name_summary_data, name_summary_headers = load_csv_safely(name_summary_file)
    
    # Count the length of summaries
    summary_lengths = [len(row.get('summary', '')) for row in name_summary_data]
    
    # Create a histogram of summary lengths
    print("\nSummary Length Distribution:")
    buckets = [0, 50, 100, 200, 500, 1000, float('inf')]
    counts = [0] * (len(buckets) - 1)
    
    for length in summary_lengths:
        for i in range(len(buckets) - 1):
            if buckets[i] <= length < buckets[i+1]:
                counts[i] += 1
                break
    
    for i in range(len(buckets) - 1):
        if buckets[i+1] == float('inf'):
            print(f"{buckets[i]}+ chars: {counts[i]} summaries")
        else:
            print(f"{buckets[i]}-{buckets[i+1]-1} chars: {counts[i]} summaries")
    
    # Check how many have non-empty summaries
    non_empty_summaries = sum(1 for length in summary_lengths if length > 20)
    print(f"\nSummaries longer than 20 chars: {non_empty_summaries} ({non_empty_summaries/len(name_summary_data)*100:.1f}%)")
    
    # Now check if the summaries.csv file exists and analyze overlap
    if os.path.exists(summaries_file):
        structured_data, structured_headers = load_csv_safely(summaries_file)
        
        # Create sets of game titles for comparison
        name_summary_titles = {row.get('name', '').strip().lower() for row in name_summary_data}
        structured_titles = {row.get('Title', '').strip().lower() for row in structured_data}
        
        # Calculate overlap statistics
        overlap = name_summary_titles.intersection(structured_titles)
        only_in_name_summary = name_summary_titles - structured_titles
        only_in_structured = structured_titles - name_summary_titles
        
        print("\nOverlap Analysis:")
        print(f"Games in name_summary file: {len(name_summary_titles)}")
        print(f"Games in structured_summaries file: {len(structured_titles)}")
        print(f"Games in both files: {len(overlap)}")
        print(f"Games only in name_summary: {len(only_in_name_summary)}")
        print(f"Games only in structured_summaries: {len(only_in_structured)}")
        
        # Check if the unmatched games might be due to slight name differences
        if len(only_in_name_summary) > 0 and len(only_in_structured) > 0:
            print("\nAnalyzing potential fuzzy matches...")
            fuzzy_matches = 0
            
            # Simple fuzzy matching (prefix matching)
            for name_title in list(only_in_name_summary)[:100]:  # Just check a sample
                for struct_title in only_in_structured:
                    # Check if one is a substring of the other (with a min length)
                    if len(name_title) > 5 and len(struct_title) > 5:
                        if name_title[:5] == struct_title[:5]:
                            fuzzy_matches += 1
                            break
            
            print(f"Potential fuzzy matches from sample: {fuzzy_matches}")
            print("Note: This is a rough estimate and may need more sophisticated matching.")
    
    # Analyze common developers and other metadata if available
    if len(name_summary_data) > 0 and 'developer' in name_summary_data[0]:
        developers = [row.get('developer', '').strip() for row in name_summary_data]
        developer_counts = Counter(developers)
        
        print("\nTop 10 Developers:")
        for developer, count in developer_counts.most_common(10):
            if developer:  # Skip empty developer names
                print(f"{developer}: {count} games")

def calculate_summary_cost_savings():
    """Calculate potential cost savings by using existing summaries"""
    
    # Load the name summary data
    name_summary_data, _ = load_csv_safely(name_summary_file)
    
    # Constants for cost calculation (based on GPT-4.1 Mini)
    AVG_TOKENS_PER_CHAR = 0.25  # About 4 characters per token
    INPUT_COST_PER_1M_TOKENS = 0.40  # $0.40 per million tokens
    OUTPUT_COST_PER_1M_TOKENS = 1.60  # $1.60 per million tokens
    SUMMARY_LENGTH_TOKENS = 300  # Average summary tokens
    
    # Count the number of usable summaries (longer than 20 chars)
    usable_summaries = sum(1 for row in name_summary_data if len(row.get('summary', '')) > 20)
    
    # Calculate potential cost savings
    input_tokens_saved = sum(len(row.get('name', '')) + len(row.get('summary', '')) 
                             for row in name_summary_data if len(row.get('summary', '')) > 20)
    input_tokens_saved = input_tokens_saved * AVG_TOKENS_PER_CHAR
    
    output_tokens_saved = usable_summaries * SUMMARY_LENGTH_TOKENS
    
    input_cost_saved = (input_tokens_saved / 1000000) * INPUT_COST_PER_1M_TOKENS
    output_cost_saved = (output_tokens_saved / 1000000) * OUTPUT_COST_PER_1M_TOKENS
    total_cost_saved = input_cost_saved + output_cost_saved
    
    print("\nPotential Cost Savings Analysis:")
    print(f"Number of usable summaries: {usable_summaries}")
    print(f"Estimated input tokens saved: {input_tokens_saved:,.0f}")
    print(f"Estimated output tokens saved: {output_tokens_saved:,.0f}")
    print(f"Estimated input cost saved: ${input_cost_saved:.2f}")
    print(f"Estimated output cost saved: ${output_cost_saved:.2f}")
    print(f"Total estimated cost saved: ${total_cost_saved:.2f}")

if __name__ == "__main__":
    print("Analyzing bigwinboard_name_summary.csv file...")
    analyze_summaries_overlap()
    calculate_summary_cost_savings()
    print("\nAnalysis complete.")
