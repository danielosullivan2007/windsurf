import os
import csv

# Input file path (same as in generate_reliable_summaries.py)
input_file = "../data/bigwinboard_cleaned.csv"

# Check if the file exists
if not os.path.exists(input_file):
    print(f"File not found: {input_file}")
    exit(1)

# Function to safely get review length
def get_review_length(review):
    if not review:
        return 0
    return len(str(review).strip())

try:
    # Use a more robust CSV parsing approach
    short_reviews = []
    all_reviews = []
    review_lengths = []
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        # Use CSV reader with proper quoting
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        
        # Get headers
        headers = next(reader)
        print(f"CSV Headers: {headers}")
        
        # Find review column index
        review_index = None
        title_index = None
        
        for i, header in enumerate(headers):
            if header.lower() in ['review', 'review_text']:
                review_index = i
            if header.lower() == 'title':
                title_index = i
        
        if review_index is None:
            # If no exact match, look for any column with 'review' in the name
            for i, header in enumerate(headers):
                if 'review' in header.lower():
                    review_index = i
                    break
        
        if title_index is None:
            # Default to first column if no title column found
            title_index = 0
            
        if review_index is None:
            print(f"Could not find review column in headers: {headers}")
            exit(1)
            
        print(f"Using column '{headers[review_index]}' (index {review_index}) for reviews")
        print(f"Using column '{headers[title_index]}' (index {title_index}) for titles")
        
        # Process each row
        row_count = 0
        short_count = 0
        
        for row in reader:
            row_count += 1
            
            # Skip rows that don't have enough columns
            if len(row) <= max(review_index, title_index):
                continue
                
            try:
                title = row[title_index] if title_index < len(row) else "Unknown"
                review = row[review_index] if review_index < len(row) else ""
                
                # Calculate review length
                review_length = get_review_length(review)
                review_lengths.append(review_length)
                
                # Track all reviews for length distribution
                all_reviews.append((title, review, review_length))
                
                # Track short reviews
                if review_length < 100:
                    short_count += 1
                    if len(short_reviews) < 10:  # Only keep 10 examples
                        short_reviews.append((title, review, review_length))
            except Exception as e:
                if row_count < 5:
                    print(f"Error processing row {row_count}: {e}")
    
    print(f"\nProcessed {row_count} rows from {input_file}")
    print(f"Found {short_count} reviews with less than 100 characters ({short_count/row_count*100:.1f}% of total)")
    
    # Display examples of short reviews
    print("\nExamples of reviews with less than 100 characters:")
    for i, (title, review, length) in enumerate(short_reviews):
        print(f"\n{i+1}. Game: {title}")
        print(f"   Review length: {length} characters")
        print(f"   Review: {review}")
    
    # Calculate distribution of review lengths
    bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
    bin_labels = ['0-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5000+']
    bin_counts = [0] * len(bins)
    
    for _, _, length in all_reviews:
        for i in range(len(bins)-1):
            if bins[i] <= length < bins[i+1]:
                bin_counts[i] += 1
                break
    
    print("\nDistribution of review lengths:")
    for i in range(len(bin_labels)):
        count = bin_counts[i]
        percentage = count / row_count * 100 if row_count > 0 else 0
        print(f"   {bin_labels[i]} characters: {count} reviews ({percentage:.1f}%)")
        
except Exception as e:
    print(f"Error analyzing reviews: {e}")

