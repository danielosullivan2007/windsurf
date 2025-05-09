import re
import pandas as pd

def clean_text(text):
    # Remove the specific JavaScript/HTML block
    pattern = r'Play Demo\s*Write a comment\s*\.write-comment-button:hover\s*{\s*background-color:\s*#b8b8b8\s*!important;\s*/\*\s*Grey color on hover\s*\*/\s*}\s*170%\s*bonus\s*300\s*free\s*spins\s*Get\s*Bonus\s*18\+\s*\|\s*Terms\s*apply\s*//\s*Wait\s*for\s*dom/jQuery\s*to\s*be\s*ready\s*jQuery\(document\)\.ready\(function\(\$\)\{.*?\}\);'
    
    # Remove the pattern using regex, with re.DOTALL to match across multiple lines
    cleaned_text = re.sub(pattern, '', str(text), flags=re.DOTALL | re.IGNORECASE)
    
    return cleaned_text.strip()

# Read the CSV file
df = pd.read_csv('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/bigwinboard_cleaned.csv')

# Apply cleaning to all text columns
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].apply(clean_text)

# Save the cleaned DataFrame
df.to_csv('/Users/danielosullivan/Desktop/windsurf_testing/windsurf/bigwinboard_cleaned.csv', index=False)

print("Cleaning complete. File saved.")
