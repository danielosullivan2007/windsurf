import csv
import re
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/bigwinboard_name_summary.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/bigwinboard_name_summary_cleaned.csv')

# Patterns that indicate a line is an artifact or not a valid game name
ARTIFACT_PATTERNS = [
    r'^\s*\}\);',
    r'^\s*\[smartslider3 slider=',
    r'Overview',
    r'Slot Overview',
    r'features',
    r'^[^,]+: Slot Overview',
    r'^<span style=',
]

# Compile regexes
artifact_regexes = [re.compile(p, re.IGNORECASE) for p in ARTIFACT_PATTERNS]

def is_artifact_line(line):
    for regex in artifact_regexes:
        if regex.search(line):
            return True
    return False

def deduplicate_csv(input_path, output_path):
    cleaned_rows = []
    removed_count = 0
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            # Remove empty rows
            if not row or all(cell.strip() == '' for cell in row):
                continue
            # If single column, check if it's an artifact
            if len(row) == 1:
                if is_artifact_line(row[0]):
                    removed_count += 1
                    continue
                # Also remove lines that are clearly not game names (e.g., contain ": Slot Overview")
                if ': Slot Overview' in row[0]:
                    removed_count += 1
                    continue
                # Remove lines that start with ");" or "[smartslider3"
                if row[0].strip().startswith('});') or row[0].strip().startswith('[smartslider3'):
                    removed_count += 1
                    continue
                cleaned_rows.append(row)
            else:
                # For rows with more than 1 column, keep as is
                cleaned_rows.append(row)
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    print(f"Deduplication complete. Removed {removed_count} artifact/duplicate rows. Cleaned file saved as: {output_path}")

if __name__ == '__main__':
    deduplicate_csv(INPUT_FILE, OUTPUT_FILE)
