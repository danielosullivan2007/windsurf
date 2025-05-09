import csv
import re
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/bigwinboard_with_summaries.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/bigwinboard_with_summaries_cleaned.csv')

# Patterns that indicate artifact/duplicate lines in the game name or summary
ARTIFACT_PATTERNS = [
    r'^\s*\}\);',
    r'^\s*\[smartslider3 slider=',
    r'^\s*<span style=',
    r': Slot Overview',
    r'Slot Overview',
    r'Overview',
    r'^\s*\}\);',
    r'^\s*\[smartslider3',
]

artifact_regexes = [re.compile(p, re.IGNORECASE) for p in ARTIFACT_PATTERNS]

def is_artifact(text):
    if not text:
        return False
    for regex in artifact_regexes:
        if regex.search(text):
            return True
    return False

def clean_bigwinboard_csv(input_path, output_path):
    cleaned_rows = []
    removed_count = 0
    with open(input_path, newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        cleaned_rows.append(header)
        for row in reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue
            name = row[0].strip() if len(row) > 0 else ''
            summary = row[-1].strip() if len(row) > 1 else ''
            if is_artifact(name) or is_artifact(summary):
                removed_count += 1
                continue
            cleaned_rows.append(row)
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_rows)
    print(f"Cleaning complete. Removed {removed_count} artifact/duplicate rows. Cleaned file saved as: {output_path}")

if __name__ == '__main__':
    clean_bigwinboard_csv(INPUT_FILE, OUTPUT_FILE)
