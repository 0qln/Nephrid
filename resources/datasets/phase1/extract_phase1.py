import csv
import sys

# File paths
INPUT_FILE = 'lichess_db_puzzle.csv'
OUTPUT_FILE = 'mate_in_1_and_2.epd'

# The specific themes we want for Curriculum Phase 1
TARGET_THEMES = {'mateIn1', 'mateIn2'}

def main():
    print(f"Starting extraction from '{INPUT_FILE}'...")

    total_processed = 0
    total_written = 0

    # Open files with standard UTF-8 encoding
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

            reader = csv.reader(infile)

            for row in reader:
                total_processed += 1

                # Skip the header row if your CSV happens to have one
                if row[0] == 'PuzzleId':
                    continue

                # Ensure the row has the expected number of columns (prevent IndexErrors)
                if len(row) < 8:
                    continue

                fen = row[1]
                # Themes are space-separated in the 8th column (index 7)
                themes = set(row[7].split(' '))

                # Check if the puzzle contains ANY of our target themes
                if TARGET_THEMES.intersection(themes):
                    outfile.write(f"{fen}\n")
                    total_written += 1

                # Print progress every 500k rows so you know it hasn't frozen
                if total_processed % 500_000 == 0:
                    print(f"Processed {total_processed:,} puzzles... (Saved {total_written:,} mates)")

    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'. Make sure it is in the same directory.")
        sys.exit(1)

    print("-" * 40)
    print("EXTRACTION COMPLETE!")
    print(f"Total puzzles scanned: {total_processed:,}")
    print(f"Phase 1 FENs saved:    {total_written:,}")
    print(f"Output file:           {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
