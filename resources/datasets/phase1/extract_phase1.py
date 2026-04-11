import csv
import sys
import os
from os import path

# File paths
PROJECT_ROOT = os.environ['PROJECT_ROOT']
INPUT_FILE = path.join(PROJECT_ROOT, 'resources/datasets/lichess_db_puzzle.csv')
OUTPUT_FILE = path.join(PROJECT_ROOT, 'resources/datasets/phase1/mate_in_1_and_2.epd')

TARGET_THEMES = {'mateIn1', 'mateIn2'}

def main():
    print(f"Starting extraction from '{INPUT_FILE}'...")

    total_processed = 0
    total_written = 0

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
             open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:

            reader = csv.reader(infile)

            for row in reader:
                total_processed += 1

                if row[0] == 'PuzzleId':
                    continue

                if len(row) < 8:
                    continue

                fen = row[1]
                moves_str = row[2] # Example: "d4e6 d6h2"
                themes = set(row[7].split(' '))

                if TARGET_THEMES.intersection(themes):
                    first_move = moves_str.split(' ')[0]
                    outfile.write(f"{fen} fm {first_move}\n")
                    total_written += 1

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
