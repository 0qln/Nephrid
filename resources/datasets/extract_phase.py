import csv
import sys
import os
import argparse
import random
from os import path

PROJECT_ROOT_ENV = os.environ.get('PROJECT_ROOT')
if PROJECT_ROOT_ENV is None:
    print("Error: PROJECT_ROOT environment variable not set.")
    sys.exit(1)
PROJECT_ROOT: str = PROJECT_ROOT_ENV  # now known to be str

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract FENs and first moves from Lichess puzzle CSV for a given phase and theme(s)."
    )
    parser.add_argument(
        '--phase',
        type=int,
        required=True,
        help="Phase number (used for output directory and file name)."
    )
    parser.add_argument(
        '--themes',
        type=str,
        required=True,
        help=("Theme filter. Use '|' for logical OR (any of the themes), "
              "e.g., 'endgame|short'. Use '&' for logical AND (all themes), "
              "e.g., 'endgame&short'. Do not mix '|' and '&' in one string.")
    )
    parser.add_argument(
        '--input',
        type=str,
        default=path.join(PROJECT_ROOT, 'resources/datasets/lichess_db_puzzle.csv'),
        help="Path to the input CSV file (default: PROJECT_ROOT/resources/datasets/lichess_db_puzzle.csv)."
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Path to the output EPD file (default: PROJECT_ROOT/resources/datasets/phase<phase>/default.epd)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1000,
        help="Random seed for reproducible shuffling (optional)."
    )
    return parser.parse_args()

def parse_theme_expression(expr: str):
    """
    Parse a theme expression like 'endgame|short' (OR) or 'endgame&short' (AND).
    Returns (operator, list_of_themes).
    operator is either 'or' or 'and'.
    Raises ValueError if both '|' and '&' appear.
    """
    expr = expr.strip()
    if not expr:
        raise ValueError("Empty theme expression")

    has_pipe = '|' in expr
    has_amp = '&' in expr

    if has_pipe and has_amp:
        raise ValueError("Cannot mix '|' and '&' in the same theme expression. Use only one operator.")

    if has_pipe:
        operator = 'or'
        themes = [t.strip() for t in expr.split('|') if t.strip()]
    elif has_amp:
        operator = 'and'
        themes = [t.strip() for t in expr.split('&') if t.strip()]
    else:
        # Single theme – treat as OR (any)
        operator = 'or'
        themes = [expr]

    if not themes:
        raise ValueError("No valid theme names found after splitting.")

    return operator, themes

def main():
    args = parse_arguments()

    phase = args.phase
    themes_raw = args.themes
    input_file = args.input
    output_file = args.output

    if output_file is None:
        output_file = path.join(PROJECT_ROOT, f'resources/datasets/phase{phase}/default.epd')

    # Ensure output directory exists
    os.makedirs(path.dirname(output_file), exist_ok=True)

    # Parse the theme expression
    try:
        operator, theme_list = parse_theme_expression(themes_raw)
    except ValueError as e:
        print(f"Error parsing --themes: {e}")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Starting extraction from '{input_file}'...")
    print(f"Phase: {phase}")
    print(f"Theme expression: '{themes_raw}' -> operator='{operator}', themes={theme_list}")
    print(f"Output: {output_file}")

    total_processed = 0
    collected_lines = []

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)

            for row in reader:
                total_processed += 1

                # Skip header
                if row[0] == 'PuzzleId':
                    continue

                if len(row) < 8:
                    continue

                fen = row[1]
                moves_str = row[2]
                puzzle_themes = set(row[7].split(' '))

                if operator == 'or':
                    # Any of the listed themes suffices
                    if any(t in puzzle_themes for t in theme_list):
                        first_move = moves_str.split(' ')[0]
                        collected_lines.append(f"{fen} sm {first_move};")
                else:  # operator == 'and'
                    # All listed themes must be present
                    if all(t in puzzle_themes for t in theme_list):
                        first_move = moves_str.split(' ')[0]
                        collected_lines.append(f"{fen} sm {first_move};")

                if total_processed % 500_000 == 0:
                    print(f"Processed {total_processed:,} puzzles... (Collected {len(collected_lines):,} matches)")

    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'.")
        sys.exit(1)

    print("-" * 40)
    print(f"Scanning complete. Total puzzles processed: {total_processed:,}")
    print(f"Matches found: {len(collected_lines):,}")

    if collected_lines:
        print("Shuffling collected lines...")
        random.shuffle(collected_lines)

        print(f"Writing to '{output_file}'...")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(collected_lines))
            if collected_lines:
                outfile.write('\n')  # trailing newline
    else:
        print("No matching puzzles found. No output file created.")

    print("EXTRACTION COMPLETE!")

if __name__ == '__main__':
    main()
