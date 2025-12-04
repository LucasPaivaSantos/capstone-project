import argparse
import sys
import traceback

from utils.data_loader import load_and_validate_csv

def main():
    # manage CLI arguments
    parser = argparse.ArgumentParser(description="Acid Geopolymer Concrete Optimizer")
    
    parser.add_argument(
    "--csv_path",
    default="data/best.csv",
    help="Path to the CSV file containing the dataset (default: data/best.csv)"
)

    args = parser.parse_args()

    # main flow
    try:
        print("\nLoading Data")
        X, y = load_and_validate_csv(args.csv_path)

    except Exception as e:
        # for erros I didn't anticipate
        print(f"\nUnexpected Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()