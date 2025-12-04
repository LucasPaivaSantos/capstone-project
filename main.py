import argparse
import importlib
import os
import pkgutil
import sys
import traceback

from utils.data_loader import load_and_validate_csv

from models.base import MODEL_REGISTRY

def load_modules(package_name):
    """
    Dinamically loads all modules in the specified package by joining their names.
    """
    package_dir = os.path.join(os.path.dirname(__file__), package_name)
    # prevents errors if the package does not exist
    if not os.path.exists(package_dir):
        return
    
    # interate through all modules in the package directory
    for _, module_name, _ in pkgutil.iter_modules([package_dir]):
        if module_name != 'base': # skip base module
            importlib.import_module(f'{package_name}.{module_name}')

def main():
    # load all modules
    load_modules('models')

    # manage CLI arguments
    parser = argparse.ArgumentParser(description="Acid Geopolymer Concrete Optimizer")
    
    parser.add_argument(
    "--csv_path",
    default="data/best.csv",
    help="Path to the CSV file containing the dataset (default: data/best.csv)")
    parser.add_argument(
    "--model", "-m",
    required=True, 
    choices=MODEL_REGISTRY.keys(),
    help="Machine Learning Model to use")

    args = parser.parse_args()

    # main flow
    try:
        print("Loading Data")
        X, y = load_and_validate_csv(args.csv_path)

        print(f"\nInitializing Model: {args.model}")
        model_cls = MODEL_REGISTRY[args.model]
        model_instance = model_cls()

    except Exception as e:
        # for erros I didn't anticipate
        print(f"\nUnexpected Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()