import argparse
import importlib
import os
import numpy as np
import pkgutil
import sys
import traceback

from utils.data_loader import load_and_validate_csv
from utils.experiment_logger import save_experiment_info

from models.base import MODEL_REGISTRY
from strategies.base import STRATEGY_REGISTRY

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
    load_modules('strategies')


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
    parser.add_argument(
    "--model_seed",
    type=int,
    default=None,
    help="Seed for ML model (if not provided, a random seed will be generated)")
    parser.add_argument(
    "--strategy", "-s",
    required=True, 
    choices=STRATEGY_REGISTRY.keys(),
    help="Validation Strategy")

    args = parser.parse_args()

    # main flow
    try:
        print("Loading Data")
        X, y = load_and_validate_csv(args.csv_path)

        # generate or use provided model_seed
        model_seed = args.model_seed if args.model_seed is not None else np.random.randint(0, 100)

        print(f"\nInitializing Model: {args.model} with seed {model_seed}")
        model_cls = MODEL_REGISTRY[args.model]
        model_instance = model_cls(seed=model_seed)

        print(f"\nExecuting Strategy: {args.strategy}")
        strategy_cls = STRATEGY_REGISTRY[args.strategy]
        strategy_instance = strategy_cls()
        
        # print model evaluation results
        model_evaluation = strategy_instance.evaluate(model_instance, X, y)
        print("\nEvaluation Metrics:")
        for metric, value in model_evaluation.items():
            print(f" - {metric}: {value}")


        # save experiment information
        save_experiment_info(
        args.csv_path,
        args.model,
        model_seed,
        args.strategy,
        model_evaluation.items()
        )

    except Exception as e:
        # for erros I didn't anticipate
        print(f"\nUnexpected Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()