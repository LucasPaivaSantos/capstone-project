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

def handle_cli_args():
    """
    Handles all CLI argument parsing and validation logic.
    Returns the parsed arguments object.
    """

    parser = argparse.ArgumentParser(description="Acid Geopolymer Concrete Optimizer")
    
    parser.add_argument(
        "--csv_path",
        default="data/best.csv",
        help="Path to the CSV file containing the dataset (default: data/best.csv)"
    )

    parser.add_argument(
        "--model", "-m",
        required=True, 
        choices=MODEL_REGISTRY.keys(),
        help="Machine Learning Model to use"
    )
    parser.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="Seed for ML model (if not provided, a random seed will be generated)"
    )

    parser.add_argument(
        "--strategy", "-s",
        required=True, 
        choices=STRATEGY_REGISTRY.keys(),
        help="Validation Strategy"
    )
    parser.add_argument(
        "--strategy_seed",
        type=int,
        default=None,
        help="Seed for validation strategy (if not provided, a random seed will be generated)"
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Test size for train-test split (e.g., 0.2). Only valid for 'train-split' strategy."
    )

    args = parser.parse_args()
    

    # if test_size is provided, strategy must be 'train-split'
    if args.test_size is not None and args.strategy != 'train-split':
        parser.error(f"The argument '--test_size' is not allowed with strategy '{args.strategy}'. It is only valid for 'train-split'.")

    # set default to 0.2 if strategy is 'train-split' and test_size not provided
    if args.strategy == 'train-split' and args.test_size is None:
        args.test_size = 0.2
        print("Note: 'train-split' selected without explicit test_size. Defaulting to 0.2")

    return args

def main():
    # load all modules
    load_modules('models')
    load_modules('strategies')

    # get arguments
    args = handle_cli_args()
    

    # main flow
    try:
        print("Loading Data")
        X, y = load_and_validate_csv(args.csv_path)

        # generate or use provided seeds
        model_seed = args.model_seed if args.model_seed is not None else np.random.randint(0, 100)
        strategy_seed = args.strategy_seed if args.strategy_seed is not None else np.random.randint(0, 100)

        if args.model == 'svr':
            print(f"\nInitializing Model: {args.model}")
        else:
            print(f"\nInitializing Model: {args.model} with seed {model_seed}")
        model_cls = MODEL_REGISTRY[args.model]
        model_instance = model_cls(seed=model_seed)

        print(f"\nExecuting Strategy: {args.strategy} with seed {strategy_seed}")
        strategy_cls = STRATEGY_REGISTRY[args.strategy]
        strategy_instance = strategy_cls(seed=strategy_seed)

        # prepare evaluation kwargs based on strategy
        eval_kwargs = {}
        if args.strategy == 'train-split':
            eval_kwargs['test_size'] = args.test_size
            print(f"Configuration: Test Size = {args.test_size}")

        # run evaluation
        model_evaluation = strategy_instance.evaluate(model_instance, X, y, **eval_kwargs)
        
        # print model evaluation results
        print("\nEvaluation Metrics:")
        for metric, value in model_evaluation.items():
            print(f" - {metric}: {value}")


        # save experiment information
        save_experiment_info(
        args.csv_path,
        args.model,
        model_seed,
        args.strategy,
        strategy_seed,
        args.test_size,
        model_evaluation.items()
        )

    except Exception as e:
        # for erros I didn't anticipate
        print(f"\nUnexpected Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()