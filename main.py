import argparse
import importlib
import os
import numpy as np
import pkgutil
import sys
import traceback

from utils.data_loader import load_and_validate_csv
from utils.experiment_logger import save_experiment_info
from optimization.optimizer import run_optimization

from models.base import MODEL_REGISTRY
from strategies.base import STRATEGY_REGISTRY

def load_modules(package_name):
    """
    Dynamically loads all modules in the specified package by joining their names.
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
    
    # data arguments
    parser.add_argument(
        "--csv_path",
        default="data/best.csv",
        help="Path to the CSV file containing the dataset (default: data/best.csv)"
    )

    # model arguments
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

    # strategy arguments
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
        help="Test size for train-test split."
    )

    parser.add_argument(
        "--n_splits",
        type=int,
        default=None,
        help="Number of folds for K-Fold cross-validation."
    )

    # optimization arguments
    parser.add_argument(
        "--optimize",
        action='store_true',
        help="Enable NSGA-II optimization"
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=10,
        help="Population Size"
    )
    parser.add_argument(
        "--n_gen",
        type=int,
        default=150,
        help="Number of Generations"
    )

    args = parser.parse_args()


    # if test_size is provided, strategy must be 'train-split'
    if args.test_size is not None and args.strategy != 'train-split':
        parser.error(f"The argument '--test_size' is not allowed with strategy '{args.strategy}'. It is only valid for 'train-split'.")

    # if n_splits is provided, strategy must be 'k-fold'
    if args.n_splits is not None and args.strategy != 'k-fold':
        parser.error(f"The argument '--n_splits' is not allowed with strategy '{args.strategy}'. It is only valid for 'k-fold'.")

    # set defaults based on strategy
    if args.strategy == 'train-split' and args.test_size is None:
        args.test_size = 0.2
        print("Note: 'train-split' selected without explicit test_size. Defaulting to 0.2")

    if args.strategy == 'k-fold' and args.n_splits is None:
        args.n_splits = 5
        print("Note: 'k-fold' selected without explicit n_splits. Defaulting to 5")

    return args

def main():
    # load modules
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
        elif args.strategy == 'k-fold':
            eval_kwargs['n_splits'] = args.n_splits
            print(f"Configuration: Number of Folds = {args.n_splits}")

        # run evaluation
        model_evaluation = strategy_instance.evaluate(model_instance, X, y, **eval_kwargs)
        
        # print model evaluation results
        print("\nEvaluation Metrics:")
        for metric, value in model_evaluation.items():
            print(f" - {metric}: {value}")

        # run optimization
        optimization_results = None
        if args.optimize:
            print("Retraining model for optimization")
            model_instance.fit(X, y)
            
            best_mixture, best_fitness, feat_names, history = run_optimization(
                model=model_instance,
                X_train=X,
                pop_size=args.pop_size,
                n_gen=args.n_gen,
                seed=model_seed # maybe another seed arg?
            )
            
            # pack results for logger
            optimization_results = (best_mixture, best_fitness, feat_names, history)
            
            print("\nOptimization Results:")
            print(f"Predicted Max Strength: {best_fitness:.4f}")
            print("Mixture:")
            for n, v in zip(feat_names, best_mixture):
                print(f"  {n}: {v:.4f}")

        # save experiment information
        save_experiment_info(
            args.csv_path,
            args.model,
            model_seed,
            args.strategy,
            strategy_seed,
            args.test_size if args.strategy == 'train-split' else args.n_splits if args.strategy == 'k-fold' else None,
            model_evaluation.items(),
            optimization_results=optimization_results
        )

    except Exception as e:
        # for erros I didn't anticipate
        print(f"\nUnexpected Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()