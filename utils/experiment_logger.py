from datetime import datetime
import os
# Import the new plotter module
from .visualization import plot_optimization_history

def save_experiment_info(csv_path, model_name, model_seed, strategy_name, 
                        strategy_seed, test_size, model_evaluation, 
                        optimization_results=None):
    """
    Saves experiment information to a timestamped directory.
    
    Args:
        csv_path (str): Path to the CSV file used
        model_name (str): Name of the model
        model_seed (int): Seed used for the model
        strategy_name (str): Name of the validation strategy
        strategy_seed (int): Seed used for the validation strategy
        test_size (float): Test size used in 'train-split' strategy
        model_evaluation (iterable): Evaluation metrics of the model
    
    Returns:
        str: Path to the experiment directory
    """
    # create results directory if it doesn't exist
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(results_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # create experiment.txt file
    experiment_file = os.path.join(experiment_dir, "experiment.txt")
    with open(experiment_file, 'w') as f:
        f.write(f"CSV Path: {csv_path}\n")
        if model_name == 'svr':
            f.write(f"Model: {model_name}\n")
        else:
            f.write(f"Model: {model_name} with seed: {model_seed}\n")
        f.write(f"Strategy: {strategy_name} with seed: {strategy_seed}\n")
        
        if strategy_name == 'train-split':
            f.write(f"Test Size: {test_size}\n")
            
        f.write("\nModel Evaluation:\n")
        for metric, value in model_evaluation:
            f.write(f" - {metric}: {value}\n")
            
        if optimization_results:
            # unpack results
            best_mixture, best_strength, feature_names, history = optimization_results
            
            f.write("\nOptimization Results\n")
            f.write(f"Predicted Max Compressive Strength: {best_strength:.4f} MPa\n")
            f.write("Optimal Mixture Proportions:\n")
            for name, value in zip(feature_names, best_mixture):
                f.write(f" - {name}: {value:.4f}\n")
            
            # call the plotter if history exists
            if history:
                plot_optimization_history(history, experiment_dir)
    
    print(f"\nExperiment info saved to: {experiment_file}")
    return experiment_dir