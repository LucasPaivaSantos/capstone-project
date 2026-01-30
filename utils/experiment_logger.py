from datetime import datetime
import os
import numpy as np
# Import the new plotter module
from .visualization import plot_optimization_history, plot_pareto_front

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
        optimization_results (tuple): Optimization results tuple or None
    
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
            best_solutions, best_fitness, feature_names, history, objectives = optimization_results
            
            f.write("\n" + "="*60 + "\n")
            f.write("MULTI-OBJECTIVE OPTIMIZATION RESULTS\n")
            f.write("="*60 + "\n")
            
            # Write objectives information
            obj_names = ["Compressive Strength (MPa)"]
            obj_names.extend([obj.get_display_name() for obj in objectives])
            
            f.write(f"\nOptimization Objectives:\n")
            for i, name in enumerate(obj_names):
                direction = "maximize" if i == 0 else objectives[i-1].get_direction()
                f.write(f" {i+1}. {name} ({direction})\n")
            
            f.write(f"\nNumber of Pareto Optimal Solutions: {len(best_solutions)}\n")
            
            # Write all solutions
            f.write("\n" + "-"*60 + "\n")
            f.write("PARETO OPTIMAL SOLUTIONS\n")
            f.write("-"*60 + "\n")
            
            for idx in range(len(best_solutions)):
                f.write(f"\nSolution #{idx+1}:\n")
                
                # Write objective values
                f.write("  Objectives:\n")
                for i, obj_name in enumerate(obj_names):
                    f.write(f"    - {obj_name}: {best_fitness[idx, i]:.4f}\n")
                
                # Write mixture proportions
                f.write("  Mixture Proportions:\n")
                for name, value in zip(feature_names, best_solutions[idx]):
                    f.write(f"    - {name}: {value:.4f}\n")
            
            # Save solutions to CSV for easy import
            csv_file = os.path.join(experiment_dir, "pareto_solutions.csv")
            with open(csv_file, 'w') as csv_f:
                # Header
                header = obj_names + feature_names
                csv_f.write(",".join(header) + "\n")
                
                # Data rows
                for idx in range(len(best_solutions)):
                    row_data = list(best_fitness[idx]) + list(best_solutions[idx])
                    csv_f.write(",".join([f"{val:.6f}" for val in row_data]) + "\n")
            
            print(f"Pareto solutions saved to: {csv_file}")
            
            # Plot optimization history
            if history:
                plot_optimization_history(history, experiment_dir)
            
            # Plot Pareto front if multi-objective
            if len(objectives) > 0:
                plot_pareto_front(best_fitness, obj_names, experiment_dir, objectives)
    
    print(f"\nExperiment info saved to: {experiment_file}")
    return experiment_dir