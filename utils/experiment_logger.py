from datetime import datetime
import os

def save_experiment_info(csv_path, model_name, seed):
    """
    Saves experiment information to a timestamped directory.
    
    Args:
        csv_path (str): Path to the CSV file used
        model_name (str): Name of the model
        seed (int): Random seed used for the model
    
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
        f.write(f"Model: {model_name} with seed: {seed}\n")
    
    print(f"\nExperiment info saved to: {experiment_file}")
    return experiment_dir