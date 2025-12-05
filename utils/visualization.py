import matplotlib.pyplot as plt
import os

def plot_optimization_history(history, save_dir):
    """
    Plots the NSGA-II optimization history (Best Fitness per Generation).
    
    Args:
        history (list): List of best fitness values from each generation.
        save_dir (str): Directory to save the plot.
    """
    if not history:
        print("No optimization history to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    # plot generation vs fitness
    generations = range(1, len(history) + 1)
    plt.plot(generations, history, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=4)
    
    plt.title('Optimization Evolution (NSGA-II)', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Predicted Compressive Strength (MPa)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # annotate the best value
    best_val = max(history)
    best_gen = history.index(best_val) + 1
    plt.annotate(f'Max: {best_val:.2f} MPa', 
                 xy=(best_gen, best_val), 
                 xytext=(best_gen, best_val + (best_val * 0.05)),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # save the file
    filename = "optimization_evolution.png"
    filepath = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"Optimization evolution graph saved to: {filepath}")