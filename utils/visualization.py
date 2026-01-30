import matplotlib.pyplot as plt
import numpy as np
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
    plt.ylabel('Best Predicted Compressive Strength (MPa)', fontsize=12)
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


def plot_pareto_front(fitness_values, objective_names, save_dir, objectives):
    """
    Plots the Pareto front for multi-objective optimization.
    
    Args:
        fitness_values (ndarray): Array of fitness values (n_solutions x n_objectives)
        objective_names (list): List of objective names
        save_dir (str): Directory to save the plot
        objectives (list): List of objective instances
    """
    n_objectives = fitness_values.shape[1]
    
    if n_objectives == 2:
        # 2D Pareto front
        plot_pareto_2d(fitness_values, objective_names, save_dir)
    elif n_objectives == 3:
        # 3D Pareto front
        plot_pareto_3d(fitness_values, objective_names, save_dir)
    else:
        # Parallel coordinates plot for 4+ objectives
        plot_parallel_coordinates(fitness_values, objective_names, save_dir)


def plot_pareto_2d(fitness_values, objective_names, save_dir):
    """
    Plots 2D Pareto front.
    
    Args:
        fitness_values (ndarray): Array of fitness values (n_solutions x 2)
        objective_names (list): List of objective names
        save_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 7))
    
    plt.scatter(fitness_values[:, 0], fitness_values[:, 1], 
                c='#1f77b4', s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Sort by first objective for line connection
    sorted_indices = np.argsort(fitness_values[:, 0])
    sorted_fitness = fitness_values[sorted_indices]
    plt.plot(sorted_fitness[:, 0], sorted_fitness[:, 1], 
             'r--', alpha=0.3, linewidth=1)
    
    plt.xlabel(objective_names[0], fontsize=12)
    plt.ylabel(objective_names[1], fontsize=12)
    plt.title('Pareto Front - Trade-off Analysis', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = "pareto_front_2d.png"
    filepath = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"2D Pareto front saved to: {filepath}")


def plot_pareto_3d(fitness_values, objective_names, save_dir):
    """
    Plots 3D Pareto front.
    
    Args:
        fitness_values (ndarray): Array of fitness values (n_solutions x 3)
        objective_names (list): List of objective names
        save_dir (str): Directory to save the plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(fitness_values[:, 0], 
                        fitness_values[:, 1], 
                        fitness_values[:, 2],
                        c=fitness_values[:, 0], 
                        cmap='viridis', 
                        s=100, 
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=1)
    
    ax.set_xlabel(objective_names[0], fontsize=11, labelpad=10)
    ax.set_ylabel(objective_names[1], fontsize=11, labelpad=10)
    ax.set_zlabel(objective_names[2], fontsize=11, labelpad=10)
    ax.set_title('3D Pareto Front - Multi-Objective Trade-offs', fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(objective_names[0], rotation=270, labelpad=15)
    
    filename = "pareto_front_3d.png"
    filepath = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"3D Pareto front saved to: {filepath}")


def plot_parallel_coordinates(fitness_values, objective_names, save_dir):
    """
    Plots parallel coordinates for 4+ objectives.
    
    Args:
        fitness_values (ndarray): Array of fitness values (n_solutions x n_objectives)
        objective_names (list): List of objective names
        save_dir (str): Directory to save the plot
    """
    import pandas as pd
    
    # Normalize fitness values to [0, 1] for better visualization
    normalized_fitness = np.zeros_like(fitness_values)
    for i in range(fitness_values.shape[1]):
        min_val = fitness_values[:, i].min()
        max_val = fitness_values[:, i].max()
        if max_val > min_val:
            normalized_fitness[:, i] = (fitness_values[:, i] - min_val) / (max_val - min_val)
        else:
            normalized_fitness[:, i] = 0.5
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_objectives = len(objective_names)
    x = range(n_objectives)
    
    # Plot each solution as a line
    for i in range(len(normalized_fitness)):
        ax.plot(x, normalized_fitness[i], alpha=0.4, linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(objective_names, rotation=45, ha='right')
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Parallel Coordinates - Pareto Solutions', fontsize=14)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    
    filename = "pareto_parallel_coordinates.png"
    filepath = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
    print(f"Parallel coordinates plot saved to: {filepath}")