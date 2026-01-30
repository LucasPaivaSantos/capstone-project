import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

class GeopolymerProblem(ElementwiseProblem):
    def __init__(self, model, feature_names, xl, xu, objectives):
        """
        Initialize the multi-objective problem.
        
        Args:
            model: Trained ML model for strength prediction
            feature_names: List of feature names
            xl: Lower bounds for variables
            xu: Upper bounds for variables
            objectives: List of objective instances (CO2, Cost, etc.)
        """
        # Number of objectives = 1 (strength) + additional objectives
        n_objectives = 1 + len(objectives)
        
        super().__init__(n_var=len(feature_names), 
                         n_obj=n_objectives, 
                         n_ieq_constr=0, 
                         xl=xl, 
                         xu=xu)
        self.model = model
        self.feature_names = feature_names
        self.objectives = objectives

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate all objectives for a given solution.
        
        Args:
            x: Decision variables (mixture proportions)
            out: Output dictionary
        """
        # Objective 1: Maximize compressive strength (minimize negative strength)
        input_data = pd.DataFrame([x], columns=self.feature_names)
        prediction = self.model.predict(input_data)[0]
        
        # Initialize objectives list
        obj_values = [-prediction]  # Negative for maximization
        
        # Calculate additional objectives
        for objective in self.objectives:
            obj_value = objective.calculate(x, self.feature_names)
            # All objectives are minimization in NSGA-II
            obj_values.append(obj_value)
        
        out["F"] = obj_values

def run_optimization(model, X_train, pop_size=100, n_gen=50, seed=1, objectives=None):
    """
    Sets up and runs the NSGA-II multi-objective optimization.
    
    Args:
        model: Trained ML model
        X_train: Training data (DataFrame)
        pop_size: Population size
        n_gen: Number of generations
        seed: Random seed
        objectives: List of objective instances to optimize
    
    Returns: 
        best_solutions: Array of Pareto optimal solutions
        best_fitness: Array of fitness values for Pareto solutions
        feature_names: List of feature names
        history: List of hypervolume or generation metrics
    """
    if objectives is None:
        objectives = []
    
    xl = X_train.min().values
    xu = X_train.max().values
    feature_names = X_train.columns.tolist()

    print(f"\nOptimization Configuration")
    print(f"Objectives: Maximize Strength" + 
          (f" + {', '.join([obj.get_display_name() for obj in objectives])}" if objectives else ""))
    print(f"Search Space Bounds:")
    for name, low, high in zip(feature_names, xl, xu):
        print(f" - {name}: [{low:.2f}, {high:.2f}]")

    problem = GeopolymerProblem(model, feature_names, xl, xu, objectives)

    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    print(f"\nRunning NSGA-II for {n_gen} generations with pop_size={pop_size}")
    
    result = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True, 
                   verbose=True)

    # Extract Pareto front solutions
    if result.X.ndim == 1:
        # Single solution
        best_solutions = result.X.reshape(1, -1)
        best_fitness = result.F.reshape(1, -1)
    else:
        # Multiple solutions (Pareto front)
        best_solutions = result.X
        best_fitness = result.F
    
    # Convert first objective back to positive (strength)
    best_fitness[:, 0] = -best_fitness[:, 0]
    
    # Extract history - track hypervolume or best strength over generations
    history_metrics = []
    if result.history:
        for gen_algo in result.history:
            # Track the best strength in each generation
            best_strength = -np.min(gen_algo.opt.get("F")[:, 0])
            history_metrics.append(float(best_strength))

    return best_solutions, best_fitness, feature_names, history_metrics