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
    def __init__(self, model, feature_names, xl, xu):
        super().__init__(n_var=len(feature_names), 
                         n_obj=1, 
                         n_ieq_constr=0, 
                         xl=xl, 
                         xu=xu)
        self.model = model
        self.feature_names = feature_names

    def _evaluate(self, x, out, *args, **kwargs):
        input_data = pd.DataFrame([x], columns=self.feature_names)
        prediction = self.model.predict(input_data)[0]
        # minimize negative strength to maximize strength
        out["F"] = -prediction

def run_optimization(model, X_train, pop_size=100, n_gen=50, seed=1):
    """
    Sets up and runs the NSGA-II optimization.
    Returns: best_solution, best_fitness, feature_names, history
    """
    
    xl = X_train.min().values
    xu = X_train.max().values
    feature_names = X_train.columns.tolist()

    print(f"\nOptimization Configuration")
    print(f"Search Space Bounds:")
    for name, low, high in zip(feature_names, xl, xu):
        print(f" - {name}: [{low:.2f}, {high:.2f}]")

    problem = GeopolymerProblem(model, feature_names, xl, xu)

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
    
    # save_history to track evolution
    result = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True, 
                   verbose=True)

    # TODO: Im not sure about this part
    # extract the best solution and its fitness
    if result.X.ndim == 2:
        idx = np.argmin(result.F.flatten())
        best_solution = result.X[idx]
        best_fitness_val = -result.F.flatten()[idx]
    else:
        best_solution = result.X
        best_fitness_val = -result.F[0] if result.F.size > 0 else -result.F

    if hasattr(best_fitness_val, "item"):
        best_fitness_val = best_fitness_val.item()
    else:
        best_fitness_val = float(best_fitness_val)

    # extract the best fitness from each generation
    history_fitness = []
    if result.history:
        for gen_algo in result.history:
            best_of_gen = -gen_algo.opt[0].F[0]
            history_fitness.append(float(best_of_gen))

    return best_solution, best_fitness_val, feature_names, history_fitness