from abc import ABC, abstractmethod

STRATEGY_REGISTRY = {}

def register_strategy(name):
    def decorator(cls):
        # register the class in the STRATEGY_REGISTRY dictionary
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

class BaseStrategy(ABC):
    """Interface for concrete classes."""

    def __init__(self, seed):
        """
        Initialize the strategy with a seed.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed

    @abstractmethod
    def evaluate(self, model, X, y, **kwargs):
        """
        Receives a model instance, features (X), and target (y).
        Returns a dictionary with metrics (e.g., RMSE, R2).
        """
        pass