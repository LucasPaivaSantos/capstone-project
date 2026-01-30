from abc import ABC, abstractmethod

OBJECTIVE_REGISTRY = {}

def register_objective(name):
    """Decorator to register objective functions."""
    def decorator(cls):
        OBJECTIVE_REGISTRY[name] = cls
        return cls
    return decorator

class BaseObjective(ABC):
    """Base class for optimization objectives."""
    
    def __init__(self):
        """Initialize the objective."""
        pass
    
    @abstractmethod
    def calculate(self, feature_values, feature_names):
        """
        Calculate the objective value.
        
        Args:
            feature_values (array): Array of feature values
            feature_names (list): List of feature names
            
        Returns:
            float: Objective value
        """
        pass
    
    @abstractmethod
    def get_direction(self):
        """
        Returns the optimization direction.
        
        Returns:
            str: 'minimize' or 'maximize'
        """
        pass
    
    @abstractmethod
    def get_display_name(self):
        """
        Returns a human-readable name for the objective.
        
        Returns:
            str: Display name
        """
        pass