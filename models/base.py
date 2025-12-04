from abc import ABC, abstractmethod

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        # register the class in the MODEL_REGISTRY dictionary
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

class BaseModel(ABC):
    """Interface for concrete classes."""
    
    @abstractmethod
    def fit(self, X, y):
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Returns predictions."""
        pass