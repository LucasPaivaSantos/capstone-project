import numpy as np
from .base import BaseObjective, register_objective

@register_objective('cost')
class CostObjective(BaseObjective):
    """
    Objective to minimize cost.
    
    Formula: 4.0 * source_material + 0.15 * fine_aggregate + 
             0.15 * coarse_aggregate + 61.52 * chemical_activator
    """
    
    def __init__(self):
        super().__init__()
        self.coefficients = {
            'source_material': 4.0,
            'fine_aggregate': 0.15,
            'coarse_aggregate': 0.15,
            'chemical_activator': 61.52
        }
    
    def calculate(self, feature_values, feature_names):
        """
        Calculate cost value.
        
        Args:
            feature_values (array): Array of feature values
            feature_names (list): List of feature names
            
        Returns:
            float: Cost value
        """
        cost_value = 0.0
        
        for i, name in enumerate(feature_names):
            # Check if this feature has a coefficient
            if name in self.coefficients:
                cost_value += self.coefficients[name] * feature_values[i]
        
        return cost_value
    
    def get_direction(self):
        """Returns 'minimize' since we want to reduce cost."""
        return 'minimize'
    
    def get_display_name(self):
        """Returns human-readable name."""
        return 'Cost ($)'