import numpy as np
from .base import BaseObjective, register_objective

@register_objective('co2')
class CO2EmissionObjective(BaseObjective):
    """
    Objective to minimize CO2 emissions.
    
    Formula: 0.265 * source_material + 0.00524 * fine_aggregate + 
             0.00533 * coarse_aggregate + 1.39 * chemical_activator
    """
    
    def __init__(self):
        super().__init__()
        self.coefficients = {
            'source_material': 0.265,
            'fine_aggregate': 0.00524,
            'coarse_aggregate': 0.00533,
            'chemical_activator': 1.39
        }
    
    def calculate(self, feature_values, feature_names):
        """
        Calculate CO2 emission value.
        
        Args:
            feature_values (array): Array of feature values
            feature_names (list): List of feature names
            
        Returns:
            float: CO2 emission value
        """
        co2_value = 0.0
        
        for i, name in enumerate(feature_names):
            # Check if this feature has a coefficient
            if name in self.coefficients:
                co2_value += self.coefficients[name] * feature_values[i]
        
        return co2_value
    
    def get_direction(self):
        """Returns 'minimize' since we want to reduce CO2 emissions."""
        return 'minimize'
    
    def get_display_name(self):
        """Returns human-readable name."""
        return 'CO2 Emission (kg)'