import os
import pandas as pd

REQUIRED_COLUMNS = [
    'chemical_activator',
    'source_material',
    'fine_aggregate',
    'coarse_aggregate',
    'compressive_strength'
]

def load_and_validate_csv(csv_path):
    """Loads CSV and returns X (features) and y (target)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: The file '{csv_path}' was not found.")
    
    try:
        # create dataframe from CSV
        data_frame = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # validate columns
    if not set(REQUIRED_COLUMNS).issubset(data_frame.columns):
        raise ValueError(f"Missing columns. Expected: {REQUIRED_COLUMNS}")
    
    print(f"Successfully loaded {len(data_frame)} rows.")

    # separate features and target
    X = data_frame.drop(columns=['compressive_strength']) # X got all the columns except 'compressive_strength'
    y = data_frame['compressive_strength'] # y got only the 'compressive_strength' column

    return X, y