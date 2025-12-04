import xgboost as xgb
import numpy as np
from .base import BaseModel, register_model

@register_model('xgboost')
class XGBoostModel(BaseModel):
    def __init__(self):
        seed = np.random.randint(0, 100)
        # seed = 100  # for reproducibility during testing
        print("xgboost seed used: ", seed)
        self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)