import xgboost as xgb
from .base import BaseModel, register_model

@register_model('xgboost')
class XGBoostModel(BaseModel):
    def __init__(self, seed):
        self.seed = seed
        print(f"XGBoost initialized with seed: {self.seed}")
        self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=self.seed)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)