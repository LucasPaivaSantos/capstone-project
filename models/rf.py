from sklearn.ensemble import RandomForestRegressor
from .base import BaseModel, register_model

@register_model('random_forest')
class RandomForestModel(BaseModel):
    def __init__(self, seed):
        super().__init__(seed)
        print(f"Random Forest initialized with seed: {self.seed}")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.seed
            )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)