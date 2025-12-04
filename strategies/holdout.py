import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from .base import BaseStrategy, register_strategy

@register_strategy('train-split')
class TrainTestSplitStrategy(BaseStrategy):
    def __init__(self, seed):
        super().__init__(seed)
        print(f"Train-Test Split Strategy initialized with seed: {self.seed}")

    def evaluate(self, model, X, y, test_size=0.2, **kwargs):
        print(f"Running Train-Test Split (Size: {test_size})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=self.seed
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {"RMSE": rmse, "R2": r2}