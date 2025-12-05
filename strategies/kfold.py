import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from .base import BaseStrategy, register_strategy

@register_strategy('k-fold')
class KFoldStrategy(BaseStrategy):
    def __init__(self, seed):
        super().__init__(seed)
        print(f"K-Fold Cross-Validation Strategy initialized with seed: {self.seed}")

    def evaluate(self, model, X, y, n_splits=5, **kwargs):
        n_splits = n_splits
        print(f"Running K-Fold Cross-Validation (Folds: {n_splits})")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        
        rmse_scores = []
        r2_scores = []
        
        fold_num = 1
        for train_index, test_index in kf.split(X):
            print(f"  Processing fold {fold_num}/{n_splits}...")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # create a fresh model instance for each fold
            from models.base import MODEL_REGISTRY
            model_cls = type(model)
            fold_model = model_cls(seed=model.seed)
            
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            
            fold_num += 1
        
        # calculate mean and standard deviation
        mean_rmse = np.mean(rmse_scores)
        std_rmse = np.std(rmse_scores)
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        
        return {
            "RMSE_mean": mean_rmse,
            "RMSE_std": std_rmse,
            "R2_mean": mean_r2,
            "R2_std": std_r2
        }