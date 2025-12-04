from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .base import BaseModel, register_model

@register_model('svr')
class SVRModel(BaseModel):
    def __init__(self, seed):
        super().__init__(seed) # seed is being used just to satisfy BaseModel interface
        print(f"SVR initialized")
        
        self.model = make_pipeline(
            StandardScaler(),
            SVR(
                kernel='rbf',  # radial basis function
                C=100.0,
                epsilon=0.1
            )
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)