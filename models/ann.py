from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base import BaseModel, register_model

@register_model('ann')
class ANNModel(BaseModel):
    def __init__(self, seed):
        super().__init__(seed)
        print(f"ANN (MLPRegressor) initialized with seed: {self.seed}")
        
        # wrap the model in a pipeline with StandardScaler for normalization
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(100, 50), # two hidden layers with 100 and 50 neurons
                activation='relu', # rectified linear unit
                solver='lbfgs', # optimizer suitable for smaller datasets
                max_iter=4000,
                random_state=self.seed
            )
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)