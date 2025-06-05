from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Model:
    """A wrapper for scikit-learn models that includes a StandardScaler pipeline."""
    def __init__(self, model):
        self.model = make_pipeline(StandardScaler(), model)
        
    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def __getattr__(self, name):
        return getattr(self.model, name)
