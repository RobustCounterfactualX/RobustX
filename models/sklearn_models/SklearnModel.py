import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from models.BaseModel import BaseModel


class SklearnModel(BaseModel):
    def __init__(self, model):
        super().__init__(model)

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def predict(self, X) -> pd.DataFrame:
        return self.model.predict(X)

    def predict_single(self, X) -> int:
        return self.predict(X)[0]

    def predict_proba(self, X):
        self.model.predict_proba(X)

    def evaluate(self, X, y) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average='weighted')
        }