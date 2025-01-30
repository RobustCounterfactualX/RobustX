import pandas as pd
import numpy as np
import torch
import joblib
from multimethod import multimethod
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from rocelib.models.TrainedModel import TrainedModel


class SKLearnModel(TrainedModel):
    def __init__(self, model_path: str):
        """
        Initialize the SKLearnModel by loading the saved scikit-learn model.

        :param model_path: Path to the saved model file (.pkl)
        """
        self.model = joblib.load(model_path)  # Load the saved model

    @classmethod
    def from_model(cls, model: BaseEstimator) -> 'SKLearnModel':
        """
        Alternative constructor to initialize SKLearnModel from a scikit-learn model instance.

        :param model: A scikit-learn model instance
        :return: An instance of SKLearnModel
        """
        instance = cls.__new__(cls)  # Create an instance without calling __init__
        instance.model = model  # Directly assign the model
        return instance

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the outcome using the scikit-learn model.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Predictions for each instance.
        """
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions, columns=["prediction"])

    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts a single outcome as an integer.

        :param x: pd.DataFrame, Instance to predict.
        :return: int, Single integer prediction.
        """
        prediction = self.predict(x)

        if prediction.empty or prediction.isna().any().any():
            raise ValueError("Prediction returned NaN or empty result. Check model training and inputs.")

        return int(prediction.iloc[0, 0])

    @multimethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Probabilities for each class.
        """
        probabilities = self.model.predict_proba(X)
        return pd.DataFrame(probabilities)

    @multimethod
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts class probabilities for a tensor input.

        :param X: torch.Tensor, Instances to predict.
        :return: torch.Tensor, Probabilities of each outcome.
        """
        X_numpy = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        probabilities = self.model.predict_proba(X_numpy)
        return torch.tensor(probabilities, dtype=torch.float32)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Evaluates the model using accuracy.

        :param X: pd.DataFrame, The feature variables.
        :param y: pd.Series, The target variable.
        :return: Accuracy of the model as a float.
        """
        predictions = self.predict(X)["prediction"].values
        return accuracy_score(y, predictions)
