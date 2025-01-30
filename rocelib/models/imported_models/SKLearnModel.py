import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score
from rocelib.models.TrainedModel import TrainedModel


class SKLearnModel(TrainedModel):
    """
    Wrapper class for scikit-learn models.
    Provides methods for training, predicting, and evaluating models.
    """

    def __init__(self, model: BaseEstimator):
        """
        Initializes the SKLearnModel with a scikit-learn model.

        :param model: A scikit-learn model instance
        """
        self.model = model

    @classmethod
    def from_model(cls, model: BaseEstimator) -> 'SKLearnModel':
        """
        Alternative constructor to initialize SKLearnModel from an existing model instance.

        :param model: A scikit-learn model instance
        :return: An instance of SKLearnModel
        """
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance.model = model
        return instance

    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Trains the scikit-learn model. Not sure if this is required

        :param X: Feature variables (DataFrame)
        :param y: Target variable (Series)
        """
        self.model.fit(X, y, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the outcomes for given feature variables.

        :param X: Feature variables (DataFrame)
        :return: Predictions as a DataFrame
        """
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions, columns=["prediction"])

    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts a single outcome as an integer.

        :param x: Single instance (DataFrame)
        :return: Integer prediction
        """
        prediction = self.predict(x).iloc[0, 0]
        return int(prediction)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities for given instances.

        :param X: Feature variables (DataFrame)
        :return: Probabilities as a DataFrame
        """
        probabilities = self.model.predict_proba(X)
        return pd.DataFrame(probabilities)

    def predict_proba_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        """
        Predicts class probabilities and returns them as a PyTorch tensor. Not sure if this is required

        :param X: Feature variables (DataFrame)
        :return: Probabilities as a torch.Tensor
        """
        probabilities = self.model.predict_proba(X)
        return torch.tensor(probabilities, dtype=torch.float32)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Evaluates the model's performance using accuracy and F1 score.

        :param X: Feature variables (DataFrame)
        :param y: True target values (Series)
        :return: Dictionary containing "accuracy" and "f1_score"
        """
        y_pred = self.predict(X)["prediction"].values
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred, average='weighted')
        }

    #TODO: check if train and predict_proba_tensor methods are required