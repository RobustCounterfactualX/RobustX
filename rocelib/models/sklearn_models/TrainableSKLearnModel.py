import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.TrainableModel import TrainableModel
from rocelib.models.TrainedModel import TrainedModel
from rocelib.models.imported_models.SKLearnModel import SKLearnModel



class TrainableSKLearnModel(TrainableModel):
    """
    A base class for scikit-learn models.

    This class wraps a scikit-learn model and provides methods for training, predicting,
    and evaluating the model. Inherits from TrainableModel.
    """

    def __init__(self, model):
        """
        Initializes the TrainableSKLearnModel with a scikit-learn model.

        @param model: The scikit-learn model instance to be wrapped.
        """
        super().__init__(model)

    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> TrainedModel:
        """
        Trains the scikit-learn model.

        @param y: The target variable, should be a DataFrame.
        """
        self.model.fit(X, y)
        return SKLearnModel.from_model(self._model)

    # def predict(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Predicts the outcomes for given feature variables.

    #     @param X: The feature variables, should be a DataFrame.
    #     @return: Predictions for each instance, returned as a DataFrame.
    #     """
    #     return self.model.predict(X)

    # def predict_single(self, X: pd.DataFrame) -> int:
    #     """
    #     Predicts the outcome for a single instance.

    #     @param X: The feature variables for a single instance, should be a DataFrame.
    #     @return: Prediction for the single instance, returned as an integer.
    #     """
    #     return self.predict(X)[0]

    # def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Predicts the probabilities of outcomes for given feature variables.

    #     @param X: The feature variables, should be a DataFrame.
    #     @return: Probabilities of each outcome, returned as a DataFrame.
    #     """
    #     return self.model.predict_proba(X)

    # def predict_proba_tensor(self, X: pd.DataFrame) -> torch.Tensor:
    #     """
    #     Predicts the probabilities of outcomes for given feature variables.

    #     @param X: The feature variables, should be a DataFrame.
    #     @return: Probabilities of each outcome, returned as a DataFrame.
    #     """
    #     return torch.tensor(self.model.predict_proba(X))

    # def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict:
    #     """
    #     Evaluates the model's performance using accuracy and F1 score.

    #     @param X: The feature variables, should be a DataFrame.
    #     @param y: The true target values, should be a DataFrame.
    #     @return: A dictionary with "accuracy" and "f1_score" of the model.
    #     """
    #     y_pred = self.predict(X)
    #     return {
    #         "accuracy": accuracy_score(y, y_pred),
    #         "f1_score": f1_score(y, y_pred, average='weighted')
    #     }
