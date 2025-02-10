from abc import ABC, abstractmethod
import pandas as pd
import torch


class BaseModel(ABC):
    """
    Abstract base class to define the essential methods that all model types must implement, providing template for
    training, predicting, and evaluating models in a standardized way.

    Attributes
    ----------
    _model : object
        The underlying model object (e.g., a scikit-learn model or a PyTorch model) that this class wraps.

    Methods
    -------
    train(X: pd.DataFrame, y: pd.DataFrame) -> None:
        Trains the model using the provided feature and target data.

    predict(X: pd.DataFrame) -> pd.DataFrame:
        Predicts the outcomes for the given instances.

    predict_single(X: pd.DataFrame) -> int:
        Predicts the outcome for a single instance and returns an integer.

    predict_proba(X: pd.DataFrame) -> pd.DataFrame:
        Predicts the probabilities of outcomes for the given instances.

    predict_proba_tensor(X: torch.Tensor) -> torch.Tensor:
        Predicts the probabilities of outcomes for tensor inputs.

    evaluate(X: pd.DataFrame, y: pd.DataFrame):
        Evaluates the model's performance on the provided feature and target data.

    Properties
    ----------
    model:
        Returns the underlying model object.
    """

    def __init__(self, model):
        """
        Initializes the BaseModel with a specific model.

        @param model: The model object to be used (e.g., a scikit-learn or PyTorch model).
        """
        self._model = model

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> None:
        """
        Trains the model using X feature variables and y target variable. Each implementing class
        can decide how to train their model and can add additional parameters, but X and y must be of
        type DataFrame.

        @param X: pd.DataFrame, The feature variables.
        @param y: pd.DataFrame, The target variable.

        @return: None
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the model to predict the outcomes for any number of instances.

        @param X: pd.DataFrame, Instances to predict.

        @return: pd.DataFrame, Predictions for each instance.
        """
        pass

    @abstractmethod
    def predict_single(self, X: pd.DataFrame) -> int:
        """
        Predicts the outcome of a single instance and returns an integer.

        @param X: pd.DataFrame, Instance to predict.

        @return: int, Prediction as an integer.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the probabilities of outcomes.

        @param X: pd.DataFrame, Instances to predict.

        @return: pd.DataFrame, Probabilities of each outcome.
        """
        pass

    @abstractmethod
    def predict_proba_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the probabilities of outcomes for tensor inputs.

        @param X: torch.Tensor, Instances to predict.

        @return: torch.Tensor, Probabilities of each outcome.
        """
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Evaluates the model's performance on the provided feature and target data.

        @param X: pd.DataFrame, The feature variables.
        @param y: pd.DataFrame, The target variable.

        @return: Evaluation result of the model.
        """
        pass

    @property
    def model(self):
        """
        Returns the underlying model object.

        @return: The model object.
        """
        return self._model
    
    def __repr__(self):
        return str(self._model)
