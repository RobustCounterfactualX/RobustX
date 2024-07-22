from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """
    Abstract class to provide minimum functionality from all types of models
    """

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Trains the model using X feature variables and y target variable, each implementing class
        can decide how to train their model and can add additional parameters but X and y must be of
        type DataFrame
        :param X: The feature variables
        :param y: The target variable
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the model to predict the outcomes of any number of instances
        :param X: Instances to predict
        :return: Predictions for each instance
        """
        pass

    @abstractmethod
    def predict_single(self, X: pd.DataFrame) -> int:
        """
        Predicts outcome of single instance and returns an integer
        :param X: Instance to predict
        :return: Prediction as integer
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts probability of outcomes
        :param X: Instances to predict
        :return: Probabilities of each outcome
        """
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Evaluates the model
        :param X: The feature variables
        :param y: The target variable
        :return: An evaluation of model
        """
        pass
