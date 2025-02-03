from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
import torch


class TrainedModel(ABC):
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the model to predict the outcomes for any number of instances.

        @param X: pd.DataFrame, Instances to predict.

        @return: pd.DataFrame, Predictions for each instance.
        """
        pass

    @abstractmethod
    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts the outcome of a single instance and returns an integer.

        @param x: pd.DataFrame, Instance to predict.

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
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Evaluates the model's performance on the provided feature and target data.

        @param X: pd.DataFrame, The feature variables.
        @param y: pd.DataFrame, The target variable.

        @return: Accuracy of the model as a float.
        """
        pass
