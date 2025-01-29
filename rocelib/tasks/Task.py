from abc import ABC, abstractmethod

import pandas as pd

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.TrainableModel import TrainableModel


class Task(ABC):
    """
    An abstract base class representing a general task that involves training a model
    on a specific dataset.

    Attributes:
        _training_data (DatasetLoader): The dataset used for training the model.
        __model (TrainableModel): The model to be trained and used for predictions.
    """

    def __init__(self, model: TrainableModel, training_data: DatasetLoader):
        """
        Initializes the Task with a model and training data.

        @param model: An instance of a model that extends TrainableModel
        @param training_data: An instance of DatasetLoader containing the training data.
        """
        self._training_data = training_data
        self.__model = model

    @abstractmethod
    def train(self, **kwargs):
        """
        Abstract method to train the model on the provided training data.
        Must be implemented by subclasses.
        """
        pass

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        """
        Abstract method to retrieve a random positive instance from the training data.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column.
        @return: A Pandas Series representing a random positive instance.
        """
        pass

    @property
    def training_data(self):
        """
        Property to access the training data.

        @return: The training data loaded from DatasetLoader.
        """
        return self._training_data

    @property
    def model(self):
        """
        Property to access the model.

        @return: The model instance that extends TrainableModel
        """
        return self.__model
