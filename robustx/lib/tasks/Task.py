from abc import ABC, abstractmethod
import pandas as pd
from robustx.datasets.DatasetLoader import DatasetLoader
from robustx.lib.models.BaseModel import BaseModel


class Task(ABC):
    """
    An abstract base class representing a general task that involves training a model
    on a specific dataset.

    Attributes:
        _training_data (DatasetLoader): The dataset used for training the model.
        __model (BaseModel): The model to be trained and used for predictions.
    """

    def __init__(self, model: BaseModel, training_data: DatasetLoader):
        """
        Initializes the Task with a model and training data.

        @param model: An instance of a model that extends BaseModel
        @param training_data: An instance of DatasetLoader containing the training data.
        """
        self._training_data = training_data
        self.__model = model


    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        """
        Abstract method to retrieve a random positive instance from the training data.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column.
        @return: A Pandas Series representing a random positive instance.
        """
        pass

    def get_negative_instances(self, neg_value=0, column_name="target") -> pd.DataFrame:
        """
        Abstract method to retrieve all the negative instances in the dataset as predicted by the model.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column.
        @return: All instances with a negative target value predicted by the model.
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

        @return: The model instance that extends BaseModel
        """
        return self.__model
