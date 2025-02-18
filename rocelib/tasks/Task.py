from abc import ABC, abstractmethod

import pandas as pd

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.TrainedModel import TrainedModel
from typing import List, Dict, Any, Union, Tuple


class Task(ABC):
    """
    An abstract base class representing a general task that involves training a model
    on a specific dataset.

    Attributes:
        _dataset (DatasetLoader): The dataset used for training the model.
        __model (TrainableModel): The model to be trained and used for predictions.
    """

    def __init__(self, model: TrainedModel, dataset: DatasetLoader, mm_models: [TrainedModel] = None):
        """
        Initializes the Task with a model and training data.

        @param model: An instance of a model that extends TrainableModel
        @param dataset: An instance of DatasetLoader containing the training data.
        """
        self._dataset = dataset
        self.__model = model
        self._CEs: Dict[str, Tuple[pd.DataFrame, float]] = {}  # Stores generated counterfactuals per method
        self.__mm_models = mm_models



    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        """
        Abstract method to retrieve a random positive instance from the training data.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column.
        @return: A Pandas Series representing a random positive instance.
        """
        pass

    @property
    def dataset(self):
        """
        Property to access the training data.

        @return: The training data loaded from DatasetLoader.
        """
        return self._dataset

    @property
    def model(self):
        """
        Property to access the model.

        @return: The model instance that extends TrainableModel
        """
        return self.__model

    @property
    def mm_models(self):
        """
        Property to access the model.

        @return: The model instance that extends TrainableModel
        """
        return self.__mm_models

    @property
    def CEs(self):
        return self._CEs
