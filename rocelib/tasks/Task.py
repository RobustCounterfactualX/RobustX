from abc import ABC, abstractmethod

import pandas as pd

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.TrainedModel import TrainedModel
from typing import List, Dict, Any, Tuple


class Task(ABC):
    """
    An abstract base class representing a general task that involves training a model
    on a specific dataset.

    Attributes:
        _dataset (DatasetLoader): The dataset used for training the model.
        __model (TrainableModel): The model to be trained and used for predictions.
    """

    def __init__(self, model: TrainedModel, dataset: DatasetLoader, mm_models: Dict[str, TrainedModel] = None):
        """
        Initializes the Task with a model and training data and optionally multiple models for MM

        @param model: An instance of a model that extends TrainedModel
        @param dataset: An instance of DatasetLoader containing the training data.
        @param dataset: A list of instances of a model that extends TrainedModel
        """
        self._dataset = dataset
        self.__model = model
        self._CEs: Dict[str, Tuple[pd.DataFrame, float]] = {}  # Stores generated counterfactuals per method
        # self._mm_CEs: List[Dict[str, Tuple[pd.DataFrame, float]]] = []
        self._mm_CEs: Dict[str, Dict[str, Tuple[pd.DataFrame, float]]] = {} #Stores generated counterfactuals per model per method
        self.__mm_models: Dict[str, TrainedModel] = mm_models

        # Set mm_flag based on whether the user added multiple models
        if mm_models and len(mm_models) > 1:
            self.mm_flag = True
        else:
            self.mm_flag = False

        self.methods = {}
        self.evaluation_metrics = {}


    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        """
        Abstract method to retrieve a random positive instance from the training data.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column.
        @return: A Pandas Series representing a random positive instance.
        """
        pass

    def generate(self, methods: List[str]) -> Dict[str, Tuple[pd.DataFrame, float]]:
        pass

    def generate_mm(self, methods: List[str]) -> Dict[str, Tuple[pd.DataFrame, float]]:
        pass

    def evaluate(self, methods: List[str], evaluations: List[str]) -> Dict[str, Dict[str, Any]]:
        pass

    def get_recourse_methods(self) -> List[str]:
        return list(self.methods.keys())

    def get_evaluation_metrics(self) -> List[str]:
        return list(self.evaluation_metrics.keys())

    @property
    def dataset(self):
        """
        Property to access the training data.

        @return: The training data loaded from DatasetLoader.
        """
        return self._dataset
    
    @property
    def ces(self):
        """
        Property to access the training data.

        @return: The training data loaded from DatasetLoader.
        """
        return self._CEs

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

    @property
    def mm_CEs(self):
        return self._mm_CEs
