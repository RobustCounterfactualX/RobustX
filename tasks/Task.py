from abc import ABC, abstractmethod
import pandas as pd
from datasets.DatasetLoader import DatasetLoader
from models.BaseModel import BaseModel


class Task(ABC):

    def __init__(self, model: BaseModel, training_data: DatasetLoader):
        self._training_data = training_data
        self.__model = model

    @abstractmethod
    def train(self):
        pass

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        pass

    @property
    def training_data(self):
        return self._training_data

    @property
    def model(self):
        return self.__model
