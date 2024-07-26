from abc import ABC, abstractmethod
import pandas as pd
from datasets.DatasetLoader import DatasetLoader


class Task(ABC):

    def __init__(self, training_data: DatasetLoader):
        self._training_data = training_data

    @abstractmethod
    def train(self):
        pass

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        pass

    @property
    def training_data(self):
        return self._training_data
