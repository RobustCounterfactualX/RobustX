from datasets.DatasetLoader import DatasetLoader
from models.BaseModel import BaseModel
import pandas as pd

from tasks.Task import Task


class ClassificationTask(Task):

    def __init__(self, model: BaseModel, training_data: DatasetLoader):
        super().__init__(training_data)
        self._model = model

    def train(self):
        self._model.train(self._training_data.X, self._training_data.y)

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        while self._model.predict_single(pos_instance) == neg_value:
            pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        return pos_instance

    @property
    def model(self):
        return self._model
