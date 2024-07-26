from datasets.DatasetLoader import DatasetLoader
from models.BaseModel import BaseModel
from tasks.Task import Task


class ModelMultiplicityTask(Task):

    def __init__(self, models: list[BaseModel], training_data: DatasetLoader):
        super().__init__(training_data)
        self._models = models

    def train(self):
        for model in self.models:
            model.train(self._training_data.X, self._training_data.y.to_frame())

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        while not self.__checkAllPositive(pos_instance.to_frame(), neg_value):
            pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        return pos_instance

    def __checkAllPositive(self, instance, neg_value):
        for model in self.models:
            if model.predict_single(instance) == neg_value:
                return False
        return True

    @property
    def models(self):
        return self._models
