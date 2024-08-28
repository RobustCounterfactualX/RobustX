import pandas as pd

from tasks.Task import Task


class ClassificationTask(Task):

    def train(self):
        self.model.train(self._training_data.X, self._training_data.y)

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        while self.model.predict_single(pos_instance) == neg_value:
            pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)
        return pos_instance

