import pandas as pd

from tasks.Task import Task


class ValidityEvaluator:

    def __init__(self, task: Task, valid_val=1):
        self.task = task
        self.valid_val = valid_val

    def checkValidity(self, instance):
        if not isinstance(instance, pd.DataFrame):
            instance = pd.DataFrame(instance).T
        return self.task.model.predict_single(instance) == self.valid_val

    def evaluate(self, instances):
        pos = 0
        cnt = 0
        instances = instances.drop(columns=["target", "loss"], errors='ignore')
        for _, instance in instances.iterrows():
            if self.checkValidity(instance):
                pos += 1
            cnt += 1
        return pos / cnt
