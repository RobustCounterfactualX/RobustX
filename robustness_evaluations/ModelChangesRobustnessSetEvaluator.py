import pandas as pd

from robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from tasks.Task import Task


class ModelChangesRobustnessSetEvaluator:

    def __init__(self, ct: Task, evaluator=ModelChangesRobustnessEvaluator):
        self.task = ct
        self.evaluator = evaluator(ct)

    def evaluate(self, instances, neg_value=0):
        res = []
        for _, instance in instances.iterrows():
            res.append(self.evaluator.evaluate(instance, neg_value=neg_value))

        return pd.DataFrame(res)
