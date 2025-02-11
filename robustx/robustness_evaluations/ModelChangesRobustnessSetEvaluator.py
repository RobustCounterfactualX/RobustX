import pandas as pd

from robustx.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from robustx.lib.tasks.Task import Task


class ModelChangesRobustnessSetEvaluator:
    """
    Class for evaluating the robustness of CEs with respect to model changes.

    This class uses a specified evaluator to assess the robustness of model predictions for
    multiple CEs.

    Attributes:
        task (Task): The task for which robustness is being evaluated.
        evaluator (ModelChangesRobustnessEvaluator): An instance of a robustness evaluator used
                                                     to assess each instance.
    """

    def __init__(self, ct: Task, evaluator=ModelChangesRobustnessEvaluator):
        """
        Initializes the ModelChangesRobustnessSetEvaluator with a task and an evaluator.

        @param ct: The task to solve, provided as a Task instance.
        @param evaluator: The evaluator class used to assess robustness of the task's model.
                          Defaults to ModelChangesRobustnessEvaluator.
        """
        self.task = ct
        self.evaluator = evaluator(ct)

    def evaluate(self, instances, neg_value=0):
        """
        Evaluates the robustness of model predictions for a set of instances.

        @param instances: A DataFrame containing the instances to evaluate.
        @param neg_value: The value considered negative in the target variable, used
                          to evaluate the robustness of the model's prediction.
        @return: A DataFrame containing the robustness evaluation results for each instance.
        """
        res = []
        for _, instance in instances.iterrows():
            res.append(self.evaluator.evaluate(instance, neg_value=neg_value))

        return pd.DataFrame(res)
