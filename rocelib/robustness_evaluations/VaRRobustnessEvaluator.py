import pandas as pd
from rocelib.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from rocelib.lib.tasks.Task import Task


class VaRRobustnessEvaluator(ModelChangesRobustnessEvaluator):
    """
    A simple and common robustness evaluation method for evaluating validity of the CE after retraining.
    Used for robustness against model changes.

    Attributes:
        task (Task): The task to solve, inherited from ModelChangesRobustnessEvaluator.
        models (List[BaseModel]): The list of models retrained on the same dataset.
    """

    def __init__(self, ct: Task, models):
        """
        Initializes the VaRRobustnessEvaluator with a given task and trained models.

        @param ct: The task for which robustness evaluations are being made.
                   Provided as a Task instance.
        @param models: The list of models retrained on the same dataset.
        """
        super().__init__(ct)
        self.models = models

    def evaluate(self, instance, desired_outcome=1):
        """
        Evaluates whether the instance (the ce) is predicted with the desired outcome by all retrained models.
        The instance is robust if this is true.

        @param instance: The instance (in most cases a ce) to evaluate.
        @param desired_outcome: The value considered positive in the target variable.
        @return: A boolean indicating robust or not.
        """
        instance = pd.DataFrame(instance.values.reshape(1, -1))
        pred_on_orig_model = self.task.model.predict_single(instance)

        # ensure basic validity
        if pred_on_orig_model != desired_outcome:
            return False
        # check predictions by new models
        for m in self.models:
            if m.predict_single(instance) != desired_outcome:
                return False
        return True
