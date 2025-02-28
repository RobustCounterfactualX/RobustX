import pandas as pd
from rocelib.evaluations.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from rocelib.tasks.Task import Task


class VaRRobustnessEvaluator(ModelChangesRobustnessEvaluator):
    """
    A robustness evaluation method for assessing the validity of counterfactual explanations (CEs)
    after retraining models. It ensures robustness against model changes.

    Attributes:
        task (Task): The task to solve, inherited from ModelChangesRobustnessEvaluator.
        models (List[BaseModel]): The list of models retrained on the same dataset.
    """

    def __init__(self, task: Task, models):
        """
        Initializes the VaRRobustnessEvaluator with a given task and a set of trained models.

        @param task: The task for which robustness evaluations are being made.
                     Provided as a Task instance.
        @param models: The list of models retrained on the same dataset.
        """
        super().__init__(task)
        self.models = models

    def evaluate_single_instance(self, index, recourse_method, desired_output=1):
        """
        Evaluates whether the instance and its counterfactual explanation (CE) are robust
        under model changes.

        @param index: The index of the instance in the dataset.
        @param recourse_method: The recourse method used for generating counterfactuals.
        @param desired_output: The desired classification outcome.
        @return: A boolean indicating whether the instance and CE remain robust.
        """
        # Retrieve the instance and counterfactual explanation (CE)
        instance = self.task.dataset.data.iloc[index]
        ce = self.task.CEs[recourse_method][0].iloc[index]

        # Convert to DataFrame for model prediction
        instance_df = pd.DataFrame(instance.values.reshape(1, -1))
        ce_df = pd.DataFrame(ce.values.reshape(1, -1))

        # Initial predictions from the original model
        pred_on_orig_model = self.task.model.predict_single(instance_df)
        pred_on_ce = self.task.model.predict_single(ce_df)

        # Ensure the original instance is classified correctly
        if pred_on_orig_model != desired_output:
            return False

        # Ensure the CE is also classified as the desired output
        if pred_on_ce != desired_output:
            return False

        # Check predictions on retrained models
        for model in self.models:
            if model.predict_single(instance_df) != desired_output or model.predict_single(ce_df) != desired_output:
                return False

        return True