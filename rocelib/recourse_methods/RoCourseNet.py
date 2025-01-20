from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.lib.distance_functions.DistanceFunctions import euclidean
from rocelib.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
import pandas as pd
import numpy as np
import torch

from rocelib.tasks.Task import Task


class RoCourseNet(RecourseGenerator):
    """
    A recourse generator using the RoCourseNet methodology, integrated with the SimpleNNModel.
    """

    def __init__(self, task: Task):
        """
        Initializes RoCourseNet with a given task and robustness evaluator.

        @param task: The task to solve, provided as a Task instance.
        """
        super().__init__(task)
        self.intabs = DeltaRobustnessEvaluator(task)
        self.model= task.model  # Ensure the model is an instance of SimpleNNModel

    def _generation_method(self, instance, gamma=0.1, column_name="target", neg_value=0,
                           distance_func=euclidean, max_iter=50, lr=0.1, delta=0.01, **kwargs) -> pd.DataFrame:
        """
        Generates a robust counterfactual explanation for a given instance using the RoCourseNet approach.

        @param instance: The instance for which to generate a counterfactual (Series or DataFrame row).
        @param gamma: The regularization strength for proximity.
        @param column_name: The name of the target column.
        @param neg_value: The negative class value in the target variable.
        @param distance_func: Function to calculate distances (default: Euclidean).
        @param max_iter: Maximum number of optimization iterations.
        @param lr: Learning rate for gradient updates.
        @param delta: Magnitude of adversarial perturbation.
        @param kwargs: Additional parameters.
        @return: A DataFrame containing the generated counterfactual explanation.
        """

        def adversarial_loss(cf, perturbation):
            """
            Attacker problem: Simulates the worst-case data shift.
            @param cf: Current counterfactual instance.
            @param perturbation: Adversarial perturbation applied to the model.
            @return: Loss value representing the impact of the attack.
            """
            shifted_weights = {k: v + perturbation.get(k, 0) for k, v in self.model.get_torch_model().state_dict().items()}
            self.model.get_torch_model().load_state_dict(shifted_weights, strict=False)
            pred_cf = self.model.predict_single(cf)
            return (pred_cf - (1 - neg_value)) ** 2

        def defender_loss(cf, original, perturbation):
            """
            Defender problem: Balances proximity and robustness.
            @param cf: Current counterfactual instance.
            @param original: Original input instance.
            @param perturbation: Adversarial perturbation applied to the model.
            @return: Loss value balancing robustness and proximity.
            """
            proximity_loss = distance_func(original.values, cf.values)
            robustness_loss = adversarial_loss(cf, perturbation)
            return robustness_loss + gamma * proximity_loss

        # Initialize CF and perturbation
        cf = instance.copy()
        perturbation = {k: torch.zeros_like(v) for k, v in self.model.get_torch_model().state_dict().items()}

        for _ in range(max_iter):
            # Step 1: Solve the adversarial problem (Attacker)
            for k, v in perturbation.items():
                v.requires_grad = True
            loss_adv = adversarial_loss(cf, perturbation)
            loss_adv.backward()
            with torch.no_grad():
                for k, v in perturbation.items():
                    v -= lr * v.grad  # Update perturbation
                    v.clamp_(-delta, delta)  # Clamp values within [-delta, delta]
                    v.grad = None  # Reset gradients

            # Step 2: Solve the robust CF optimization (Defender)
            cf_tensor = torch.tensor(cf.values, dtype=torch.float32, requires_grad=True)
            loss_def = defender_loss(cf_tensor, instance, perturbation)
            loss_def.backward()
            with torch.no_grad():
                cf_tensor -= lr * cf_tensor.grad  # Update counterfactual
                cf_tensor.grad = None  # Reset gradients

        # Convert CF back to DataFrame
        cf = pd.DataFrame(cf_tensor.detach().numpy(), index=instance.index).T

        # Validate CF
        pred_cf = self.model.predict_single(cf)
        if pred_cf <= neg_value:
            print("Failed to generate valid counterfactual.")
            return pd.DataFrame(instance).T

        return cf
