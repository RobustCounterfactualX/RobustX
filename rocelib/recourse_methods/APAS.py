import numpy as np
import pandas as pd
import torch
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.ApproximateDeltaRobustnessEvaluator import \
    ApproximateDeltaRobustnessEvaluator
from rocelib.tasks.Task import Task


class APAS(RecourseGenerator):
    """
    A counterfactual explanation generator that uses any CEGenerator class and a ApproximateDeltaRobustnessEvaluator evaluator
    to find counterfactual explanations that are approximately robust against model changes.

    Inherits from the CEGenerator class and implements the _generation_method to generate counterfactual examples
    with approximate robustness checks using a specified confidence alpha. The method iterates over positive instances
    and evaluates their robustness, returning those with stable counterfactuals.

    This is a similar implementation of Marzari et. al "Rigorous Probabilistic Guarantees for Robust Counterfactual Explanations", ECAI 2024

    Attributes:
        CE_generator specific to this class, but utilizes the task and model from the RecourseCE base class.
        alpha = confidence level in the robustness evaluator
    """

    def __init__(self, task: Task, ce_generator=KDTreeNNCE, alpha=0.999):
        """
        Initializes the APAS CE generator with a given task and a CE generator.

        @param task: The task to generate counterfactual explanations for.
        """

        super().__init__(task)
        self.rg = ce_generator(task)
        self.alpha = alpha

    def _generation_method(self,
                           original_input,
                           target_column_name="target",
                           desired_outcome=0,
                           delta_max=0.5,
                           maximum_iterations=1000, verbose=False,
                           **kwargs):

        """
        Generates the first counterfactual explanation for a given input using the APΔS method, i.e., a combination of exponential and binary search with a probabilistic delta robustness model changes check.

        @param target_column_name: The name of the target column.
        @param desired_outcome: The value considered for the generation of the counterfactual in the target_column_name.
        @param delta_max: Maximum perturbation allowed in the model for the robustness_check.
        @param maximum_iterations: The maximum number of iterations to run the APΔS method.

        @return: the first robust counterfactual explanation to Δ-model changes.
        """

        iterations = 0
        robustness_check = ApproximateDeltaRobustnessEvaluator(self.task, self.alpha)

        for i in range(maximum_iterations):
            if verbose: print(f"Iteration {i}/{maximum_iterations}")
            ce = self.rg._generation_method(instance=original_input)

            # check if column names contains ['predicted', 'Loss'] columns
            if 'predicted' in ce.columns and 'Loss' in ce.columns:
                ce = ce.drop(columns=['predicted', 'Loss']).astype(np.float32)

            ce = torch.tensor(ce.values[0], dtype=torch.float32)

            robustness = robustness_check.evaluate(ce, desired_outcome=desired_outcome, delta=delta_max)
            if robustness:
                return ce

            iterations += 1

        print("No robust counterfactual explanation found for the given perturbation.")
        return pd.DataFrame(original_input).T
