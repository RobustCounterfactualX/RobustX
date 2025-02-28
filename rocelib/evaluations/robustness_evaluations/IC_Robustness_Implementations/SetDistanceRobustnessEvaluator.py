from rocelib.evaluations.robustness_evaluations.InputChangesRobustnessEvaluator import InputChangesRobustnessEvaluator
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from sklearn.metrics import DistanceMetric
import numpy as np
import pandas as pd


class SetDistanceRobustnessEvaluator(InputChangesRobustnessEvaluator):
    """
    Compare the set distance between two sets of counterfactuals
    """

    def evaluate_single_instance(self, instance, counterfactual, recourse_method):
        """
        Compare the counterfactuals for the original instance and those for the perturbed instance.

        @param instance: An input instance.
        @param counterfactual: One or more CE points for the instance.
        @param recourse_method: The method used for counterfactual generation.
        @return: Set distance metric.
        """
        # Ensure generator is an instance
        generator: RecourseGenerator = self.task.methods[recourse_method](self.task)

        # Ensure counterfactual is a DataFrame, then convert to NumPy
        if isinstance(counterfactual, pd.Series):
            counterfactual = counterfactual.to_frame().T

        counterfactual = counterfactual.values  # Convert to NumPy array

        # Perturb the input instance
        perturbed = self.perturb_input(instance)

        # Generate counterfactuals for perturbed input
        ce_perturbed = generator.generate_for_instance(instance=perturbed)

        # Ensure ce_perturbed is a DataFrame, then convert to NumPy
        if isinstance(ce_perturbed, pd.Series):
            ce_perturbed = ce_perturbed.to_frame().T

        if ce_perturbed is not None and not ce_perturbed.empty:
            ce_perturbed = ce_perturbed.values  # Convert to NumPy array
        else:
            return np.inf  # Assign large value to indicate failure

        # Compute Euclidean set distance
        dist = DistanceMetric.get_metric('euclidean')
        dists = dist.pairwise(counterfactual, ce_perturbed)

        # Compute set distance
        return (np.sum(np.min(dists, axis=1)) / (2 * len(counterfactual))) + \
                (np.sum(np.min(dists, axis=0)) / (2 * len(ce_perturbed)))

