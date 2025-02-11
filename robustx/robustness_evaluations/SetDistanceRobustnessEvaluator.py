from robustx.robustness_evaluations.InputChangesRobustnessEvaluator import InputChangesRobustnessEvaluator
from robustx.generators.CEGenerator import CEGenerator
from sklearn.metrics import DistanceMetric
import numpy as np


class SetDistanceRobustnessEvaluator(InputChangesRobustnessEvaluator):
    """
    Compare the set distance between two sets of counterfactuals
    """

    def evaluate(self, instance, counterfactual, generator: CEGenerator):
        """
        Compare the counterfactuals for the original instance and those for the perturbed instance.

        @param instance: An input instance.
        @param counterfactual: One or more CE points for the instance.
        @param generator: CE generator.
        """

        perturbed = self.perturb_input(instance)
        ce_perturbed = generator.generate_for_instance(perturbed)

        dist = DistanceMetric.get_metric('euclidean')
        dists = dist.pairwise(counterfactual, ce_perturbed)

        # compute set distance
        return np.sum(np.min(dists, axis=1)) / (2 * len(counterfactual)) + np.sum(np.min(dists, axis=0)) / (
                    2 * len(ce_perturbed))
