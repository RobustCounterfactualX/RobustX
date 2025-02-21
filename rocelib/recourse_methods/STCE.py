import numpy as np
import pandas as pd
import torch

from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator


class TrexNN(RecourseGenerator):
    """
    A recourse generator that uses the T-Rex method for finding robust counterfactual explanations.

    Inherits from the RecourseGenerator class and implements the _generation_method to find counterfactual examples
    with robustness checks using a specified base method and evaluator. The method iterates over positive instances
    and evaluates their robustness, returning those with stable counterfactuals.

    Attributes:
        None specific to this class, but utilizes the task and model from the RecourseGenerator base class.
    """

    def _generation_method(self, instance,
                           robustness_check: ModelChangesRobustnessEvaluator.__class__ = DeltaRobustnessEvaluator,
                           column_name="target",
                           neg_value=0, K=40,
                           threshold=0.4, **kwargs):
        """
        Generates a counterfactual explanation using the T-Rex method.

        @param instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param robustness_check: The robustness evaluator to check model changes with respect to input perturbations.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @param K: The number of samples for stability evaluation.
        @param threshold: The threshold for counterfactual stability.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the counterfactual explanation if found, otherwise the original instance.
        """
        positives = self.task.dataset.data[self.task.dataset.data[column_name]
                                                 == neg_value].drop(columns=[column_name])

        # Compute Euclidean distances between the instance and each positive sample
        instance_values = instance.values.flatten()  # Drop target column from instance
        positives['distance'] = positives.apply(lambda x: np.linalg.norm(x.values - instance_values), axis=1)

        # Sort positives by distance
        positives = positives.sort_values(by='distance')

        positives = positives.drop(columns=["distance"])

        evaluator = robustness_check(self.task)

        for _, positive in positives.iterrows():

            if evaluator.evaluate_single_instance(positive, **kwargs):

                val = self.counterfactual_stability(positive)
                if val > threshold:
                    return pd.DataFrame(positive).T

        return pd.DataFrame(instance).T

    def counterfactual_stability(self, xp):
        """
        Evaluates the stability of a given counterfactual instance.

        @param xp: The instance for which to evaluate counterfactual stability.
        @return: A tensor representing the stability score of the counterfactual.
        """
        k = 1000

        # Predict probability for the given instance
        score_x = self.task.model.predict_proba(xp)

        # Prepare a DataFrame with the predicted score
        score_x = pd.DataFrame([score_x[1]] * k)
        score_x.reset_index(drop=True, inplace=True)

        # Generate Gaussian samples based on the input instance
        gaussian_samples = np.random.normal(xp.to_numpy(), 0.1, (k, len(xp.T)))

        # Get model scores for the Gaussian samples
        model_scores = pd.DataFrame(self.task.model.predict_proba(gaussian_samples)[1])
        model_scores.columns = range(model_scores.shape[1])

        # Calculate the stability score using tensor operations
        res = torch.tensor(np.sum((model_scores - (model_scores - score_x).abs()) / len(model_scores)),
                           requires_grad=True)
        return res
