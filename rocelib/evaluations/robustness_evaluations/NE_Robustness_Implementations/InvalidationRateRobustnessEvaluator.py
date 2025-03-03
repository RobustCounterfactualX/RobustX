import numpy as np

from rocelib.evaluations.robustness_evaluations.NoisyExecutionRobustnessEvaluator import \
    NoisyExecutionRobustnessEvaluator
from rocelib.tasks.Task import Task

import random


class InvalidationRateRobustnessEvaluator(NoisyExecutionRobustnessEvaluator):
    """
     An Evaluator class which evaluates robustness using the IR method described in [Pawelczyk et al., 2023a]

    Attributes:
        task (Task): The task to solve, inherited from ModelChangesRobustnessEvaluator.
    """

    def __init__(self, ct: Task):
        """
        Initializes the ... with a given task.

        @param ct: The task to solve, provided as a Task instance.
        """
        super().__init__(ct)
        self.dataset_mins = self.task.dataset.X.min().to_frame().transpose().values
        self.dataset_maxs = self.task.dataset.X.max().to_frame().transpose().values

    def evaluate_single_instance(self, instance, counterfactual, **kwargs):
        """
        Evaluates whether the model's prediction for a given instance is robust to ...

        @param instance: a single input that can be evaluated by the model
        @param counterfactual: a counterfactual explanation - possibly of instance
        @return: A boolean indicating whether the model's prediction is robust
        """

        # use this to generate unique noise for every value in a df with more than one CEs
        # random_values = np.random.normal(loc=0, scale=5, size=df.shape)
        # df_new = df + random_values

        cols_to_drop = [col for col in ["predicted", "Loss", "loss", "HiringDecision"] if col in counterfactual.index.to_list()]
        counterfactual = counterfactual.drop(labels=cols_to_drop, errors='ignore')

        feature_count = len(counterfactual)

        mean = np.zeros(feature_count)
        stddev = 0.1
        cov_matrix = (stddev**2) * np.identity(feature_count)
        noise = np.random.multivariate_normal(mean, cov_matrix, size=1)  # size 1 as only 1 CE

        pred = self.task.model.predict_single(counterfactual)
        denormalised_noise = noise * (self.dataset_maxs - self.dataset_mins)
        pred_noisy = self.task.model.predict_single(counterfactual + denormalised_noise.flatten())

        return pred == pred_noisy
