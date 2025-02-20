import numpy as np

from rocelib.evaluations.robustness_evaluations.NoisyExecutionRobustnessEvaluator import \
    NoisyExecutionRobustnessEvaluator
from rocelib.tasks.Task import Task

import random


class InvalidationRateRobustnessEvaluator(NoisyExecutionRobustnessEvaluator):
    """
     An Evaluator class which evaluates ...

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the validity of CEs

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are valid

    -------
    """

    def __init__(self, ct: Task):
        """
        Initializes the ... with a given task.

        @param ct: The task to solve, provided as a Task instance.
        """
        super().__init__(ct)
        self.dataset_mins = self.task.dataset.X.min().to_frame().transpose().values
        self.dataset_maxs = self.task.dataset.X.max().to_frame().transpose().values

    def evaluate_single_instance(self, index, recourse_method, **kwargs):
        """
        Evaluates whether the model's prediction for a given instance is robust to ...

        @param index: The index of the instance to evaluate.
        @param recourse_method: The particular recourse method used for evaluation (not needed in this implementation)
        @return: A boolean indicating whether the model's prediction is robust
        """

        # use this to generate unique noise for every value in a df with more than one CEs
        # random_values = np.random.normal(loc=0, scale=5, size=df.shape)
        # df_new = df + random_values

        # TODO more tests

        # instance is a single CE
        instance = self.task.dataset.data.iloc[index]
        instance = instance.drop(labels=["target"])

        feature_count = len(instance)

        mean = np.zeros(feature_count)
        stddev = 0.1
        cov_matrix = (stddev**2) * np.identity(feature_count)
        noise = np.random.multivariate_normal(mean, cov_matrix, size=1)  # size 1 as only 1 CE

        pred = self.task.model.predict_single(instance)
        denormalised_noise = noise * (self.dataset_maxs - self.dataset_mins)
        pred_noisy = self.task.model.predict_single(instance + denormalised_noise.flatten())

        return pred == pred_noisy
