import pandas as pd
import numpy as np

from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator
from rocelib.tasks.Task import Task

import random


class InvalidationRateRobustnessEvaluator(RecourseEvaluator):
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

    def evaluate(self, instance, **kwargs):
        """
        Evaluates whether the model's prediction for a given instance is robust to ...

        @param instance: The instance to evaluate.
        @param desired_output: The desired output for the model (0 or 1).
                               The evaluation will check if the model's output matches this.
        @param delta: The maximum allowable perturbation in the input features.
        @param bias_delta: Additional bias to apply to the delta changes.
        @param M: A large constant used in MILP formulation for modeling constraints.
        @param epsilon: A small constant used to ensure numerical stability.
        @return: A boolean indicating whether the model's prediction is robust given the desired output.
        """

        # IMPORTANT: used dataset should be normalised so each feature is in range [0, 1]
        # instance is a single CE

        # use this to generate unique noise for every value in a df with more than one CEs
        # random_values = np.random.normal(loc=0, scale=5, size=df.shape)
        # df_new = df + random_values

        # TODO more tests
        # TODO merge, then change to be like new DeltaRE (probably evaluate more than 1 CE at a time, return %)

        instance = instance.drop(columns=["predicted", "Loss"], errors='ignore')

        mean = np.zeros(len(instance.columns))
        stddev = 0.1
        cov_matrix = (stddev**2) * np.identity(len(instance.columns))

        noise = np.random.multivariate_normal(mean, cov_matrix, size=len(instance))
        pred = self.task.model.predict_single(instance)

        denormalised_noise = noise * (self.dataset_maxs - self.dataset_mins)
        pred_noisy = self.task.model.predict_single(instance + denormalised_noise)

        return pred == pred_noisy

        # calculate expectation of M(x') - M(x' + s)
        # where M is the model which generates an output label
        # x' is a predicted counterfactual
        # sigma is a sampled noise vector



