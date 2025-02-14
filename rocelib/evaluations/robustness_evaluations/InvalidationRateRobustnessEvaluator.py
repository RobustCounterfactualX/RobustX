import pandas as pd

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
        self.dataset_mins = self.task.dataset.min()
        self.dataset_maxs = self.task.dataset.max()

    def evaluate(self, instance, **kwargs):  # , desired_output=1, delta=0.5, bias_delta=0, M=1000000000, epsilon=0.0001):
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
        # instance is a single CE

        # use this to generate unique noise for every value in a df with more than one CEs
        # random_values = np.random.normal(loc=0, scale=5, size=df.shape)
        # df_new = df + random_values

        mean = 0
        stddev = 0.1
        I = 1  # references in paper but i have no idea what it is
        noise = pd.DataFrame([random.gauss(mean, stddev * I) for _ in range(len(instance))])
        pred = self.task.model().predict(instance)
        denormalised_noise = noise * (self.dataset_maxs - self.dataset_mins) + self.dataset_mins
        # denormalised_noise = [n * (maxi - mini) + mini for n, mini, maxi in zip(noise, self.dataset_mins, self.dataset_maxs)]
        pred_noisy = self.task.model().predict(instance + denormalised_noise)

        return pred == pred_noisy
        # cf2 = method.generate_for_instance(instance + noise, neg_value=0, column_name="target")

        # calculate expectation of M(x') - M(x' + s)
        # where M is the model which generates an output label
        # x' is a predicted counterfactual
        # sigma is a sampled noise vector



