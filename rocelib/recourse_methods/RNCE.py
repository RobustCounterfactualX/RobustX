import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.tasks.Task import Task
from functools import lru_cache


class RNCE(RecourseGenerator):
    """
    A recourse generator that finds robust nearest counterfactual examples using KDTree.

    Inherits from the RecourseGenerator class and implements the _generation_method to find counterfactual examples 
    that are robust to perturbations. It leverages KDTree for nearest neighbor search and uses a robustness evaluator 
    to identify robust instances in the training data.

    Attributes:
        intabs (DeltaRobustnessEvaluator): An evaluator for checking the robustness of instances to perturbations.
    """

    def __init__(self, task: Task):
        """
        Initializes the RNCE recourse generator with a given task and robustness evaluator.

        @param task: The task to solve, provided as a Task instance.
        """
        super().__init__(task)
        self.intabs = DeltaRobustnessEvaluator(task)

    def _generation_method(self, x, robustInit=True, optimal=True, column_name="target", neg_value=0, delta=0.005,
                           bias_delta=0.005, k=1, **kwargs):
        """
        Generates a counterfactual explanation using nearest neighbor search.

        @param x: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param robustInit: If True, only robust instances are considered for counterfactual generation.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @param delta: The tolerance for robustness in the feature space.
        @param bias_delta: The bias tolerance for robustness in the feature space.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the counterfactual explanation.
        """
        S = self.getCandidates(robustInit, delta, bias_delta, column_name=column_name, neg_value=neg_value)
        if S.empty:
            print("No instance in the dataset is robust for the given perturbations!")
            return pd.DataFrame(x).T

        treer = KDTree(S, leaf_size=40)
        x_df = pd.DataFrame(x).T
        idxs = np.array(treer.query(x_df)[1]).flatten()
        res = pd.DataFrame(S.iloc[idxs[0]]).T
        return res

    @lru_cache()
    def getCandidates(self, robustInit, delta, bias_delta, column_name="target", neg_value=0):
        """
        Retrieves candidate instances from the dataset that are robust to perturbations.

        @param robustInit: If True, only robust instances are considered.
        @param delta: The tolerance for robustness in the feature space.
        @param bias_delta: The bias tolerance for robustness in the feature space.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing robust instances from the dataset.
        """
        S = []

        for _, instance in self.task.dataset.data.iterrows():
            instance_x = instance.drop(column_name)
            if robustInit:
                if self.intabs.evaluate_single_instance(instance_x, delta=delta, bias_delta=bias_delta):
                    S.append(instance_x)
            else:
                if self.task.model.predict_single(instance_x):
                    S.append(instance_x)

        return pd.DataFrame(S)
