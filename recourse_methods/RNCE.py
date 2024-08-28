import numpy as np
import pandas as pd

from intabs.IntervalAbstractionPyTorch import IntervalAbstractionPytorch
from recourse_methods.RecourseGenerator import RecourseGenerator
from robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from tasks.Task import Task
from sklearn.neighbors import KDTree


class RNCE(RecourseGenerator):

    def __init__(self, task: Task, delta=0.005, bias_delta=0):
        super().__init__(task)
        self.delta = delta
        self.bias_delta = bias_delta

        self.intabs = DeltaRobustnessEvaluator(task)

    def _generation_method(self, x, robustInit=True, optimal=True, column_name="target", neg_value=0):
        S = self.getCandidates(robustInit, column_name=column_name)
        treer = KDTree(S, leaf_size=40)
        x_df = pd.DataFrame(x).T
        idxs = np.array(treer.query(x_df)[1]).flatten()
        return S.iloc[idxs[0]]

    def getCandidates(self, robustInit, column_name="target"):

        S = []

        for _, instance in self.task.training_data.data.iterrows():
            instance_x = instance.drop(column_name)
            if robustInit:
                if self.intabs.evaluate(instance_x, delta=self.delta):
                    S.append(instance_x)

            else:
                if self.task.model.predict_single(instance_x):
                    S.append(instance_x)

        return pd.DataFrame(S)

    # def getRobustCE(self, x, kdtree: KDTree, optimal):
    #
    #     a = 1
    #
    #     s = 0.05
    #
    #     neighbors_distances, neighbors_indices = kdtree.query([x], k=kdtree.n_samples_)
    #     neighbors_indices = neighbors_indices[0]
    #
    #     x_prime = None
    #
    #     # Iterate through neighbors until no more neighbors are left
    #     for nn_index in neighbors_indices:
    #         x_nn_prime = self.task.training_data.data.iloc[nn_index]  # Get the next nearest neighbor
    #         # Do something with x_nn_prime
    #         if self.intabs.evaluate(x_nn_prime):
    #             x_prime = x_nn_prime
    #
    #     if optimal:
    #         while a > 0:
    #             x_line = a * x_prime + (1-a) * x
    #             a -= s
    #
    #             if self.intabs.evaluate(x_line):
    #                 x_prime = x_line
    #
    #     return x_prime
