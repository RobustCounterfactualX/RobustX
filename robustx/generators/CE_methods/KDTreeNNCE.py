import pandas as pd
import numpy as np
import torch
from sklearn.neighbors import KDTree

from robustx.generators.CEGenerator import CEGenerator


class KDTreeNNCE(CEGenerator):
    """
    A counterfactual explanation generator that uses KD-Tree for nearest neighbor counterfactual explanations.

    Inherits from the CEGenerator class and implements the _generation_method to find
    counterfactual explanations using KD-Tree for nearest neighbors.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
        __customFunc (callable, optional): A custom distance function, inherited from CEGenerator.
    """

    def _generation_method(self, instance, gamma=0.1,
                           column_name="target", neg_value=0, **kwargs) -> pd.DataFrame:
        """
        Generates a counterfactual explanation using KD-Tree for nearest neighbor search.

        @param instance: The instance for which to generate a counterfactual.
        @param distance_func: The function used to calculate the distance between points.
        @param custom_distance_func: Optional custom distance function. (Not used in this method)
        @param gamma: The distance threshold for convergence. (Not used in this method)
        @param column_name: The name of the target column. (Not used in this method)
        @param neg_value: The value considered negative in the target variable.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the nearest counterfactual explanation or None if no positive instances.
        """
        model = self.task.model

        preds = model.predict(self.task.training_data.X)
        idxs_1 = np.where(preds.values.flatten()>=0.5)[0]
        idxs_0 = np.array([i for i in range(len(preds.values.flatten())) if i not in idxs_1])

        # Determine the target label
        nnce_y = 1 - neg_value

        # Filter out instances that have the desired counterfactual label
        positive_instances = self.task.training_data.X.values
        if nnce_y:
            positive_instances = positive_instances[idxs_1]
        else:
            positive_instances = positive_instances[idxs_0]

        # If there are no positive instances, return None
        if len(positive_instances) == 0:
            return instance

        # Build KD-Tree
        kd_tree = KDTree(positive_instances)

        # Query the KD-Tree for the nearest neighbour
        dist, idx = kd_tree.query(instance.values.astype(float).reshape(1, -1), k=1, return_distance=True)
        nearest_instance = positive_instances[idx[0]]

        return pd.DataFrame(nearest_instance, columns=self.task.training_data.X.columns)
