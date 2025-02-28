import pandas as pd
import torch
from sklearn.neighbors import KDTree

from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator


class KDTreeNNCE(RecourseGenerator):
    """
    A recourse generator that uses KD-Tree for nearest neighbor counterfactual explanations.

    Inherits from the RecourseGenerator class and implements the _generation_method to find
    counterfactual explanations using KD-Tree for nearest neighbors.

    Attributes:
        _task (Task): The task to solve, inherited from RecourseGenerator.
        __customFunc (callable, optional): A custom distance function, inherited from RecourseGenerator.
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

        # Convert X values of dataset to tensor
        X_tensor = torch.tensor(self.task.dataset.X.values, dtype=torch.float32)

        # Get all model predictions of model, turning them to 0s or 1s
        model_labels = model.predict(X_tensor)
        model_labels = (model_labels >= 0.5).astype(int)

        # Determine the target label
        y = neg_value
        nnce_y = 1 - y

        # Convert instance to DataFrame if it is a Series
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T

        # Prepare the data
        preds = self.task.dataset.X.copy()
        preds["predicted"] = model_labels

        # Filter out instances that have the desired counterfactual label
        positive_instances = preds[preds["predicted"] == nnce_y].drop(columns=["predicted"])

        # If there are no positive instances, return None
        if positive_instances.empty:
            return instance

        # Build KD-Tree
        kd_tree = KDTree(positive_instances.values)

        # Query the KD-Tree for the nearest neighbour
        dist, idx = kd_tree.query(instance.values, k=1, return_distance=True)
        nearest_instance = positive_instances.iloc[idx[0]]

        nearest_instance.loc[:, "predicted"] = nnce_y

        # Add the distance as a new column
        nearest_instance.loc[:, "Loss"] = dist[0]

        return nearest_instance
