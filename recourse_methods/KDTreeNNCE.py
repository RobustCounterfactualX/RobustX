import torch
import pandas as pd
from sklearn.neighbors import KDTree

from recourse_methods.RecourseGenerator import RecourseGenerator


class KDTreeNNCE(RecourseGenerator):

    def generate_for_instance(self, instance, distance_func="euclidean", custom_func=None, gamma=0.1,
                              column_name="target", neg_value=0):

        model = self.ct.model

        # Convert X values of dataset to tensor
        X_tensor = torch.tensor(self.ct.training_data.X.values, dtype=torch.float32)

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
        preds = self.ct.training_data.X.copy()
        preds["predicted"] = model_labels

        # Filter out instances that have the desired counterfactual label
        positive_instances = preds[preds["predicted"] == nnce_y].drop(columns=["predicted"])

        # If there are no positive instances, return None
        if positive_instances.empty:
            return None

        # Build KD-Tree
        kd_tree = KDTree(positive_instances.values, metric=distance_func)

        # Query the KD-Tree for the nearest neighbour
        dist, idx = kd_tree.query(instance.values, k=1, return_distance=True)
        nearest_instance = positive_instances.iloc[idx[0]]

        nearest_instance["predicted"] = nnce_y

        # Add the distance as a new column
        nearest_instance["Loss"] = dist[0]

        return nearest_instance
