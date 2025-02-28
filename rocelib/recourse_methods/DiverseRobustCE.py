from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric
import pandas as pd
import numpy as np
import torch
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator


class DiverseRobustCE(RecourseGenerator):
    """
    A counterfactual explanation generator that returns a set of diverse counterfactuals for the purpose of robustness
    against input perturbations, i.e. similar inputs will receive similar counterfactuals.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
    """

    def _generation_method(self, instance, column_name="target", neg_value=0, n_ces=5, alpha=0.5, beta=0.25) -> pd.DataFrame:
        """
        Generate diverse counterfactuals.

        Args:
            instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
            column_name: The name of the target column.
            neg_value: The value considered negative in the target variable.
            n_ces: Number of diverse counterfactuals to return.
            alpha: Hyperparameter - controls the distance of candidate counterfactuals.
            beta: Hyperparameter - controls the minimum separation between counterfactuals.

        Returns:
            A DataFrame containing generated counterfactuals.
        """
        # Ensure `instance` is a DataFrame
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T

        num_features = self.task.dataset.X.shape[1]  # Expected feature count
        ces = np.zeros((n_ces, num_features))  # Ensure counterfactuals match dataset shape

        m = self.task.model

        # Convert dataset to tensor and get model predictions
        X_tensor = torch.tensor(self.task.dataset.X.values, dtype=torch.float32)
        model_labels = m.predict(X_tensor)
        model_labels = (model_labels >= 0.5).astype(int)

        y_target = 1 - neg_value

        # Ensure instance matches dataset columns
        if instance.shape[1] != self.task.dataset.X.shape[1]:
            raise ValueError(
                f"Instance shape mismatch: expected {self.task.dataset.X.shape[1]} features, got {instance.shape[1]}."
            )

        # Prepare dataset
        preds = self.task.dataset.X.copy()
        preds["predicted"] = model_labels

        # Filter instances that match desired counterfactual label
        positive_instances = preds[preds["predicted"] == y_target].drop(columns=["predicted"], errors="ignore")

        # Ensure `positive_instances` has the expected number of features
        if positive_instances.shape[1] != num_features:
            raise ValueError(
                f"Feature count mismatch: dataset has {num_features} features, but positive instances have {positive_instances.shape[1]}."
            )

        # If no valid positive instances exist, return the input instance
        if positive_instances.empty:
            return instance

        # Build KDTree using the correct feature set
        kd_tree = KDTree(positive_instances.values)

        # Query the nearest neighbour
        dists, idxs = kd_tree.query(instance.values, k=1, return_distance=True)
        ces[0, :] = positive_instances.iloc[idxs.flatten()[0]].values  # Ensure full feature set

        # Get lowest distance
        lowest_dist = dists.flatten()[0]

        # Query multiple nearest neighbors
        k = min(int(self.task.dataset.X.shape[0] / 2), len(positive_instances))  # Ensure k is within bounds
        dists, idxs = kd_tree.query(instance.values, k=k, return_distance=True)

        # Ensure valid indices
        valid_idxs = np.where(dists.flatten() <= lowest_dist * (1 + alpha))[0]
        idxs = idxs.flatten()[valid_idxs]

        # Greedily add diverse counterfactuals
        idx_to_add = 1
        idx_in_candidates = 1
        dist_calc = DistanceMetric.get_metric('minkowski')

        while idx_to_add < n_ces and idx_in_candidates < len(idxs):
            this_cand = positive_instances.iloc[idxs[idx_in_candidates]].values  # Extract as full row
            this_dist = dist_calc.pairwise(instance.values, this_cand.reshape(1, -1))[0, 0]

            if this_dist >= (1 + beta) * lowest_dist:
                ces[idx_to_add, :] = this_cand
                idx_to_add += 1
            idx_in_candidates += 1

        # Trim unused slots
        ces = ces[:idx_to_add, :]

        # Convert to DataFrame with correct feature names
        return pd.DataFrame(ces, columns=self.task.dataset.X.columns)


    def _binary_linear_search(self, x, ce, y_target, dist_calc, min_dist):
        xp = ce
        while dist_calc.pairwise(x.reshape(1, -1), ce.reshape(1, -1))[0, 0] > 0.1 * min_dist:
            xp = (x + ce) / 2
            if self.task.model.predict_single(pd.DataFrame(xp.reshape(1, -1))) != y_target:
                x = xp
            else:
                ce = xp
        return xp
