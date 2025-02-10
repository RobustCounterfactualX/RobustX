from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric
import pandas as pd
import numpy as np
import torch
from robustx.generators.CEGenerator import CEGenerator


class DiverseRobustCE(CEGenerator):
    """
    A counterfactual explanation generator that returns a set of diverse counterfactuals for the purpose of robustness
    against input perturbations, i.e. similar inputs will receive similar counterfactuals.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
    """

    def _generation_method(self, instance, column_name="target", neg_value=0, n_ces=5, alpha=0.5,
                           beta=0.25) -> pd.DataFrame:
        """
        Generate diverse CEs. Can return one or more counterfactuals in a pandas dataframe
        Args:
            instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
            column_name: The name of the target column.
            neg_value: The value considered negative in the target variable.
            n_ces: Number of diverse counterfactuals to return
            alpha: Hyperparameter, the candidate counterfactuals can be (1+alpha) times more distant\
                    to the input than the closest point in the nearest neighbour.
            beta: Hyperparameter, the distance between each selected CE should be at least (1+beta) times the minimum\
                    distance between the closest point in the nearest neighbour and the input

        Returns: CE

        """
        ces = np.zeros((n_ces, len(instance.values)))

        m = self.task.model

        # reuse kdtree nnce
        X_tensor = torch.tensor(self.task.training_data.X.values, dtype=torch.float32)

        # Get all model predictions of model, turning them to 0s or 1s
        # Get all model predictions of model, turning them to 0s or 1s
        model_labels = m.predict(X_tensor)
        model_labels = (model_labels >= 0.5).astype(int)

        y_target = 1 - neg_value
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T

        # Prepare the data
        preds = self.task.training_data.X.copy()
        preds["predicted"] = model_labels

        # Filter out instances that have the desired counterfactual label
        positive_instances = preds[preds["predicted"] == y_target].drop(columns=["predicted"])

        # If there are no positive instances, return None
        if positive_instances.empty:
            return instance

        # Build KD-Tree
        kd_tree = KDTree(positive_instances.values)

        # Query the KD-Tree for the nearest neighbour
        dists, idxs = kd_tree.query(instance.values, k=1, return_distance=True)
        ces[0] = positive_instances.values[idxs.flatten()[0]]

        # get the lowest distance
        lowest_dist = dists.flatten()[0]

        # Query the KD-Tree again
        k = int(self.task.training_data.X.shape[0] / 2)
        dists, idxs = kd_tree.query(instance.values, k=k, return_distance=True)
        idxs = idxs.flatten()[np.where(dists <= lowest_dist * (1 + alpha))[1]]

        # greedily add CEs
        idx_to_add = 1
        idx_in_candidates = 1
        dist_calc = DistanceMetric.get_metric('minkowski')  # same as the one used in kd tree
        while idx_to_add < n_ces and idx_in_candidates < len(idxs):
            this_cand = positive_instances.values[idxs[idx_in_candidates]]
            this_dist = dist_calc.pairwise(instance.values.reshape(1, -1), this_cand.reshape(1, -1))[0, 0]
            if this_dist >= (1 + beta) * lowest_dist:
                ces[idx_to_add] = this_cand
                idx_to_add += 1
            idx_in_candidates += 1

        # filter out placeholder CEs if any
        if idx_to_add < n_ces:
            ces = ces[:idx_to_add]

        # # run binary linear-search to further reduce distance
        # for i, ce in enumerate(ces):
        #     ces[i] = self._binary_linear_search(instance.values, ce, y_target, dist_calc, lowest_dist)
        return pd.DataFrame(ces)

    def _binary_linear_search(self, x, ce, y_target, dist_calc, min_dist):
        xp = ce
        while dist_calc.pairwise(x.reshape(1, -1), ce.reshape(1, -1))[0, 0] > 0.1 * min_dist:
            xp = (x + ce) / 2
            if self.task.model.predict_single(pd.DataFrame(xp.reshape(1, -1))) != y_target:
                x = xp
            else:
                ce = xp
        return xp
