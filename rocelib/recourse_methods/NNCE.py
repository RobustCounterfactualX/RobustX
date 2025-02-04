import numpy as np
import pandas as pd
import torch

from rocelib.lib.distance_functions.DistanceFunctions import euclidean
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator


class NNCE(RecourseGenerator):
    """
    A recourse generator that uses a nearest-neighbor counterfactual explanation (NNCE) approach.

    Inherits from RecourseGenerator and calculates counterfactual explanations based on the nearest neighbor
    in the training data with the desired prediction.

    Attributes:
        _task (Task): The task to solve, inherited from RecourseGenerator.
        __customFunc (callable, optional): A custom distance function, inherited from RecourseGenerator.
    """

    def _generation_method(self, instance, gamma=0.1, column_name="target", neg_value=0,
                           distance_func=euclidean, **kwargs) -> pd.DataFrame:
        """
        Generates a nearest-neighbor counterfactual explanation for a provided instance.

        @param instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param gamma: The threshold for the distance between the instance and the counterfactual. (Not used in this method)
        @param column_name: The name of the target column. (Not used in this method)
        @param neg_value: The value considered negative in the target variable.
        @param distance_func: The function used to calculate the distance between two points. Defaults to euclidean.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the nearest-neighbor counterfactual explanation for the provided instance.
        """
        model = self.task.model

        # Convert X values of dataset to tensor
        X_tensor = torch.tensor(self.task.dataset.X.values, dtype=torch.float32)

        # Get all model predictions of model, turning them to 0s or 1s
        model_labels = model.predict(X_tensor)
        model_labels = (model_labels >= 0.5)

        # Determine the target label
        y = neg_value
        nnce_y = 1 - y

        # Set initial CE and minimum distance of CE
        nnce = None
        nnce_dist = np.inf

        if isinstance(instance, pd.Series):
            negative_df = instance.to_frame()
        else:
            negative_df = instance

        preds = self.task.dataset.X
        preds["predicted"] = model_labels

        # Iterate through each model prediction
        for _, pred in preds.iterrows():

            # Skip if the current instance is not the desired outcome
            if pred["predicted"] != nnce_y:
                continue

            # Calculate distance between negative instance and current sample
            sample_dist = distance_func(negative_df, pd.DataFrame(pred.drop(["predicted"])))

            # If distance is less than any other encountered yet, we have found a new NNCE
            if sample_dist < nnce_dist:
                nnce = pred
                nnce_dist = sample_dist

        nnce = pd.DataFrame(nnce).T
        nnce["Loss"] = nnce_dist

        return nnce
