from recourse_methods.RecourseGenerator import RecourseGenerator
from lib.distance_functions.DistanceFunctions import l1, euclidean
import torch
import numpy as np
import pandas as pd


class NNCE(RecourseGenerator):

    def generate_for_instance(self, instance, distance_func="euclidean", custom_func=None, gamma=0.1,
                              column_name="target", neg_value=0):

        model = self.ct.model

        # Convert X values of dataset to tensor
        X_tensor = torch.tensor(self.ct.training_data.X.values, dtype=torch.float32)

        # Get all model predictions of model, turning them to 0s or 1s
        model_labels = model.predict(X_tensor)
        model_labels = (model_labels >= 0.5)

        # Determine the target label
        # y = 1 if model.predict_single(instance) >= 0.5 else 0
        y = neg_value
        nnce_y = 1 - y

        # Set initial CE and minimum distance of CE
        nnce = None
        nnce_dist = np.inf

        # Decide distance function to use
        if distance_func == "l1" or distance_func == "manhattan":
            dist = l1
        else:
            dist = euclidean

        if isinstance(instance, pd.Series):
            negative_df = instance.to_frame()
        else:
            negative_df = instance

        preds = self.ct.training_data.X
        preds["predicted"] = model_labels

        # Iterate through each model prediction
        for _, pred in preds.iterrows():

            # Skip if the current instance is not the desired outcome
            if pred["predicted"] != nnce_y:
                continue

            # Calculate distance between negative instance and current sample
            sample_dist = dist(negative_df, pd.DataFrame(pred.drop(["predicted"])))

            # If distance is less than any other encountered yet, we have found a new NNCE
            if sample_dist < nnce_dist:
                nnce = pred
                nnce_dist = sample_dist

        nnce = pd.DataFrame(nnce).T
        nnce["Loss"] = nnce_dist

        return nnce
