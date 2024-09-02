import numpy as np

from lib.distance_functions.DistanceFunctions import euclidean
from tasks.Task import Task


class DistanceEvaluator:

    def __init__(self, ct: Task, distance_func=euclidean, valid_val=1):
        self.task = ct
        self.distance_func = distance_func
        self.valid_val = valid_val

    def evaluate(self, recourses):
        df1 = recourses.drop(columns=["target", "loss"], errors='ignore')
        df2 = self.task.training_data.get_negative_instances(neg_value=1 - self.valid_val)

        # Ensure the DataFrames have the same shape
        assert df1.shape == df2.shape, "DataFrames must have the same shape"

        distances = []

        # Iterate over each row in the DataFrames
        for i in range(len(df1)):
            row1 = df1.iloc[i:i + 1]  # Get the i-th row as a DataFrame
            row2 = df2.iloc[i:i + 1]  # Get the i-th row as a DataFrame

            # Calculate Euclidean distance between corresponding rows
            dist = euclidean(row1, row2)
            distances.append(dist)

        # Calculate and return the average distance
        return np.mean(distances)
