import numpy as np

from RoCELib.evaluations.RecourseEvaluator import RecourseEvaluator
from RoCELib.lib.distance_functions.DistanceFunctions import euclidean


class DistanceEvaluator(RecourseEvaluator):
    """
     An Evaluator class which evaluates the average distance of recourses from their original instance

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the distance of CEs

    distance_func: Function
        A function which takes in 2 dataframes and returns an integer representing distance, defaulted to euclidean

    valid_val: int
        Stores what the target value of a valid counterfactual is defined as

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the average distance of each x' from x

    -------
    """

    def evaluate(self, recourses, valid_val=1, distance_func=euclidean, column_name="target", subset=None, **kwargs):
        """
        Determines the average distance of the CEs from their original instances
        @param recourses: pd.DataFrame, dataset containing CEs in same order as negative instances in dataset
        @param valid_val: int, what the target value of a valid counterfactual is defined as, default 1
        @param distance_func: Function, function which takes in 2 dataframes and returns an integer representing
                              distance, defaulted to euclidean
        @param column_name: name of target column
        @param subset: optional DataFrame, contains instances to generate CEs on
        @param kwargs: other arguments
        @return: int, average distance of CEs from their original instances
        """
        df1 = recourses.drop(columns=[column_name, "loss"], errors='ignore')

        if subset is None:
            df2 = self.task.training_data.get_negative_instances(neg_value=1 - valid_val, column_name=column_name)
        else:
            df2 = subset

        # Ensure the DataFrames have the same shape
        assert df1.shape == df2.shape, "DataFrames must have the same shape"

        distances = []

        # Iterate over each row in the DataFrames
        for i in range(len(df1)):
            row1 = df1.iloc[i:i + 1]  # Get the i-th row as a DataFrame
            row2 = df2.iloc[i:i + 1]  # Get the i-th row as a DataFrame

            # Calculate distance between corresponding rows
            dist = distance_func(row1, row2)
            distances.append(dist)

        # Calculate and return the average distance
        return np.mean(distances)
