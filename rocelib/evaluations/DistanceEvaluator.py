import numpy as np

from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator
from rocelib.evaluations.robustness_evaluations.Evaluator import Evaluator
from rocelib.lib.distance_functions.DistanceFunctions import euclidean


class DistanceEvaluator(Evaluator):
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

    def evaluate(self, recourse_method, valid_val=1, distance_func=euclidean, column_name="target", subset=None, **kwargs):
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
        recourses = self.task.CEs[recourse_method][0]
        
        df1 = recourses.drop(columns=[column_name, "loss", "predicted"], errors='ignore')
        # df1 = df1.drop(columns=[column_name, "predicted"], errors='ignore')


        if subset is None:
            df2 = self.task.dataset.get_negative_instances()
        else:
            df2 = subset

        # Drop any extra target columns from df2
        df2 = df2.drop(columns=[column_name, "predicted"], errors='ignore')

        # **Ensure both DataFrames have the same columns before assertion**
        df1 = df1[df2.columns]  # Align df1 columns to match df2

        print(f"Final Shapes - df1: {df1.shape}, df2: {df2.shape}")

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
