from robustx.lib.distance_functions.DistanceFunctions import euclidean
from robustx.generators.CEGenerator import CEGenerator
import pandas as pd


class BinaryLinearSearch(CEGenerator):
    """
    A counterfactual explanation generator that uses binary linear search to find counterfactual explanations.

    Inherits from the CEGenerator class and implements the _generation_method to perform
    binary linear search for generating counterfactuals.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
        __customFunc (callable, optional): A custom distance function, inherited from CEGenerator.
    """

    def _generation_method(self, instance, gamma=0.1, column_name="target", neg_value=0,
                           distance_func=euclidean, **kwargs) -> pd.DataFrame:
        """
        Generates a counterfactual explanation using binary linear search.

        @param instance: The instance for which to generate a counterfactual.
        @param distance_func: The function used to calculate the distance between points.
        @param gamma: The distance threshold for convergence.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the counterfactual explanation.
        """
        if self.custom_distance_func is not None:
            distance_func = self.custom_distance_func

        # Get initial counterfactual
        c = self.task.get_random_positive_instance(neg_value, column_name).T

        # Make sure column names are same so return result has same indices
        negative = instance.to_frame()
        c.columns = negative.columns

        model = self.task.model

        # Loop until CE is under gamma threshold
        while distance_func(negative, c) > gamma:

            # Calculate new CE by finding midpoint
            new_neg = c.add(negative, axis=0) / 2

            # Reassign endpoints based on model prediction
            if model.predict_single(new_neg.T) == model.predict_single(negative.T):
                negative = new_neg
            else:
                c = new_neg

        # Form the dataframe
        ct = c.T

        # Store model prediction in return CE (this should ALWAYS be the positive value)
        res = model.predict_single(ct)

        ct[column_name] = res

        newct = c.T
        nt = instance

        loss = distance_func(nt, newct)

        # Store the loss
        ct["loss"] = loss

        return ct
