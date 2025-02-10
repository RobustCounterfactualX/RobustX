import pandas as pd
import numpy as np
from robustx.lib.tasks.Task import Task


class ClassificationTask(Task):
    """
    A specific task type for classification problems that extends the base Task class.

    This class provides methods for training the model and retrieving positive instances
    from the training data.

    Attributes:
        model: The model to be trained and used for predictions.
        _training_data: The dataset used for training the model.
    """

    def get_random_positive_instance(self, neg_value, column_name="target") -> pd.Series:
        """
        Retrieves a random positive instance from the training data that does not have the specified negative value.

        This method continues to sample from the training data until a positive instance
        is found whose predicted label is not equal to the negative value.

        @param neg_value: The value considered negative in the target variable.
        @param column_name: The name of the target column used to identify positive instances.
        @return: A Pandas Series representing a random positive instance.
        """
        # Get a random positive instance from the training data
        pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)

        # Loop until a positive instance whose prediction is positive is found
        while self.model.predict_single(pos_instance) == neg_value:
            pos_instance = self._training_data.get_random_positive_instance(neg_value, column_name=column_name)

        return pos_instance

    def get_negative_instances(self, neg_value=0, column_name="target") -> pd.DataFrame:
        """
        Filters all the negative instances in the dataset as predicted by the model and returns them
        @param neg_value: What target value counts as a "negative" instance
        @param column_name: Target column's name
        @return: All instances with a negative target value predicted by the model
        """
        preds = self.model.predict(self.training_data.X).values.flatten()
        if neg_value == 0:
            idxs = np.where(preds < 0.5)[0]
            negatives = self.training_data.data.drop(columns=[column_name])
            negatives = pd.DataFrame(negatives.values[idxs], columns=negatives.columns)
        else:
            idxs = np.where(preds >= 0.5)[0]
            negatives = self.training_data.data.drop(columns=[column_name])
            negatives = pd.DataFrame(negatives.values[idxs], columns=negatives.columns)
        return negatives