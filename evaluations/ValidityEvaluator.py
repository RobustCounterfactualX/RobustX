import pandas as pd

from evaluations.RecourseEvaluator import RecourseEvaluator


class ValidityEvaluator(RecourseEvaluator):
    """
     An Evaluator class which evaluates the proportion of recourses which are valid

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the validity of CEs

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are valid

    -------
    """
    def checkValidity(self, instance, valid_val):
        """
        Checks if a given CE is valid
        @param instance: pd.DataFrame / pd.Series / torch.Tensor, the CE to check validity of
        @param valid_val: int, the target column value which denotes a valid CE
        @return:
        """
        # Convert to DataFrame
        if not isinstance(instance, pd.DataFrame):
            instance = pd.DataFrame(instance).T

        # Return if prediction is valid
        return self.task.model.predict_single(instance) == valid_val

    def evaluate(self, recourses, valid_val=1, column_name="target", **kwargs):
        """
        Evaluates the proportion of CEs are valid
        @param recourses: pd.DataFrame, set of CEs which we want to evaluate
        @param valid_val: int, target column value which denotes a valid instance
        @param column_name: str, name of target column
        @param kwargs: other arguments
        @return: int, proportion of CEs which are valid
        """
        valid = 0
        cnt = 0

        # Remove redundant columns
        instances = recourses.drop(columns=[column_name, "loss"], errors='ignore')

        for _, instance in instances.iterrows():

            # Increment validity counter if CE is valid
            if instance is not None and not instance.empty and self.checkValidity(instance, valid_val):
                valid += 1

            # Increment total counter
            cnt += 1

        return valid / cnt
