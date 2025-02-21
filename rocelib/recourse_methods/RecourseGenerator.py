from abc import ABC, abstractmethod

import pandas as pd

from rocelib.tasks.Task import Task


class RecourseGenerator(ABC):
    """
    Abstract class for generating counterfactual explanations for a given task.

    This class provides a framework for generating counterfactuals based on a distance function
    and a given task. It supports default distance functions such as Euclidean and Manhattan,
    and allows for custom distance functions.

    Attributes:
        _task (Task): The task to solve.
        __customFunc (callable, optional): A custom distance function.
    """

    def __init__(self, ct: Task, custom_distance_func=None):
        """
        Initializes the RecourseGenerator with a task and an optional custom distance function.

        @param ct: The Task instance to solve.
        @param custom_distance_func: An optional custom distance function.
        """
        self._task = ct
        self.__customFunc = custom_distance_func

    @property
    def task(self):
        return self._task

    def generate(self, instances: pd.DataFrame, neg_value=0,
                 column_name="target", **kwargs) -> pd.DataFrame:
        """
        Generates counterfactuals for a given DataFrame of instances.

        @param instances: A DataFrame of instances for which you want to generate recourses.
        @param distance_func: The method to calculate the distance between two points. Options are 'l1' / 'manhattan', 'l2' / 'euclidean', and 'custom'.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame of the recourses for the provided instances.
        """
        cs = []
        for _, instance in instances.iterrows():
            cs.append(self.generate_for_instance(instance, neg_value=neg_value,
                                                 column_name=column_name, **kwargs))
        res = pd.concat(cs)

        return res

    def generate_for_instance(self, instance, neg_value=0,
                              column_name="target", **kwargs) -> pd.DataFrame:
        """
        Generates a counterfactual for a provided instance.

        @param instance: The instance for which you would like to generate a counterfactual.
        @param distance_func: The method to calculate the distance between two points. Options are 'l1' / 'manhattan', 'l2' / 'euclidean', and 'custom'.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the recourse for the instance.
        """

        return self._generation_method(instance, neg_value=neg_value, column_name=column_name, **kwargs)

    def generate_for_all(self, neg_value=0, column_name="target", **kwargs) -> pd.DataFrame:
        """
        Generates counterfactuals for all instances with a given negative value in their target column.

        @param neg_value: The value in the target column which counts as a negative instance.
        @param column_name: The name of the target variable.
        @param distance_func: The method to calculate the distance between two points. Options are 'l1' / 'manhattan', 'l2' / 'euclidean', and 'custom'.
        @return: A DataFrame of the recourses for all negative values.
        """
        negatives = self.task.dataset.get_negative_instances()

        recourses = self.generate(
            negatives,
            column_name=column_name,
            neg_value=neg_value,
            **kwargs
        )

        recourses.index = negatives.index
        return recourses

    @abstractmethod
    def _generation_method(self, instance,
                           column_name="target", neg_value=0, **kwargs):
        """
        Abstract method to be implemented by subclasses for generating counterfactuals.

        @param instance: The instance for which to generate a counterfactual.
        @param distance_func: The function to calculate distances.
        @param column_name: The name of the target column.
        @param neg_value: The value considered negative in the target variable.
        @return: A DataFrame containing the generated counterfactuals.
        """
        pass

    @property
    def custom_distance_func(self):
        """
        Returns custom distance function passed at instantiation
        @return: distance Function, (DataFrame, DataFrame) -> Int
        """
        return self.__customFunc
