from abc import ABC, abstractmethod

import pandas as pd

from lib.distance_functions.DistanceFunctions import euclidean, manhattan
from tasks.Task import Task


class RecourseGenerator(ABC):

    def __init__(self, ct: Task, custom_distance_func=None):
        """
        :param ct: Task to solve
        :param custom_distance_func: Optional custom distance function
        """
        self._task = ct
        self.__customFunc = custom_distance_func

    @property
    def task(self):
        return self._task

    def generate(self, instances, distance_func="euclidean", neg_value=0,
                 column_name="target") -> pd.DataFrame:
        """
        Generates counterfactuals for a given DataFrame of instances
        :param instances: A DataFrame of instances for which you want to generate recourses
        :param distance_func: The way you would like to calculate the distance between two points, default is l2
                              ('l1' / 'manhattan', 'l2' / 'euclidean', 'custom')
        :param column_name: The target column name
        :param neg_value: The value which is considered negative in the target variable
        :return: A DataFrame of the recourses for the provided instances
        """
        cs = []

        for _, instance in instances.iterrows():
            cs.append(self.generate_for_instance(instance, distance_func, neg_value=neg_value,
                                                 column_name=column_name))

        res = pd.concat(cs)

        return res

    def generate_for_instance(self, instance, distance_func="euclidean", neg_value=0,
                              column_name="target") -> pd.DataFrame:
        """
        Generates a counterfactual for a provided instance
        :param instance: The instance for which you would like to generate a counterfactual for
        :param distance_func: The way you would like to calculate the distance between two points, default is l2
                              ('l1' / 'manhattan', 'l2' / 'euclidean', 'custom')
        :param column_name: The target column name
        :param neg_value: The value which is considered negative in the target variable
        :return: A DataFrame containing the recourse for the instance
        """
        func = euclidean
        if distance_func == "l1" or distance_func == "manhattan":
            func = manhattan
        elif distance_func == "custom":
            func = self.custom_distance_func

        return self._generation_method(instance, func, neg_value=neg_value, column_name=column_name)

    def generate_for_all(self, neg_value=0, column_name="target", distance_func="euclidean") -> pd.DataFrame:
        """
        Generates for all instances with a given negative value in their target column
        :param neg_value: The value in the target column which counts as a negative instance
        :param column_name: The name of the target variable
        :param distance_func: The way you would like to calculate the distance between two points, default is l2
                              ('l1' / 'manhattan', 'l2' / 'euclidean', 'custom')
        :return: A DataFrame of the recourses for all negative values
        """
        negatives = self.task.training_data.get_negative_instances(neg_value, column_name=column_name)

        recourses = self.generate(
            negatives,
            distance_func,
            column_name=column_name,
            neg_value=neg_value
        )

        recourses.index = negatives.index
        return recourses

    @abstractmethod
    def _generation_method(self, instance, distance_func,
                           column_name="target", neg_value=0):
        pass

    @property
    def custom_distance_func(self):
        return self.__customFunc
