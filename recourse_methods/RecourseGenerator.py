from abc import ABC, abstractmethod

from tasks.ClassificationTask import ClassificationTask
import pandas as pd


class RecourseGenerator(ABC):

    def __init__(self, ct: ClassificationTask, custom_func=None):
        self._classificationTask = ct
        self.__customFunc = custom_func

    @property
    def ct(self):
        return self._classificationTask

    def generate(self, instances, distance_func="euclidean", custom_func=None, neg_value=0,
                 column_name="target") -> pd.DataFrame:
        """
        Generates counterfactuals for a given DataFrame of instances
        :param instances: A DataFrame of instances for which you want to generate recourses
        :param distance_func: The way you would like to calculate the distance between two points, default is l2
                              ('l1' / 'manhattan', 'l2' / 'euclidean', 'custom')
        :param custom_func: If a custom distance function was chosen, the function should be passed in here
        :param column_name: The target column name
        :param neg_value: The value which is considered negative in the target variable
        :return: A DataFrame of the recourses for the provided instances
        """
        cs = []

        for _, instance in instances.iterrows():
            cs.append(self.generate_for_instance(instance, distance_func, custom_func, neg_value=neg_value,
                                                 column_name=column_name))

        res = pd.concat(cs)

        return res

    @abstractmethod
    def generate_for_instance(self, instance, distance_func="euclidean", custom_func=None, neg_value=0,
                              column_name="target") -> pd.DataFrame:
        """
        Generates a counterfactual for a provided instance
        :param instance: The instance for which you would like to generate a counterfactual for
        :param distance_func: The way you would like to calculate the distance between two points, default is l2
                              ('l1' / 'manhattan', 'l2' / 'euclidean', 'custom')
        :param custom_func: If a custom distance function was chosen, the function should be passed in here
        :param column_name: The target column name
        :param neg_value: The value which is considered negative in the target variable
        :return: A DataFrame containing the recourse for the instance
        """
        pass

    def generate_for_all(self, neg_value=0, column_name="target", distance_func="euclidean",
                         custom_func=None) -> pd.DataFrame:
        """
        Generates for all instances with a given negative value in their target column
        :param neg_value: The value in the target column which counts as a negative instance
        :param column_name: The name of the target variable
        :param distance_func: The way you would like to calculate the distance between two points, default is l2
                              ('l1' / 'manhattan', 'l2' / 'euclidean', 'custom')
        :param custom_func: If a custom distance function was chosen, the function should be passed in here
        :return: A DataFrame of the recourses for all negative values
        """
        negatives = self.ct.training_data.get_negative_instances(neg_value, column_name=column_name)

        recourses = self.generate(
            negatives,
            distance_func,
            custom_func,
            column_name=column_name,
            neg_value=neg_value
        )

        recourses.index = negatives.index
        return recourses
