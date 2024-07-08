from abc import ABC, abstractmethod
import pandas as pd


class DatasetLoader(ABC):
    """
    An abstract class used to outline the minimal functionality of a dataset loader

    ...

    Attributes
    -------
    data: pd.DataFrame
        Stores the dataset as a DataFrame, only has value once load_data() called

    X: pd.DataFrame
        Stores the feature column as a DataFrame, only has value once load_data() called

    y: pd.DataFrame


    Methods
    -------
    load_data()
        Classes implementing DatasetLoader should specify a way to load all the data

    get_preprocessed_features()
        Classes implementing DatasetLoader should specify a way to preprocess the features in the data and return them
    """

    @abstractmethod
    def load_data(self):
        """
        Loads data into data attribute
        :return: None
        """
        pass

    @abstractmethod
    def get_preprocessed_features(self) -> pd.DataFrame:
        """
        Returns a preprocessed version of the dataset by altering the feature variables
        :return: pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """
        Returns whole dataset as DataFrame
        :return: pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def X(self) -> pd.DataFrame:
        """
        Returns only feature variables as DataFrame
        :return: pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def y(self) -> pd.Series:
        """
        Returns only target variable as Series
        :return: pd.Series
        """
        pass
