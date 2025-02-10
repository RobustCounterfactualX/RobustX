from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..DatasetLoader import DatasetLoader


class ExampleDatasetLoader(DatasetLoader, ABC):
    """
    An abstract extension of DatasetLoader class which stores example datasets provided within the library

        ...

    Attributes / Properties
    ------------------------

    _categorical: list[str]
        Stores the list of categorical column names

    _numerical: list[str]
        Stores the list of numerical column names

    __missing_num: any
        Value representing missing numerical data

    __missing_cat: any
        Value representing missing categorical data

    -------

    Methods
    -------

    categorical -> list[str]:
        Returns the list of categorical features

    numerical -> list[str]:
        Returns the list of numerical features

    load_data() -> None:
        Abstract method to load data into the dataset

    get_default_preprocessed_features() -> pd.DataFrame:
        Abstract method to get the default preprocessed dataset

    get_preprocessed_features() -> pd.DataFrame:
        Returns the dataset preprocessed according to user specifications (imputing, scaling, encoding)

    default_preprocess() -> None:
        Preprocesses and updates the dataset using the default preprocessing method

    preprocess() -> None:
        Preprocesses and updates the dataset based on user-provided parameters
    -------
    """

    def __init__(self, categoricals, numericals, missing_val_num=np.nan, missing_val_cat=np.nan, seed=None):
        """
        Initializes the ExampleDatasetLoader with categorical and numerical features, as well as values for missing data.

        @param categoricals: list[str], List of categorical features
        @param numericals: list[str], List of numerical features
        @param missing_val_num: optional, Value to represent missing numerical data (default: np.nan)
        @param missing_val_cat: optional, Value to represent missing categorical data (default: np.nan)
        """
        super().__init__(seed=seed)
        self._categorical = categoricals
        self._numerical = numericals
        self.__missing_num = missing_val_num
        self.__missing_cat = missing_val_cat

    @property
    def categorical(self):
        """
        Returns all categorical column names
        @return: list[str]
        """
        return self._categorical

    @property
    def numerical(self) -> list[str]:
        """
        Returns all numerical column names
        @return: list[str]
        """
        return self._numerical

    @abstractmethod
    def load_data(self):
        """
        Loads data into data attribute
        @return: None
        """
        pass

    @abstractmethod
    def get_default_preprocessed_features(self) -> pd.DataFrame:
        """
        Returns a preprocessed version of the dataset by using a default/standard preprocessing pipeline
        @return: pd.DataFrame
        """
        pass

    def get_preprocessed_features(
            self,
            impute_strategy_numeric: str = 'mean',
            impute_strategy_categoric: str = 'most_frequent',
            fill_value_categoric: str = None,
            fill_value_numeric: str = None,
            scale_method: str = 'standard',
            encode_categorical: bool = True,
            selected_features: list = None
    ) -> pd.DataFrame:
        """
        Returns a preprocessed version of the dataset based on what the user inputs
        @param impute_strategy_numeric:  strategy for imputing missing numeric values ('mean', 'median')
        @param impute_strategy_categoric: strategy for imputing missing categoric values ('most_frequent', 'constant')
        @param fill_value_categoric: value to use for constant imputing strategy for categorical features
        @param fill_value_numeric: value to use for constant imputing strategy for numerical features
        @param scale_method: method for scaling numerical features ('standard', 'minmax', None)
        @param encode_categorical: whether to encode categorical features (True/False)
        @param selected_features: list of features to select, if None all features are used
        @return: pd.DataFrame
        """
        # Extract only the selected features and separate into numerical, categorical
        if selected_features is not None:
            data_selected = self.data[selected_features]
            numeric_features = list(set(self.numerical) & set(selected_features))
            categoric_features = list(set(self.categorical) & set(selected_features))
        else:
            numeric_features = self.numerical
            categoric_features = self.categorical
            data_selected = self.data

        if len(self.numerical) > 0:
            # Impute numerical features
            if impute_strategy_categoric == 'constant':
                numerical_imputer = SimpleImputer(strategy=impute_strategy_numeric, missing_values=self.__missing_num,
                                                  fill_value=fill_value_numeric)
            else:
                numerical_imputer = SimpleImputer(strategy=impute_strategy_numeric, missing_values=self.__missing_num)

            numerical_data_imputed = pd.DataFrame(numerical_imputer.fit_transform(data_selected[numeric_features]),
                                                  columns=numeric_features)

            # Scale numerical features
            if scale_method == 'standard':
                scaler = StandardScaler()
                numerical_data_scaled = pd.DataFrame(scaler.fit_transform(numerical_data_imputed),
                                                     columns=numeric_features)
            elif scale_method == 'minmax':
                scaler = MinMaxScaler()
                numerical_data_scaled = pd.DataFrame(scaler.fit_transform(numerical_data_imputed),
                                                     columns=numeric_features)
            else:
                numerical_data_scaled = numerical_data_imputed
        else:
            numerical_data_scaled = pd.DataFrame()

        if len(self.categorical) > 0:
            # Impute categorical features
            if impute_strategy_categoric == 'constant':
                categorical_imputer = SimpleImputer(strategy=impute_strategy_categoric,
                                                    missing_values=self.__missing_cat,
                                                    fill_value=fill_value_categoric)
            else:
                categorical_imputer = SimpleImputer(strategy=impute_strategy_categoric,
                                                    missing_values=self.__missing_cat)

            categorical_data_imputed = pd.DataFrame(categorical_imputer.fit_transform(self.data[categoric_features]),
                                                    columns=categoric_features)

            # Encode categorical features
            if encode_categorical:
                categorical_data_encoded = pd.get_dummies(categorical_data_imputed, drop_first=True,
                                                          columns=categoric_features)
            else:
                categorical_data_encoded = categorical_data_imputed
        else:
            categorical_data_encoded = pd.DataFrame()

        # Join preprocessed categorical and numerical features
        preprocessed_data = pd.concat([categorical_data_encoded, numerical_data_scaled], axis=1)

        return preprocessed_data

    def default_preprocess(self):
        """
        Changes the data attribute to be preprocessed using the default method
        @return: None
        """
        preprocessed = self.get_default_preprocessed_features()
        self.data = pd.concat([preprocessed, self.y], axis=1).drop_duplicates()

    def preprocess(
            self,
            impute_strategy_numeric: str = 'mean',
            impute_strategy_categoric: str = 'most_frequent',
            fill_value_categoric: str = None,
            fill_value_numeric: str = None,
            scale_method: str = 'standard',
            encode_categorical: bool = True,
            selected_features: list = None
    ):
        """
        Changes the data attribute to be preprocessed based on parameters
        @param impute_strategy_numeric:  strategy for imputing missing numeric values ('mean', 'median')
        @param impute_strategy_categoric: strategy for imputing missing categoric values ('most_frequent', 'constant')
        @param fill_value_categoric: value to use for constant imputing strategy for categorical features
        @param fill_value_numeric: value to use for constant imputing strategy for numerical features
        @param scale_method: method for scaling numerical features ('standard', 'minmax', None)
        @param encode_categorical: whether to encode categorical features (True/False)
        @param selected_features: list of features to select, if None all features are used
        @return: None
        """
        preprocessed = self.get_preprocessed_features(
            impute_strategy_numeric=impute_strategy_numeric,
            impute_strategy_categoric=impute_strategy_categoric,
            fill_value_categoric=fill_value_categoric,
            fill_value_numeric=fill_value_numeric,
            scale_method=scale_method,
            encode_categorical=encode_categorical,
            selected_features=selected_features
        )
        self.data = pd.concat([preprocessed, self.y], axis=1).drop_duplicates()
