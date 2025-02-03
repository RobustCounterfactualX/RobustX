import pandas as pd

from ..DatasetLoader import DatasetLoader


class CsvDatasetLoader(DatasetLoader):
    """
    A DatasetLoader class which opens a CSV and loads it into DatasetLoader format

        ...

    Attributes / Properties
    -------

    _data: pd.DataFrame
        Stores the dataset as a DataFrame, only has value once load_data() called

    _target_col: str
        Stores name of target variable

    X: pd.DataFrame
        Stores the feature columns as a DataFrame, only has value once load_data() called

    y: pd.DataFrame
        Stores the target column as a DataFrame, only has value once load_data() called

    -------

    Methods
    -------

    get_negative_instances() -> pd.DataFrame:
        Filters all the negative instances in the dataset and returns them

    get_random_positive_instance() -> pd.Series:
        Returns a random instance where the target variable is NOT the neg_value

    -------
    """

    def __init__(self, csv, target_column, header=0, names=None):
        """
        @param csv: str, Path to csv
        @param target_column: str, Name of column storing target variable
        @param header: optional int, Row number(s) containing column labels and marking the start of the data (zero-indexed).
        @param names: optional list[str], Column labels
        """
        super().__init__()
        self._target_col = target_column
        self.__load_data(csv, header, names)
        if target_column not in self._data.columns:
            raise ValueError(f"Target column {target_column} not found in dataset")

    def __load_data(self, csv, header, names):
        """
        Loads data into protected self._data attribute
        @param csv: str, Path to csv
        @param header: optional int, Row number(s) containing column labels and marking the start of the data (zero-indexed).
        @param names: optional list[str], Column labels
        @return:
        """
        try:

            if names is None:
                self._data = pd.read_csv(csv, header=header)
            else:
                self._data = pd.read_csv(csv, header=header, names=names)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {csv} not found")



    @property
    def X(self):
        return self._data.drop(columns=[self._target_col])

    @property
    def y(self) -> pd.Series:
        return self._data[[self._target_col]]
