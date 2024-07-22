import pandas as pd

from datasets.DatasetLoader import DatasetLoader


class CsvDatasetLoader(DatasetLoader):
    """
    A DatasetLoader class responsible for loading CSVs
    """

    def __init__(self, csv, target_column, header=0, names=None):
        """
        Parameters
        ----------
        csv : str
            Path to the csv
        target_column : str
            Name of column storing target variable
        header : int, optional
            Row number(s) containing column labels and marking the start of the data (zero-indexed).
        names: list[str], optional
            Column labels
        """
        super().__init__()
        self._target_col = target_column
        self.__load_data(csv, header, names)

    def __load_data(self, csv, header, names):
        if names is None:
            self._data = pd.read_csv(csv, header=header)
        else:
            self._data = pd.read_csv(csv, header=header, names=names)

    @property
    def X(self):
        return self._data.drop(columns=[self._target_col])

    @property
    def y(self) -> pd.Series:
        return self._data[[self._target_col]]
