import pandas as pd

from rocelib.datasets.DatasetLoader import DatasetLoader


class CsvDatasetLoader(DatasetLoader):
    def __init__(self, file_path, target_column_label, neg_value, header=0, names=None):
        super().__init__(target_column_label, neg_value)
        self.file_path = file_path
        self._data = None
        self._header = header
        self._names = names
        self.load_data()


    def load_data(self):
        """Loads the CSV data and validates the target column specification."""
        try:
            # Load CSV
            self._data = pd.read_csv(self.file_path, header=self._header, names=self._names)

            # Ensure the CSV is not empty
            if self._data.empty:
                raise ValueError("The loaded dataset is empty.")


            # Validate target column specification
            if self._target_column_label not in self._data.columns:
                raise ValueError(
                    f"Target column label '{self._target_column_label}' not found in dataset columns.")


        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{self.file_path}' was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty or cannot be read as a valid CSV.")
        except pd.errors.ParserError:
            raise ValueError("The file could not be parsed as a valid CSV format.")
    @property
    def X(self):
        """Returns the feature matrix (X) by excluding the target column."""
        return self._data.drop(columns=[self._target_column_label])

    @property
    def y(self):
        """Returns the target column (y)."""
        return self._data[self._target_column_label]
