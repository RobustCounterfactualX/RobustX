import pandas as pd

from rocelib.datasets.DatasetLoader import DatasetLoader


class CsvDatasetLoader(DatasetLoader):
    def __init__(self, file_path, target_column_label=None, target_column_index=None, header=0, names=None):
        super().__init__(target_column_index=target_column_index, target_column_label=target_column_label)
        self.file_path = file_path
        self._data = None
        self._header = header
        self._names = names


    def load_data(self):
        """Loads the CSV data and validates the target column specification."""
        try:
            # Load CSV
            self._data = pd.read_csv(self.file_path, header=self._header, names=self._names)

            # Ensure the CSV is not empty
            if self._data.empty:
                raise ValueError("The loaded dataset is empty.")


            # Validate target column specification
            if self._target_column_label is not None:
                print(f'{self._target_column_label}   {self._data.columns}')
                print(self._target_column_label not in self._data.columns)
                if self._target_column_label not in self._data.columns:
                    print('got here')
                    raise ValueError(
                        f"Target column label '{self._target_column_label}' not found in dataset columns.")

            elif self._target_column_index is not None:
                num_columns = len(self._data.columns)
                if not (0 <= self._target_column_index < num_columns):
                    raise IndexError(
                        f"Target column index {self._target_column_index} is out of bounds. Dataset has {num_columns} columns.")

        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{self.file_path}' was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty or cannot be read as a valid CSV.")
        except pd.errors.ParserError:
            raise ValueError("The file could not be parsed as a valid CSV format.")
    @property
    def X(self):
        """Returns the feature matrix (X) by excluding the target column."""
        if self._data is None:
            raise ValueError("Dataset not loaded. Call `load_data()` first.")

        if self._target_column_label:
            return self._data.drop(columns=[self._target_column_label])
        elif self._target_column_index is not None:
            return self._data.drop(columns=[self._data.columns[self._target_column_index]])
        return self._data  # If no target column is defined, return full dataset.

    @property
    def y(self):
        """Returns the target column (y)."""
        if self._data is None:
            raise ValueError("Dataset not loaded. Call `load_data()` first.")

        if self._target_column_label:
            return self._data[self._target_column_label]
        elif self._target_column_index is not None:
            return self._data.iloc[:, self._target_column_index]
        return None  # If no target column is defined, return None
