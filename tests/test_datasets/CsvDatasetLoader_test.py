import pytest
import pandas as pd
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader


def test_csv_dataset_loader():
    """Test if CsvDatasetLoader correctly loads and matches an example dataset."""
    ionosphere = get_example_dataset("ionosphere")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]

    csv = CsvDatasetLoader(url, target_column_label="target", names=column_names)
    csv.load_data()

    assert csv._data is not None, "CSV Test: Dataset should be loaded"
    assert csv.X.shape[1] == 34, "CSV Test: Features should have 34 columns"
    assert csv.y.shape[0] == csv.X.shape[0], "CSV Test: Target and features should have the same row count"
    assert csv.__eq__(ionosphere), "CSV Test: Data does not match the example dataset"


def test_csv_dataset_loader_no_label_or_index():
    """Ensure ValueError is raised if neither target_column_label nor target_column_index is provided."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]

    with pytest.raises(ValueError, match="Either target_column_label or target_column_index must be provided"):
        CsvDatasetLoader(url, names=column_names).load_data()


def test_csv_dataset_loader_invalid_file_error():
    """Ensure FileNotFoundError is raised when a non-existent file is provided."""
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]

    with pytest.raises(FileNotFoundError, match="The file 'not_a_file.csv' was not found."):
        CsvDatasetLoader("not_a_file.csv", target_column_label="target", names=column_names).load_data()


def test_csv_dataset_loader_invalid_target_label_error():
    """Ensure ValueError is raised when the target column label does not exist."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]
    target_column_label = 'wrong_label'
    with pytest.raises(ValueError, match=f"Target column label '{target_column_label}' not found in dataset columns."):
        CsvDatasetLoader(url, target_column_label=target_column_label, names=column_names).load_data()


def test_csv_dataset_loader_invalid_target_index_error():
    """Ensure IndexError is raised when the target column index is out of bounds."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]

    with pytest.raises(IndexError, match="Target column index 100 is out of bounds. Dataset has"):
        CsvDatasetLoader(url, target_column_index=100, names=column_names).load_data()


def test_csv_dataset_loader_x_and_y_properties():
    """Test that X and y properties return correct data."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]

    loader = CsvDatasetLoader(url, target_column_label="target", names=column_names)
    loader.load_data()

    assert isinstance(loader.X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(loader.y, pd.Series), "y should be a Series"
    assert loader.X.shape[1] == 34, "X should have 34 feature columns"
    assert loader.y.shape[0] == loader.X.shape[0], "y and X should have the same number of rows"


def test_csv_dataset_loader_empty_file():
    """Ensure ValueError is raised when the CSV file is empty."""
    empty_file = "empty_file.csv"

    # Create an empty CSV file for testing
    pd.DataFrame().to_csv(empty_file, index=False)

    with pytest.raises(ValueError, match="The file is empty or cannot be read as a valid CSV."):
        CsvDatasetLoader(empty_file, target_column_label="target").load_data()


