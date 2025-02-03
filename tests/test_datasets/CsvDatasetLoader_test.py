from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader


def test_csv_dataset_loader():
    ionosphere = get_example_dataset("ionosphere")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]
    csv = CsvDatasetLoader(url, target_column="target", names=column_names)
    assert csv.__eq__(ionosphere), "CSV Test: data does not match"

def test_csv_dataset_loader_invalid_file_error():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]
    try:
        CsvDatasetLoader("not_a_file", target_column="target", names=column_names)
        assert False, "CSV Test: file does not exist"
    except FileNotFoundError:
        assert True, "CSV Test: file does not exist"

def test_csv_dataset_loader_invalid_target_error():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]
    try:
        CsvDatasetLoader(url, target_column="not_a_column", names=column_names)
        assert False, "CSV Test: target column does not exist"
    except ValueError:
        assert True, "CSV Test: target column does not exist"

def test_csv_dataset_loader_no_names():
    ionosphere = get_example_dataset("ionosphere")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    csv = CsvDatasetLoader(url, target_column="target")
    assert csv.__eq__(ionosphere), "CSV Test: data does not match"