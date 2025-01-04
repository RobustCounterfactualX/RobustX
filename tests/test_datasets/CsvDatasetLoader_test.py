from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader


def test_csv_dataset_loader():
    ionosphere = get_example_dataset("ionosphere")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
    column_names = [f"feature_{i}" for i in range(34)] + ["target"]
    csv = CsvDatasetLoader(url, target_column="target", names=column_names)
    assert csv.__eq__(ionosphere), "CSV Test: data does not match"
