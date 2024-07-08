from datasets.ExampleDatasets import get_example_dataset


def test_iris_dataset_loader() -> None:
    iris = get_example_dataset("iris")
    assert not iris.data.empty, "Iris Test: data is empty when loaded initially"
    preprocessed = iris.get_preprocessed_features()
    assert not preprocessed.empty, "Iris test: preprocessed features are empty"
