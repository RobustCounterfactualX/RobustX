from datasets.ExampleDatasets import get_example_dataset


def test_ionosphere_dataset_loader():
    ionosphere = get_example_dataset("ionosphere")
    assert not ionosphere.data.empty, "Ionosphere Test: data is empty when loaded initially"
    preprocessed = ionosphere.get_preprocessed_features()
    assert not preprocessed.empty, "Ionosphere test: preprocessed features are empty"
