from rocelib.datasets.ExampleDatasets import get_example_dataset

ionosphere = get_example_dataset("ionosphere")


def test_ionosphere_dataset_loader() -> None:
    assert not ionosphere.data.empty, "Ionosphere Test: data is empty when loaded initially"


def test_default_preprocessing() -> None:
    preprocessed = ionosphere.get_default_preprocessed_features()
    print(preprocessed.head())
    assert not preprocessed.empty, "Ionosphere test: default preprocessed features are empty"


def test_custom_preprocessing() -> None:
    preprocessed = ionosphere.get_preprocessed_features(impute_strategy_numeric='mean',
                                                        impute_strategy_categoric='most_frequent',
                                                        scale_method='minmax',
                                                        encode_categorical=True)
    print(preprocessed.head())
    assert not preprocessed.empty, "Ionosphere test: preprocessed features are empty"
