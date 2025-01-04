from rocelib.datasets.ExampleDatasets import get_example_dataset


adult = get_example_dataset("adult")
print(adult.data.head())


def test_adult_dataset_loader() -> None:
    assert not adult.data.empty, "Adult Test: data is empty when loaded initially"


def test_default_preprocessing() -> None:
    preprocessed = adult.get_default_preprocessed_features()
    print(preprocessed.head())
    assert not preprocessed.empty, "Adult test: default preprocessed features are empty"


def test_custom_preprocessing() -> None:
    preprocessed = adult.get_preprocessed_features(impute_strategy_numeric='mean',
                                                   impute_strategy_categoric='most_frequent', scale_method='minmax',
                                                   encode_categorical=True)
    print(preprocessed.head())
    assert not preprocessed.empty, "Adult test: preprocessed features are empty"
