from rocelib.datasets.ExampleDatasets import get_example_dataset

iris = get_example_dataset("iris")


def test_iris_dataset_loader() -> None:
    assert not iris.data.empty, "Iris Test: data is empty when loaded initially"


def test_default_preprocessing() -> None:
    preprocessed = iris.get_default_preprocessed_features()
    print(preprocessed.head())
    assert not preprocessed.empty, "Iris test: default preprocessed features are empty"


def test_custom_preprocessing() -> None:
    preprocessed = iris.get_preprocessed_features(impute_strategy_numeric='mean',
                                                  impute_strategy_categoric='most_frequent', scale_method='minmax',
                                                  encode_categorical=True)
    print(preprocessed.head())
    assert not preprocessed.empty, "Iris test: preprocessed features are empty"
