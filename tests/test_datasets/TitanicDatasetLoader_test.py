from rocelib.datasets.ExampleDatasets import get_example_dataset

titanic = get_example_dataset("titanic")


def test_titanic_dataset_loader() -> None:
    assert not titanic.data.empty, "Titanic Test: data is empty when loaded initially"


def test_default_preprocessing() -> None:
    preprocessed = titanic.get_default_preprocessed_features()
    print(preprocessed.head())
    assert not preprocessed.empty, "Titanic test: default preprocessed features are empty"


def test_custom_preprocessing() -> None:
    preprocessed = titanic.get_preprocessed_features(impute_strategy_numeric='mean',
                                                     impute_strategy_categoric='constant',
                                                     fill_value_categoric='IMPUTED_VAL',
                                                     scale_method='minmax',
                                                     selected_features=["Sex", "Embarked", "Pclass",
                                                                        "Age", "SibSp", "Parch", "Fare"],
                                                     encode_categorical=True)
    assert not preprocessed.empty, "Titanic test: preprocessed features are empty"
