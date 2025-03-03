import os

import joblib
import pandas as pd
import pytest

from rocelib.models.imported_models.SKLearnModel import SKLearnModel


def test_imported_sklearn_model_file_predict_single_same_as_original(testing_models) -> None:
    # Create Model
    ct = testing_models.get("ionosphere", "ionosphere", "logistic regression", 34, 8, 1)

    # Save Model
    joblib.dump(ct.model.model, "./model.pkl")

    # Import Model
    trained_model = SKLearnModel("./model.pkl")

    for _, instance in ct.dataset.data.iterrows():
        instance_x = instance.drop("target")

        if isinstance(instance_x, pd.Series):
            instance_x = instance_x.to_frame().T

        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)

    os.remove("./model.pkl")


def test_imported_sklearn_model_file_predict_all_same_as_original(testing_models) -> None:
    # Create Model
    ct = testing_models.get("ionosphere", "ionosphere", "logistic regression", 34, 8, 1)

    # Save Model
    joblib.dump(ct.model.model, "./model.pkl")

    # Import Model
    trained_model = SKLearnModel("./model.pkl")

    predictions_1 = pd.DataFrame(ct.model.predict(ct.dataset.data.drop("target", axis=1)))
    predictions_2 = pd.DataFrame(trained_model.predict(ct.dataset.data.drop("target", axis=1)))

    # Ensure same column names for consistency
    predictions_2.columns = predictions_1.columns

    assert predictions_1.equals(predictions_2)

    os.remove("./model.pkl")


def test_imported_sklearn_model_from_instance_predict_single_same_as_original(testing_models) -> None:
    # Create Model
    ct = testing_models.get("ionosphere", "ionosphere", "logistic regression", 34, 8, 1)

    sklearn_model = ct.model.model  # Extract the actual scikit-learn model

    # Import Model
    trained_model = SKLearnModel.from_model(sklearn_model)

    predictions_1 = ct.model.predict(ct.dataset.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.dataset.data.drop("target", axis=1))

    predictions_1_df = pd.DataFrame(predictions_1).reset_index(drop=True)
    predictions_2_df = pd.DataFrame(predictions_2).reset_index(drop=True)

    predictions_2_df.columns = predictions_1_df.columns

    assert predictions_1_df.equals(predictions_2_df)


def test_throws_file_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        trained_model = SKLearnModel("./garbage.pkl")


def test_throws_wrong_file_type_error() -> None:
    with pytest.raises(ValueError):
        trained_model = SKLearnModel("./test.txt")


def test_throws_type_error() -> None:
    with pytest.raises(TypeError):
        trained_model = SKLearnModel(42)


def test_throws_type_error_again() -> None:
    with pytest.raises(TypeError):
        trained_model = SKLearnModel.from_model(2)