from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.imported_models.KerasModel import KerasModel
from rocelib.models.keras_models.TrainableKerasModel import TrainableKerasModel
from rocelib.tasks.ClassificationTask import ClassificationTask
import os
import pytest


def test_import_keras_model_file() -> None:
    # Create Model
    model = TrainableKerasModel(34, 8, 1)
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    trained_model = model.train(dl)

    ct = ClassificationTask(trained_model, dl)

    # Save Model
    ct.model.model.save("./model.keras")

    # Import Model

    trained_model = KerasModel("./model.keras")

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))

    assert predictions_1.equals(predictions_2)

    os.remove('./model.keras')


def test_imported_keras_model_from_instance_predict_single_same_as_original() -> None:
    # Create Model
    model = TrainableKerasModel(34, 8, 1)
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    trained_model = model.train(dl)

    ct = ClassificationTask(trained_model, dl)



    keras_model = ct.model.model

    # Import Model
    trained_model = KerasModel.from_model(keras_model)

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))

    assert predictions_1.equals(predictions_2)


def test_throws_error_when_file_not_found() -> None:
    with pytest.raises(ValueError):
        trained_model = KerasModel("./garbage.keras")


def test_throws_error_when_wrong_file_type() -> None:
    with pytest.raises(ValueError):
        trained_model = KerasModel("./test.h5")


def test_throws_type_error() -> None:
    with pytest.raises(TypeError):
        trained_model = KerasModel(29)


def test_throws_type_error_again() -> None:
    with pytest.raises(TypeError):
        trained_model = KerasModel.from_model(2)




