import pandas as pd

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.imported_models.KerasModel import KerasModel
from rocelib.models.keras_models.SimpleKerasNNModel import SimpleKerasNNModel
from rocelib.tasks.ClassificationTask import ClassificationTask
import os


def test_import_keras_model_file() -> None:
    # Create Model
    model = SimpleKerasNNModel(34, 8, 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    # Save Model
    model.model.save("./model.keras")

    # Import Model

    trained_model = KerasModel("./model.keras")

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))

    assert predictions_1.equals(predictions_2)

    os.remove('./model.keras')


def test_import_keras_model_instance() -> None:
    # Create Model
    model = SimpleKerasNNModel(34, 8, 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    keras_model = ct.model.get_keras_model()

    # Import Model
    trained_model = KerasModel.from_model(keras_model)

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))

    assert predictions_1.equals(predictions_2)
    