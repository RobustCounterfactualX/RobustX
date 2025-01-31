import pandas as pd

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.imported_models.KerasModel import KerasModel
from rocelib.models.keras_models.TrainableKerasModel import TrainableKerasModel
from rocelib.tasks.ClassificationTask import ClassificationTask
import os


def test_import_keras_model_file() -> None:
    # Create Model
    model = TrainableKerasModel(34, 8, 1)
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    model.train(dl.X, dl.y)

    ct = ClassificationTask(model, dl)


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
    model = TrainableKerasModel(34, 8, 1)
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    model.train(dl.X, dl.y)

    ct = ClassificationTask(model, dl)



    keras_model = ct.model.get_keras_model()

    # Import Model
    trained_model = KerasModel.from_model(keras_model)

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))

    assert predictions_1.equals(predictions_2)
    