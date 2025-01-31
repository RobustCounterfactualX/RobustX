import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.models.imported_models.SKLearnModel import SKLearnModel
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel


def test_import_sklearn_model_file() -> None:
    """
    Tests loading a scikit-learn model from a saved file and comparing predictions.
    """
    # Create and train a model
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()
    trained_model = model.train(dl.X, dl.y)
    ct = ClassificationTask(trained_model, dl)



    # Save model to file
    joblib.dump(ct.model, "./model.pkl")

    # Load model using SKLearnModel
    trained_model = SKLearnModel("./model.pkl")

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)

    # Cleanup
    os.remove("./model.pkl")


def test_import_sklearn_model_instance() -> None:
    """
    Tests creating SKLearnModel from a scikit-learn model instance and comparing predictions.
    """
    # Create and train a model
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()

    trained_model = model.train(dl.X, dl.y)
    ct = ClassificationTask(trained_model, dl)


    # Extract trained scikit-learn model
    sklearn_model = ct.model

    # Import Model
    trained_model = SKLearnModel.from_model(sklearn_model)

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)
