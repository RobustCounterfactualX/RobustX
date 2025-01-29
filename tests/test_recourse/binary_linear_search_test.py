import numpy as np
import pandas as pd

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_binary_linear_search_nn() -> None:
    # Create a new classification task and train the model on our data
    model = TrainablePyTorchModel(10, [7], 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    model.train(dl.X, dl.y)
    ct = ClassificationTask(model, dl)


    # Use BinaryLinearSearch to generate a recourse for each negative value
    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="HiringDecision")
    print(res)

    assert not res.empty


def test_binary_linear_search_dt() -> None:
    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    model.train(dl.X, dl.y)

    ct = ClassificationTask(model, dl)


    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0, column_name="target")

    assert not res.empty


def test_binary_linear_search_lr() -> None:
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    model.train(dl.X, dl.y)

    ct = ClassificationTask(model, dl)



    def euclidean_copy(x: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(np.sum((x.values - c.values) ** 2))

    recourse = BinaryLinearSearch(ct, custom_distance_func=euclidean_copy)

    res = recourse.generate_for_all(neg_value=0, column_name="target")

    assert not res.empty
