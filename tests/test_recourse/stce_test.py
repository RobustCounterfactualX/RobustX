import pandas as pd

from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.recourse_methods.STCE import TrexNN
from rocelib.tasks.ClassificationTask import ClassificationTask



# def test_stce(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     res = ct.generate(["STCE"])

#     assert not res["STCE"][0].empty
