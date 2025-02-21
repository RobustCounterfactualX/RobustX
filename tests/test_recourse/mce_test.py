from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["MCE"])

    assert not res["MCE"][0].empty

