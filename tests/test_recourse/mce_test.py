from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    recourse = MCE(ct)
    res = recourse.generate_for_all(neg_value=0, column_name="target")

    assert not res.empty

