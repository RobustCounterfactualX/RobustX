from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.Wachter import Wachter
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_wachter(testing_models) -> None:
    ct, _, _ = testing_models.get(Dataset.IONOSPHERE, ModelType.NEURALNET, 34, 8, 1)

    recourse = Wachter(ct)

    res = recourse.generate_for_all(neg_value=0)

    val = ValidityEvaluator(ct)

    x = val.evaluate(res)

    assert x > 0.85


