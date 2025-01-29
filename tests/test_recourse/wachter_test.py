from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.Wachter import Wachter
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_wachter() -> None:

    dl = get_example_dataset("ionosphere")

    model = TrainablePyTorchModel(34, [8], 1)

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = Wachter(ct)

    res = recourse.generate_for_all(neg_value=0)

    val = ValidityEvaluator(ct)

    x = val.evaluate(res)

    assert x > 0.95


