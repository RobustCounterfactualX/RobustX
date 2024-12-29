from RoCELib.datasets.ExampleDatasets import get_example_dataset
from RoCELib.evaluations.ValidityEvaluator import ValidityEvaluator
from RoCELib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from RoCELib.recourse_methods.Wachter import Wachter
from RoCELib.tasks.ClassificationTask import ClassificationTask


def test_wachter() -> None:

    dl = get_example_dataset("ionosphere")

    model = SimpleNNModel(34, [8], 1)

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = Wachter(ct)

    res = recourse.generate_for_all(neg_value=0)

    val = ValidityEvaluator(ct)

    x = val.evaluate(res)

    assert x > 0.95


