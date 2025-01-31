from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.generators.recourse_methods.Wachter import Wachter
from rocelib.lib.tasks.ClassificationTask import ClassificationTask


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


