from datasets.ExampleDatasets import get_example_dataset
from evaluations.ValidityEvaluator import ValidityEvaluator
from models.pytorch_models.SimpleNNModel import SimpleNNModel
from recourse_methods.Wachter import Wachter
from tasks.ClassificationTask import ClassificationTask


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


