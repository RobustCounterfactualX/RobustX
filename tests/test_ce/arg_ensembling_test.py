from robustx.datasets.ExampleDatasets import get_example_dataset
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from robustx.generators.robust_CE_methods.ArgEnsembling import ArgEnsembling


def test_proplace() -> None:
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()
    ct = ClassificationTask(model, dl)
    ct.train()
    models = [model, model, model]
    points = dl.get_negative_instances(neg_value=0, column_name="target")[0:1]
    recourse_gen = ArgEnsembling(dl, models)
    ce = recourse_gen.generate(points)

    assert ce is not None
    ce_vals = ce.values

    assert len(ce_vals) == 3
