from robustx.datasets.ExampleDatasets import get_example_dataset
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.generators.robust_CE_methods.PROPLACE import PROPLACE
from robustx.lib.tasks.ClassificationTask import ClassificationTask
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator


def test_proplace() -> None:
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    delta = 0.003

    recourse_gen = PROPLACE(ct)

    _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]
    ev = DeltaRobustnessEvaluator(ct)
    for _, neg in dl.get_negative_instances(neg_value=0).head(3).iterrows():
        res = recourse_gen.generate_for_instance(neg, delta=delta, delta_bias=delta)
        rob = ev.evaluate(res, desired_output=1, delta=0.003, bias_delta=0.003, M=1000,
                          epsilon=0.000001)
        assert rob


if __name__ == "__main__":
    test_proplace()
