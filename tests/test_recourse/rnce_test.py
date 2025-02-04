from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.RNCE import RNCE
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator



def test_rnce() -> None:
    model = TrainablePyTorchModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()

    trained_model = model.train(dl)
    ct = ClassificationTask(trained_model, dl)


    delta = 0.01

    recourse = RNCE(ct)

    res = recourse.generate_for_all(neg_value=0)

    val = ValidityEvaluator(ct)

    x = val.evaluate(res)

    assert x > 0.95

    # _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]

    # for _, neg in dl.get_negative_instances(neg_value=0).head(10).iterrows():
    #     res = recourse.generate_for_instance(neg, delta=delta)
    #     assert recourse.intabs.evaluate(res, delta=delta)
