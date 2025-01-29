from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances():
    model = TrainablePyTorchModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = MCE(ct)

    # _, neg = list(dl.get_negative_instances(neg_value=0).iterrows())[0]

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():

        res = recourse.generate_for_instance(neg)

        if not res.empty:
            prediction = model.predict_single(res)

            assert prediction
