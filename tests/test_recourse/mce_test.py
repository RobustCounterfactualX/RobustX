from RoCELib.datasets.ExampleDatasets import get_example_dataset
from RoCELib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from RoCELib.recourse_methods.MCE import MCE
from RoCELib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances():
    model = SimpleNNModel(34, [8], 1)
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
