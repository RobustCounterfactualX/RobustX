from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.generators.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.lib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances():
    model1 = SimpleNNModel(34, [8], 1)
    model2 = SimpleNNModel(34, [16, 8], 1)
    model3 = SimpleNNModel(34, [16, 8, 4], 1)

    dl = get_example_dataset("ionosphere")

    ct1 = ClassificationTask(model1, dl)
    ct2 = ClassificationTask(model2, dl)
    ct3 = ClassificationTask(model3, dl)

    dl.default_preprocess()
    ct1.train()
    ct2.train()
    ct3.train()

    recourse = ModelMultiplicityMILP(dl, [model1, model2, model3])

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():

        res = recourse.generate_for_instance(neg)

        if not res.empty:
            prediction1 = model1.predict_single(res)

            prediction2 = model2.predict_single(res)

            prediction3 = model3.predict_single(res)

            assert prediction1
            assert prediction2
            assert prediction3
