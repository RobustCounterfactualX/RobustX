from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_mce_predicts_positive_instances():
    model1 = TrainablePyTorchModel(34, [8], 1)
    model2 = TrainablePyTorchModel(34, [16, 8], 1)
    model3 = TrainablePyTorchModel(34, [16, 8, 4], 1)

    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    ct1.train(dl.X, dl.y)
    ct2.train(dl.X, dl.y)
    ct3.train(dl.X, dl.y)

    ct1 = ClassificationTask(model1, dl)
    ct2 = ClassificationTask(model2, dl)
    ct3 = ClassificationTask(model3, dl)



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
