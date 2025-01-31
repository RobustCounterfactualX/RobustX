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
    trained_model1 = model1.train(dl.X, dl.y)
    ct1 = ClassificationTask(trained_model1, dl)
    trained_model2 = model2.train(dl.X, dl.y)
    ct2 = ClassificationTask(trained_model2, dl)
    trained_model3 = model3.train(dl.X, dl.y)
    ct3 = ClassificationTask(trained_model3, dl)
    # TODO : changed MILP to trainedModel
    


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
