from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.tasks.ClassificationTask import ClassificationTask


# def test_mce_predicts_positive_instances(testing_models): TODO
#     dl = get_example_dataset("ionosphere")
#     trained_model_1 = TrainablePyTorchModel(34, [8], 1).train(dl.X, dl.y)
#     trained_model_2 = TrainablePyTorchModel(34, [16, 8], 1).train(dl.X, dl.y)
#     trained_model_3 = TrainablePyTorchModel(34, [16, 8, 4], 1).train(dl.X, dl.y)

#     recourse = ModelMultiplicityMILP(dl, [trained_model_1, trained_model_2, trained_model_3])
#     res = recourse.generate_for_all()

#     assert not res.empty