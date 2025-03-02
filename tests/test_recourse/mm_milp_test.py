from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.ModelMultiplicityMILP import ModelMultiplicityMILP
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.tasks.TaskBuilder import TaskBuilder


def test_mm_milp(testing_models):
    dl = get_example_dataset("ionosphere")
    trained_model_1 = TrainablePyTorchModel(34, [8], 1).train(dl.X, dl.y)
    trained_model_2 = TrainablePyTorchModel(34, [16, 8], 1).train(dl.X, dl.y)
    trained_model_3 = TrainablePyTorchModel(34, [16, 8, 4], 1).train(dl.X, dl.y)

    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    ces = ct.generate(["MMMILP"])

    assert not ces["MMMILP"][0].empty