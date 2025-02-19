from rocelib.tasks.TaskBuilder import TaskBuilder
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.imported_models.PytorchModel import PytorchModel


def test_add_multiple_models():
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_keras_model(34, 8, 1, dl).add_data(dl).build()

    assert isinstance(ct.model, PytorchModel)
    assert len(ct.mm_models) == 2
