from rocelib.tasks.TaskBuilder import TaskBuilder
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.imported_models.PytorchModel import PytorchModel


def test_add_multiple_models():
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl, "my_model").add_keras_model(34, 8, 1, dl).add_keras_model(34, 8, 1, dl).add_data(dl).build()

    assert isinstance(ct.model, PytorchModel)
    assert len(ct.mm_models.keys()) == 3
    assert list(ct.mm_models.keys()) == ["my_model", "keras_model_0", "keras_model_1"]
    assert ct.mm_flag

def test_add_singular_model():
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    assert len(ct.mm_models.keys()) == 1
    assert not ct.mm_flag

def test_generate_mm() -> None:
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()

    recourse_methods = ["NNCE","KDTreeNNCE","MCE"]
    ces = ct.generate_mm(recourse_methods)

    for recourse_method in recourse_methods:
        for model in ["pytorch_model_0", "pytorch_model_1"]:
            assert not ces[recourse_method][model][0].empty



    #TODO: for some reason generate_mm doesn't work for BinaryLinearSearch
