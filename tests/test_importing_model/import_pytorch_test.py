from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.models.imported_models.PytorchModel import PytorchModel
import torch
import os


def test_import_pytorch_model_file() -> None:
    #Create Model
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    #Save Model
    torch.save(ct.model.get_torch_model(), "./model.pt")

    #Import Model

    trained_model = PytorchModel("./model.pt")

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)

    os.remove('./model.pt')




def test_import_pytorch_model_instance() -> None:
    #Create Model
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    torch_model = ct.model.get_torch_model()

    #Import Model
    trained_model = PytorchModel.from_model(torch_model)

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)