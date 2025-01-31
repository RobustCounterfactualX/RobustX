from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.models.imported_models.PytorchModel import PytorchModel
import torch
import os


def test_imported_pytorch_model_file_same_as_original() -> None:
    #Create Model
    model = TrainablePyTorchModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    trained_model = model.train(dl.X, dl.y)

    ct = ClassificationTask(trained_model, dl)



    #Save Model
    torch.save(ct.model.model, "./model.pt")

    #Import Model

    imported_model = PytorchModel("./model.pt")

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == imported_model.predict_single(instance_x)

    os.remove('./model.pt')




def test_imported_pytorch_model_from_instance_same_as_original() -> None:
    #Create Model
    model = TrainablePyTorchModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    dl.default_preprocess()
    trained_model = model.train(dl.X, dl.y)

    ct = ClassificationTask(trained_model, dl)


    torch_model = ct.model.model

    #Import Model
    imported_model = PytorchModel.from_model(torch_model)

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == imported_model.predict_single(instance_x)