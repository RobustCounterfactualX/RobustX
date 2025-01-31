from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.models.imported_models.PytorchModel import PytorchModel
import torch
import os
import pytest


def trained_classification_task():
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()
    return ct

def test_imported_pytorch_model_file_predict_single_same_as_original() -> None:
    #Create Model
    ct = trained_classification_task()

    #Save Model
    torch.save(ct.model.get_torch_model(), "./model.pt")

    #Import Model

    trained_model = PytorchModel("./model.pt")

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)

    os.remove('./model.pt')

def test_imported_pytorch_model_file_predict_all_same_as_original() -> None:
    #Create Model
    ct = trained_classification_task()

    #Save Model
    torch.save(ct.model.get_torch_model(), "./model.pt")

    #Import Model
    trained_model = PytorchModel("./model.pt")

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))
    assert predictions_1.equals(predictions_2)

    os.remove('./model.pt')


def test_imported_pytorch_model_from_instance_predict_single_same_as_original() -> None:
    #Create Model
    ct = trained_classification_task()

    torch_model = ct.model.get_torch_model()

    #Import Model
    trained_model = PytorchModel.from_model(torch_model)

    for _, instance in ct.training_data.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)

def test_imported_pytorch_model_from_instance_predict_all_same_as_original() -> None:
    #Create Model
    ct = trained_classification_task()

    torch_model = ct.model.get_torch_model()

    #Import Model
    trained_model = PytorchModel.from_model(torch_model)

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))
    assert predictions_1.equals(predictions_2)


def test_imported_pytorch_model_from_instance_predict_all_same_as_original() -> None:
    #Create Model
    ct = trained_classification_task()

    torch_model = ct.model.get_torch_model()

    #Import Model
    trained_model = PytorchModel.from_model(torch_model)

    predictions_1 = ct.model.predict(ct.training_data.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.training_data.data.drop("target", axis=1))
    assert predictions_1.equals(predictions_2)

def test_throws_file_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        trained_model = PytorchModel("./garbage.pt")

def test_throws_wrong_file_type_error() -> None:
    with pytest.raises(TypeError):
        trained_model = PytorchModel("./test.txt")

def test_throws_error_if_underlying_model_not_pytorch() -> None:
    pass
    #TODO


#TODO: add error handling
#model relies on some other class

