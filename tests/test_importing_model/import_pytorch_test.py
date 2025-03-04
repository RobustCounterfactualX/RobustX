import os

import pytest
import torch

from rocelib.models.imported_models.PytorchModel import PytorchModel
from ..test_helpers.TestingModels import TestingModels


def test_imported_pytorch_model_file_predict_single_same_as_original():
    # Create Model
    testing_models = TestingModels()
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    # Save Entire Model
    torch.save(ct.model.model, "./model.pt")

    # Load Model
    imported_model = PytorchModel("./model.pt")

    # Test Single Prediction
    sample_input = torch.randn(1, 34)  # Adjust shape based on model input
    original_prediction = ct.model.model(sample_input)
    imported_prediction = imported_model.model(sample_input)

    assert torch.equal(original_prediction, imported_prediction), "Predictions do not match"

    os.remove('./model.pt')



def test_imported_pytorch_model_file_predict_all_same_as_original():
    # Create Model
    testing_models = TestingModels()
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    # Save Entire Model
    torch.save(ct.model.model, "./model.pt")

    # Load Model
    imported_model = PytorchModel("./model.pt")

    # Test Multiple Predictions
    test_data = torch.randn(10, 34)  # Adjust shape based on model input
    original_predictions = ct.model.model(test_data)
    imported_predictions = imported_model.model(test_data)

    assert torch.equal(original_predictions, imported_predictions), "Batch predictions do not match"
    os.remove('./model.pt')



def test_throws_file_not_found_error():
    with pytest.raises(ValueError, match="Model file not found"):
        PytorchModel("./garbage.pt")


def test_throws_wrong_file_type_error():
    with pytest.raises(ValueError, match="Invalid file format"):
        PytorchModel("./test.txt")


def test_throws_error_if_underlying_model_not_pytorch():
    testing_models = TestingModels()
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    # Save the model incorrectly (saving wrapper instead of actual model)
    torch.save(ct.model, "./model.pt")

    with pytest.raises(RuntimeError, match="Expected a PyTorch model"):
        PytorchModel("./model.pt")

