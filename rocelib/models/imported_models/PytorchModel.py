import pandas as pd
import torch
import os
import numpy as np

from rocelib.exceptions.invalid_pytorch_model_error import InvalidPytorchModelError
from rocelib.models.TrainedModel import TrainedModel


class PytorchModel(TrainedModel):
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the PytorchModel by loading the saved PyTorch model.
        :param model_path: Path to the saved PyTorch model file (.pt or .pth)
        :param device: Device to load the model on ('cpu' or 'cuda')
        """
        if not isinstance(model_path, str):
            raise TypeError(f"Expected 'model_path' to be a string, got {type(model_path)}")

        # Check file format before checking existence
        if not model_path.endswith((".pt", ".pth")):
            raise ValueError(f"Invalid file format: {model_path}. Expected a .pt or .pth file.")

        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        try:
            self.device = torch.device(device)
            self.model = torch.load(model_path, map_location=self.device)

            if not isinstance(self.model, torch.nn.Module):
                raise TypeError("Expected a PyTorch model (torch.nn.Module), but got a different object.")

            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model from {model_path}: {e}")

    # exception handling if the user supplied an incorrect model
    def check_model_is_torch_class(self, model):
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(
                f"Expected a PyTorch model (torch.nn.Module), but got {type(model).__name__}"
                "The loaded model is not a torch model, but instead relies on a self defined class. \n"
                "Please save your model again, ensuring to save the underlying torch model, rather than your wrapper class\n"
                "Then try and load your model in again")

    @classmethod
    def from_model(cls, model, device: str = "cpu") -> 'PytorchModel':
        """
        Alternative constructor to initialize PytorchModel from a PyTorch model instance.

        :param model: A PyTorch model instance
        :param device: Device to load the model on ('cpu' or 'cuda')
        :return: An instance of PytorchModel
        """
        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance.device = torch.device(device)
        instance.model = model.to(instance.device)
        instance.model.eval()  # Set to evaluation mode

        (cls.input_dim, cls.hidden_dim, cls.output_dim) = get_model_dimensions_and_hidden_layers(model)

        return instance

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the outcome using a PyTorch model from Pandas DataFrame input.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Predictions for each instance.
        """
        # Ensure X is a tensor
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            X_tensor = torch.tensor(X_values, dtype=torch.float32).to(self.device)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device)  # Ensure it's on the correct device
        else:
            raise TypeError(f"Unsupported type for X: {type(X)}")

        return self.model(X_tensor).detach().cpu().numpy()

    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts a single outcome as an integer.

        :param x: pd.DataFrame, Instance to predict.
        :return: int, Single integer prediction.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.values, dtype=torch.float32)
        return 0 if self.predict_proba(x).iloc[0, 0] > 0.5 else 1

    def predict_proba(self, x: torch.Tensor) -> pd.DataFrame:
        """
        Predicts probabilities of outcomes.

        @param x: Input data as a torch tensor.
        @return: Probabilities of each outcome as a pandas DataFrame.
        """
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        res = self.model(x)
        res = pd.DataFrame(res.detach().numpy())

        temp = res[0]

        # The probability that it is 0 is 1 - the probability returned by model
        res[0] = 1 - res[0]

        # The probability it is 1 is the probability returned by the model
        res[1] = temp
        return res

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Evaluates the model using accuracy or other relevant metrics.

        :param X: pd.DataFrame, The feature variables.
        :param y: pd.DataFrame, The target variable.
        :return: Accuracy of the model as a float.
        """
        predictions = self.predict(X)
        accuracy = (predictions.view(-1) == torch.tensor(y.values)).float().mean()
        return accuracy.item()


def get_model_dimensions_and_hidden_layers(model):
    """
    Returns the input dimension, output dimension, and number of hidden layers in a PyTorch model.

    :param model: A PyTorch model instance
    :return: (input_dim, output_dim, num_hidden_layers)
    """
    layers = list(model.children())  # Get all model layers
    if not layers:
        raise ValueError("The model has no layers.")

    first_layer = layers[0]  # First layer (input)

    # Extract input and output dimensions
    input_dim = getattr(first_layer, 'in_features', None)

    hidden_dims = []
    for layer in layers:
        if hasattr(layer, 'out_features'):
            hidden_dims.append(layer.out_features)
    output_dim = hidden_dims.pop()

    return input_dim, hidden_dims, output_dim
