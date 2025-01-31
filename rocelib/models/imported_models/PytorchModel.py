import pandas as pd
import numpy as np
import torch
from multimethod import multimethod

from exceptions.invalid_pytorch_model_error import InvalidPytorchModelError
from rocelib.models.TrainedModel import TrainedModel


class PytorchModel(TrainedModel):
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the PytorchModel by loading the saved PyTorch model.

        :param model_path: Path to the saved PyTorch model file (.pt or .pth)
        :param device: Device to load the model on ('cpu' or 'cuda')
        """
        if not model_path.endswith((".pt", ".pth")):
            raise TypeError(f"Invalid file type: {model_path}. Expected a '.pt' or '.pth' file.")

        self.device = torch.device(device)
        self.model = torch.load(model_path, map_location=self.device)  # Load full model

        self.check_model_is_torch_class(self.model)

        self.model.eval()  # Set to evaluation mode

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
        return instance

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the outcome using a PyTorch model from Pandas DataFrame input.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Predictions for each instance.
        """
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return pd.DataFrame(predictions)

    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts a single outcome as an integer.

        :param x: pd.DataFrame, Instance to predict.
        :return: int, Single integer prediction.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.values, dtype=torch.float32)
        return 0 if self.predict_proba(x).iloc[0, 0] > 0.5 else 1

    @multimethod
    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Probabilities for each class.
        """
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = torch.tensor(x.values, dtype=torch.float32)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return self.predict_proba(x)

    @multimethod
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts the class probabilities for a tensor input.

        :param X: torch.Tensor, Instances to predict.
        :return: torch.Tensor, Probabilities of each outcome.
        """
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

    def check_model_is_torch_class(self, model):
        if not isinstance(model, torch.nn.Module):
            raise InvalidPytorchModelError(
                f"Expected a PyTorch model (torch.nn.Module), but got {type(model).__name__}"
                "The loaded model is not a torch model, but instead relies on a self defined class. \n"
                "Please save your model again, ensuring to save the underlying torch model, rather than your wrapper class\n"
                "Then try and load your model in again")