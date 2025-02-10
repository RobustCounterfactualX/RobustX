import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from robustx.lib.models.BaseModel import BaseModel


class CustomPyTorchModel(BaseModel):
    """
    A custom PyTorch model that can be trained, used for predictions, and evaluated for performance.

    Attributes
    ----------
    model: nn.Module
        The PyTorch model architecture.
    criterion: nn.CrossEntropyLoss
        The loss function used for training.
    optimizer: optim.Adam
        The optimizer used for training the model.

    Methods
    -------
    train(X: pd.DataFrame, y: pd.DataFrame, epochs: int = 10, batch_size: int = 32) -> None:
        Trains the model on the provided data for a specified number of epochs and batch size.

    predict(X: pd.DataFrame) -> pd.DataFrame:
        Predicts outcomes for multiple instances of the provided feature data.

    predict_single(X: pd.DataFrame) -> int:
        Predicts the outcome for a single instance of the provided feature data.

    predict_proba(X: pd.DataFrame) -> pd.DataFrame:
        Predicts the probability of each outcome for multiple instances.

    evaluate(X: pd.DataFrame, y: pd.DataFrame) -> float:
        Evaluates the model's accuracy on the provided feature and target data.
    """

    def __init__(self, model, criterion=nn.CrossEntropyLoss(), optimizer_class=optim.Adam, learning_rate=0.001):
        """
        Initializes the CustomPyTorchModel with a specified model, loss function, and optimizer.

        @param model: The PyTorch model architecture.
        @param criterion: The loss function, default is CrossEntropyLoss.
        @param optimizer_class: The optimizer class, default is Adam.
        @param learning_rate: The learning rate for the optimizer, default is 0.001.
        """
        super().__init__(model)
        self.criterion = criterion
        self.optimizer = optimizer_class(self._model.parameters(), lr=learning_rate)

    def train(self, X: pd.DataFrame, y: pd.DataFrame, epochs=10, batch_size=32, **kwargs):
        """
        Train the PyTorch model using the provided data.

        @param X: Feature variables as a pandas DataFrame.
        @param y: Target variable as a pandas DataFrame.
        @param epochs: Number of training epochs, default is 10.
        @param batch_size: Size of each mini-batch, default is 32.
        """
        # Convert pandas DataFrames to torch tensors
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)  # Assuming y is for classification

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                outputs = self._model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader)}")

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outcomes for the given features X (multiple instances).

        @param X: Input data as a pandas DataFrame.
        @return: Predictions as a pandas DataFrame.
        """
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return pd.DataFrame(predicted.numpy(), columns=['Prediction'])

    def predict_single(self, X: pd.DataFrame) -> int:
        """
        Predict the outcome for a single instance.

        @param X: Input data for a single instance as a pandas DataFrame.
        @return: Predicted class as an integer.
        """
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return int(predicted.item())

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the probability of outcomes for multiple instances.

        @param X: Input data as a pandas DataFrame.
        @return: Probabilities as a pandas DataFrame.
        """
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self._model(X_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return pd.DataFrame(probabilities.numpy())

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Evaluate the model's performance on a test set (X, y).

        @param X: Feature variables as a pandas DataFrame.
        @param y: Target variable as a pandas DataFrame.
        @return: Accuracy of the model as a float.
        """
        self._model.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        with torch.no_grad():
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_tensor).sum().item()
            total = y_tensor.size(0)
        accuracy = correct / total
        return accuracy
