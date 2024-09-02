import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from models.BaseModel import BaseModel


class CustomPyTorchModel(BaseModel):
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), optimizer_class=optim.Adam, learning_rate=0.001):
        super().__init__(model)
        self.criterion = criterion
        self.optimizer = optimizer_class(self._model.parameters(), lr=learning_rate)

    def train(self, X: pd.DataFrame, y: pd.DataFrame, epochs=10, batch_size=32):
        """
        Train the PyTorch model using provided data.
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
        """
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return pd.DataFrame(predicted.numpy(), columns=['Prediction'])

    def predict_single(self, X: pd.DataFrame) -> int:
        """
        Predict outcome for a single instance.
        """
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            outputs = self._model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return int(predicted.item())

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict probability of outcomes for multiple instances.
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
