import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.BaseModel import BaseModel


class SimpleNNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X, y, epochs=100):
        self.model.train()
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, X) -> pd.DataFrame:
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.values, dtype=torch.float32)
        return pd.DataFrame(self.model(X).detach().numpy())

    def predict_single(self, x) -> int:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.values, dtype=torch.float32)
        return 0 if self.predict_proba(x).iloc[0, 0] > 0.5 else 1

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = (predictions.view(-1) == torch.tensor(y.values)).float().mean()
        return accuracy.item()

    def predict_proba(self, x: torch.Tensor) -> pd.DataFrame:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.values, dtype=torch.float32)
        res = self.model(x)
        res = pd.DataFrame(res.detach().numpy())

        # The probability that it is 0 is 1 - the probability returned by model
        res[0] = 1 - res[0]

        # The probability it is 1 is the probability returned by the model
        res[1] = 1 - res.iloc[0]
        return res
