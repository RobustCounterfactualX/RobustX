import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.BaseModel import BaseModel


class SimpleNNModel(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        super().__init__(self.__create_model())
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def __create_model(self):
        model = nn.Sequential()

        if self.hidden_dim:
            model.append(nn.Linear(self.input_dim, self.hidden_dim[0]))
            model.append(nn.ReLU())

            for i in range(0, len(self.hidden_dim) - 1):
                model.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
                model.append(nn.ReLU())

            model.append(nn.Linear(self.hidden_dim[-1], self.output_dim))

        else:
            model.append(nn.Linear(self.input_dim, self.output_dim))

        if self.output_dim == 1:
            model.append(nn.Sigmoid())

        return model

    def train(self, X, y, epochs=100):
        self.model.train()
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self._model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def set_weights(self, weights):
        """Set custom weights for the model."""
        # Initialize layer index for Sequential model
        layer_idx = 0
        for i, layer in enumerate(self._model):
            if isinstance(layer, nn.Linear):
                # Extract weights and biases from the weights dictionary
                with torch.no_grad():
                    layer.weight = nn.Parameter(weights[f'fc{layer_idx}_weight'])
                    layer.bias = nn.Parameter(weights[f'fc{layer_idx}_bias'])
                layer_idx += 1

    def predict(self, X) -> pd.DataFrame:
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.values, dtype=torch.float32)
        return pd.DataFrame(self._model(X).detach().numpy())

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
        res = self._model(x)
        res = pd.DataFrame(res.detach().numpy())

        # The probability that it is 0 is 1 - the probability returned by model
        res[0] = 1 - res[0]

        # The probability it is 1 is the probability returned by the model
        res[1] = 1 - res.iloc[0]
        return res

    def get_torch_model(self):
        return self._model