import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rocelib.models.TrainableModel import TrainableModel
from rocelib.models.TrainedModel import TrainedModel
from rocelib.models.imported_models.PytorchModel import PytorchModel



class TrainablePyTorchModel(TrainableModel):
    """
    A simple neural network model using PyTorch. This model can be customized with different numbers of hidden layers and units.

    Attributes
    ----------
    input_dim: int
        The number of input features for the model.
    hidden_dim: list of int
        The number of units in each hidden layer. An empty list means no hidden layers.
    output_dim: int
        The number of output units for the model.
    criterion: nn.BCELoss
        The loss function used for training.
    optimizer: optim.Adam
        The optimizer used for training the model.

    Methods
    -------
    __create_model() -> nn.Sequential:
        Creates and returns the PyTorch model architecture.

    train(X: pd.DataFrame, y: pd.DataFrame, epochs: int = 100) -> None:
        Trains the model on the provided data for a specified number of epochs.

    set_weights(weights: Dict[str, torch.Tensor]) -> None:
        Sets custom weights for the model.

    predict(X: pd.DataFrame) -> pd.DataFrame:
        Predicts the outcomes for the provided instances.

    predict_single(x: pd.DataFrame) -> int:
        Predicts the outcome of a single instance and returns an integer.

    evaluate(X: pd.DataFrame, y: pd.DataFrame) -> float:
        Evaluates the model's accuracy on the provided data.

    predict_proba(x: torch.Tensor) -> pd.DataFrame:
        Predicts the probability of outcomes for the provided instances.

    predict_proba_tensor(x: torch.Tensor) -> torch.Tensor:
        Predicts the probability of outcomes for the provided instances using tensor input.

    get_torch_model() -> nn.Module:
        Returns the underlying PyTorch model.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the TrainablePyTorchModel with specified dimensions.

        @param input_dim: Number of input features.
        @param hidden_dim: List specifying the number of neurons in each hidden layer.
        @param output_dim: Number of output neurons.
        """
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

    def train(self, X, y, epochs=100, **kwargs) -> TrainedModel:
        """
        Trains the neural network model.

        @param dataset_loader: Feature and target variables as a DatasetLoader
        @param epochs: Number of training epochs.
        """
        self.model.train()
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self._model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
        return PytorchModel.from_model(self.get_torch_model())
        

    def set_weights(self, weights) -> TrainedModel:
        """
        Sets custom weights for the model.

        @param weights: Dictionary containing weights and biases for each layer.
        """
        # Initialize layer index for Sequential model
        layer_idx = 0
        for i, layer in enumerate(self._model):
            if isinstance(layer, nn.Linear):
                # Extract weights and biases from the weights dictionary
                with torch.no_grad():
                    layer.weight = nn.Parameter(weights[f'fc{layer_idx}_weight'])
                    layer.bias = nn.Parameter(weights[f'fc{layer_idx}_bias'])
                layer_idx += 1
        return PytorchModel.from_model(self.get_torch_model())


    def get_torch_model(self):
        """
        Retrieves the underlying PyTorch model.

        @return: The PyTorch model.
        """
        return self._model
