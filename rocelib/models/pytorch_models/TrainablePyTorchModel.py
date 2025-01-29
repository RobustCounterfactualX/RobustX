import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rocelib.models.TrainableModel import TrainableModel


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

    def train(self, X, y, epochs=100, **kwargs):
        """
        Trains the neural network model.

        @param X: Feature variables as a pandas DataFrame.
        @param y: Target variable as a pandas DataFrame.
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
        
        

    def set_weights(self, weights):
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

    def predict(self, X) -> pd.DataFrame:
        """
        Predicts outcomes for the given input data.

        @param X: Input data as a pandas DataFrame or torch tensor.
        @return: Predictions as a pandas DataFrame.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.values, dtype=torch.float32)
        return pd.DataFrame(self._model(X).detach().numpy())

    def predict_single(self, x) -> int:
        """
        Predicts the outcome for a single instance.

        @param x: Single input instance as a pandas DataFrame or torch tensor.
        @return: Prediction as an integer (0 or 1).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x.values, dtype=torch.float32)
        return 0 if self.predict_proba(x).iloc[0, 0] > 0.5 else 1

    def evaluate(self, X, y):
        """
        Evaluates the model's accuracy.

        @param X: Feature variables as a pandas DataFrame.
        @param y: Target variable as a pandas DataFrame.
        @return: Accuracy of the model as a float.
        """
        predictions = self.predict(X)
        accuracy = (predictions.view(-1) == torch.tensor(y.values)).float().mean()
        return accuracy.item()

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
        res = self._model(x)
        res = pd.DataFrame(res.detach().numpy())

        temp = res[0]

        # The probability that it is 0 is 1 - the probability returned by model
        res[0] = 1 - res[0]

        # The probability it is 1 is the probability returned by the model
        res[1] = temp
        return res

    def predict_proba_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts probabilities of outcomes using the model.

        @param x: Input data as a torch tensor.
        @return: Probabilities of each outcome as a torch tensor.
        """
        return self._model(x)

    def get_torch_model(self):
        """
        Retrieves the underlying PyTorch model.

        @return: The PyTorch model.
        """
        return self._model
