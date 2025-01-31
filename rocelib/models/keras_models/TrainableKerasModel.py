from typing import Dict

import pandas as pd
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.models import Sequential
from keras.optimizers import Adam

from rocelib.models.TrainableModel import TrainableModel
from rocelib.models.TrainedModel import TrainedModel
from rocelib.models.imported_models.KerasModel import KerasModel




class TrainableKerasModel(TrainableModel):
    """
    A simple feedforward neural network model using Keras for binary classification.

    This model includes one hidden layer with ReLU activation and an output layer with a sigmoid activation function.
    It utilizes the Adam optimizer with a learning rate of 0.001 and the binary cross-entropy loss function.

    Attributes
    ----------
    model : keras.models.Sequential
        The Keras Sequential model instance containing the neural network architecture.

    Methods
    -------
    __init__(input_dim: int, hidden_dim: int, output_dim: int):
        Initializes the neural network model with the specified dimensions.

    train(X: pd.DataFrame, y: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> None:
        Trains the model using the provided feature and target variables.

    predict(X: pd.DataFrame) -> pd.DataFrame:
        Predicts the outcomes for a set of instances.

    predict_single(x: pd.DataFrame) -> int:
        Predicts the outcome for a single instance and returns the class label.

    evaluate(X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        Evaluates the model on the provided feature and target variables.

    predict_proba(x: pd.DataFrame) -> pd.DataFrame:
        Predicts the probabilities of outcomes for a set of instances.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        @param input_dim: The number of input features for the model.
        @param hidden_dim: The number of neurons in the hidden layer.
        @param output_dim: The number of output neurons (1 for binary classification).
        """
        model = Sequential([
            Dense(hidden_dim, input_dim=input_dim, activation='relu'),
            Dense(output_dim, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=[Accuracy()])
        super().__init__(model)

    def train(self, X: pd.DataFrame, y: pd.DataFrame, epochs: int = 100, batch_size: int = 32, **kwargs) -> TrainedModel:
        """
        Trains the model on the provided data.

        @param X: The feature variables as a DataFrame.
        @param y: The target variable as a DataFrame.
        @param epochs: The number of epochs to train the model (default is 100).
        @param batch_size: The batch size used in training (default is 32).
        """
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        return KerasModel.from_model(self.get_keras_model())

    def get_keras_model(self):
        """
        Retrieves the underlying Keras model.

        @return: The Keras model.
        """
        return self._model
