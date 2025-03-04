import keras
import pandas as pd
import os
import numpy as np

from rocelib.models.TrainedModel import TrainedModel


class KerasModel(TrainedModel):
    def __init__(self, model_path: str):
        """
        Initialize the KerasModel by loading the saved Keras model.
        :param model_path: Path to the saved Keras model file (.keras)
        """
        if not isinstance(model_path, str):
            raise TypeError(f"Expected 'model_path' to be a string, got {type(model_path)}")

        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        if not model_path.endswith(".keras"):
            raise ValueError(f"Invalid file format: {model_path}. Expected a .keras file.")

        try:
            self.model = keras.saving.load_model(model_path)
            (self.input_dim, self.hidden_dim, self.output_dim) = get_model_dimensions_and_hidden_layers(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model from {model_path}: {e}")

    @classmethod
    def from_model(cls, model: keras.Model) -> 'KerasModel':
        """
        Alternative constructor to initialize KerasModel from a Keras model instance.
        :param model: A Keras model instance
        :return: An instance of KerasModel
        """
        if not isinstance(model, keras.Model):
            raise TypeError(f"Expected a keras.Model, got {type(model)}")
        instance = cls.__new__(cls)
        instance.model = model
        (cls.input_dim, cls.hidden_dim, cls.output_dim) = get_model_dimensions_and_hidden_layers(model)
        return instance

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the outcome using a Keras model from Pandas DataFrame input.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Predictions for each instance.
        """
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions)

    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts a single outcome as an integer.

        :param x: pd.DataFrame, Instance to predict.
        :return: int, Single integer prediction.
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T  # Convert Series to row DataFrame

        expected_input = self.model.input_shape[-1]

        if x.shape != (1, expected_input):
            raise ValueError(
                f"Expected input shape (1, {expected_input}), got {x.shape}. "
                "For multiple predictions, use the 'predict' method instead "
            )
        prediction = self.predict(x)
        return 0 if prediction.iloc[0, 0] > 0.5 else 1

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities.

        :param x: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Probabilities for each class.
        """
        probabilities = self.model.predict(x)
        probabilities_df = pd.DataFrame(probabilities)
        probabilities_df[0] = 1 - probabilities_df[0]
        probabilities_df[1] = 1 - probabilities_df[0]
        return probabilities_df

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Evaluates the model using accuracy or other relevant metrics.

        :param X: pd.DataFrame, The feature variables.
        :param y: pd.DataFrame, The target variable.
        :return: Accuracy of the model as a float.
        """
        _, accuracy = self.model.evaluate(X, y)
        return accuracy

def get_model_dimensions_and_hidden_layers(model):
    """
    Returns the input dimension, output dimension, and number of hidden layers in a Keras model.

    :param model: A Keras model instance
    :return: (input_dim, hidden_dims, output_dim)
    """
    layers = model.layers  # Get all model layers
    if not layers:
        raise ValueError("The model has no layers.")

    # Extract input and output dimensions from the model itself
    input_dim = model.input_shape[-1]  # Get input dimension
    output_dim = model.output_shape[-1]  # Get output dimension

    # Extract hidden layer sizes (excluding input & output layers)
    hidden_dims = [layer.units for layer in layers if hasattr(layer, "units")]

    return input_dim, hidden_dims[:-1], output_dim  # Exclude last hidden dim (output layer)