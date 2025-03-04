import joblib
import pandas as pd
import os
import numpy as np
from sklearn.base import BaseEstimator
from rocelib.models.TrainedModel import TrainedModel

class SKLearnModel(TrainedModel):
    def __init__(self, model_path: str):
        """
        Initialize the SKLearnModel by loading the saved sklearn model.
        :param model_path: Path to the saved sklearn model file (.pkl)
        """
        if not isinstance(model_path, str):
            raise TypeError(f"Expected 'model_path' to be a string, got {type(model_path)}")

        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")

        if not model_path.endswith(".pkl"):
            raise ValueError(f"Invalid file format: {model_path}. Expected a .pkl file.")

        try:
            self.model = joblib.load(model_path)
            self.check_model_is_sklearn_class(self.model)
            (self.input_dim, self.hidden_dim, self.output_dim) = get_model_dimensions_and_hidden_layers(self.model)

        except Exception as e:
            raise RuntimeError(f"Failed to load Scikit-learn model from {model_path}: {e}")

    def check_model_is_sklearn_class(self, model):
        if not isinstance(model, BaseEstimator):
            raise TypeError(f"Expected an sklearn model (BaseEstimator), but got {type(model).__name__}.")

    @classmethod
    def from_model(cls, model: BaseEstimator) -> 'SKLearnModel':
        """
        Alternative constructor to initialize SKLearnModel from an existing sklearn model.

        :param model: A trained sklearn model instance
        :return: An instance of SKLearnModel
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError(f"Expected 'model' to be an instance of sklearn BaseEstimator, but got {type(model)}")

        instance = cls.__new__(cls)  # Create a new instance without calling __init__
        instance.model = model
        (cls.input_dim, cls.hidden_dim, cls.output_dim) = get_model_dimensions_and_hidden_layers(model)
        return instance

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the outcome using an sklearn model from Pandas DataFrame input.

        :param X: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Predictions for each instance.
        """
        predictions = self.model.predict(X)
        return pd.DataFrame(predictions, columns=["prediction"])

    def predict_single(self, x: pd.DataFrame) -> int:
        """
        Predicts a single outcome as an integer.

        :param x: pd.DataFrame, Instance to predict.
        :return: int, Single integer prediction.
        """
        if isinstance(x, pd.Series):
            x = x.to_frame().T  # Convert Series to single-row DataFrame

        # Convert DataFrame to NumPy array (forcing 2D)
        x_array = np.array(x).reshape(1, -1)  # Ensure (1, n_features) shape

        prediction = self.model.predict(x_array)  # Ensure sklearn gets the correct format
        return int(prediction[0])

    def predict_proba(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probabilities.

        :param x: pd.DataFrame, Instances to predict.
        :return: pd.DataFrame, Probabilities for each class.
        """
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(x)
            return pd.DataFrame(probabilities, columns=[0, 1])
        else:
            raise AttributeError("This model does not support probability prediction.")

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        """
        Evaluates the model using accuracy score.

        :param X: pd.DataFrame, The feature variables.
        :param y: pd.DataFrame, The target variable.
        :return: Accuracy of the model as a float.
        """
        accuracy = self.model.score(X, y)
        return accuracy

    def check_model_is_sklearn_class(self, model):
        """
        Check if the loaded model is an sklearn-compatible model.

        :param model: Model instance to check.
        """
        if not isinstance(model, BaseEstimator):
            raise TypeError(
                f"Expected an sklearn model (BaseEstimator), but got {type(model).__name__}. "
                "Ensure you are loading a properly trained sklearn model."
            )


def get_model_dimensions_and_hidden_layers(model):
    """
    Returns the input dimension, output dimension, and number of hidden layers in an sklearn model.

    :param model: An sklearn model instance
    :return: (input_dim, output_dim, hidden_dims)
    """

    if hasattr(model, "coef_"):  # Covers Logistic Regression, Linear Regression, SVM (linear kernel)
        input_dim = model.coef_.shape[1]
        output_dim = model.coef_.shape[0]
        hidden_dims = []

    elif hasattr(model, "coefs_"):  # Covers MLPClassifier and MLPRegressor
        input_dim = model.coefs_[0].shape[0]
        output_dim = model.coefs_[-1].shape[1]
        hidden_dims = [layer.shape[1] for layer in model.coefs_[:-1]]

    elif hasattr(model, "support_vectors_"):  # Covers SVM (SVC, SVR)
        input_dim = model.support_vectors_.shape[1]
        output_dim = 1 if hasattr(model, "dual_coef_") else model.n_classes_
        hidden_dims = []

    elif hasattr(model, "tree_"):  # Covers DecisionTreeClassifier and DecisionTreeRegressor
        input_dim = model.n_features_in_
        output_dim = model.n_classes_ if hasattr(model, "n_classes_") else 1
        hidden_dims = []  # Decision trees do not have hidden layers

    elif hasattr(model, "estimators_"):  # Covers RandomForest, GradientBoosting, etc.
        input_dim = model.n_features_in_
        output_dim = model.n_classes_ if hasattr(model, "n_classes_") else 1
        hidden_dims = []  # Ensembles don’t have traditional hidden layers

    else:
        raise ValueError(f"Unsupported sklearn model type: {type(model)}")

    return input_dim, hidden_dims, output_dim