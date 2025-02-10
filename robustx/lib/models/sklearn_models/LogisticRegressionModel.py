from sklearn.linear_model import LogisticRegression
from robustx.lib.models.sklearn_models.SklearnModel import SklearnModel


class LogisticRegressionModel(SklearnModel):
    """
    A Logistic Regression Classifier model wrapper for scikit-learn.

    Inherits from SklearnModel and initializes LogisticRegression as the underlying model.
    """

    def __init__(self):
        super().__init__(LogisticRegression(solver='liblinear'))
