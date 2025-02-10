from sklearn.svm import SVC
from robustx.lib.models.sklearn_models.SklearnModel import SklearnModel


class SVMModel(SklearnModel):
    """
    A SVM model wrapper for scikit-learn.

    Inherits from SklearnModel and initializes SVC as the underlying model.
    """

    def __init__(self):
        super().__init__(SVC())
