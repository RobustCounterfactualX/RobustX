from sklearn.svm import SVC

from rocelib.models.sklearn_models.TrainableSKLearnModel import TrainableSKLearnModel


class TrainableSVMModel(TrainableSKLearnModel):
    """
    A SVM model wrapper for scikit-learn.

    Inherits from TrainableSKLearnModel and initializes SVC as the underlying model.
    """

    def __init__(self):
        super().__init__(SVC())
