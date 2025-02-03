from sklearn.linear_model import LogisticRegression

from rocelib.models.sklearn_models.TrainableSKLearnModel import TrainableSKLearnModel


class TrainableLogisticRegressionModel(TrainableSKLearnModel):
    """
    A Logistic Regression Classifier model wrapper for scikit-learn.

    Inherits from TrainableSKLearnModel and initializes LogisticRegression as the underlying model.
    """

    def __init__(self):
        super().__init__(LogisticRegression(solver='liblinear'))
