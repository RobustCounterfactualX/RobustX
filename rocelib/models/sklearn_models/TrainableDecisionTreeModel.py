from sklearn.tree import DecisionTreeClassifier

from rocelib.models.sklearn_models.TrainableSKLearnModel import TrainableSKLearnModel


class TrainableDecisionTreeModel(TrainableSKLearnModel):
    """
    A Decision Tree Classifier model wrapper for scikit-learn.

    Inherits from TrainableSKLearnModel and initializes a DecisionTreeClassifier as the underlying model.
    """

    def __init__(self):
        super().__init__(DecisionTreeClassifier())
