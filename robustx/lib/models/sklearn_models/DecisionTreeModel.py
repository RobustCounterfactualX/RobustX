from sklearn.tree import DecisionTreeClassifier

from robustx.lib.models.sklearn_models.SklearnModel import SklearnModel


class DecisionTreeModel(SklearnModel):
    """
    A Decision Tree Classifier model wrapper for scikit-learn.

    Inherits from SklearnModel and initializes a DecisionTreeClassifier as the underlying model.
    """

    def __init__(self):
        super().__init__(DecisionTreeClassifier())
