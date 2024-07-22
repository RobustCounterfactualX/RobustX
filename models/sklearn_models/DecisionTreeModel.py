from sklearn.tree import DecisionTreeClassifier
from models.sklearn_models.SklearnModel import SklearnModel


class DecisionTreeModel(SklearnModel):

    def __init__(self):
        super().__init__(DecisionTreeClassifier())
