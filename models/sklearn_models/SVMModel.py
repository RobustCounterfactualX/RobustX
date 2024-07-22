from sklearn.svm import SVC
from models.sklearn_models.SklearnModel import SklearnModel


class SVMModel(SklearnModel):

    def __init__(self):
        super().__init__(SVC())
