from sklearn.linear_model import LogisticRegression
from models.sklearn_models.SklearnModel import SklearnModel


class LogisticRegressionModel(SklearnModel):

    def __init__(self):
        super().__init__(LogisticRegression(solver='liblinear'))
