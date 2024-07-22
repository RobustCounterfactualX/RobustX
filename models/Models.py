from models.pytorch_models.SimpleNNModel import SimpleNNModel
from models.sklearn_models.DecisionTreeModel import DecisionTreeModel
from models.sklearn_models.LogisticRegressionModel import LogisticRegressionModel
from models.sklearn_models.SVMModel import SVMModel


def get_sklearn_model(name):
    if name == "log_reg":
        return LogisticRegressionModel()
    elif name == "decision_tree":
        return DecisionTreeModel()
    elif name == "svm":
        return SVMModel()


def get_pytorch_model(name):
    if name == "simple_nn":
        return SimpleNNModel()
