from rocelib.models.sklearn_models.TrainableDecisionTreeModel import TrainableDecisionTreeModel
from rocelib.models.sklearn_models.TrainableLogisticRegressionModel import TrainableLogisticRegressionModel
from rocelib.models.sklearn_models.TrainableSVMModel import TrainableSVMModel


def get_sklearn_model(name: str):
    """
    Retrieves an instance of a scikit-learn model based on the provided name.

    @param name: The name of the desired model. Options are:
        - "log_reg" for Logistic Regression
        - "decision_tree" for Decision Tree
        - "svm" for Support Vector Machine

    @return: An instance of the requested scikit-learn model. The model class should be a subclass of TrainableModel.

    @raises ValueError: If the provided model name does not match any of the predefined options.
    """
    if name == "log_reg":
        return TrainableLogisticRegressionModel()
    elif name == "decision_tree":
        return TrainableDecisionTreeModel()
    elif name == "svm":
        return TrainableSVMModel()
    else:
        raise ValueError(f"Unknown model name: {name}")
