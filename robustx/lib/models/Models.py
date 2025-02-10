from robustx.lib.models.sklearn_models.DecisionTreeModel import DecisionTreeModel
from robustx.lib.models.sklearn_models.LogisticRegressionModel import LogisticRegressionModel
from robustx.lib.models.sklearn_models.SVMModel import SVMModel


def get_sklearn_model(name: str):
    """
    Retrieves an instance of a scikit-learn model based on the provided name.

    @param name: The name of the desired model. Options are:
        - "log_reg" for Logistic Regression
        - "decision_tree" for Decision Tree
        - "svm" for Support Vector Machine

    @return: An instance of the requested scikit-learn model. The model class should be a subclass of BaseModel.

    @raises ValueError: If the provided model name does not match any of the predefined options.
    """
    if name == "log_reg":
        return LogisticRegressionModel()
    elif name == "decision_tree":
        return DecisionTreeModel()
    elif name == "svm":
        return SVMModel()
    else:
        raise ValueError(f"Unknown model name: {name}")
