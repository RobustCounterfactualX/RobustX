from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.TrainedModel import TrainedModel
from rocelib.models.TrainableModel import TrainableModel
from rocelib.models.imported_models.PytorchModel import PytorchModel
from rocelib.models.imported_models.SKLearnModel import SKLearnModel
from rocelib.models.imported_models.KerasModel import KerasModel
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.models.keras_models.TrainableKerasModel import TrainableKerasModel
from rocelib.models.sklearn_models.TrainableDecisionTreeModel import TrainableDecisionTreeModel
from rocelib.models.sklearn_models.TrainableLogisticRegressionModel import TrainableLogisticRegressionModel
from rocelib.models.sklearn_models.TrainableSVMModel import TrainableSVMModel

import os

from rocelib.tasks.Task import Task
from rocelib.tasks.ClassificationTask import ClassificationTask


class TaskBuilder:
    """
    Builder pattern for creating Task instances, ensuring a model and training data are provided.
    """

    def __init__(self):
        """
        Initializes the TaskBuilder with no model or dataset set.
        """
        self._model = None
        self._data = None
        self._task_type = ClassificationTask  # Default task type will be ClassificationTask

    def set_task_type(self, task_type: str): #TODO
        """
        Sets the task type. Defaults to ClassificationTask if not set.

        @param task_type: The Task class to instantiate (e.g., ClassificationTask).
        @return: self (TaskBuilder) for method chaining.
        """
        task_types = {
            "classification": ClassificationTask,
        }
        if task_type not in task_types:
            raise ValueError(f"Unsupported task type: {task_type}. Please use one of: {task_types.keys()}")

        self._task_type = task_types[task_type]

        return self

    def add_model_from_path(self, model_path: str):
        """
        Adds a pre-trained model from a path.

        @param model_path: Path to the saved PyTorch model.
        @return: self (TaskBuilder) for method chaining.
        """
        _, ext = os.path.splitext(model_path)
    
        model_classes = {
            ".pt": PytorchModel,
            ".pth": PytorchModel,
            ".keras": KerasModel,
            ".pkl": SKLearnModel,
        }

        if ext not in model_classes:
            raise ValueError(f"Unsupported model format: {ext}")

        self._model = model_classes[ext](model_path)
        return self

    def add_pytorch_model(self, input_dim, hidden_dim, output_dim, training_data: DatasetLoader):
        """
        Adds a new Trainable PyTorch Model.

        @param input_dim: Number of input features.
        @param hidden_dim: List specifying the number of neurons in each hidden layer.
        @param output_dim: Number of output neurons.
        @return: self (TaskBuilder) for method chaining.
        """
        self._model = TrainablePyTorchModel(input_dim, hidden_dim, output_dim).train(training_data.X, training_data.y)
        return self
    
    def add_sklearn_model(self, model_type: str, training_data: DatasetLoader): # TODO - enum of sk learn types
        """
        Adds a new Trainable PyTorch Model.

        @param input_dim: Number of input features.
        @param hidden_dim: List specifying the number of neurons in each hidden layer.
        @param output_dim: Number of output neurons.
        @return: self (TaskBuilder) for method chaining.
        """
        sk_types = {
            "decision tree": TrainableDecisionTreeModel,
            "logistic regression": TrainableLogisticRegressionModel,
            "svm": TrainableSVMModel,
        }
        if model_type not in sk_types:
            raise ValueError(f"Unsupported sk model type: {model_type}. Please use one of: {sk_types.keys()}")

        self._model = sk_types[model_type]().train(training_data.X, training_data.y)

        return self
    
    def add_keras_model(self, input_dim, hidden_dim, output_dim, training_data: DatasetLoader):
        """
        Adds a new Trainable PyTorch Model.

        @param input_dim: Number of input features.
        @param hidden_dim: List specifying the number of neurons in each hidden layer.
        @param output_dim: Number of output neurons.
        @return: self (TaskBuilder) for method chaining.
        """
        self._model = TrainableKerasModel(input_dim, hidden_dim, output_dim).train(training_data.X, training_data.y)
        return self


    def add_data(self, data: DatasetLoader):
        self._data = data
        return self

    def build(self) -> Task:
        """
        Validates and constructs the Task instance.

        @return: Task instance.
        """
        if not self._model:
            raise ValueError("A model must be added before building the Task.")

        if not self._data:
            raise ValueError("Data must be added before building the Task.")
        
        return self._task_type(self._model, self._data)