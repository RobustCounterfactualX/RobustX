from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.tasks.Task import Task

from rocelib.tasks.TaskBuilder import TaskBuilder
from enums.dataset_enums import Dataset
from enums.model_enums import ModelType


class TestingModels:
    def __init__(self):
        # dictionary of singletons (dataset, model_type, args) -> Task
        self.models = {}

    def get(self, training_dataset: str, dataset: str, model_type: str, *args) -> Task:
        if (training_dataset, model_type, args) not in self.models:
            print(self.models)
            return self.create_and_train(training_dataset, model_type, dataset, args)
        else:
            print(self.models)

            return self.models[(training_dataset, model_type, args)]

    def create_and_train(self, training_dataset: str, model_type: str, dataset: str, args) -> Task:
        tb = TaskBuilder()
        dl_training = get_example_dataset(training_dataset)
        dl = get_example_dataset(dataset)

        if model_type == "pytorch":
            if len(args) < 2:
                raise TypeError(
                    f"Expected at least 2 layer dimension, received {len(args)}: "
                    f"{args}"
                )
            input_layer = args[0]
            hidden_layer = list(args[1:-1])
            output_layer = args[2]

            tb.add_pytorch_model(input_layer, hidden_layer, output_layer, dl_training)
        elif model_type == "keras":
            if len(args) < 2:
                raise TypeError(
                    f"Expected at least 2 layer dimension, received {len(args)}: "
                    f"{args}"
                )
            input_layer = args[0]
            hidden_layer = list(args[1:-1])
            output_layer = args[2]

            tb.add_keras_model(input_layer, hidden_layer, output_layer, dl_training)
        elif model_type == "decision tree":
            tb.add_sklearn_model(model_type, dl_training)
        elif model_type == "logistic regression":
            tb.add_sklearn_model(model_type, dl_training)
        elif model_type == "svm":
            tb.add_sklearn_model(model_type, dl_training)
        else:
            tb.add_model_from_path(model_type)
        
        tb.add_data(dl)
        
        ct = tb.build()

        self.models[(training_dataset, model_type, args)] = ct

        return ct