from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.tasks.ClassificationTask import ClassificationTask
from enums.dataset_enums import Dataset
from enums.model_enums import ModelType


class TestingModels:
    def __init__(self):
        # dictionary of singletons
        self.models = {}

    def get(self, dataset: Dataset, model_type: ModelType, *args) -> (ClassificationTask, DatasetLoader):
        if (dataset, model_type, args) not in self.models:
            return self.create_and_train(dataset, model_type, args)
        else:
            return self.models[(dataset, model_type, args)]

    def create_and_train(self, dataset: Dataset, model_type: ModelType, args) -> (ClassificationTask, DatasetLoader):
        if dataset == Dataset.IONOSPHERE:
            dl = get_example_dataset("ionosphere")
            dl.default_preprocess()

        elif dataset == Dataset.RECRUITMENT:
            dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

        else:
            dl = None
            # TODO: throw error for not recognised dataset

        if model_type == ModelType.NEURALNET:
            input_layer = args[0]
            output_layer = args[-1]
            hidden_layer = list(args[1:-1])

            model = TrainablePyTorchModel(input_layer, hidden_layer , output_layer)

            # TODO: currently if args = (2), NN(2, [], 2) is generated
            # check if we can have one layer NN, if not throw an error here
            # if len(args) < 2

        elif model_type == ModelType.DECISION_TREE:
            model = get_sklearn_model("decision_tree")

        elif model_type == ModelType.LOGISTIC_REGRESSION:
            model = get_sklearn_model("log_reg")

        else:
            model = None
            # TODO: throw error here

        trained_model = model.train(dl.X, dl.y)
        ct = ClassificationTask(trained_model, dl)

        # TODO: should dl and trained_model be a public attr of ct, so can do ct.dataset and just return ct?
        self.models[(dataset, model_type, args)] = (ct, dl, trained_model)

        return ct, dl, trained_model
