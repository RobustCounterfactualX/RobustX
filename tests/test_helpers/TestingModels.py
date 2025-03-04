from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.tasks.Task import Task
from rocelib.tasks.TaskBuilder import TaskBuilder


class TestingModels:
    """
    A class used as a singleton and shared between all test instances containing models (wrapped in a Task) that are
    used by one or more tests to prevent retraining and improve the speed of the test suite.

    Attributes:
        models (dict): a dictionary mapping model parameters to a classification task containing the requested,
        preprocessed dataset and the corresponding trained, usable model
    """
    def __init__(self):
        """
        Initializes the 'models' dictionary. Key value types: (dataset, model_type, args) -> Task

        """
        self.models = {}

    def get(self, training_dataset: str, dataset: str, model_type: str, *args) -> Task:
        """
        The public method that tests use to interface with the class. This is done by requesting a certain model -
        if it exists then it's container Task will be returned, otherwise a new Task will be created, and the requested
        model will be trained and added to it.

        @param training_dataset: The name of the dataset used for training the model.
        @param dataset: The name of the dataset we want to evaluate our model on.
        @param model_type: Specifies which ML framework the model is implemented with (e.g. pytorch, keras, etc.)
        @param args: Additional model parameters specific to some model types (e.g. layer sizes for a neural network).
        @return: A Pandas Series representing a random positive instance.
        """
        if (training_dataset, model_type, args) not in self.models:
            print(self.models)
            return self.__create_and_train(training_dataset, model_type, dataset, args)
        else:
            print(self.models)

            return self.models[(training_dataset, model_type, args)]

    def __create_and_train(self, training_dataset: str, model_type: str, dataset: str, args) -> Task:
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
