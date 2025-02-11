from abc import abstractmethod, ABC

from robustx.lib.tasks.Task import Task
from robustx.datasets.DatasetLoader import DatasetLoader
from robustx.lib.models.BaseModel import BaseModel


class ModelMultiplicityRobustnessEvaluator(ABC):
    """
    Abstract base class for evaluating the robustness of CEs with respect to model multiplicity.
    """

    def __init__(self, models: list[BaseModel], data: DatasetLoader):
        """
        Initializes the ModelMultiplicityRobustnessEvaluator.

        @param models: A list of models instantiating the model multiplicity problem
        @param data: The dataset.
        """
        self.models = models
        self.data = data

    @abstractmethod
    def evaluate(self, instance, counterfactuals):
        """
        Abstract method to evaluate the robustness of the counterfactuals for the
        input instance under model multiplicity.

        Must be implemented by subclasses.

        @param instance: A single input instance for which the counterfactual is generated
        @param counterfactuals: A DataFrame of counterfactuals for the input by the models.
        """
        pass
