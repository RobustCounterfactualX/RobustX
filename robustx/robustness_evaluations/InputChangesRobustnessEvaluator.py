from abc import abstractmethod, ABC
import numpy as np
from robustx.lib.tasks.Task import Task
from robustx.generators.CEGenerator import CEGenerator


class InputChangesRobustnessEvaluator(ABC):
    """
    Abstract base class for evaluating the robustness of CEs with respect to input changes.
    """

    def __init__(self, ct: Task):
        """
        Initializes the InputChangesRobustnessEvaluator with a given task.

        @param ct: The task for which robustness evaluations are being made.
                   Provided as a Task instance.
        """
        self.task = ct

    @abstractmethod
    def evaluate(self, instance, counterfactual, generator: CEGenerator):
        """
        Compare the counterfactuals for the original instance and those for the perturbed instance.

        @param instance: An input instance.
        @param counterfactual: One or more CE points for the instance.
        @param generator: CE generator.
        """
        pass

    def perturb_input(self, instance):
        """
        Default method for perturbing an input instance by adding small Gaussian noise.

        @param instance: An input instance.
        """

        return instance + np.random.normal(0, 0.1, instance.shape)
