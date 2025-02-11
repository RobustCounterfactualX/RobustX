from abc import abstractmethod, ABC

from robustx.lib.tasks.Task import Task


class ModelChangesRobustnessEvaluator(ABC):
    """
    Abstract base class for evaluating the robustness of CEs with respect to model changes.

    This class defines an interface for evaluating how robust a CE's validity are
    when the model parameters are changed.
    """

    def __init__(self, ct: Task):
        """
        Initializes the ModelChangesRobustnessEvaluator with a given task.

        @param ct: The task for which robustness evaluations are being made.
                   Provided as a Task instance.
        """
        self.task = ct

    @abstractmethod
    def evaluate(self, instance, neg_value=0):
        """
        Abstract method to evaluate the robustness of a model's prediction on a given instance.

        Must be implemented by subclasses.

        @param instance: The instance for which to evaluate robustness.
                         This could be a single data point for the model.
        @param neg_value: The value considered negative in the target variable.
                          Used to determine if the counterfactual flips the prediction.
        @return: Result of the robustness evaluation. The return type should be defined by the subclass.
        """
        pass
