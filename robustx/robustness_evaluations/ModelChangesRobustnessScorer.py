from abc import abstractmethod, ABC

from robustx.lib.tasks.Task import Task


class ModelChangesRobustnessScorer(ABC):
    """
    Abstract base class for scoring the robustness of CEs with respect to model changes.

    This class defines an interface for assigning a robustness score to a model's predictions for a CE
    when the model parameters are changed.
    """

    def __init__(self, ct: Task):
        """
        Initializes the ModelChangesRobustnessScorer with a given task.

        @param ct: The task for which robustness scores are being calculated.
                   Provided as a Task instance.
        """
        self.task = ct

    @abstractmethod
    def score(self, instance, neg_value=0):
        """
        Abstract method to calculate the robustness score for a model's prediction on a given instance.

        Must be implemented by subclasses.

        @param instance: The instance for which to calculate the robustness score.
                         This could be a single data point for the model.
        @param neg_value: The value considered negative in the target variable.
                          Used to determine if the counterfactual flips the prediction.
        @return: The calculated robustness score. The return type should be defined by the subclass.
        """
        pass
