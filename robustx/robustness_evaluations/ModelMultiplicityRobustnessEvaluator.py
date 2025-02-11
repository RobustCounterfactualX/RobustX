from abc import abstractmethod, ABC

from robustx.lib.tasks.Task import Task


class ModelMultiplicityRobustnessEvaluator(ABC):
    """
    Abstract base class for evaluating the robustness of CEs with respect to model multiplicity.
    """

    def __init__(self, ct: Task):
        """
        Initializes the ModelMultiplicityRobustnessEvaluator with a given task.

        @param ct: The task for which robustness evaluations are being made.
                   Provided as a Task instance.
        """
        self.task = ct
