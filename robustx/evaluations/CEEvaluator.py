from abc import ABC, abstractmethod

from robustx.lib.tasks.Task import Task


class CEEvaluator(ABC):
    """
    An abstract class used to evaluate CE methods for a given task

    ...

    Attributes
    -------
    task: Task
        The task for which the CE is being evaluated
    """

    def __init__(self, task: Task):
        """
        Initializes the CEEvaluator with the given task

        @param task: Task, the task to be evaluated with the CE methods
        """
        self.task = task

    @abstractmethod
    def evaluate(self, counterfactual_explanations, **kwargs):
        """
        Abstract method to evaluate the provided counterfactual explanations

        @param counterfactual_explanations: The CE methods that are to be evaluated
        @param kwargs: Additional keyword arguments for the evaluation process
        """
        pass
