from abc import ABC, abstractmethod

from rocelib.tasks.Task import Task


class RecourseEvaluator(ABC):
    """
    An abstract class used to evaluate recourse methods for a given task

    ...

    Attributes
    -------
    task: Task
        The task for which the recourse is being evaluated
    """

    def __init__(self, task: Task):
        """
        Initializes the RecourseEvaluator with the given task

        @param task: Task, the task to be evaluated with the recourse methods
        """
        self.task = task

    @abstractmethod
    def evaluate(self, recourses, **kwargs):
        """
        Abstract method to evaluate the provided recourses

        @param recourses: The recourse methods that are to be evaluated
        @param kwargs: Additional keyword arguments for the evaluation process
        """
        pass
