from abc import abstractmethod, ABC
from rocelib.tasks.Task import Task

class Evaluator(ABC):
    def __init__(self, task: Task, recourse_methods: [str]):
        self.task = task
        self.recourse_methods = recourse_methods

    @abstractmethod
    def evaluate(self):
        """
        Returns: a dictionary from recourse method -> score
        """
        pass

    @abstractmethod
    def evaluate_single_instance(self, instance):
        pass
