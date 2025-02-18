from abc import abstractmethod, ABC
from rocelib.tasks.Task import Task

class Evaluator(ABC):
    def __init__(self, task: Task, recourse_methods: [str]):
        self.task = task
        self.recourse_methods = recourse_methods


    def evaluate(self):
        """
        Returns: a dictionary from recourse method -> score
        """
        for recourse_method in self.recourse_methods:
            for instance in self.task.dataset:
                self.evaluate_single_instance(instance, recourse_method)
    

    @abstractmethod
    def evaluate_single_instance(self, instance, recourse_method):
        pass

        

