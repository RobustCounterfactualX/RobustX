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
            for index in range(len(self.task.dataset.data)):
                self.evaluate_single_instance(index, recourse_method)
    

    @abstractmethod
    def evaluate_single_instance(self, index, recourse_method):
        pass

        

