from abc import abstractmethod, ABC
from rocelib.tasks.Task import Task

class Evaluator(ABC):
    def __init__(self, task: Task):
        self.task = task
        

    @abstractmethod
    def evaluate(self, recourse_method, **kwargs):
        """
        Returns: a list of evaluation scores
        """
        pass
    


        

