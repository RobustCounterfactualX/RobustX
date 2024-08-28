from abc import abstractmethod, ABC

from tasks.Task import Task


class ModelChangesRobustnessEvaluator(ABC):

    def __init__(self, ct: Task):
        self.task = ct

    @abstractmethod
    def evaluate(self, instance, neg_value=0):
        pass
