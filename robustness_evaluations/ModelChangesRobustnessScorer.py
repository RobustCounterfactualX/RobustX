from abc import abstractmethod, ABC

from tasks.Task import Task


class ModelChangesRobustnessScorer(ABC):

    def __init__(self, ct: Task):
        self.task = ct

    @abstractmethod
    def score(self, instance, neg_value=0):
        pass
