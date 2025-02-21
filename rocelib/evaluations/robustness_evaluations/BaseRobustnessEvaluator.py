from abc import abstractmethod, ABC

from rocelib.evaluations.robustness_evaluations.Evaluator import Evaluator
from rocelib.tasks.Task import Task


class BaseRobustnessEvaluator(Evaluator):
    """
    Abstract base class for evaluating the robustness of model predictions in general

    This class defines an interface for evaluating how robust a model's predictions with all the different robustness types.
    
    """
    def evaluate(self, recourse_method, **kwargs):
        """
        Returns: a list of evaluation scores
        """
        evaluations = []
        # for index in range(len(self.task.dataset.data)):
        #     evaluations.append(self.evaluate_single_instance(index, recourse_method, **kwargs))
        return evaluations

    @abstractmethod
    def evaluate_single_instance(self, instance, counterfactual, **kwargs):
        pass
