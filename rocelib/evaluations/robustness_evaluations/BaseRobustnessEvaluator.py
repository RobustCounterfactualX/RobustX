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
        for index, (_, instance) in enumerate(self.task.dataset.get_negative_instances().iterrows()):
            counterfactual = self.task.ces[recourse_method][0].iloc[index]
            evaluations.append(self.evaluate_single_instance(instance, counterfactual, **kwargs))
        return evaluations
    
    @abstractmethod
    def evaluate_single_instance(self, instance, counterfactual, **kwargs):
        pass
