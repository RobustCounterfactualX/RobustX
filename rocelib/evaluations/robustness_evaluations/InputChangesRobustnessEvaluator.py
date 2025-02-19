from abc import abstractmethod, ABC

from rocelib.evaluations.robustness_evaluations.BaseRobustnessEvaluator import BaseRobustnessEvaluator, Evaluator
from rocelib.tasks.Task import Task


class InputChangesRobustnessEvaluator(BaseRobustnessEvaluator):
    """
    Abstract base class for evaluating the robustness of model predictions with respect to Input Changes and acts 
    as a holder for concrete implementations

    """
    pass

   

