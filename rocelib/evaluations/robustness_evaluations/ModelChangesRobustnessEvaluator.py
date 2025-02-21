from abc import abstractmethod, ABC

from rocelib.evaluations import RecourseEvaluator
from rocelib.evaluations.robustness_evaluations.BaseRobustnessEvaluator import BaseRobustnessEvaluator
from rocelib.tasks.Task import Task


class ModelChangesRobustnessEvaluator(BaseRobustnessEvaluator):
    """
    Abstract base class for evaluating the robustness of model predictions with respect to Model changes and acts 
    as a holder for concrete implementations

    """

    pass