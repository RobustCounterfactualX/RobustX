from abc import abstractmethod, ABC

from rocelib.evaluations import RecourseEvaluator
from rocelib.evaluations.robustness_evaluations import BaseRobustnessEvaluator, Evaluator
from rocelib.tasks.Task import Task


class NoisyExecutionRobustnessEvaluator(BaseRobustnessEvaluator, ABC):
    """
    Abstract base class for evaluating the robustness of model predictions with respect to Noisy execution and acts 
    as a holder for concrete implementations

    """
    pass