from abc import abstractmethod, ABC

from rocelib.evaluations import RecourseEvaluator
from rocelib.evaluations.robustness_evaluations import Evaluator
from rocelib.tasks.Task import Task


class ModelChangesRobustnessEvaluator(ABC, Evaluator):
    """
    Abstract base class for evaluating the robustness of model predictions with respect to model changes.

    This class defines an interface for evaluating how robust a model's predictions are
    when the model parameters are changed.
    """

    pass