from tqdm import tqdm
from rocelib.generators.RecourseGenerator import RecourseGenerator
from rocelib.generators.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.generators.recourse_methods.Wachter import Wachter
from rocelib.robustness_evaluations import ApproximateDeltaRobustnessEvaluator
from rocelib.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from rocelib.lib.tasks.Task import Task
import pandas as pd
import numpy as np


class APAS(RecourseGenerator):
    """
    A recourse generator that uses the Mixed-Integer Linear Programming (MILP) method and a DeltaRobustnessEvaluator evaluator
    to find counterfactual explanations that are robust against model changes.

    Inherits from the RecourseGenerator class and implements the _generation_method to find counterfactual examples
    with robustness checks using a specified base method and evaluator. The method iterates over positive instances
    and evaluates their robustness, returning those with stable counterfactuals.

    Attributes:
        None specific to this class, but utilizes the task and model from the RecourseGenerator base class.
    """

    def __init__(self, task: Task, recourse_generator: KDTreeNNCE):
        """
        Initializes the APAS recourse generator with a given task and a CE generator.

        @param task: The task to generate counterfactual explanations for.
        """

        super().__init__(task)
        self.rg = recourse_generator(task)
        

    def _generation_method(self,
                            original_input, 
                            target_column_name="target",
                            desired_outcome=0, 
                            robustness_check= ApproximateDeltaRobustnessEvaluator,
                            delta_max=0.5,
                            maximum_iterations=1000,
                           **kwargs):
        
        """
        Generates the first counterfactual explanation for a given input using the APΔS method, i.e., a combination of exponential and binary search with a Wachter delta robustness model changes check.

        @param target_column_name: The name of the target column.
        @param desired_outcome: The value considered for the generation of the counterfactual in the target_column_name.
        @param robustness_check: The robustness evaluator to check model changes with respect to model changes (default).
        @param delta_max: Maximum perturbation allowed in the model for the robustness_check.
        @param maximum_iterations: The maximum number of iterations to run the APΔS method.

        @return: the first robust counterfactual explanation to Δ-model changes.
        """
        
        
        iterations = 0
        print("original_input\n", original_input)
        print("\nwith prediction: ", self._task.model.predict_single(original_input))
     
        for _ in range(maximum_iterations):
            ce = self.rg._generation_method(instance=original_input, neg_value=1-desired_outcome)
            ce = ce.drop(columns=['predicted', 'Loss'])            
            valid = self._task.model.predict_single(ce) == desired_outcome
            if not valid:
                print("Counterfactual explanation is invalid...")
                quit()

            robustness = robustness_check.evaluate(ce.T, desired_outcome=desired_outcome, delta=delta_max)
            if robustness:
                return ce
            
            iterations += 1

        return None




       
