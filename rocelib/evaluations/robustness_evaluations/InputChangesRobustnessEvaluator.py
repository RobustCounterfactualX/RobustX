from abc import abstractmethod, ABC
import numpy as np
from rocelib.evaluations.robustness_evaluations.BaseRobustnessEvaluator import BaseRobustnessEvaluator, Evaluator
from rocelib.tasks.Task import Task


class InputChangesRobustnessEvaluator(BaseRobustnessEvaluator):
    """
    Abstract base class for evaluating the robustness of model predictions with respect to Input Changes and acts 
    as a holder for concrete implementations

    """
    def evaluate(self, recourse_method, **kwargs):
        """
        Computes the average distance returned by evaluate_single_instance.

        Returns: a float representing the average distance across all instances.
        """

        total_distance = 0
        total_count = 0
                
        for index, (_, instance) in enumerate(self.task.dataset.get_negative_instances().iterrows()):
            counterfactual = self.task.ces[recourse_method][0].iloc[index]
            distance = self.evaluate_single_instance(instance, counterfactual, recourse_method, **kwargs)
            
            if distance is not None:  # Ensure valid distance values
                total_distance += distance
                total_count += 1

        # Compute and return the average distance (avoid division by zero)
        return total_distance / total_count if total_count > 0 else 0
    
    def perturb_input(self, instance):
        """
        Default method for perturbing an input instance by adding small Gaussian noise.

        @param instance: An input instance.
        """

        return instance + np.random.normal(0, 0.1, instance.shape)

   

