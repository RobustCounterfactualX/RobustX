import pandas as pd

from rocelib.intabs.IntervalAbstractionPyTorch import IntervalAbstractionPytorch
from rocelib.evaluations.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator

from rocelib.tasks import Task
import numpy as np
import torch.nn as nn

class ApproximateDeltaRobustnessEvaluator(ModelChangesRobustnessEvaluator):
    """
    A robustness evaluator that uses a Approximate Plausible Δ model shifts (APΔS) approach to evaluate
    the robustness of a model's predictions when a delta perturbation is applied.

    This class inherits from ModelChangesRobustnessEvaluator and uses the a probabilistic approach 
    to determine if the model's prediction remains stable under model perturbations.

    Attributes:
        task (Task): The task to solve, inherited from ModelChangesRobustnessEvaluator.
        alpha (Float):Confidence in the prediction.
        R (Float): Fraction of samples for which the predictions should remain stable.
    """

    def __init__(self, ct: Task, alpha=0.999, R=0.995):
        """
        Initializes the DeltaRobustnessEvaluator with a given task.

        @param ct: The task to solve, provided as a Task instance.
        """
        super().__init__(ct)
        self.alpha = alpha
        self.R = R
        self.number_of_samples = int(np.ceil(np.log(1 - self.alpha) / np.log(self.R)))

    def evaluate(self, ce, desired_outcome=0, delta=0.5, bias_delta=0):
        """
        Evaluates whether the model's prediction for a given instance is robust to changes in the input.

        @param ce: The counterfactual explanation (numpy array or tensor) to evaluate.
        @param desired_outcome: The desired output for the model (0 or 1).
        @param delta: The maximum allowable perturbation in the input features.
        @param bias_delta: Additional bias to apply to the delta changes.
        @return: Boolean indicating whether the model's prediction is robust given the desired output.
        """

        # Store initial weights
        old_weights = {}
        old_biases = {}
        i = 0

        # for _, layer in enumerate(self.task.model):
        for layer in self.task.model.model.children():
            if isinstance(layer, nn.Linear):
                old_weights[i] = layer.weight.detach().numpy()
                old_biases[i] = layer.bias.detach().numpy()
                i += 1

        for _ in range(self.number_of_samples):
            input_features = ce.detach().numpy() if hasattr(ce, "detach") else ce.copy()
            if isinstance(input_features, pd.DataFrame):
                input_features = input_features.drop(columns=["predicted", "Loss"])

            for l in range(len(old_weights)):
                layer_weights = old_weights[l]
                layer_biases = old_biases[l]

                # Apply perturbations
                weights_perturbation = np.random.uniform(-delta, delta, layer_weights.shape)
                biases_perturbation = (
                    np.random.uniform(-bias_delta, bias_delta, layer_biases.shape)
                    if bias_delta > 0 else np.zeros_like(layer_biases)
                )

                perturbed_weights = layer_weights + weights_perturbation
                perturbed_biases = layer_biases + biases_perturbation

                # Compute pre-activation result
                preactivated_res = np.dot(input_features, perturbed_weights.T) + perturbed_biases

                # Apply activation functions
                input_features = (
                    np.maximum(0.0, preactivated_res) if l != len(old_weights) - 1
                    else 1 / (1 + np.exp(-preactivated_res))  # Sigmoid for final layer
                )

            # Check robustness condition
            final_output = input_features.item()
            if (final_output < 0.5 and desired_outcome == 1) or (final_output >= 0.5 and desired_outcome == 0):
                return False  # Not robust

        return True  # Robust

    def evaluate_single_instance(self, index, recourse_method, **kwargs):
        """
        Evaluates whether the model's prediction for a given indexed instance and its CE are robust.

        @param index: The index of the instance to evaluate.
        @param recourse_method: The recourse method used to generate the counterfactual.
        @param kwargs: Additional parameters (desired_output, delta, bias_delta).
        @return: Boolean indicating whether the CE and instance predictions are robust.
        """

        # Extract parameters from kwargs
        desired_outcome = kwargs.get("desired_outcome", 0)
        delta = kwargs.get("delta", 0.5)
        bias_delta = kwargs.get("bias_delta", 0)

        # Retrieve counterfactual explanation (CE)
        try:
            ce = self.task._CEs[recourse_method][0].iloc[index]
        except KeyError:
            print(f"Error: Recourse method '{recourse_method}' not found.")
            return False

        # Convert CE to numpy array
        ce = ce.to_numpy().reshape(1, -1)

        # Perform robustness evaluation using the main evaluate() function
        return self.evaluate(ce, desired_outcome=desired_outcome, delta=delta, bias_delta=bias_delta)

    # def evaluate(self, ce, desired_outcome=0, delta=0.5, bias_delta=0):
    #     """
    #     Evaluates whether the model's prediction for a given instance is robust to changes in the input.

    #     @param instance: The instance to evaluate.
    #     @param desired_output: The desired output for the model (0 or 1).
    #                            The evaluation will check if the model's output matches this.
    #     @param delta: The maximum allowable perturbation in the input features.
    #     @param bias_delta: Additional bias to apply to the delta changes.
    #     @param M: A large constant used in MILP formulation for modeling constraints.
    #     @param epsilon: A small constant used to ensure numerical stability.
    #     @return: A boolean indicating whether the model's prediction is robust given the desired output.
    #     """

    #     # Store initial weights
    #     old_weights = {}
    #     old_biases = {}
    #     i = 0

    #     for _, layer in enumerate(self.task.model.get_torch_model()):
    #         if isinstance(layer, nn.Linear):
    #             old_weights[i] = layer.weight.detach().numpy()
    #             old_biases[i] = layer.bias.detach().numpy()
    #             i += 1

    #     for _ in range(int(self.number_of_samples)):
           
    #         input_features = ce.detach().numpy()

    #         for l in range(0,len(old_weights)):
    #             layer_weights = old_weights[l]
    #             layer_biases = old_biases[l]
                
    #             weights_perturbation = np.random.uniform(-delta, delta, layer_weights.shape)
    #             if bias_delta > 0: biases_perturbation = np.random.uniform(-bias_delta, bias_delta, layer_biases.shape)

    #             layer_weights = layer_weights+weights_perturbation
    #             if bias_delta > 0:
    #                 layer_biases = layer_biases+biases_perturbation
               
    #             preactivated_res = np.dot(input_features, layer_weights.T) + layer_biases

    #             if l != len(old_weights)-1:
    #                 #relu
    #                 activated_res = np.maximum(0.0, preactivated_res)
    #             else:
    #                 #sigmoid
    #                 activated_res = 1/(1 + np.exp(-preactivated_res))
                
    #             input_features = activated_res

    #         #print(input_features)
    #         if (input_features.item() < 0.5 and desired_outcome == 1) or (input_features.item() >= 0.5 and desired_outcome == 0):
    #             return 0
            
    #     return 1
    

