from robustx.lib.intabs.IntervalAbstractionPyTorch import IntervalAbstractionPytorch
from robustx.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from robustx.lib.tasks.Task import Task
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
        self.number_of_samples = np.ceil(np.log(1 - self.alpha) / np.log(self.R))

    def evaluate(self, ce, desired_outcome=0, delta=0.5, bias_delta=0):
        """
        Evaluates whether the model's prediction for a given instance is robust to changes in the input.

        @param instance: The instance to evaluate.
        @param desired_output: The desired output for the model (0 or 1).
                               The evaluation will check if the model's output matches this.
        @param delta: The maximum allowable perturbation in the input features.
        @param bias_delta: Additional bias to apply to the delta changes.
        @param M: A large constant used in MILP formulation for modeling constraints.
        @param epsilon: A small constant used to ensure numerical stability.
        @return: A boolean indicating whether the model's prediction is robust given the desired output.
        """

        # Store initial weights
        old_weights = {}
        old_biases = {}
        i = 0

        for _, layer in enumerate(self.task.model.get_torch_model()):
            if isinstance(layer, nn.Linear):
                old_weights[i] = layer.weight.detach().numpy()
                old_biases[i] = layer.bias.detach().numpy()
                i += 1

        for _ in range(int(self.number_of_samples)):
           
            input_features = ce.detach().numpy()

            for l in range(0,len(old_weights)):
                layer_weights = old_weights[l]
                layer_biases = old_biases[l]
                
                weights_perturbation = np.random.uniform(-delta, delta, layer_weights.shape)
                if bias_delta > 0: biases_perturbation = np.random.uniform(-bias_delta, bias_delta, layer_biases.shape)

                layer_weights = layer_weights+weights_perturbation
                if bias_delta > 0:
                    layer_biases = layer_biases+biases_perturbation
               
                preactivated_res = np.dot(input_features, layer_weights.T) + layer_biases

                if l != len(old_weights)-1:
                    #relu
                    activated_res = np.maximum(0.0, preactivated_res)
                else:
                    #sigmoid
                    activated_res = 1/(1 + np.exp(-preactivated_res))
                
                input_features = activated_res

            #print(input_features)
            if (input_features.item() < 0.5 and desired_outcome == 1) or (input_features.item() >= 0.5 and desired_outcome == 0):
                return 0
            
        return 1
