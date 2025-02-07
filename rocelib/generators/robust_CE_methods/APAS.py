from rocelib.generators.CEGenerator import CEGenerator
from rocelib.robustness_evaluations.ApproximateDeltaRobustnessEvaluator import ApproximateDeltaRobustnessEvaluator
from rocelib.lib.tasks.Task import Task


class APAS(CEGenerator):
    """
    A counterfactual explanation generator that uses any CEGenerator class and a ApproximateDeltaRobustnessEvaluator evaluator
    to find counterfactual explanations that are approximately robust against model changes.

    Inherits from the CEGenerator class and implements the _generation_method to generate counterfactual examples
    with approximate robustness checks using a specified confidence alpha. The method iterates over positive instances
    and evaluates their robustness, returning those with stable counterfactuals.

    This is a similar implementation of Marzari et. al "Rigorous Probabilistic Guarantees for Robust Counterfactual Explanations", ECAI 2024

    Attributes:
        CE_generator specific to this class, but utilizes the task and model from the RecourseCE base class.
        alpha = confidence level in the robustness evaluator
    """

    def __init__(self, task: Task, CE_generator: CEGenerator, alpha: float):
        """
        Initializes the APAS CE generator with a given task and a CE generator.

        @param task: The task to generate counterfactual explanations for.
        """

        super().__init__(task)
        self.rg = CE_generator(task)
        self.alpha = alpha


    def _generation_method(self,
                            original_input, 
                            target_column_name="target",
                            desired_outcome=0,
                            delta_max=0.5,
                            maximum_iterations=1000,
                           **kwargs):
        
        """
        Generates the first counterfactual explanation for a given input using the APΔS method, i.e., a combination of exponential and binary search with a probabilistic delta robustness model changes check.

        @param target_column_name: The name of the target column.
        @param desired_outcome: The value considered for the generation of the counterfactual in the target_column_name.
        @param delta_max: Maximum perturbation allowed in the model for the robustness_check.
        @param maximum_iterations: The maximum number of iterations to run the APΔS method.

        @return: the first robust counterfactual explanation to Δ-model changes.
        """
        
        
        iterations = 0
        robustness_check = ApproximateDeltaRobustnessEvaluator(self.task, self.alpha)

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




       
