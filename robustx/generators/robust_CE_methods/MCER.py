from robustx.generators.CE_methods.MCE import MCE
from robustx.generators.CEGenerator import CEGenerator
from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from robustx.lib.tasks.Task import Task
import pandas as pd


class MCER(CEGenerator):
    """
    A counterfactual explanation generator that uses the Mixed-Integer Linear Programming (MILP) method and a robustness evaluator
    to find counterfactual explanations that are robust against perturbations.

    Inherits from CEGenerator and combines MCE with a robustness evaluation mechanism.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
        __customFunc (callable, optional): A custom distance function, inherited from CEGenerator.
        mce (MCE): An instance of the MCE class for generating counterfactuals using MILP.
        evaluator (DeltaRobustnessEvaluator): An instance of the DeltaRobustnessEvaluator for evaluating robustness.
    """

    def __init__(self, ct: Task, evaluator=DeltaRobustnessEvaluator):
        """
        Initializes the MCER CEGenerator with a given task and evaluator.

        @param ct: The task to solve, provided as a Task instance.
        @param evaluator: The evaluator class used to assess the robustness of the counterfactuals. Defaults to DeltaRobustnessEvaluator.
        """
        super().__init__(ct)
        self.mce = MCE(ct)
        self.evaluator = evaluator(ct)

    def _generation_method(self, instance, column_name="target", neg_value=0, M=100000, epsilon=0.0001,
                           threshold=50, increment=0.03, delta=0.005, bias_delta=0.005, **kwargs) -> pd.DataFrame:
        """
        Generates a robust counterfactual explanation for a provided instance by iterating over different minimum distances
        and evaluating robustness until the threshold is reached or a robust counterfactual is found.

        @param instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
        @param column_name: The name of the target column. (Not used in this method)
        @param neg_value: The value considered negative in the target variable.
        @param M: A large constant used for modeling constraints in the MCE method.
        @param epsilon: A small constant used for modeling constraints in the MCE method.
        @param threshold: The maximum number of iterations to find a robust counterfactual.
        @param increment: The amount by which the minimum distance is incremented in each iteration.
        @param delta: The robustness parameter used in the evaluator.
        @param bias_delta: The bias delta parameter used in the evaluator.
        @param kwargs: Additional keyword arguments.
        @return: A DataFrame containing the robust counterfactual explanation for the provided instance.
        """

        # Initial ce
        ce = instance

        # Number of iterations
        i = 0

        # Minimum distance away from decision boundary
        minimum_distance = 0

        # Used for comparison later
        instance_df = pd.DataFrame(instance).T

        # Iterate up to threshold
        while i < threshold:

            # Use MCE to generate counterfactual explanation
            ce = self.mce.generate_for_instance(instance, neg_value=neg_value,
                                                column_name=column_name, minimum_distance=minimum_distance)

            # MCE returns original instance if solution doesn't exist
            if ce.equals(instance_df):
                # print("No possible solution for given parameters - maybe your delta is TOO HIGH!")
                return ce

            # If solution exists, check robustness, if robust return
            if self.evaluator.evaluate(ce, desired_output=1 - neg_value, delta=delta, bias_delta=bias_delta, M=M,
                                       epsilon=epsilon):
                return ce

            # Increment iteration counter and minimum distance from boundary to get more robust CEs
            i += 1
            minimum_distance += increment

        # print("Exceeded threshold before finding robust CE - maybe your delta is TOO HIGH!")
        return ce
