from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator
from rocelib.evaluations.robustness_evaluations.Evaluator import Evaluator
from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.evaluations.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator


class RobustnessProportionEvaluator(Evaluator):
    """
     An Evaluator class which evaluates the proportion of recourses which are robust

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the robustness of CEs

    robustness_evaluator: ModelChangesRobustnessEvaluator
        An instance of ModelChangesRobustnessEvaluator to evaluate the robustness of the CEs

    valid_val: int
        Stores what the target value of a valid counterfactual is defined as

    target_col: str
        Stores what the target column name is

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are robust for the given parameters

    -------
    """

    def evaluate(self, recourse_method, delta=0.05, bias_delta=0, M=1000000, epsilon=0.001, valid_val=1, column_name="target",
                 robustness_evaluator: ModelChangesRobustnessEvaluator.__class__ = DeltaRobustnessEvaluator,
                 **kwargs):
        """
        Evaluate the proportion of CEs which are robust for the given parameters
        @param recourses: pd.DataFrame, the CEs to evaluate
        @param delta: int, delta needed for robustness evaluator
        @param bias_delta: int, bias delta needed for robustness evaluator
        @param M: int, large M needed for robustness evaluator
        @param epsilon: int, small epsilon needed for robustness evaluator
        @param column_name: str, what the target column name is
        @param valid_val: int, what the target value of a valid counterfactual is defined as
        @param robustness_evaluator: ModelChangesRobustnessEvaluator.__class__, the CLASS of the evaluator to use
        @return: Proportion of CEs which are robust
        """
        recourses = self.task._CEs[recourse_method][0]
        robust = 0
        cnt = 0

        # Drop categorical or non-numeric columns before evaluation
        instances = recourses.drop(columns=[column_name, "loss", "predicted"], errors="ignore")

        # Align columns to dataset features
        expected_features = self.task.dataset.X.columns
        instances = instances[expected_features]  # Select only the expected feature columns

        robustness_evaluator = robustness_evaluator(self.task)

        for _, instance in instances.iterrows():

            # Increment robust if CE is robust under given parameters
            if instance is not None and robustness_evaluator.evaluate_single_instance(instance, desired_output=valid_val,
                                                                      delta=delta, bias_delta=bias_delta,
                                                                      M=M, epsilon=epsilon):
                robust += 1

            # Increment total number of CEs encountered
            cnt += 1

        return robust / cnt
