from gurobipy import Model, GRB

from rocelib.lib.OptSolver import OptSolver
from rocelib.evaluations.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from rocelib.tasks.Task import Task


class DeltaRobustnessEvaluator(ModelChangesRobustnessEvaluator):
    """
    A robustness evaluator that uses a Mixed-Integer Linear Programming (MILP) approach to evaluate
    the robustness of a model's predictions when perturbations are applied.

    This class inherits from ModelChangesRobustnessEvaluator and uses the Gurobi optimizer
    to determine if the model's prediction remains stable under perturbations.

    Attributes:
        task (Task): The task to solve, inherited from ModelChangesRobustnessEvaluator.
        opt (OptSolver): An optimizer instance for setting up and solving the MILP problem.
    """

    def __init__(self, task: Task):
        """
        Initializes the DeltaRobustnessEvaluator with a given task.

        @param ct: The task to solve, provided as a Task instance.
        """
        super().__init__(task)
        self.opt = OptSolver(task)

    def evaluate_single_instance(self, instance, counterfactual=None, desired_output=1, delta=0.005, bias_delta=0.005, M=10000, epsilon=0.0001):    
        """
        Evaluates whether the model's prediction for a given instance is robust to changes in the input.

        @param index: The index of the instance to evaluate.
        @param desired_output: The desired output for the model (0 or 1).
                               The evaluation will check if the model's output matches this.
        @param delta: The maximum allowable perturbation in the input features.
        @param bias_delta: Additional bias to apply to the delta changes.
        @param M: A large constant used in MILP formulation for modeling constraints.
        @param epsilon: A small constant used to ensure numerical stability.
        @return: A boolean indicating whether the model's prediction is robust given the desired output.
        """
        if counterfactual is not None and not counterfactual.empty:
            instance = counterfactual

        # instance = self.task.dataset.data.iloc[index]
        # Initialize the Gurobi model
        self.opt.gurobiModel = Model()

        # Set up the optimization problem with delta perturbations
        self.opt.setup(instance, delta=delta, bias_delta=bias_delta, M=M)

        # Set the objective to minimize or maximize based on the desired output
        if desired_output:
            self.opt.gurobiModel.setObjective(self.opt.outputNode, GRB.MINIMIZE)
        else:
            self.opt.gurobiModel.setObjective(self.opt.outputNode, GRB.MAXIMIZE)

        # Update the Gurobi model before optimization
        self.opt.gurobiModel.update()

        # Run the optimization
        self.opt.gurobiModel.optimize()

        # Get the status of the optimization solution
        status = self.opt.gurobiModel.status

        # If no optimal solution was found, return False (indicating non-robustness)
        if status != GRB.status.OPTIMAL:
            return False

        # Evaluate the robustness based on the output node's value and desired output
        if desired_output:
            return self.opt.outputNode.getAttr(GRB.Attr.X) > 0
        else:
            return self.opt.outputNode.getAttr(GRB.Attr.X) < 0