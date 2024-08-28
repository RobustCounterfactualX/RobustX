import pandas as pd
from gurobipy import Model, GRB
from gurobipy.gurobipy import quicksum

from intabs.IntervalAbstractionPyTorch import IntervalAbstractionPytorch
from robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from tasks.Task import Task
from lib.OptSolver import OptSolver


# TODO: Validity and distance evaluators
# TODO: Naive gradient method TODO: Make a Robustness Evaluator copy too which evaluates a group of points,
#  a collection of these DeltaRobustnessEvaluators? Instantiate the point evaluator for each point.
class DeltaRobustnessEvaluator(ModelChangesRobustnessEvaluator):

    def __init__(self, ct: Task):
        super().__init__(ct)
        self.opt = OptSolver(ct)

    def evaluate(self, instance, desired_output=1, delta=0.5, bias_delta=0, M=1000000000, epsilon=0.0001):

        # TODO: Ask if there's a better way to do this
        self.opt.gurobiModel = Model()

        self.opt.evaluate(instance, desired_output=desired_output, delta=delta, bias_delta=bias_delta, M=M,
                          epsilon=epsilon)

        # Desired output is 1, so we find minimum value to make sure it always classifies 1
        if desired_output:
            self.opt.gurobiModel.setObjective(self.opt.outputNode, GRB.MINIMIZE)
        else:
            self.opt.gurobiModel.setObjective(self.opt.outputNode, GRB.MAXIMIZE)

        self.opt.gurobiModel.update()

        self.opt.gurobiModel.optimize()

        status = self.opt.gurobiModel.status

        # If no solution was obtained that means the INN could not be modelled
        if status != GRB.status.OPTIMAL:
            return False

        # Return whether it is robust based on the desired output, assuming sigmoid will be applied
        if desired_output:
            return self.opt.outputNode.getAttr(GRB.Attr.X) > 0
        else:
            return self.opt.outputNode.getAttr(GRB.Attr.X) < 0
