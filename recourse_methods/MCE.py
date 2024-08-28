import pandas as pd
from gurobipy import Model, GRB
from gurobipy.gurobipy import quicksum

from lib.OptSolver import OptSolver
from recourse_methods.RecourseGenerator import RecourseGenerator
from tasks.Task import Task


class MCE(RecourseGenerator):

    def __init__(self, ct: Task):
        super().__init__(ct)
        self.opt = OptSolver(ct)

    def _generation_method(self, instance, distance_func, column_name="target", neg_value=0, M=1000, epsilon=0.0001):

        if isinstance(instance, pd.DataFrame):
            ilist = instance.iloc[0].tolist()
        else:
            ilist = instance.tolist()

        self.opt.gurobiModel = Model()

        self.opt.evaluate(instance=instance, desired_output=1-neg_value, delta=0, M=M, epsilon=epsilon, fix_inputs=False)

        if not neg_value:
            self.opt.gurobiModel.addConstr(self.opt.outputNode - epsilon >= 0.0, name="output_node_lb_>=0")
        else:
            self.opt.gurobiModel.addConstr(self.opt.outputNode + epsilon <= 0.0, name="output_node_ub_<=0")

        objective = self.opt.gurobiModel.addVar(name="objective")

        self.opt.gurobiModel.addConstr(objective == quicksum(
            (self.opt.inputNodes[f'v_0_{i}'] - ilist[i]) ** 2 for i in range(len(self.task.training_data.X.columns))))

        self.opt.gurobiModel.update()

        self.opt.gurobiModel.optimize()

        status = self.opt.gurobiModel.status

        # If no solution was obtained that means the INN could not be modelled
        if status != GRB.status.OPTIMAL:
            return pd.DataFrame()

        ce = []

        for v in self.opt.gurobiModel.getVars():
            if 'v_0_' in v.varName:
                ce.append(v.getAttr(GRB.Attr.X))
        return pd.DataFrame(ce).T
