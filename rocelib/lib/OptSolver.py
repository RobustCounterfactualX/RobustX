import pandas as pd

from gurobipy import Model, GRB
from gurobipy.gurobipy import quicksum

from rocelib.intabs.IntervalAbstractionPyTorch import IntervalAbstractionPytorch
from rocelib.tasks.Task import Task


class OptSolver:
    """
    A solver class that uses Gurobi to optimize a model based on a given task and instance.

    Attributes / Properties
    -------
    task: Task
        The task to be optimized.
    gurobiModel: Model
        The Gurobi optimization model.
    inputNodes: dict
        Dictionary to store Gurobi variables for input nodes.
    outputNode: Gurobi variable
        The Gurobi variable representing the output node.

    Methods
    -------
    setup(instance, desired_output=1, delta=0.5, bias_delta=0, M=1000000000, epsilon=0.0001, fix_inputs=True):
        Sets up the Gurobi model with constraints based on the provided instance and parameters.

    -------
    """

    def __init__(self, ct: Task):
        """
        Initializes the OptSolver with a given Task.

        @param ct: Task, The task to be optimized.
        """
        self.task = ct
        self.gurobiModel = Model()
        self.inputNodes = None
        self.outputNode = None

    def setup(self, instance, delta=0, bias_delta=0, M=1000, fix_inputs=True):
        """
        Sets up the Gurobi model with constraints based on the provided instance and parameters.

        @param instance: pd.DataFrame or list, The input data instance for which to set up the model.
        @param desired_output: int, Optional, The desired output value (default is 1).
        @param delta: float, Optional, The delta value used in constraints (default is 0.5).
        @param bias_delta: float, Optional, The bias delta value used in constraints (default is 0).
        @param M: float, Optional, A large constant used in constraints (default is 1000000000).
        @param epsilon: float, Optional, The epsilon value used in constraints (default is 0.0001).
        @param fix_inputs: bool, Optional, Whether to fix input values or use variable bounds (default is True).

        @return: None
        """

        # Turn off the Gurobi output
        self.gurobiModel.setParam('OutputFlag', 0)

        # Convert instance to a list
        if isinstance(instance, pd.DataFrame):
            try:
                ilist = instance.iloc[0].tolist()
            except Exception as e:
                raise Exception("Empty instance provided")
        else:
            ilist = instance.tolist()

        intabs = IntervalAbstractionPytorch(self.task.model, delta, bias_delta=bias_delta)

        self.inputNodes = {}
        all_nodes = {}
        activation_states = {}

        if fix_inputs:

            # Create the Gurobi variables for the inputs
            for i in range(len(self.task.dataset.X.columns)):
                key = f"v_0_{i}"
                self.inputNodes[key] = self.gurobiModel.addVar(lb=-float('inf'), name=key)
                all_nodes[key] = self.inputNodes[key]

                self.gurobiModel.addConstr(self.inputNodes[key] == ilist[i], name=f"constr_input_{i}")

        else:

            # Create the Gurobi variables for the inputs
            for i, col in enumerate(self.task.dataset.X.columns):
                key = f"v_0_{i}"

                # Calculate the minimum and maximum values for the current column
                col_min = self.task.dataset.X[col].min()
                col_max = self.task.dataset.X[col].max()

                # Use the calculated min and max for the bounds of the variable
                self.inputNodes[key] = self.gurobiModel.addVar(lb=col_min, ub=col_max, name=key)
                all_nodes[key] = self.inputNodes[key]

        self.gurobiModel.update()

        num_layers = len(intabs.layers)

        # Iterate through all "hidden" layers
        for layer in range(num_layers - 2):

            # Go through each node in the current layer
            for node in range(intabs.layers[layer + 1]):

                var_name = f"v_{layer + 1}_{node}"
                activation_name = f"xi_{layer + 1}_{node}"

                all_nodes[var_name] = self.gurobiModel.addVar(lb=-float('inf'), name=var_name)
                activation_states[activation_name] = self.gurobiModel.addVar(vtype=GRB.BINARY, name=activation_name)

                self.gurobiModel.update()

                # 1) Add v_i_j >= 0 constraint
                self.gurobiModel.addConstr(all_nodes[var_name] >= 0, name="constr1_" + var_name)

                # 2) Add v_i_j <= M ( 1 - xi_i_j )
                self.gurobiModel.addConstr(M * (1 - activation_states[activation_name]) >= all_nodes[var_name],
                                           name="constr2_" + var_name)

                # 3) Add v_i_j <= sum((W_i_j + delta)v_i-1_j + ... + M xi_i_j)
                self.gurobiModel.addConstr(quicksum((
                    intabs.weight_intervals[f'weight_l{layer}_n{prev_node_index}_to_l{layer + 1}_n{node}'][1] *
                    all_nodes[f"v_{layer}_{prev_node_index}"] for prev_node_index in range(intabs.layers[layer])
                )) + intabs.bias_intervals[f'bias_into_l{layer + 1}_n{node}'][1] + M * activation_states[
                                               activation_name] >= all_nodes[var_name],
                                           name="constr3_" + var_name)

                # 4) Add v_i_j => sum((W_i_j - delta)v_i-1_j + ...)
                self.gurobiModel.addConstr(quicksum((
                    intabs.weight_intervals[f'weight_l{layer}_n{prev_node_index}_to_l{layer + 1}_n{node}'][0] *
                    all_nodes[f"v_{layer}_{prev_node_index}"] for prev_node_index in range(intabs.layers[layer])
                )) + intabs.bias_intervals[f'bias_into_l{layer + 1}_n{node}'][0] <= all_nodes[var_name],
                                           name="constr4_" + var_name)

                self.gurobiModel.update()

        # Create a singular output node
        self.outputNode = self.gurobiModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                  name='output_node')

        # Constraint 1: node <= ub(W)x + ub(B)
        self.gurobiModel.addConstr(quicksum((
            intabs.weight_intervals[f'weight_l{num_layers - 2}_n{prev_node_index}_to_l{num_layers - 1}_n{0}'][1] *
            all_nodes[f"v_{num_layers - 2}_{prev_node_index}"] for prev_node_index in range(intabs.layers[num_layers - 2])
        )) + intabs.bias_intervals[f'bias_into_l{num_layers - 1}_n{0}'][1] >= self.outputNode,
                                   name="output_node_C1")

        # Constraint 2: node => lb(W)x + lb(B)
        self.gurobiModel.addConstr(quicksum((
            intabs.weight_intervals[f'weight_l{num_layers - 2}_n{prev_node_index}_to_l{num_layers - 1}_n{0}'][0] *
            all_nodes[f"v_{num_layers - 2}_{prev_node_index}"] for prev_node_index in range(intabs.layers[num_layers - 2])
        )) + intabs.bias_intervals[f'bias_into_l{num_layers - 1}_n{0}'][0] <= self.outputNode,
                                   name="output_node_C2")
