import pandas as pd
from gurobipy import Model, GRB
from gurobipy.gurobipy import quicksum

from intabs.IntervalAbstractionPyTorch import IntervalAbstractionPytorch
from robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator
from tasks.Task import Task


class OptSolver(ModelChangesRobustnessEvaluator):

    def __init__(self, ct: Task):
        super().__init__(ct)
        self.gurobiModel = Model()
        self.inputNodes = None
        self.outputNode = None

    def evaluate(self, instance, desired_output=1, delta=0.5, bias_delta=0, M=1000000000, epsilon=0.0001,
                 fix_inputs=True):

        # Turn off the Gurobi output
        self.gurobiModel.setParam('OutputFlag', 0)

        if isinstance(instance, pd.DataFrame):
            ilist = instance.iloc[0].tolist()
        else:
            ilist = instance.tolist()

        intabs = IntervalAbstractionPytorch(self.task.model, delta, bias_delta=bias_delta)

        self.inputNodes = {}

        all_nodes = {}

        activation_states = {}

        if fix_inputs:
            # Create the Gurobi variables for the inputs
            for i in range(len(self.task.training_data.X.columns)):
                key = f"v_0_{i}"
                self.inputNodes[key] = self.gurobiModel.addVar(lb=-float('inf'), name=key)
                all_nodes[key] = self.inputNodes[key]

                # activation_name = f"xi_0_{i}"
                # activation_states[activation_name] = self.gurobiModel.addVar(vtype=GRB.BINARY, name=activation_name)
                self.gurobiModel.addConstr(self.inputNodes[key] == ilist[i], name=f"constr_input_{i}")

        else:
            # Create the Gurobi variables for the inputs
            for i, col in enumerate(self.task.training_data.X.columns):
                key = f"v_0_{i}"

                # Calculate the minimum and maximum values for the current column
                col_min = self.task.training_data.X[col].min()
                col_max = self.task.training_data.X[col].max()

                # Use the calculated min and max for the bounds of the variable
                self.inputNodes[key] = self.gurobiModel.addVar(lb=col_min, ub=col_max, name=key)
                all_nodes[key] = self.inputNodes[key]

        self.gurobiModel.update()

        num_layers = len(intabs.layers)

        # Iterate through all "hidden" layers, the first value in intabs.layers is the input layer and the
        # last value in intabs.layers is the output layer. The actual layer index whose variables we want to
        # create is layer at index layer+1
        for layer in range(num_layers - 2):

            # Go through each layer in the layer whose variables we want to create
            for node in range(intabs.layers[layer + 1]):
                # The interval abstraction denotes the input layer as layer 0
                # Thus to get the hidden layers' index right, we must have the
                # var_name to be layer+1

                # Create Gurobi variables for each node and their activation state
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
                    all_nodes[
                        f"v_{layer}_{prev_node_index}"] for prev_node_index in range(intabs.layers[layer])
                )) + intabs.bias_intervals[f'bias_into_l{layer + 1}_n{node}'][1] + M * activation_states[
                                               activation_name] >= all_nodes[var_name],
                                           name="constr3_" + var_name)

                # 4) Add v_i_j => sum((W_i_j - delta)v_i-1_j + ...)
                self.gurobiModel.addConstr(quicksum((
                    intabs.weight_intervals[f'weight_l{layer}_n{prev_node_index}_to_l{layer + 1}_n{node}'][0] *
                    all_nodes[
                        f"v_{layer}_{prev_node_index}"] for prev_node_index in range(intabs.layers[layer])
                )) + intabs.bias_intervals[f'bias_into_l{layer + 1}_n{node}'][0] <= all_nodes[var_name],
                                           name="constr4_" + var_name)

                self.gurobiModel.update()

        # TODO: Currently assuming binary classification
        self.outputNode = self.gurobiModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                  name='output_node')

        # constraint 1: node <= ub(W)x + ub(B)
        self.gurobiModel.addConstr(quicksum((
            intabs.weight_intervals[f'weight_l{num_layers - 2}_n{prev_node_index}_to_l{num_layers - 1}_n{0}'][1] *
            all_nodes[
                f"v_{num_layers - 2}_{prev_node_index}"] for prev_node_index in range(intabs.layers[num_layers - 2])
        )) + intabs.bias_intervals[f'bias_into_l{num_layers - 1}_n{0}'][1] >= self.outputNode,
                                   name="output_node_C1")

        # constraint 2: node => lb(W)x + lb(B)
        self.gurobiModel.addConstr(quicksum((
            intabs.weight_intervals[f'weight_l{num_layers - 2}_n{prev_node_index}_to_l{num_layers - 1}_n{0}'][0] *
            all_nodes[
                f"v_{num_layers - 2}_{prev_node_index}"] for prev_node_index in range(intabs.layers[num_layers - 2])
        )) + intabs.bias_intervals[f'bias_into_l{num_layers - 1}_n{0}'][0] <= self.outputNode,
                                   name="output_node_C2")
