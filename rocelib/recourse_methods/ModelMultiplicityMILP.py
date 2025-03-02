import pandas as pd
from gurobipy import Model
from gurobipy.gurobipy import quicksum, GRB

from rocelib.datasets.DatasetLoader import DatasetLoader
from rocelib.models.imported_models.PytorchModel import PytorchModel
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.tasks.Task import Task


def create_weights_and_bias_dictionary(model):
    """
    Extract the learned weights and biases from a PyTorch-based model and store them in dictionaries.
    The first layer in the model is considered the input layer (layer 0).

    Parameters
    ----------
    model : PytorchModel
        A PytorchModel object containing the trained PyTorch model.

    Returns
    -------
    tuple of (dict, dict)
        Two dictionaries:
        - weight_dict, keyed by 'weight_l<layer_idx>_n<src_idx>_to_l<layer_idx+1>_n<dest_idx>'
        - bias_dict, keyed by 'bias_into_l<layer_idx+1>_n<dest_idx>'
    """

    params = {}
    for name, param in model.model.named_parameters():
        params[name] = param.detach().numpy()

    weight_dict = {}
    bias_dict = {}

    # Loop through layers
    for layer_idx in range(0, len(params) // 2):

        # Get weights and biases
        weights = params[f'{layer_idx * 2}.weight']
        biases = params[f'{layer_idx * 2}.bias']

        for dest_idx in range(weights.shape[0]):

            # Set the interval for biases
            bias_key = f'bias_into_l{layer_idx + 1}_n{dest_idx}'
            bias_dict[bias_key] = biases[dest_idx]

            for src_idx in range(weights.shape[1]):
                # Set the interval for weights
                weight_key = f'weight_l{layer_idx}_n{src_idx}_to_l{layer_idx + 1}_n{dest_idx}'
                weight = weights[dest_idx, src_idx]
                weight_dict[weight_key] = weight

    return weight_dict, bias_dict


class ModelMultiplicityMILP(RecourseGenerator):
    """
        Formulates and solves a Mixed Integer Linear Program (MILP) to find a recourse instance
        that satisfies the classification constraints of multiple neural network models simultaneously.
    """

    def __init__(self, ct: Task, custom_distance_func=None):
        """
        Initializes the Argumentative Ensembling CE generator with a dataset and a list of models.

        Args:
            dl: dataset loader
            models: the list of models forming the model multiplicity problem setting
        """
        super().__init__(ct)
        self.gurobiModel = Model()
        self.models = ct.mm_models.values()
        self.inputNodes = None
        self.outputNodes = {}


    def _generation_method(self, instance, column_name="target", neg_value=0, M=1000, epsilon=0.1, **kwargs):

        """
            Generate a recourse instance by formulating and solving the MILP with constraints
            derived from each of the provided PyTorch models.

            Parameters
            ----------
            instance : pd.DataFrame or list
                The original instance (features) for which recourse is to be generated.
            column_name : str, optional
                Unused in this implementation, by default "target".
            neg_value : int, optional
                Decides the sign of the output constraints; 0 for non-negative and 1 for non-positive, by default 0.
            M : float, optional
                A large constant for big-M constraints, by default 1000.
            epsilon : float, optional
                A small offset for output constraints, by default 0.1.
            kwargs : dict
                Additional arguments (not used in this method).

            Returns
            -------
            pd.DataFrame
                A single-row DataFrame containing the values of the recourse instance if a solution is found.
                If the MILP is infeasible (no solution), returns an empty DataFrame.
        """

        self.gurobiModel = Model()

        # Turn off the Gurobi output
        self.gurobiModel.setParam('OutputFlag', 0)

        if isinstance(instance, pd.DataFrame):
            ilist = instance.iloc[0].tolist()
        else:
            ilist = instance.tolist()

        # Dictionary to store input variables, shared across all models
        self.inputNodes = {}
        activation_states = {}
        all_nodes = {}

        # Create Gurobi variables for the inputs (shared for all models)
        for i, col in enumerate(self.task.dataset.X.columns):
            key = f"v_0_{i}"

            # Calculate the minimum and maximum values for the current column
            col_min = self.task.dataset.X[col].min()
            col_max = self.task.dataset.X[col].max()

            # Use the calculated min and max for the bounds of the variable
            self.inputNodes[key] = self.gurobiModel.addVar(lb=col_min, ub=col_max, name=key)
            all_nodes[key] = self.inputNodes[key]

        for model_idx, model in enumerate(self.models):
            weights, biases = create_weights_and_bias_dictionary(model)

            layers = [model.input_dim] + model.hidden_dim + [model.output_dim]

            # Iterate through all "hidden" layers, the first value in intabs.layers is the input layer and the
            # last value in intabs.layers is the output layer. The actual layer index whose variables we want to
            # create is layer at index layer+1
            for layer in range(len(layers) - 2):

                # Go through each layer in the layer whose variables we want to create
                for node in range(layers[layer + 1]):
                    # Create Gurobi variables for each node and their activation state
                    var_name = f"model{model_idx}_v_{layer + 1}_{node}"
                    activation_name = f"model{model_idx}_xi_{layer + 1}_{node}"

                    all_nodes[var_name] = self.gurobiModel.addVar(lb=-float('inf'), name=var_name)
                    activation_states[activation_name] = self.gurobiModel.addVar(vtype=GRB.BINARY, name=activation_name)

                    self.gurobiModel.update()

                    # 1) Add v_i_j >= 0 constraint
                    self.gurobiModel.addConstr(all_nodes[var_name] >= 0, name=f"model{model_idx}_constr1_" + var_name)

                    # 2) Add v_i_j <= M ( 1 - xi_i_j )
                    self.gurobiModel.addConstr(M * (1 - activation_states[activation_name]) >= all_nodes[var_name],
                                               name=f"model{model_idx}_constr2_" + var_name)

                    qr = quicksum((
                        weights[f'weight_l{layer}_n{prev_node_index}_to_l{layer + 1}_n{node}'] *
                        all_nodes[
                            f"model{model_idx}_v_{layer}_{prev_node_index}" if layer else f"v_0_{prev_node_index}"] for
                    prev_node_index in range(layers[layer])
                    )) + biases[f'bias_into_l{layer + 1}_n{node}']

                    # 3) Add v_i_j <= sum((W_i_j + delta)v_i-1_j + ... + M xi_i_j)
                    self.gurobiModel.addConstr(qr + M * activation_states[
                        activation_name] >= all_nodes[var_name],
                                               name=f"model{model_idx}_constr3_" + var_name)

                    self.gurobiModel.addConstr(qr <= all_nodes[var_name])

                    self.gurobiModel.update()

            outputNode = self.gurobiModel.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                 name=f'model{model_idx}_output_node')

            self.outputNodes[f'model{model_idx}_output_node'] = outputNode

            # constraint 1: node <= ub(W)x + ub(B)
            self.gurobiModel.addConstr(quicksum((
                weights[f'weight_l{len(layers) - 2}_n{prev_node_index}_to_l{len(layers) - 1}_n0'] *
                all_nodes[
                    f"model{model_idx}_v_{len(layers) - 2}_{prev_node_index}" if len(
                        layers) - 2 else f"v_0_{prev_node_index}"] for prev_node_index in range(layers[len(layers) - 2])
            )) + biases[f'bias_into_l{len(layers) - 1}_n0'] == outputNode,
                                       name=f'model{model_idx}_output_node_C1')

            if not neg_value:
                self.gurobiModel.addConstr(outputNode - epsilon >= 0,
                                           name=f"model{model_idx}_output_node_lb_>=0")
            else:
                self.gurobiModel.addConstr(outputNode + epsilon <= 0,
                                           name=f"model{model_idx}_output_node_ub_<=0")

            self.gurobiModel.update()

        objective = self.gurobiModel.addVar(name="objective")

        self.gurobiModel.addConstr(objective == quicksum(
            (self.inputNodes[f'v_0_{i}'] - ilist[i]) ** 2 for i in range(len(self.task.dataset.X.columns))))

        self.gurobiModel.update()

        self.gurobiModel.optimize()

        status = self.gurobiModel.status

        # If no solution was obtained that means the INN could not be modelled
        if status != GRB.status.OPTIMAL:
            return pd.DataFrame()

        ce = []

        for v in self.gurobiModel.getVars():
            if 'v_0_' in v.varName:
                ce.append(v.getAttr(GRB.Attr.X))
        # return pd.DataFrame(ce).T
        return pd.DataFrame([ce], columns=self.task.dataset.X.columns)