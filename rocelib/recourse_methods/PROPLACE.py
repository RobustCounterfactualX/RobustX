import pandas as pd
from gurobipy import Model, GRB
from gurobipy.gurobipy import quicksum

from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.recourse_methods.RNCE import RNCE
from rocelib.tasks.Task import Task
from rocelib.intabs.WeightBiasDictionary import create_weights_and_bias_dictionary


class PROPLACE(RecourseGenerator):
    """
    A counterfactual explanation generator that finds provably robust counterfactual explanations for MLPs.

    Inherits from the CEGenerator class and implements the _generation_method to perform MILP based
    robust optimisation.

    Attributes:
        _task (Task): The task to solve, inherited from CEGenerator.
        achieved: boolean variable indicating whether to end the robust optimisation procedure
        x_prime_star: optimal counterfactual
        x_prime_current: current counterfactual as the robust optimisation proceeds
        layers: list of model dimensions for each layer
        neg_value: the undesirable class label
        delta: hyperparameter, the infinity-norm magnitude of model parameter perturbations
        orig_w: weight dictionary of the original model
        orig_b: bias dictionary of the original model
        w_primes: the list of the worst perturbations' weight dictionaries computed by robust optimisation
        b_primes: the list of the worst perturbations' bias dictionaries computed by robust optimisation
        rnce: the RNCE recourse generator, used upfront for obtaining the plausible region
    """

    def __init__(self, ct: Task):
        """
        Initializes the PROPLACE recourse generator with a given task.

        @param ct: The task to solve, provided as a Task instance.
        """
        super().__init__(ct)
        # create model parameter dict for optimisation solver
        self.achieved = False
        self.x_prime_star = None
        self.x_prime_current = None
        self.layers = [ct.model.input_dim] + ct.model.hidden_dim + [ct.model.output_dim]
        self.neg_value = None
        self.delta = None
        self.orig_w, self.orig_b = create_weights_and_bias_dictionary(ct.model)
        self.w_primes = [self.orig_w]
        self.b_primes = [self.orig_b]
        self.rnce = RNCE(ct)

    def _generation_method(self, instance, column_name="target", neg_value=0, delta=0.005,
                           bias_delta=0.005, epsilon=0.0001, k=10, **kwargs) -> pd.DataFrame:
        """
        The main procedure. First get RNCE points for constructing the plausible region, then perform robust
        optimisation by iteratively running the master problem and the adversarial problem.

        Args:
            instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
            column_name: The name of the target column.
            neg_value: The value considered negative in the target variable.
            delta: The tolerance for robustness in the model parameter space.
            bias_delta: The tolerance for robustness in the model parameter space.
            epsilon: The tolerance for the optimisation solver
            k: number of points RNCE returns
            **kwargs: Additional keyword arguments.

        Returns: A DataFrame containing the counterfactual explanation.

        """
        

        # get the boundary points of the convex hull
        candidates = self.rnce._generation_method(instance, robustInit=True, column_name=column_name,
                                                  neg_value=neg_value, delta=delta, bias_delta=bias_delta, k=k).values
        
        if candidates.shape[0] == 0:
            raise ValueError("RNCE returned an empty candidate set, meaning no plausible region was found.")
        # Convert instance to a list
        if isinstance(instance, pd.DataFrame):
            try:
                ilist = instance.iloc[0].tolist()
            except Exception as e:
                raise Exception("Empty instance provided")
        else:
            ilist = instance.tolist()
        self.neg_value = neg_value
        self.delta = delta
        # start robust optimisation
        while not self.achieved:
            self.x_prime_current = self._master_prob(ilist, candidates)
            if self.x_prime_current.empty:
                raise ValueError("Master problem returned an empty DataFrame, meaning it couldn't find a valid counterfactual.")
            self._adv_prob()  # add worst case perturbation to self.wprimes or have found best solution
        res = self.x_prime_current

        self.delta = None
        self.achieved = False
        self.x_prime_star = None
        self.x_prime_current = None
        self.w_primes = [self.orig_w]
        self.b_primes = [self.orig_b]
        return res

    def _master_prob(self, ilist, candidates):
        """
        Master problem or the outer optimisation problem which computes the CE that is robust to all model
        perturbations in self.w_primes (self.b_primes)

        Args:
            ilist: Instance list
            candidates: The boundary points of the convex hull (plausible region)

        Returns: a counterfactual point

        """
        gurobi_model = Model()
        gurobi_model.setParam('OutputFlag', 0)
        input_vars, gurobi_model = self._master_prob_add_inputs(gurobi_model)
        gurobi_model = self._master_prob_add_plausibility(gurobi_model, candidates, input_vars)
        for i, wp in enumerate(self.w_primes):
            bp = self.b_primes[i]
            gurobi_model = self._master_prob_add_one_model(gurobi_model, wp, bp, i, input_vars)
        # set objective
        # objective = gurobi_model.addVar(name="objective")
        # gurobi_model.addConstr(objective == quicksum(
        #     (input_vars[f'v_0_{i}'] - ilist[i]) ** 2 for i in range(len(self.task.training_data.X.columns))))

        obj_vars_l1 = []
        for i in range(len(self.task.dataset.X.columns)):
            gurobi_model.update()
            key = f"v_0_{i}"
            this_obj_var_l1 = gurobi_model.addVar(vtype=GRB.SEMICONT, lb=-GRB.INFINITY, name=f"objl1_feat_{i}")
            gurobi_model.addConstr(this_obj_var_l1 >= ilist[i] - input_vars[key])
            gurobi_model.addConstr(this_obj_var_l1 >= input_vars[key] - ilist[i])
            obj_vars_l1.append(this_obj_var_l1)
        gurobi_model.setObjective(quicksum(obj_vars_l1), GRB.MINIMIZE)

        # gurobi_model.update()
        # gurobi_model.setObjective(objective, GRB.MINIMIZE)
        gurobi_model.update()
        gurobi_model.Params.NonConvex = 2
        gurobi_model.optimize()

        status = gurobi_model.status
        # If no solution was obtained that means the INN could not be modelled
        if status != GRB.status.OPTIMAL:
            print("Gurobi Model Optimization Failed!")
            print("Gurobi Status:", status)

            if gurobi_model.status == GRB.INFEASIBLE:
                print("Model is infeasible! Computing IIS...")
                gurobi_model.computeIIS()
                gurobi_model.write("infeasible_model.ilp")  # Save IIS information
                print("IIS written to infeasible_model.ilp")


            return pd.DataFrame()

        ce = []

        for v in gurobi_model.getVars():
            if 'v_0_' in v.varName:
                ce.append(v.getAttr(GRB.Attr.X))
        return pd.DataFrame(ce).T

    def _master_prob_add_inputs(self, gurobi_model):
        """
        Add input variables into the optimisation model for the master problem

        Args:
            gurobi_model: Gurobi optimisation model

        Returns: Dictionary of input variables, and the updated gurobi optimisation model

        """
        input_vars = {}
        for i, col in enumerate(self.task.dataset.X.columns):
            key = f"v_0_{i}"

            # Calculate the minimum and maximum values for the current column
            col_min = self.task.dataset.X[col].min()
            col_max = self.task.dataset.X[col].max()

            # Use the calculated min and max for the bounds of the variable
            input_vars[key] = gurobi_model.addVar(vtype=GRB.CONTINUOUS, lb=col_min, ub=col_max, name=key)
            gurobi_model.update()
        return input_vars, gurobi_model

    def _master_prob_add_plausibility(self, gurobi_model, candidates, input_vars):
        """
        Add plausibility region constraints into the optimisation model for the master problem

        Args:
            gurobi_model: Gurobi optimisation model
            candidates: The boundary points of the convex hull (plausible region)
            input_vars: Dictionary of input variables

        Returns: The updated gurobi optimisation model

        """
        k = candidates.shape[0]
        l_var = []
        # add lambdas for each vertex of the convex hull
        for i in range(k):
            l_var.append(gurobi_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1))
        gurobi_model.addConstr(quicksum(item for item in l_var) == 1)
        for feat_idx in range(self.layers[0]):
            gurobi_model.addConstr(
                input_vars[f'v_0_{feat_idx}'] == quicksum(candidates[i][feat_idx] * l_var[i] for i in range(k)))
        return gurobi_model

    def _master_prob_add_one_model(self, gurobi_model, wp, bp, model_idx, input_nodes):
        """
        Add one MLP forward pass constraints into the optimisation model for the master problem

        Args:
            gurobi_model: Gurobi optimisation model
            wp: The dictionary of model weights of the perturbed model
            bp:The dictionary of model biases of the perturbed model
            model_idx: The index of the perturbed model
            input_nodes: Dictionary of input variables

        Returns: The updated gurobi optimisation model

        """
        activation_states = {}
        all_nodes = {}
        for key in input_nodes.keys():
            all_nodes[key] = input_nodes[key]
        for layer in range(len(self.layers) - 2):

            # Go through each layer in the layer whose variables we want to create
            for node in range(self.layers[layer + 1]):
                # Create Gurobi variables for each node and their activation state
                var_name = f"model{model_idx}_v_{layer + 1}_{node}"
                activation_name = f"model{model_idx}_xi_{layer + 1}_{node}"

                all_nodes[var_name] = gurobi_model.addVar(lb=-float('inf'), name=var_name)
                activation_states[activation_name] = gurobi_model.addVar(vtype=GRB.BINARY, name=activation_name)

                gurobi_model.update()

                # 1) Add v_i_j >= 0 constraint
                gurobi_model.addConstr(all_nodes[var_name] >= 0, name=f"model{model_idx}_constr1_" + var_name)

                # 2) Add v_i_j <= M ( 1 - xi_i_j )
                gurobi_model.addConstr(1000 * (1 - activation_states[activation_name]) >= all_nodes[var_name],
                                       name=f"model{model_idx}_constr2_" + var_name)

                qr = quicksum((
                    wp[f'weight_l{layer}_n{prev_node_index}_to_l{layer + 1}_n{node}'] *
                    all_nodes[
                        f"model{model_idx}_v_{layer}_{prev_node_index}" if layer else f"v_0_{prev_node_index}"] for
                    prev_node_index in range(self.layers[layer])
                )) + bp[f'bias_into_l{layer + 1}_n{node}']

                # 3) Add v_i_j <= sum((W_i_j + delta)v_i-1_j + ... + M xi_i_j)
                gurobi_model.addConstr(qr + 1000 * activation_states[
                    activation_name] >= all_nodes[var_name],
                                       name=f"model{model_idx}_constr3_" + var_name)

                gurobi_model.addConstr(qr <= all_nodes[var_name])
                gurobi_model.update()

        output_node = gurobi_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                          name=f'model{model_idx}_output_node')

        # constraint 1: node <= ub(W)x + ub(B)
        gurobi_model.addConstr(quicksum((
            wp[f'weight_l{len(self.layers) - 2}_n{prev_node_index}_to_l{len(self.layers) - 1}_n0'] *
            all_nodes[
                f"model{model_idx}_v_{len(self.layers) - 2}_{prev_node_index}" if len(
                    self.layers) - 2 else f"v_0_{prev_node_index}"] for prev_node_index in
            range(self.layers[len(self.layers) - 2])
        )) + bp[f'bias_into_l{len(self.layers) - 1}_n0'] == output_node,
                               name=f'model{model_idx}_output_node_C1')

        if not self.neg_value:
            gurobi_model.addConstr(output_node - 0.0001 >= 0,
                                   name=f"model{model_idx}_output_node_lb_>=0")
        else:
            gurobi_model.addConstr(output_node + 0.0001 <= 0,
                                   name=f"model{model_idx}_output_node_ub_<=0")

        gurobi_model.update()
        return gurobi_model

    def _adv_prob(self):
        """
        The adversarial problem or the inner optimisation problem which finds the model parameter perturbations which
        most invalidates the current robust recourse (self.x_prime_current). Add model perturbations to the storage.

        Returns: None

        """
        gurobi_model = Model()
        gurobi_model.setParam('OutputFlag', 0)
        aux_vars = dict()
        node_vars = dict()
        node_vars[0], gurobi_model = self._adv_add_inputs(gurobi_model)
        node_vars, aux_vars, gurobi_model = self._adv_add_nodes(node_vars, aux_vars, gurobi_model)
        # set objective: "minimise" output node
        if not self.neg_value:  # target 1
            gurobi_model.setObjective(node_vars[len(self.layers) - 1][0], GRB.MINIMIZE)
        else:
            gurobi_model.setObjective(node_vars[len(self.layers) - 1][0], GRB.MAXIMIZE)
        gurobi_model.Params.NonConvex = 2
        gurobi_model.optimize()

        bound = gurobi_model.getVarByName("output_node").X
        if bound >= 0 and self.neg_value == 0 or bound < 0 and self.neg_value:
            self.achieved = True
            return

        # not achieved, add worse case perturbation into perturbations
        wp_new = {}
        bp_new = {}
        for k in list(self.orig_w.keys()):
            wp_new[k] = gurobi_model.getVarByName(k).X
        for k in list(self.orig_b.keys()):
            bp_new[k] = gurobi_model.getVarByName(k).X
        self.w_primes.append(wp_new)
        self.b_primes.append(bp_new)

    def _adv_add_inputs(self, gurobi_model):
        """
        Add input variables into the optimisation model for the adv problem

        Args:
            gurobi_model: Gurobi optimisation model

        Returns: Dictionary of input variables, and the updated gurobi optimisation model

        """
        input_vars = {}
        for feat_idx in range(int(self.layers[0])):
            input_vars[feat_idx] = gurobi_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                       name="x_0_" + str(feat_idx))
            gurobi_model.addConstr(input_vars[feat_idx] == float(self.x_prime_current.values[0][feat_idx]))
            gurobi_model.update()
        return input_vars, gurobi_model

    def _adv_add_nodes(self, node_vars, aux_vars, gurobi_model):
        """
        Add forward pass constraints into the gurobi optimisation model for the adv problem

        Args:
            node_vars: Dictionary of node variables
            aux_vars: Dictionary of auxiliary variables
            gurobi_model: Gurobi optimisation model

        Returns: Dictionary of node variables, Dictionary of auxiliary variables, Gurobi optimisation model

        """
        for layer in range(1, len(self.layers)):
            node_var = dict()
            aux_var = dict()
            for node_d in range(self.layers[layer]):
                gurobi_model.update()
                # hidden layers
                w_vars = {}
                for node_s in range(self.layers[layer - 1]):
                    w_var = gurobi_model.addVar(vtype=GRB.CONTINUOUS,
                                                lb=self.orig_w[
                                                       f"weight_l{layer - 1}_n{node_s}_to_l{layer}_n{node_d}"] - self.delta,
                                                ub=self.orig_w[
                                                       f"weight_l{layer - 1}_n{node_s}_to_l{layer}_n{node_d}"] + self.delta,
                                                name=f"weight_l{layer - 1}_n{node_s}_to_l{layer}_n{node_d}")
                    w_vars[(node_s, node_d)] = w_var
                # Bi = Bi +- delta
                b_var = gurobi_model.addVar(vtype=GRB.CONTINUOUS,
                                            lb=self.orig_b[f'bias_into_l{layer}_n{node_d}'] - self.delta,
                                            ub=self.orig_b[f'bias_into_l{layer}_n{node_d}'] + self.delta,
                                            name=f'bias_into_l{layer}_n{node_d}')
                if layer != len(self.layers) - 1:
                    node_var[node_d] = gurobi_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS)
                    aux_var[node_d] = gurobi_model.addVar(vtype=GRB.BINARY)
                    gurobi_model.update()
                    # constraint 1: node >= 0
                    gurobi_model.addConstr(node_var[node_d] >= 0)
                    # constraint 2: node <= M(1-a)
                    gurobi_model.addConstr(1000 * (1 - aux_var[node_d]) >= node_var[node_d])
                    # constraint 3: node <= ub(W)x + ub(B) + Ma
                    gurobi_model.addConstr(
                        node_var[node_d] <= quicksum(w_vars[(node1, node_d)] * node_vars[layer - 1][node1]
                                                     for node1 in range(self.layers[layer - 1])) + b_var + 1000 *
                        aux_var[
                            node_d])
                    gurobi_model.addConstr(
                        node_var[node_d] >= quicksum(w_vars[(node1, node_d)] * node_vars[layer - 1][node1]
                                                     for node1 in range(self.layers[layer - 1])) + b_var)
                else:
                    node_var[node_d] = gurobi_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS,
                                                           name='output_node')
                    # constraint 1: node <= ub(W)x + ub(B)
                    gurobi_model.addConstr(quicksum(
                        ((self.orig_w[f"weight_l{layer - 1}_n{node1}_to_l{layer}_n{node_d}"] + self.delta) *
                         node_vars[layer - 1][node1]) for
                        node1 in range(self.layers[layer - 1])) + self.orig_b[
                                               f'bias_into_l{layer}_n{node_d}'] + self.delta >= node_var[node_d])
                    # constraint 2: node >= lb(W)x + lb(B)
                    gurobi_model.addConstr(quicksum(
                        ((self.orig_w[f"weight_l{layer - 1}_n{node1}_to_l{layer}_n{node_d}"] - self.delta) *
                         node_vars[layer - 1][node1]) for
                        node1 in range(self.layers[layer - 1])) + self.orig_b[
                                               f'bias_into_l{layer}_n{node_d}'] - self.delta <= node_var[node_d])
                    gurobi_model.update()
            node_vars[layer] = node_var
            if layer != (len(self.layers) - 1):
                aux_vars[layer] = aux_var
        return node_vars, aux_vars, gurobi_model
