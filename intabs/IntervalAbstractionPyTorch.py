from models.pytorch_models.SimpleNNModel import SimpleNNModel


class IntervalAbstractionPytorch:
    def __init__(self, model: SimpleNNModel, delta: float, bias_delta=None):
        self.layers = [model.input_dim] + model.hidden_dim + [model.output_dim]
        self.model = model
        self.delta = delta
        if bias_delta is None:
            self.bias_delta = delta
        else:
            self.bias_delta = bias_delta
        self.weight_intervals, self.bias_intervals = self.create_weights_and_bias_dictionary()

    def create_weights_and_bias_dictionary(self):
        """
        layer 0 is input layer
        """

        # Extract the weights and biases as numpy arrays for each layer
        params = {}
        for name, param in self.model.get_torch_model().named_parameters():
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
                bias_key = f'bias_into_l{layer_idx+1}_n{dest_idx}'
                bias_dict[bias_key] = [biases[dest_idx] - self.bias_delta, biases[dest_idx] + self.bias_delta]

                for src_idx in range(weights.shape[1]):

                    # Set the interval for weights
                    weight_key = f'weight_l{layer_idx}_n{src_idx}_to_l{layer_idx + 1}_n{dest_idx}'
                    weight = weights[dest_idx, src_idx]
                    weight_dict[weight_key] = [weight - self.delta, weight + self.delta]

        return weight_dict, bias_dict
