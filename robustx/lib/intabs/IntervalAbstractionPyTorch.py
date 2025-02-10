from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel


class IntervalAbstractionPytorch:
    """
    Converts a provided PyTorch neural network into a representation of an Interval Neural Network

        ...

    Attributes / Properties
    -------

    bias_intervals: Dict[str, (int, int)]
        The intervals for each of the biases in the NN with key 'bias_into_l{layer_idx}_n{dest_idx}', where input layer
        has a layer_idx of 0

    weight_intervals: Dict[str, (int, int)]
        The intervals for each of the weights in the NN with key 'weight_into_l{layer_idx}_n{dest_idx}', where input
        layer has a layer_idx of 0

    layers: list[int]
        Stores the number of nodes in each layer in a list

    model: BaseModel
        The original model

    delta: int
        The perturbation to each weight in the Neural Network

    bias_delta: int
        The perturbation to each bias in the Neural Network

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are robust for the given parameters

    -------
    """
    def __init__(self, model: SimpleNNModel, delta: float, bias_delta=None):
        """
        @param model: SimpleNNModel, the Neural network to create an INN of
        @param delta: int, perturbation to weights
        @param bias_delta: int, perturbation to bias, default is delta itself
        """
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
        Generates the intervals for each weight and bias in the Neural Network
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
                bias_key = f'bias_into_l{layer_idx + 1}_n{dest_idx}'
                bias_dict[bias_key] = [biases[dest_idx] - self.bias_delta, biases[dest_idx] + self.bias_delta]

                for src_idx in range(weights.shape[1]):
                    # Set the interval for weights
                    weight_key = f'weight_l{layer_idx}_n{src_idx}_to_l{layer_idx + 1}_n{dest_idx}'
                    weight = weights[dest_idx, src_idx]
                    weight_dict[weight_key] = [weight - self.delta, weight + self.delta]

        return weight_dict, bias_dict
