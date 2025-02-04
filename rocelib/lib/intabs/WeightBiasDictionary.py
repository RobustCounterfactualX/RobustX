def create_weights_and_bias_dictionary(model):
    """
    Util function for extracting the dictionary for model parameters. layer 0 is input layer
    """

    # Extract the weights and biases as numpy arrays for each layer
    params = {}
    for name, param in model.get_torch_model().named_parameters():
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
