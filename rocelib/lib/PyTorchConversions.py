import tensorflow as tf
import torch
import torch.nn as nn


### Work in Progress ###
def keras_to_pytorch(keras_model):
    """
    Converts a given keras model to a PyTorch model - still a work in progress
    @param keras_model: model to convert
    @return: PyTorch model
    """
    # Create an empty list to hold PyTorch layers
    layers = []

    # Loop through each layer in the Keras model
    for layer in keras_model.layers:
        # Map Keras Dense layers to PyTorch Linear layers
        if isinstance(layer, tf.keras.layers.Dense):
            input_dim = layer.input_shape[-1]
            output_dim = layer.units
            layers.append(nn.Linear(input_dim, output_dim))

        # Map Keras Conv2D layers to PyTorch Conv2D layers
        elif isinstance(layer, tf.keras.layers.Conv2D):
            in_channels = layer.input_shape[-1]
            out_channels = layer.filters
            kernel_size = layer.kernel_size
            padding = layer.padding
            stride = layer.strides
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding))

        # Map Keras Activation layers to PyTorch activation functions
        elif isinstance(layer, tf.keras.layers.Activation):
            activation = layer.activation
            if activation == tf.keras.activations.relu:
                layers.append(nn.ReLU())
            elif activation == tf.keras.activations.sigmoid:
                layers.append(nn.Sigmoid())
            elif activation == tf.keras.activations.softmax:
                layers.append(nn.Softmax(dim=1))

        # Map Keras Flatten layers to PyTorch Flatten layers
        elif isinstance(layer, tf.keras.layers.Flatten):
            layers.append(nn.Flatten())

        # Map Keras MaxPooling2D to PyTorch MaxPool2d
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            pool_size = layer.pool_size
            stride = layer.strides
            padding = layer.padding
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=padding))

    # Create a PyTorch sequential model from the list of layers
    pytorch_model = nn.Sequential(*layers)

    # Transfer weights from Keras to PyTorch
    for keras_layer, pytorch_layer in zip(keras_model.layers, pytorch_model):
        if isinstance(keras_layer, tf.keras.layers.Dense):
            pytorch_layer.weight.data = torch.from_numpy(keras_layer.get_weights()[0].T)  # Transpose weight
            pytorch_layer.bias.data = torch.from_numpy(keras_layer.get_weights()[1])

        elif isinstance(keras_layer, tf.keras.layers.Conv2D):
            pytorch_layer.weight.data = torch.from_numpy(keras_layer.get_weights()[0].transpose(3, 2, 0,
                                                                                                1))
            pytorch_layer.bias.data = torch.from_numpy(keras_layer.get_weights()[1])

    return pytorch_model
