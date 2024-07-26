from lib.distance_functions.DistanceFunctions import euclidean
from recourse_methods.RecourseGenerator import RecourseGenerator
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, concatenate, ReLU
from scipy.optimize import minimize
import pandas as pd


def construct_product_network(models):
    inputs = Input(shape=(models[0].input_shape[1],))

    outputs = [model(inputs) for model in models]
    combined_output = concatenate(outputs)

    intermediate_layer1 = ReLU()(Dense(2 * len(models))(combined_output))
    intermediate_layer2 = ReLU()(Dense(2 * len(models))(intermediate_layer1))
    final_output = Dense(len(models) + 1)(intermediate_layer2)

    product_model = Model(inputs=inputs, outputs=final_output)
    return product_model


class ProductNetwork(RecourseGenerator):

    def generate_for_instance(self, instance, distance_func="euclidean", custom_distance_func=None, neg_value=0,
                              column_name="target") -> pd.DataFrame:

        models = self.task.models

        product_model = construct_product_network(models)

        def constraint_function(x_prime, product_model, target_class):
            # Ensure the product model classifies x_prime as the target class
            return np.argmax(product_model.predict(np.array([x_prime]))[0]) == target_class

        # Find counterfactual
        def find_counterfactual(instance, product_model, target_class):
            result = minimize(
                euclidean,
                x0=instance,
                args=(instance),
                constraints={'type': 'eq',
                             'fun': lambda x_prime: constraint_function(x_prime, product_model, target_class)}
            )
            return result.x

        # Assume the target class is 1 - original class
        original_class = np.argmax(models[0].predict(instance)[0])  # Assuming all models agree on the initial class
        target_class = 1 - original_class

        # Find and print the counterfactual
        counterfactual = find_counterfactual(instance, product_model, target_class)

        return counterfactual
    # Function to construct the product network


