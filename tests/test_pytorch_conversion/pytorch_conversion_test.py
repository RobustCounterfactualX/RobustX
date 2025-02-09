# import tensorflow as tf

# from datasets.ExampleDatasets import get_example_dataset
# from lib.PyTorchConversions import keras_to_pytorch
# from models.pytorch_models.TrainableCustomPyTorchModel import TrainableCustomPyTorchModel
# from models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
# from recourse_methods.KDTreeNNCE import KDTreeNNCE
# from robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
# from tasks.ClassificationTask import ClassificationTask


# def test_dense_relu_conversion():
#     # Keras model with Dense and ReLU layers
#     keras_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(8, input_shape=(34,)),
#         tf.keras.layers.Activation("relu"),
#         tf.keras.layers.Dense(1),
#         tf.keras.layers.Activation("sigmoid")
#     ])

#     # Convert the Keras model to PyTorch
#     pytorch_model = keras_to_pytorch(keras_model)

#     model = TrainablePyTorchModel(34, [8], 1)
#     # dl = CsvDatasetLoader('../assets/recruitment_data.csv', "HiringDecision")
#     dl = get_example_dataset("ionosphere")
#     ct_pytorch = ClassificationTask(model, dl)

#     ct_converted = ClassificationTask(TrainableCustomPyTorchModel(pytorch_model), dl)

#     dl.default_preprocess()

#     ct_pytorch.train()

#     ct_converted.train()

#     kdtree = KDTreeNNCE(ct_pytorch)

#     kdtree_converted = KDTreeNNCE(ct_converted)

#     opt = DeltaRobustnessEvaluator(ct_pytorch)

#     opt_converted = DeltaRobustnessEvaluator(ct_converted)

#     for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
#         ce = kdtree.generate_for_instance(neg)
#         ce_converted = kdtree_converted.generate_for_instance(neg)
#         robust = opt.evaluate(ce, delta=0.0000000000000000001)
#         robust_converted = opt_converted.evaluate(ce_converted, delta=0.0000000000000000001)
#         assert robust == robust_converted
#         print("######################################################")
#         print("CE was: ", ce)
#         print("This CE was" + ("" if robust else " not") + " robust")
#         print("######################################################")

#     # # Verify the layer types in the PyTorch model
#     # assert isinstance(pytorch_model[0], nn.Linear)  # Dense -> Linear
#     # assert isinstance(pytorch_model[1], nn.ReLU)  # ReLU activation
#     # assert isinstance(pytorch_model[2], nn.Linear)  # Dense -> Linear
#     # assert isinstance(pytorch_model[3], nn.ReLU)  # ReLU activation
#     # assert isinstance(pytorch_model[4], nn.Linear)  # Dense -> Linear
#     #
#     # # Check the weight shapes of the first Dense/Linear layer
#     # keras_weights = keras_model.layers[0].get_weights()[0]  # Keras weights of the first Dense layer
#     # pytorch_weights = pytorch_model[0].weight.detach().numpy()
#     # assert keras_weights.T.shape == pytorch_weights.shape
