import numpy as np
import pandas as pd
import torch

def test_correct_recourses_generated_for(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE", "BinaryLinearSearch"])
    assert not ces["MCE"][0].empty
    assert not ces["BinaryLinearSearch"][0].empty

def test_correct_evaluations_generated(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE", "BinaryLinearSearch"])
    evals = ct.evaluate(["MCE"], ["Distance"])
    assert isinstance(evals["MCE"]["Distance"], np.float64)

# def test_correct_robustness_evaluations_generated(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate(["MCE", "BinaryLinearSearch"])
#     evals = ct.evaluate(["MCE"], ["Distance"])
#     assert not ces["MCE"]["Distance"].empty

# def test_generate_np_conversion(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate(["MCE"], "NPArray")
#     assert isinstance(ces[0][0], np.ndarray)
    
def test_generate_df_conversion(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE"], "DataFrame")
    assert isinstance(ces["MCE"][0], pd.DataFrame)

#TODO
# def test_generate_torch_conversion(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate(["MCE"], "Tensor")
#     assert isinstance(ces[0][0], torch.Tensor)
    


