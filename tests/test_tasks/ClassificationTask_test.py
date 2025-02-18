import numpy as np
import pandas as pd
import torch


def test_generate_np_conversion(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE"], "NPArray")
    assert isinstance(ces[0], np.ndarray)
    
def test_generate_df_conversion(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE"], "DataFrame")
    assert isinstance(ces[0], pd.DataFrame)

def test_generate_torch_conversion(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE"], "Tensor")
    assert isinstance(ces[0], torch.Tensor)
    


