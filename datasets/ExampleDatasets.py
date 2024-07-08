from datasets.provided_datasets.IonosphereDatasetLoader import IonosphereDatasetLoader
from datasets.provided_datasets.IrisDatasetLoader import IrisDatasetLoader


def get_example_dataset(name: str):
    """
    Returns a DatasetLoader class given the name of an example dataset
    :param name: the name of the dataset you wish to load,the options are:
               - iris
               - ionosphere
    :return: DatasetLoader
    """
    if name == "iris":
        ds = IrisDatasetLoader()
        ds.load_data()
        return ds
    elif name == "ionosphere":
        ds = IonosphereDatasetLoader()
        ds.load_data()
        return ds
    else:
        raise ValueError(f"Unknown dataset: {name}")


