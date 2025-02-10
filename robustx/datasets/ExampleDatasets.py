from ..datasets.provided_datasets.AdultDatasetLoader import AdultDatasetLoader
from ..datasets.provided_datasets.IonosphereDatasetLoader import IonosphereDatasetLoader
from ..datasets.provided_datasets.IrisDatasetLoader import IrisDatasetLoader
from ..datasets.provided_datasets.TitanicDatasetLoader import TitanicDatasetLoader


def get_example_dataset(name: str, seed=None):
    """
    Returns a DatasetLoader class given the name of an example dataset

    @param name: the name of the dataset you wish to load, the options are:
                 - iris
                 - ionosphere
                 - adult
                 - titanic

    @return: DatasetLoader
    """
    if name == "iris":
        ds = IrisDatasetLoader(seed)
        ds.load_data()
        return ds
    elif name == "ionosphere":
        ds = IonosphereDatasetLoader(seed)
        ds.load_data()
        return ds
    elif name == "adult":
        ds = AdultDatasetLoader(seed)
        ds.load_data()
        return ds
    elif name == "titanic":
        ds = TitanicDatasetLoader(seed)
        ds.load_data()
        return ds
    else:
        raise ValueError(f"Unknown dataset: {name}")
