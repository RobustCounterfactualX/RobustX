from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.recourse_methods.NNCE import NNCE


def test_kdtree_nnce(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "logistic regression")
    res = ct.generate(["KDTreeNNCE"])

    assert not res["KDTreeNNCE"][0].empty


def test_kdtree_nnce_same_as_nnce(testing_models) -> None:
    # ct, dl, _ = testing_models.get(Dataset.RECRUITMENT, ModelType.NEURALNET, 10, 7, 1) #TODO
    ct = testing_models.get("recruitment", "recruitment", "pytorch", 10, 7, 1)

    kdrecourse = KDTreeNNCE(ct)
    nncerecourse = NNCE(ct)
    negs = ct.dataset.get_negative_instances()

    for _, neg in negs.iterrows():
        a = kdrecourse.generate_for_instance(neg)
        b = nncerecourse.generate_for_instance(neg)

        assert a.index == b.index
