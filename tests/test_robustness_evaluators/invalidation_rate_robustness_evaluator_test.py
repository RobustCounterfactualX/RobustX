# just copied this test from delta_robustness tests to see if it works
from rocelib.evaluations.robustness_evaluations.NE_Robustness_Implementations.InvalidationRateRobustnessEvaluator import \
    InvalidationRateRobustnessEvaluator
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE


def test_ionosphere_kdtree_robustness(testing_models):
    # Instantiate the neural network and the IntervalAbstractionPytorch class
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    kdtree = KDTreeNNCE(ct)

    opt = InvalidationRateRobustnessEvaluator(ct)

    for _, neg in ct.dataset.get_negative_instances(neg_value=0).iterrows():
        ce = kdtree.generate_for_instance(neg)
        robust = opt.evaluate(ce, delta=0.0000000000000000001)
        print("######################################################")
        print("CE was: ", ce)
        print("This CE was" + ("" if robust else " not") + " robust")
        print("######################################################")