from rocelib.evaluations.robustness_evaluations.MC_Robustness_Implementations.ApproximateDeltaRobustnessEvaluator import \
    ApproximateDeltaRobustnessEvaluator
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE


def test_approximate_delta_robustness(testing_models):
    # Instantiate the neural network and the IntervalAbstractionPytorch class
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    kdtree = KDTreeNNCE(ct)

    opt = ApproximateDeltaRobustnessEvaluator(ct)

    for _, neg in ct.dataset.get_negative_instances().iterrows():
        ce = kdtree.generate_for_instance(neg)
        robust = opt.evaluate(ce, delta=0.0000000000000000001)
        print("######################################################")
        print("CE was: ", ce)
        print("This CE was" + ("" if robust else " not") + " robust")
        print("######################################################")
