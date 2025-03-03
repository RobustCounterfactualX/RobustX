from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.recourse_methods.APAS import APAS
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from sklearn.model_selection import train_test_split


def test_apas(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    res = ct.generate(["APAS"])
    assert not res["APAS"][0].empty

def test_apas_delta_robustness(testing_models) -> None:
    dl = get_example_dataset("ionosphere")
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    counterfactual_label = 0

    # create a train-test split
    target_column = "target"
    X_train, X_test, y_train, y_test = train_test_split(dl.data.drop(columns=[target_column]), dl.data[target_column],
                                                        test_size=0.2, random_state=0)

    # retrieves a random instance from the training data that does not produce the specified counterfactual_label value, i.e., a valid instance
    pos_instance = X_test[y_test == counterfactual_label].iloc[0]
    if ct.model.predict_single(pos_instance) == counterfactual_label:
        print("The selected instance is valid")

    # instanciate the robust_recourse_generator method. The APAS method is used to generate robust recourse
    confidence = 0.999
    robust_ce_generator = APAS(ct, KDTreeNNCE, confidence)

    # generate robust recourse
    delta = 0.05
    robust_ce = robust_ce_generator._generation_method(pos_instance, target_column="target",
                                                       desired_outcome=counterfactual_label, delta_max=delta)

    if robust_ce is None:
        print(f"\nNo counterfactual explanation robust to Δ={delta} model changes was found.")
    else:
        print(
            f"\nA counterfactual explanation robust to Δ={delta} model changes with probability ≥ {round((confidence) * 100, 4)}% is:\n",
            robust_ce)
        print("\nwith prediction: ", ct.model.predict_single(robust_ce))

