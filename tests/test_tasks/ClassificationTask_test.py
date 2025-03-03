import numpy as np
import pandas as pd


def test_correct_recourses_generated_for(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE", "BinaryLinearSearch"])
    assert not ces["MCE"][0].empty
    assert not ces["BinaryLinearSearch"][0].empty

def test_correct_evaluations_generated(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE", "BinaryLinearSearch"])
    evals = ct.evaluate(["MCE", "BinaryLinearSearch"], ["Distance", "Validity"])
    assert isinstance(evals["MCE"]["Distance"], np.float64)

def test_correct_robustness_evaluations_generated(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE", "BinaryLinearSearch"])
    evals = ct.evaluate(["MCE"], ["DeltaRobustnessEvaluator"])
    assert len(evals["MCE"]) == 1

def test_robustness_and_standards_evaluations_generated(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    ces = ct.generate(["MCE", "BinaryLinearSearch", "RNCE"])
    evals = ct.evaluate(["MCE", "RNCE"], ["DeltaRobustnessEvaluator", "Distance", "Validity"])

    assert len(evals["MCE"]) == 3

# def test_recourse_generated_for_all_methods_if_no_methods_list_provided(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate()
#     assert len(ces) == len(ct.get_recourse_methods())
#
# def test_evaluations_generated_for_all_methods_if_no_methods_list_provided(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate()
#     evals = ct.evaluate(evaluations=["Distance"])
#     assert len(evals) == len(ct.get_recourse_methods())
#
# def test_evaluations_generated_for_all_evaluation_metrics_if_no_evaluations_list_provided(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     eval_metrics = [metric for metric in ct.get_evaluation_metrics() if 'multiplicity' not in metric.lower()]
#
#     ces = ct.generate()
#     evals = ct.evaluate(methods=["MCE"])
#     assert len(evals["MCE"]) == len(eval_metrics)

# def test_visualisation_radar_chart(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate(["MCE", "BinaryLinearSearch"])
#     evals = ct.evaluate(["MCE", "BinaryLinearSearch"], ["Distance", "Validity", "RobustnessProportionEvaluator", "ExtraMetric"], visualisation=True)
#     assert evals is not None  # Ensure evaluation does not return None
#     # No assertion for plots since they are visual, but no errors should occur

def test_visualisation_bar_chart(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    ces = ct.generate(["MCE", "BinaryLinearSearch"])
    evals = ct.evaluate(["MCE", "BinaryLinearSearch"], ["Distance", "Validity"], visualisation=True)
    assert evals is not None  # Ensure evaluation does not return None

def test_evalutes_empty_results(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    evals = ct.evaluate([], ["Distance", "Validity"])
    assert evals == {}  # Ensure empty evaluations return empty dictionary

def test_invalid_metrics_raises_exception(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    try:
        ct.evaluate(["MCE"], ["InvalidMetric"], visualisation=True)
    except ValueError as e:
        assert "Invalid evaluation metrics" in str(e)

def test_generate_df_conversion(testing_models) -> None:
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    ces = ct.generate(["MCE"], "DataFrame")
    assert isinstance(ces["MCE"][0], pd.DataFrame)

#TODO
# def test_generate_torch_conversion(testing_models) -> None:
#     ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
#     ces = ct.generate(["MCE"], "Tensor")
#     assert isinstance(ces["MCE"][0], torch.Tensor)
#


