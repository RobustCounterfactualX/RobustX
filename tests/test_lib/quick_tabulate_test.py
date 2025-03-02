# NO LONGER RELEVANT WITH NEW GENERATION AND EVALUATION
# def test_quick_tabulate():
#     #
#     # methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
#     # model = TrainablePyTorchModel(10, [8], 1)
#     # dl = CsvDatasetLoader(csv="../assets/standardized_recruitment_data.csv", target_column="HiringDecision")
#     # subset = dl.get_negative_instances(column_name="HiringDecision", neg_value=0).head(20)
#     #
#     # quick_tabulate(dl, model, methods, subset=subset, preprocess=False, neg_value=0, column_name="HiringDecision", delta=0.01)

#     methods = {"BLS": BinaryLinearSearch, "MCE": MCE, "MCER": MCER, "Wachter": Wachter, "RNCE": RNCE}
#     model = TrainablePyTorchModel(34, [10], 1)
#     dl = get_example_dataset("ionosphere")
#     # dl.default_preprocess()

#     quick_tabulate(dl, model, methods, neg_value=0, column_name="target", delta=0.05)

#     assert True