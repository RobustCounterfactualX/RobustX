# TODO
def test_rnce(testing_models):
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    res = ct.generate(["RNCE"])
    assert not res["RNCE"][0].empty

    # recourse = RNCE(ct)
    # res = recourse.generate_for_all(neg_value=0, column_name="target")
    # assert not res.empty
