from quant_platform.feature_set_registry import get_feature_set, list_feature_sets


def test_governed_feature_set_is_pinned_and_hashed() -> None:
    feature_set = get_feature_set("governed-baseline")
    assert len(feature_set["features"]) == 20
    assert feature_set["features"]["KLEN"] == "($high-$low)/$open"
    assert len(feature_set["definition_sha256"]) == 64
    assert list_feature_sets()[0]["definition_sha256"] == feature_set["definition_sha256"]
