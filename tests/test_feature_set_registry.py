import pytest

from quant_platform.feature_set_registry import get_feature_set, list_feature_sets

pytestmark = pytest.mark.no_database


def test_governed_feature_set_is_pinned_and_hashed() -> None:
    feature_set = get_feature_set("governed-baseline")
    assert len(feature_set["features"]) == 20
    assert feature_set["features"]["KLEN"] == "($high-$low)/$open"
    assert feature_set["definition_sha256"] == (
        "893a3ff486b93a3b83e460616b73eb958d33c605d6ff91eee52b6a716b649256"
    )
    assert list_feature_sets()[0]["definition_sha256"] == feature_set["definition_sha256"]


def test_unified_feature_sets_are_available_without_changing_alpha20() -> None:
    assert len(get_feature_set("qlib-alpha158")["features"]) == 158
    assert len(get_feature_set("qlib-alpha360")["features"]) == 360
    assert len(get_feature_set("platform-seed-v1")["features"]) == 24
    assert len(get_feature_set("unified-research-v1")["features"]) == 533
    assert get_feature_set("qlib-alpha158")["definition_sha256"] == (
        "b2be170df9af46e66490a437ee75fd2ac0a78f09d80a980d26d71c7f55abaf74"
    )
    assert get_feature_set("qlib-alpha360")["definition_sha256"] == (
        "fa8142772fc429bdeb5fb43d57d68f7db57fd472d9b389853be13c9d2e7ca8f8"
    )
