import pytest
import reid_mvp


def test_sum_as_string():
    assert reid_mvp.sum_as_string(1, 1) == "2"
