from lib import base
from lib.functions_2d import circle_2d, cos_2d


def test_circle_400():
    base.seed()
    result = circle_2d(200)
    result.show(classification=True)
    assert 0 < result.loss < 0.12


def test_cos_circle_400():
    base.seed()
    result = cos_2d(500)
    result.show(classification=True)
    assert 0 < result.loss < 0.2
