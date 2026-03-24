from lib import base
from lib.functions_2d import circle_2d


def test_circle_400():
    base.seed()
    result = circle_2d(200)
    result.show(classification=True)
    assert 0 < result.loss < 0.12
