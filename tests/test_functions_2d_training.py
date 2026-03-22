from lib import base
from lib.functions_2d import Cos2D, circle_2d


def test_circle_400():
    base.seed()
    result = circle_2d(200)
    result.show(classification=True)
    assert 0 < result.loss < 0.12


def test_cos_circle_400():
    cos = Cos2D(epochs=4000, layers=[4, 8, 9, 10, 11, 12, 13, 14])
    loss = cos.train()
    assert 0 < loss < 0.2, loss
