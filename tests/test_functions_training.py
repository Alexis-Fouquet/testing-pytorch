from lib import base
from lib.functions import (
    almost_linear,
    cos_fct,
    linear_classic,
    linear_classic_noised,
    mean_value,
    min_linear,
    min_max_linear,
)


def test_train_linear_400():
    base.seed()
    result = linear_classic(40)
    result.show()
    assert 0 < result.loss < 0.1


def test_noise_linear_400():
    base.seed()
    result = linear_classic_noised(50)
    result.show()
    assert 0 < result.loss < 0.1


def test_mean_400():
    base.seed()
    result = mean_value(200)
    result.show(classification=True)
    assert 0 < result.loss < 0.12


def test_max_400():
    base.seed()
    result = almost_linear(40)
    result.show()
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss


def test_min_400():
    base.seed()
    result = min_linear(80)
    result.show()
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss


def test_min_max_400():
    base.seed()
    result = min_max_linear(250)
    result.show()
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss


def test_cos_400():
    base.seed()
    result = cos_fct(200)
    result.show()
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss
