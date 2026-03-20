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


def test_train_linear_1():
    base.seed()
    assert 3 > linear_classic().loss > 0.15


def test_train_linear_200():
    base.seed()
    result = linear_classic(20)
    if 0.1 < result.loss < 0.3:
        return
    result.plot()
    assert False, result.loss


def test_train_linear_400():
    base.seed()
    result = linear_classic(40)
    assert 0 < result.loss < 0.1


def test_noise_linear_400():
    base.seed()
    assert 0 < linear_classic_noised(50).loss < 0.1


def test_mean_1():
    base.seed()
    assert 3 > mean_value().loss > 0.1


def test_mean_200():
    base.seed()
    assert 0 < mean_value(50).loss < 0.61


def test_mean_400():
    base.seed()
    assert 0 < mean_value(500).loss < 0.12


def test_max_1():
    base.seed()
    result = almost_linear()
    if 3 > result.loss > 0.2:
        return
    result.plot()
    assert False, result.loss


def test_max_200():
    base.seed()
    assert 0.1 < almost_linear(2).loss < 0.3


def test_max_400():
    base.seed()
    result = almost_linear(40)
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss


def test_min_1():
    base.seed()
    result = min_linear()
    if 3 > result.loss > 0.2:
        return
    result.plot()
    assert False, result.loss


def test_min_200():
    base.seed()
    result = min_linear(40)
    if 0.1 < result.loss < 0.25:
        return
    result.plot()
    assert False, result.loss


def test_min_400():
    base.seed()
    result = min_linear(80)
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss


def test_min_max_1():
    base.seed()
    result = min_max_linear()
    if 3 > result.loss > 0.2:
        return
    result.plot()
    assert False, result.loss


def test_min_max_200():
    base.seed()
    result = min_max_linear(40)
    if 0.1 < result.loss < 0.25:
        return
    result.plot()
    assert False, result.loss


def test_min_max_400():
    base.seed()
    result = min_max_linear(250)
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss


def test_cos_1():
    base.seed()
    result = cos_fct()
    if 3 > result.loss > 0.2:
        return
    result.plot()
    assert False, result.loss


def test_cos_200():
    base.seed()
    result = cos_fct(40)
    if 0.1 < result.loss < 0.25:
        return
    result.plot()
    assert False, result.loss


def test_cos_400():
    base.seed()
    result = cos_fct(200)
    if 0 < result.loss < 0.1:
        return
    result.plot()
    assert False, result.loss
