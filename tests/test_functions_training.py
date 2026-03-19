from lib import base
from lib.functions import almost_linear, linear_classic, linear_classic_noised, mean_value


def test_train_linear_1():
    base.seed()
    assert 3 > linear_classic().loss > 0.15


def test_train_linear_200():
    base.seed()
    assert 0.1 < linear_classic(2).loss < 0.2


def test_train_linear_400():
    base.seed()
    assert 0 < linear_classic(40).loss < 0.1


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
    assert 3 > almost_linear().loss > 0.2


def test_max_200():
    base.seed()
    assert 0.1 < almost_linear(2).loss < 0.3


def test_max_400():
    base.seed()
    assert 0 < almost_linear(40).loss < 0.1
