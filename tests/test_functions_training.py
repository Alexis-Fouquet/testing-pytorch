from lib import base
from lib.functions import linear_classic, linear_classic_noised, mean_value


def test_train_linear_1():
    base.seed()
    assert 3 > linear_classic().loss > 0.3


def test_train_linear_200():
    base.seed()
    assert 0.2 < linear_classic(50).loss < 0.3


def test_train_linear_400():
    base.seed()
    assert 0 < linear_classic(400).loss < 0.1


def test_noise_linear_400():
    base.seed()
    assert 0 < linear_classic_noised(50).loss < 0.1


def test_mean_1():
    base.seed()
    assert 3 > mean_value().loss > 0.5


def test_mean_200():
    base.seed()
    assert 0 < mean_value(400).loss < 0.61


def test_mean_400():
    base.seed()
    assert 0 < mean_value(1600).loss < 0.12
