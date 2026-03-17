from lib.functions import linear_classic, linear_classic_noised, mean_value


def test_train_linear_1():
    assert 3 > linear_classic().loss > 0.6


def test_train_linear_200():
    assert 0.2 < linear_classic(400).loss < 0.6


def test_train_linear_400():
    # 1500 is a lot, just back luck
    assert 0 < linear_classic(1500).loss < 0.21


def test_noise_linear_1():
    assert 3 > linear_classic_noised().loss > 0.6


def test_noise_linear_200():
    assert 0.1 < linear_classic_noised(400).loss < 0.61


def test_noise_linear_400():
    assert 0 < linear_classic_noised(1500).loss < 0.11


def test_mean_1():
    assert 3 > mean_value().loss > 0.6


def test_mean_200():
    assert 0 < mean_value(400).loss < 0.61


def test_mean_linear_400():
    assert 0 < mean_value(1600).loss < 0.12
