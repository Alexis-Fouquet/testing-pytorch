from lib.functions import linear_classic


def test_train_linear_1():
    assert 1 > linear_classic().loss > 0.6


def test_train_linear_200():
    assert 0.2 < linear_classic(200).loss < 0.6


def test_train_linear_400():
    assert 0 < linear_classic(400).loss < 0.2
