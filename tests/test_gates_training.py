from lib import base
from lib.gates import gates_and_2


def test_and_1():
    base.seed()
    assert 3 > gates_and_2().loss > 0.4


def test_and_200():
    base.seed()
    assert 0 < gates_and_2(400).loss < 0.5


def test_and_400():
    base.seed()
    assert 0 < gates_and_2(3000).loss < 0.25
