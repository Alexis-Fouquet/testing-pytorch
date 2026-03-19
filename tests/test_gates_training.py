from lib import base
from lib.gates import gates_and_2, gates_and_5, gates_complex_v1, gates_or_5


def test_and_1():
    base.seed()
    assert 3 > gates_and_2().loss > 0.4


def test_and_200():
    base.seed()
    assert 0 < gates_and_2(100).loss < 0.5


def test_and_400():
    base.seed()
    assert 0 < gates_and_2(400).loss < 0.10


def test_and5_1():
    base.seed()
    assert 3 > gates_and_5().loss > 0.4


def test_and5_200():
    base.seed()
    assert 0 < gates_and_5(100).loss < 0.5


def test_and5_400():
    base.seed()
    assert 0 < gates_and_5(400).loss < 0.10


def test_or_1():
    base.seed()
    assert 3 > gates_or_5().loss > 0.3


def test_or_200():
    base.seed()
    assert 0 < gates_or_5(100).loss < 0.5


def test_or_400():
    base.seed()
    assert 0 < gates_or_5(400).loss < 0.10


def test_gv1_1():
    base.seed()
    assert 3 > gates_complex_v1().loss > 0.3


def test_gv1_200():
    base.seed()
    assert 0 < gates_complex_v1(100).loss < 0.5


def test_gv1_400():
    base.seed()
    assert 0 < gates_complex_v1(400).loss < 0.10
