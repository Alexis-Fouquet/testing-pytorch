from lib import base
from lib.build_models.gates import (
    gates_and_2,
    gates_and_5,
    gates_complex_v1,
    gates_or_5,
    gates_xor,
)


def test_and_400():
    base.seed()
    result = gates_and_2(400)
    result.show(classification=True)
    assert 0 < result.loss < 0.1


def test_and5_400():
    base.seed()
    result = gates_and_5(400)
    result.show(plot=False)
    assert 0 < result.loss < 0.10


def test_or_400():
    base.seed()
    result = gates_or_5(400)
    result.show(plot=False)
    assert 0 < result.loss < 0.10


def test_gv1_400():
    base.seed()
    result = gates_complex_v1(400)
    result.show(plot=False)
    assert 0 < result.loss < 0.1


def test_xor():
    base.seed()
    result = gates_xor(400)
    result.show(classification=True)
    if 0 < result.loss < 0.1:
        return
    print(result.model)
    assert False, result.loss
