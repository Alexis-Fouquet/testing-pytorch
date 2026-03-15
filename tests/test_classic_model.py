from lib import base
from lib.models import classic


def test_init():
    base.seed()
    model = classic.ClassicModel(1, 1)
    assert model.input_size == 1
    assert model.output_size == 1
    w = model.weights.squeeze().item()
    b = model.biases.squeeze().item()
    assert round(w, 4) == 0.1288
    assert round(b, 4) == 0.3367
