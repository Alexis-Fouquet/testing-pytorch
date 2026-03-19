from lib import base, device
from lib.models import classic


def test_init():
    base.seed()
    model = classic.ClassicModel(1, 1, device.global_device)
    assert model.input_size == 1
    assert model.output_size == 1
    w = model.weights.squeeze().item()
    b = model.biases.squeeze().item()
    assert 1 >= round(w, 4) >= 0
    assert 1 >= round(b, 4) >= 0
