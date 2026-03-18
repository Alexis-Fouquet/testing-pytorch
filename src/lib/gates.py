from torch import Size, bernoulli, ones

from lib.models.classic import ClassicModel
from lib.base import global_device, train
from lib.training_result import TrainingResult

def gates_and_2(epochs: int = 1):
    """
    Trying to simulate a simple and gate between two inputs
    """

    points = 100
    part = 0.75

    x = bernoulli(ones([2, points]) / 2).to(global_device).float()
    y = (x[0, :] > 0.9) & (x[1, :] > 0.9)
    y = y.float()
    assert x.size() == Size([2, points]), x.size()
    assert y.size() == Size([points]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:, :part_sep], x[:, part_sep:]
    out_training, out_test = y[:part_sep], y[part_sep:]
    assert in_training.size() == Size([2, part_sep]), (
        in_training.size(),
        [2, part_sep],
    )
    assert out_training.size() == Size([part_sep]), (in_training.size(), [part_sep])

    model = ClassicModel(2, 1, global_device)
    loss = train(model, in_training, out_training, in_test, out_test, epochs=epochs)
    return TrainingResult(model, loss)


def gates_and_5():
    pass


def gates_or_5():
    pass


def gates_complex_v1():
    pass


def gates_complex_v2():
    pass
