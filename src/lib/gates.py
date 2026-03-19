from torch import Size, bernoulli, ones

from lib.base import global_device, train
from lib.models.sigmoid import SigmoidModel
from lib.training_result import TrainingResult


def gates_and_2(epochs: int = 1):
    """
    Trying to simulate a simple and gate between two inputs
    """

    points = 100
    part = 0.75

    x = bernoulli(ones([points, 2]) / 2).to(global_device).float()
    y = (x[:, 0] > 0.9) & (x[:, 1] > 0.9)
    y = y.float().unsqueeze(dim=1)
    assert x.size() == Size([points, 2]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]
    assert in_training.size() == Size([part_sep, 2]), (
        in_training.size(),
        [part_sep, 2],
    )
    assert out_training.size() == Size([part_sep, 1]), (
        in_training.size(),
        [part_sep, 1],
    )

    model = SigmoidModel(2, 1, global_device)
    loss = train(
        model, in_training, out_training, in_test, out_test, epochs=epochs, lr=0.4
    )
    return TrainingResult(model, loss)


def gates_and_5(epochs: int = 1):
    """
    Trying to simulate a simple and gate between four inputs (1 noised)
    """

    points = 100
    part = 0.75

    x = bernoulli(ones([points, 5]) / 2).to(global_device).float()
    y = (x[:, 0] > 0.9) & (x[:, 1] > 0.9) & (x[:, 2] > 0.9) & (x[:, 3] > 0.9)
    y = y.float().unsqueeze(dim=1)
    assert x.size() == Size([points, 5]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]
    assert in_training.size() == Size([part_sep, 5]), (
        in_training.size(),
        [part_sep, 5],
    )
    assert out_training.size() == Size([part_sep, 1]), (
        in_training.size(),
        [part_sep, 1],
    )

    model = SigmoidModel(5, 1, global_device)
    loss = train(
        model, in_training, out_training, in_test, out_test, epochs=epochs, lr=0.4
    )
    return TrainingResult(model, loss)


def gates_or_5(epochs: int = 1):
    """
    Trying to simulate a simple or gate between four inputs (1 noised)
    """

    points = 100
    part = 0.75

    x = bernoulli(ones([points, 5]) / 2).to(global_device).float()
    y = (x[:, 0] > 0.9) | (x[:, 2] > 0.9) | (x[:, 3] > 0.9)
    y = y.float().unsqueeze(dim=1)
    assert x.size() == Size([points, 5]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]
    assert in_training.size() == Size([part_sep, 5]), (
        in_training.size(),
        [part_sep, 5],
    )
    assert out_training.size() == Size([part_sep, 1]), (
        in_training.size(),
        [part_sep, 1],
    )

    model = SigmoidModel(5, 1, global_device)
    loss = train(
        model, in_training, out_training, in_test, out_test, epochs=epochs, lr=0.4
    )
    return TrainingResult(model, loss)


def gates_complex_v1(epochs: int = 1):
    """
    Trying to simulate a complex gate
    """

    points = 100
    part = 0.75

    x = bernoulli(ones([points, 5]) / 2).to(global_device).float()
    y = (x[:, 0] > 0.9) & (x[:, 2] > 0.9) | (x[:, 3] > 0.9)
    y = y.float().unsqueeze(dim=1)
    assert x.size() == Size([points, 5]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]
    assert in_training.size() == Size([part_sep, 5]), (
        in_training.size(),
        [part_sep, 5],
    )
    assert out_training.size() == Size([part_sep, 1]), (
        in_training.size(),
        [part_sep, 1],
    )

    model = SigmoidModel(5, 1, global_device)
    loss = train(
        model, in_training, out_training, in_test, out_test, epochs=epochs, lr=0.4
    )
    return TrainingResult(model, loss)


def gates_complex_v2():
    pass
