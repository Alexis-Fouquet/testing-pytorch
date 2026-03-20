from torch import Size, bernoulli, ones

from lib.device import global_device
from lib.base import train
from lib.models.sigmoid import SigmoidModel


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
    return train(
        model,
        in_training,
        out_training,
        in_test,
        out_test,
        "and2",
        epochs=epochs,
        lr=0.4,
    )


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
    return train(
        model,
        in_training,
        out_training,
        in_test,
        out_test,
        "and4n1",
        epochs=epochs,
        lr=0.4,
    )


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
    return train(
        model,
        in_training,
        out_training,
        in_test,
        out_test,
        "or4n1",
        epochs=epochs,
        lr=0.4,
    )


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
    return train(
        model,
        in_training,
        out_training,
        in_test,
        out_test,
        "gate1",
        epochs=epochs,
        lr=0.4,
    )


def gates_complex_v2():
    pass
