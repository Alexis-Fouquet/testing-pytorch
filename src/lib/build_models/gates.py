from torch import Size, Tensor, bernoulli, concat, ones

from lib.data.tensor_data import TensorDatasetSaved
from lib.device import global_device
from lib.base import train
from lib.modules.sequential import SeqModel
from lib.modules.sigmoid import SigmoidModel
from lib.training_params import TrainingParams


def gates_and_2(epochs: int = 1):
    """
    Trying to simulate a simple and gate between two inputs
    """

    points = 50
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
        TensorDatasetSaved(in_training, out_training),
        TensorDatasetSaved(in_test, out_test),
        "and2",
        TrainingParams(epochs=epochs, lr=0.4),
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
        TensorDatasetSaved(in_training, out_training),
        TensorDatasetSaved(in_test, out_test),
        "and4n1",
        TrainingParams(epochs=epochs, lr=0.4),
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
        TensorDatasetSaved(in_training, out_training),
        TensorDatasetSaved(in_test, out_test),
        "or4n1",
        TrainingParams(epochs=epochs, lr=0.4),
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
        TensorDatasetSaved(in_training, out_training),
        TensorDatasetSaved(in_test, out_test),
        "gate1",
        TrainingParams(epochs=epochs, lr=0.4),
    )


def gates_xor(epochs: int = 1):
    """
    Trying to simulate a xor gate
    """

    points = 8
    part = 0.5

    p = Tensor([[0, 0], [1, 0], [0, 1], [1, 1]]).to(global_device)
    x = concat((p, p))
    assert x.size() == Size([points, 2]), (x.size(), x, p)
    y = (x[:, 0] > 0.9) ^ (x[:, 1] > 0.9)
    y = y.float().unsqueeze(dim=1)
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

    # Difficult with only 2
    hidden = 4
    model = SeqModel(
        [
            SigmoidModel(2, hidden, global_device),
            SigmoidModel(hidden, 1, global_device),
        ],
        global_device,
    )
    return train(
        model,
        TensorDatasetSaved(in_training, out_training),
        TensorDatasetSaved(in_test, out_test),
        "xor2",
        TrainingParams(epochs=epochs, lr=1.5),
    )
