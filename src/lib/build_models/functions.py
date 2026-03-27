from torch import Size, cos, maximum, minimum, pi, rand, randn

from lib.data_saver.tensor_loader import DeviceTensorLoader
from lib.modules.classic import ClassicModel
from lib.base import train
from lib.device import global_device
from lib.modules.sequential import SeqModel
from lib.modules.sigmoid import SigmoidModel
from lib.search_hparams import TrainingParams
from lib.training_result import TrainingResult
from lib.utils.linear_generator import LinearGenerator, add_noise


def linear_classic(epochs: int = 1) -> TrainingResult:
    # Inspired from online examples
    lin = LinearGenerator(0.7, -0.05)
    points = 500
    part = 0.75

    x = rand([points, 1])
    y = lin.generate(x)
    assert x.size() == Size([points, 1]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]

    model = ClassicModel(1, 1, global_device)
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "linear",
        TrainingParams(epochs=epochs),
    )


def linear_classic_noised(epochs: int = 1):
    """
    Calculate a linear function with noise.
    """

    lin = LinearGenerator(0.2, 0.03)
    points = 500
    part = 0.75

    x = rand([points, 1])
    assert x.size() == Size([points, 1]), x.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    # Do not noise the test dataset as we want a linear output
    out_training, out_test = (
        lin.generate(in_training, noise=0.001),
        lin.generate(in_test),
    )
    assert out_training.size() == in_training.size(), out_training.size()
    assert out_test.size() == in_test.size(), out_test.size()

    model = ClassicModel(1, 1, global_device)
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "noised",
        TrainingParams(epochs=epochs),
    )


def mean_value(epochs: int = 1):
    """
    Just calculates the mean value between 2 inputs.
    """

    points = 500
    part = 0.75
    noise = 0.001

    x = rand([points, 2]).to(global_device)
    y = ((x[:, 0] + x[:, 1]) / 2).unsqueeze(dim=1)
    assert y.size() == Size([points, 1]), y.size()
    y = y + randn([points, 1]).to(global_device) * noise
    assert x.size() == Size([points, 2]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]
    assert in_training.size() == Size([part_sep, 2]), (
        in_training.size(),
        [part_sep, 2],
    )
    assert out_training.size() == Size([part_sep, 1]), (in_training.size(), [part_sep])

    model = ClassicModel(2, 1, global_device)
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "mean",
        TrainingParams(epochs=epochs, lr=0.1),
    )


def almost_linear(epochs: int = 1):
    """
    The max between two linear functions.
    """

    lin1 = LinearGenerator(0.05, 0.05)
    lin2 = LinearGenerator(-0.8, 1.8)
    points = 500
    part = 0.75

    x = rand([points, 1])
    y = maximum(lin1.generate(x), lin2.generate(x))
    assert x.size() == Size([points, 1]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]

    # Do not noise the test dataset as we want a linear output
    out_training, out_test = add_noise(y[:part_sep, :], 0.001), y[part_sep:, :]

    model = SeqModel(
        [
            ClassicModel(1, 2, global_device),
            SigmoidModel(2, 1, global_device),
        ],
        global_device,
    )
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "max",
        TrainingParams(epochs=epochs, lr=0.05),
    )


def min_linear(epochs: int = 1):
    """
    The min between two linear functions.
    """

    lin1 = LinearGenerator(0.9, 0.07)
    lin2 = LinearGenerator(1.8, -1.8)
    points = 500
    part = 0.75

    x = rand([points, 1])
    y = minimum(lin1.generate(x), lin2.generate(x))
    assert x.size() == Size([points, 1]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]

    # Do not noise the test dataset as we want a linear output
    out_training, out_test = add_noise(y[:part_sep, :], 0.006), y[part_sep:, :]

    model = SeqModel(
        [
            ClassicModel(1, 1, global_device),
            SigmoidModel(1, 1, global_device),
        ],
        global_device,
    )
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "min",
        TrainingParams(epochs=epochs, lr=0.05),
    )


def min_max_linear(epochs: int = 1):
    """
    The min between two linear functions.
    """

    lin1 = LinearGenerator(0.9, 0.07)
    lin2 = LinearGenerator(1.4, -1.8)
    lin3 = LinearGenerator(-1.0, 2.0)
    points = 500
    part = 0.75

    x = rand([points, 1])
    y = minimum(lin1.generate(x), maximum(lin2.generate(x), lin3.generate(x)))
    assert x.size() == Size([points, 1]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]

    # Do not noise the test dataset as we want a linear output
    out_training, out_test = add_noise(y[:part_sep, :], 0.002), y[part_sep:, :]

    model = SeqModel(
        [
            ClassicModel(1, 2, global_device),
            SigmoidModel(2, 2, global_device),
            SigmoidModel(2, 1, global_device),
        ],
        global_device,
    )
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "minmax",
        TrainingParams(epochs=epochs, lr=0.05),
    )


def cos_fct(epochs: int = 1):
    points = 500
    part = 0.75

    x = rand([points, 1])
    y = (cos(x * 2 * 2 * pi) + 1) / 2
    assert x.size() == Size([points, 1]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]

    # Do not noise the test dataset as we want a linear output
    out_training, out_test = add_noise(y[:part_sep, :], 0.002), y[part_sep:, :]

    size = 10
    last_size = 4
    # Changing the number of layers will lead to a failure
    # Why?
    model = SeqModel(
        [
            ClassicModel(1, size, global_device),
            SigmoidModel(size, size, global_device),
            SigmoidModel(size, size, global_device),
            SigmoidModel(size, size, global_device),
            SigmoidModel(size, last_size, global_device),
            SigmoidModel(last_size, 1, global_device),
        ],
        global_device,
    )
    return train(
        model,
        DeviceTensorLoader(in_training, out_training),
        DeviceTensorLoader(in_test, out_test),
        "cos",
        TrainingParams(epochs=epochs, lr=0.05),
    )
