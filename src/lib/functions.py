from torch import Size, rand, randn

from lib.models.classic import ClassicModel
from lib.base import global_device, train
from lib.training_result import TrainingResult


def linear_classic(epochs: int = 1) -> TrainingResult:
    # Inspired from online examples
    init = 0.7
    direction = -0.05
    points = 500
    part = 0.75

    x = (rand([points]) + 1) / 2
    y = x * direction + init

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep], x[part_sep:]
    out_training, out_test = y[:part_sep], y[part_sep:]

    model = ClassicModel(1, 1, global_device)
    loss = train(model, in_training, out_training, in_test, out_test, epochs=epochs)
    return TrainingResult(model, loss)


def linear_classic_noised(epochs: int = 1):
    """
    Calculate a linear function with noise.
    """

    init = 0.2
    direction = 0.03
    points = 500
    part = 0.75
    noise = 0.001

    x = (rand([points]) + 1) / 2
    y = x * direction + init + randn([points]) * noise

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep], x[part_sep:]
    out_training, out_test = y[:part_sep], y[part_sep:]

    model = ClassicModel(1, 1, global_device)
    loss = train(model, in_training, out_training, in_test, out_test, epochs=epochs)
    return TrainingResult(model, loss)


def mean_value(epochs: int = 1):
    """
    Just calculates the mean value between 2 inputs.
    """

    points = 500
    part = 0.75
    noise = 0.001

    x = (rand([2, points]).to(global_device) + 1) / 2
    y = (x[0, :] + x[1, :]).squeeze() / 2 + randn([points]).to(global_device) * noise
    assert x.size() == Size([2, points]), x.size()
    assert y.size() == Size([points]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:, :part_sep], x[:, part_sep:]
    out_training, out_test = y[:part_sep], y[part_sep:]
    assert in_training.size() == Size([2, part_sep]), (in_training.size(), [2, part_sep])
    assert out_training.size() == Size([part_sep]), (in_training.size(), [part_sep])

    model = ClassicModel(2, 1, global_device)
    loss = train(model, in_training, out_training, in_test, out_test, epochs=epochs)
    return TrainingResult(model, loss)


def almost_linear():
    # Needs multiple layers
    pass


def other_v1():
    # Needs multiple layers
    pass
