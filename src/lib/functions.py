from torch import rand

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
    in_training, in_test = x[part_sep:], x[:part_sep]
    out_training, out_test = y[part_sep:], y[:part_sep]

    model = ClassicModel(1, 1, global_device)
    loss = train(model, in_training, out_training, in_test, out_test, epochs=epochs)
    return TrainingResult(model, loss)

def linear_classic_noised():
    pass

def almost_linear():
    pass

def other_v1():
    pass
