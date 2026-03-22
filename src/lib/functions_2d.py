from torch import Size, cos, rand, sqrt

from lib.base import train
from lib.device import global_device
from lib.models.sequential import SeqModel
from lib.models.sigmoid import SigmoidModel
from lib.search_hparams import ModelTraining
from lib.training_result import TrainingResult


def circle_2d(epochs: int = 1):
    points = 1000
    part = 0.75

    x = rand([points, 2])
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.1
    y = y.float().unsqueeze(dim=1)
    assert x.size() == Size([points, 2]), x.size()
    assert y.size() == Size([points, 1]), y.size()

    part_sep = (int)(points * part)
    in_training, in_test = x[:part_sep, :], x[part_sep:, :]
    # WARNING: do not add noise in classification (< 0 with crash)
    out_training, out_test = y[:part_sep, :], y[part_sep:, :]

    size = 10
    last_size = 4

    model = SeqModel(
        [
            SigmoidModel(2, size, global_device),
            SigmoidModel(size, size, global_device),
            SigmoidModel(size, last_size, global_device),
            SigmoidModel(last_size, 1, global_device),
        ],
        global_device,
    )
    return train(
        model,
        in_training,
        out_training,
        in_test,
        out_test,
        "2d_circle",
        epochs=epochs,
        lr=0.05,
    )


class Cos2D(ModelTraining):
    def internal_train(self, epochs: int, lr: float, layers: int) -> TrainingResult:
        points = 1000
        part = 0.75

        x = rand([points, 2])
        y = cos(sqrt((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) * 15) > 0
        y = y.float().unsqueeze(dim=1)
        assert x.size() == Size([points, 2]), x.size()
        assert y.size() == Size([points, 1]), y.size()

        part_sep = (int)(points * part)
        in_training, in_test = x[:part_sep, :], x[part_sep:, :]
        # WARNING: do not add noise in classification (< 0 with crash)
        out_training, out_test = y[:part_sep, :], y[part_sep:, :]

        size = 40
        last_size = 10

        # Best result -- why?
        layers_arg = (
            [
                SigmoidModel(2, size, global_device),
            ]
            + [SigmoidModel(size, size, global_device) for _ in range(layers)]
            + [
                SigmoidModel(size, last_size, global_device),
                SigmoidModel(last_size, 1, global_device),
            ]
        )
        model = SeqModel(
            layers_arg,
            global_device,
        )
        return train(
            model,
            in_training,
            out_training,
            in_test,
            out_test,
            "cos_circle",
            epochs=epochs,
            lr=lr,
            layers=layers,
        )
