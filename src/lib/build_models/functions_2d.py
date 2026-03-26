from rich import print
from torch import Size, cos, rand, sqrt, dstack

from lib.base import train
from lib.data.tensor_data import TensorDatasetSaved
from lib.device import global_device
from lib.modules.sequential import SeqModel
from lib.modules.sigmoid import SigmoidModel
from lib.training_params import TrainingParams
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
        TensorDatasetSaved(in_training, out_training),
        TensorDatasetSaved(in_test, out_test),
        "2d_circle",
        TrainingParams(epochs=epochs, lr=0.05),
    )


class Cos2D(ModelTraining):
    def internal_train(self, params: TrainingParams) -> TrainingResult | None:
        name = "cos_circle"
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

        layers_arg = (
            [
                SigmoidModel(2, size, global_device),
            ]
            + [SigmoidModel(size, size, global_device) for _ in range(params.layers)]
            + [
                SigmoidModel(size, last_size, global_device),
                SigmoidModel(last_size, 1, global_device),
            ]
        )
        model = SeqModel(
            layers_arg,
            global_device,
        )

        if params.save_exist(name):
            print(f"> Model {params.get_full_name(name)} already trained")
            params.load(name, model)
            return None
        result = train(
            model,
            TensorDatasetSaved(in_training, out_training),
            TensorDatasetSaved(in_test, out_test),
            name,
            params,
        )
        params.save(name, model)
        return result


class ThreeElements(ModelTraining):
    def internal_train(self, params: TrainingParams) -> TrainingResult | None:
        name = "three_elements"
        points = 1000
        part = 0.75

        x = rand([points, 2])
        y0 = sqrt((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2) > 0.3
        y1 = y0 & (x[:, 0] > 0.5)
        y1 = y1.float()
        y2 = y0 & (x[:, 0] < 0.5)
        y2 = y2.float()
        y = dstack([y1, y2]).squeeze()
        assert x.size() == Size([points, 2]), x.size()
        assert y.size() == Size([points, 2]), y.size()

        part_sep = (int)(points * part)
        in_training, in_test = x[:part_sep, :], x[part_sep:, :]
        out_training, out_test = y[:part_sep, :], y[part_sep:, :]

        size = 40

        layers_arg = (
            [
                SigmoidModel(2, size, global_device),
            ]
            + [SigmoidModel(size, size, global_device) for _ in range(params.layers)]
            + [
                SigmoidModel(size, 2, global_device),
            ]
        )
        model = SeqModel(
            layers_arg,
            global_device,
        )

        if params.save_exist(name):
            print(f"> Model {params.get_full_name(name)} already trained")
            params.load(name, model)
            return None
        result = train(
            model,
            TensorDatasetSaved(in_training, out_training),
            TensorDatasetSaved(in_test, out_test),
            name,
            params,
        )
        params.save(name, model)
        return result
