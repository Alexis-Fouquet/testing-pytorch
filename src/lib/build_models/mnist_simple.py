from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from lib.base import train
from lib.device import global_device
from lib.modules.sequential import SeqModel
from lib.modules.sigmoid import SigmoidModel
from lib.search_hparams import ModelTraining
from lib.training_params import TrainingParams
from lib.training_result import TrainingResult


def get_datasets() -> tuple[Dataset, Dataset]:
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    return (training_data, test_data)


class MnistSimple(ModelTraining):
    def internal_train(self, params: TrainingParams) -> TrainingResult | None:
        name = "mnist_simple"

        (training, test) = get_datasets()

        size = 40

        layers_arg = (
            [
                SigmoidModel(28, size, global_device),
            ]
            + [SigmoidModel(size, size, global_device) for _ in range(params.layers)]
            + [
                SigmoidModel(size, 10, global_device),
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

        result = train(model, training, test, name, params)
        params.save(name, model)
        return result
