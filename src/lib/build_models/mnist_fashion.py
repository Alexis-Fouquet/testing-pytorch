from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Lambda

from lib.base import train
from lib.data_saver.tensor_loader import dataset_to_dtl
from lib.device import global_device
from lib.modules.sequential import SeqModel
from lib.modules.sigmoid import SigmoidModel
from lib.search_hparams import ModelTraining
from lib.training_params import TrainingParams
from lib.training_result import TrainingResult


def transform(x: Tensor) -> Tensor:
    # There is only one input each time
    return x.flatten()


def get_datasets() -> tuple[Dataset, Dataset]:
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose(
            [
                ToTensor(),
                Lambda(transform),
            ]
        ),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=Compose(
            [
                ToTensor(),
                Lambda(transform),
            ]
        ),
    )

    return (training_data, test_data)


class MnistFashion(ModelTraining):
    def internal_train(self, params: TrainingParams) -> TrainingResult | None:
        name = "mnist_fashion"

        training, test = get_datasets()
        training, test = dataset_to_dtl(training), dataset_to_dtl(test)

        size = 512

        layers_arg = (
            [
                SigmoidModel(28 * 28, size, global_device),
            ]
            + [SigmoidModel(size, size, global_device) for _ in range(params.layers)]
            + [
                SigmoidModel(size, 10, global_device),
            ]
        )
        model = SeqModel(layers_arg, global_device, loss=CrossEntropyLoss())

        if params.save_exist(name):
            params.already_trained(name)
            return None

        result = train(model, training, test, name, params)
        params.save(name, model)
        return result
