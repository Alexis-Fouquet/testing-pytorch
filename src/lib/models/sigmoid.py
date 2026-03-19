from torch import nn, randn, sigmoid, round
from torch._prims_common import Tensor
from lib.models.base_model import BaseModel


class SigmoidModel(BaseModel):
    """
    A neural network (layer) with a sigmoid.
    """

    def __init__(self, input_size: int, output_size: int, device: str) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        super().__init__(device, nn.BCELoss())
        assert input_size > 0
        assert output_size > 0
        self.input_size = input_size
        self.output_size = output_size

        self.biases = nn.Parameter(
            randn(output_size, device=device), requires_grad=True
        )
        self.weights = nn.Parameter(
            randn([input_size, output_size], device=device), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x @ self.weights + self.biases)

    def print_epoch(
        self, truth: Tensor, out: Tensor, loss_tr: Tensor, loss_te: Tensor, epoch: int
    ):
        out = (out >= 0.5).float()
        accu = round((out == truth).float().mean() * 10000) / 100
        print(
            f"Epoch {epoch} with tr {loss_tr} and te {loss_te} -- Acc {accu}% of {truth.shape}"
        )
