from torch import nn, randn
from torch import Tensor

from lib.models import base_model


class ClassicModel(base_model.BaseModel):
    """
    A neural network (layer) as simple as possible.
    """

    def __init__(self, input_size: int, output_size: int, device: str) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        super().__init__(device, nn.L1Loss())
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
        return x @ self.weights + self.biases

    def print_epoch(
        self, truth: Tensor, out: Tensor, loss_tr: Tensor, loss_te: Tensor, epoch: int
    ):
        out = out
        truth = truth
        print(f"Epoch {epoch} with tr {loss_tr} and te {loss_te}")
