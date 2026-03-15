import torch
from torch import nn
from torch import Tensor


class ClassicModel(nn.Module):
    """
    A neural network (layer) as simple as possible.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        super().__init__()
        assert input_size > 0
        assert output_size > 0
        self.input_size = input_size
        self.output_size = output_size

        self.biases = nn.Parameter(torch.randn(output_size), requires_grad=True)
        self.weights = nn.Parameter(
            torch.randn([input_size, output_size]), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.weights * x + self.biases
