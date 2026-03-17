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

        super().__init__(device)
        assert input_size > 0
        assert output_size > 0
        self.input_size = input_size
        self.output_size = output_size

        self.biases = nn.Parameter(
            randn(output_size, device=device), requires_grad=True
        )
        self.weights = nn.Parameter(
            randn([output_size, input_size], device=device), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        if len(x.size()) > 2:
            x = x.squeeze()
        if len(x.size()) < 2:
            x = x.unsqueeze(dim=0)
        return self.weights @ x + self.biases
