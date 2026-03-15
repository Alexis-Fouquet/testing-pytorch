from torch import nn


class BaseModel(nn.Module):
    def __init__(self, device: str) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        super().__init__()
        self.device_str = device
