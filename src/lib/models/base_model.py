from torch import nn


class BaseModel(nn.Module):
    def __init__(self, device: str) -> None:
        """
        A simple base enabling to get the device of the model.
        May change in the future.
        """

        super().__init__()
        self.device_str = device
