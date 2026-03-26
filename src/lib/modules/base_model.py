from torch import nn
from torch.nn.modules.loss import _Loss


class BaseModel(nn.Module):
    def __init__(self, device: str, fn_loss: _Loss) -> None:
        """
        A simple base enabling to get the device of the model.
        May change in the future.
        """

        super().__init__()
        self.device_str = device
        self.fn_loss = fn_loss
