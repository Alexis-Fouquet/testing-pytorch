from abc import abstractmethod
from torch import nn, Tensor
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

    @abstractmethod
    def print_epoch(
        self, truth: Tensor, out: Tensor, loss_tr: Tensor, loss_te: Tensor, epoch: int
    ):
        pass
