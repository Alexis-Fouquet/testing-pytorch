from typing import Any, Mapping
from torch import Tensor, nn
from lib.modules.base_model import BaseModel
from torch.nn.modules.loss import _Loss


class SeqModel(BaseModel):
    """
    A neural network (layer) with a sigmoid.
    """

    def __init__(self, models: list, device: str, loss: _Loss | None = None) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        assert len(models) > 0
        internal_loss = models[0].fn_loss if loss is None else loss
        BaseModel.__init__(self, device, internal_loss)
        self.models = models
        self.internal_loss = internal_loss
        # TODO
        # self.internal_seq = cast(nn.Module, compile(nn.Sequential(*models).to(device)))
        self.internal_seq = nn.Sequential(*models).to(device)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        return self.internal_seq.load_state_dict(state_dict, strict, assign)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        return self.internal_seq.state_dict(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.internal_seq(x)

    @property
    def fn_loss(self):
        return self.internal_loss
