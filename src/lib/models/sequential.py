from torch import nn
from lib.models.base_model import BaseModel


class SeqModel(BaseModel, nn.Sequential):
    """
    A neural network (layer) with a sigmoid.
    """

    def __init__(self, models: list, device: str) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        BaseModel.__init__(self, device, models[0].fn_loss)
        self.models = models
        nn.Sequential.__init__(self, *models)

    @property
    def fn_loss(self):
        return self.models[0].fn_loss
