from torch import nn
from lib.models.base_model import BaseModel


class SigmoidModel(BaseModel, nn.Sequential):
    """
    A neural network (layer) with a sigmoid.
    """

    def __init__(self, models: list[BaseModel], device: str) -> None:
        """
        Creates the neural network layer, with the sizes indicated.
        """

        BaseModel.__init__(self, device, models[0].fn_loss)
        nn.Sequential.__init__(self, *models)
