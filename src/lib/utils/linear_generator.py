from torch import Size, randn
from torch._prims_common import Tensor


def add_noise(y: Tensor, noise: float = 0):
    return randn(y.shape) * noise


class LinearGenerator:
    def __init__(self, init: float, coef: float) -> None:
        self.init = init
        self.coef = coef

    def generate(self, x: Tensor, noise: float = 0) -> Tensor:
        y = add_noise(x * self.coef + self.init, noise)
        return y

