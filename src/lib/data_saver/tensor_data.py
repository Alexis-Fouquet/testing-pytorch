from torch._prims_common import Tensor
from torch.utils.data import TensorDataset


class TensorDatasetSaved(TensorDataset):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)
        self.x = x.cpu()
        self.y = y.cpu()
