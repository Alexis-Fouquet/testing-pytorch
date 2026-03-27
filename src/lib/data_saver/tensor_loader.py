from typing import Sized, cast, override, Self

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from lib.data_saver.loader import BaseLoader
from lib.device import global_device


class DeviceTensorLoader(BaseLoader):
    def __init__(self, x: Tensor, y: Tensor, shuffle: bool = False) -> None:
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x.to(global_device, non_blocking=True)
        self.y = y.to(global_device, non_blocking=True)
        self.iter = False
        self.shuffle = shuffle

    @override
    def __len__(self) -> int:
        # Always one batch
        return 1

    @override
    def __iter__(self) -> Self:
        self.iter = True
        return self

    @override
    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.iter:
            self.iter = False
            return self.x, self.y
        raise StopIteration


def dataset_to_dtl(data: Dataset, shuffle: bool = False) -> DeviceTensorLoader:
    size = len(cast(Sized, data))
    # Note: cannot pin if already on GPU
    # Note: cannot use workers with pytest
    loader = DataLoader(
        data,
        batch_size=size,
        shuffle=False,
        pin_memory=False,
        num_workers=5,
    )

    for x, y in loader:
        return DeviceTensorLoader(x, y, shuffle=shuffle)
    raise RuntimeError
