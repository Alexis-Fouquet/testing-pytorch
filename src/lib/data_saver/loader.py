from abc import abstractmethod
from typing import Self
from torch import Tensor


class BaseLoader:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Self:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __next__(self) -> tuple[Tensor, Tensor]:
        pass
